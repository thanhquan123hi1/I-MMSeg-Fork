"""Regression checks for the original model's single-GPU training adapter.

Run the full production-model CUDA check explicitly:
I_MMSEG_RUN_MODEL_SMOKE=1 python -m pytest -q -s tests/test_single_gpu.py
I_MMSEG_SMOKE_BATCH_SIZE controls the physical batch (default 1).
"""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.nn import functional as F

import trainer
from utils import ContrastiveLoss, DiceLoss

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("original_training", Path(__file__).with_name("reference_original.py"))
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
torch.set_num_threads(4)


def inputs(batch=2, groups=1):
    features = [torch.randn(batch, c, s, s, requires_grad=True)
                for c, s in [(8, 16), (12, 8), (16, 4)]]
    texts = [torch.randn(4 * groups, c, requires_grad=True) for c in [8, 12, 16]]
    labels = torch.zeros(batch, 32, 32, dtype=torch.long)
    labels[:, :16, 16:] = 1
    labels[:, 16:, :16] = 2  # legacy: scar
    labels[:, 16:, 16:] = 3  # legacy: edema
    return features, texts, labels


@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("ignore", [[2, 3], [0]])
def test_contrastive_loss_and_gradients_equal_original(groups, ignore):
    features, texts, labels = inputs(groups=groups)
    ref_features = [t.detach().clone().requires_grad_() for t in features]
    ref_texts = [t.detach().clone().requires_grad_() for t in texts]
    torch.manual_seed(1234)
    actual = ContrastiveLoss()(features, labels, texts, ignore, sample_num=1)
    torch.manual_seed(1234)
    expected = original.ContrastiveLoss(prompt_num=groups)(ref_features, labels, ref_texts, ignore, sample_num=1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    expected.backward()
    for a, b in zip(features + texts, ref_features + ref_texts):
        assert torch.isfinite(a.grad).all()
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


@pytest.mark.parametrize("batch", [1, 16])
def test_no_eligible_anchors_has_zero_connected_gradients(batch):
    features, texts, labels = inputs(batch=batch)
    loss = ContrastiveLoss()(features, torch.zeros_like(labels), texts, [0])
    assert loss.requires_grad and loss.item() == 0
    loss.backward()
    for tensor in features + texts:
        assert tensor.grad is not None and torch.count_nonzero(tensor.grad) == 0


def test_one_empty_scale_does_not_discard_valid_scales():
    features = [torch.randn(1, 8, s, s, requires_grad=True) for s in [8, 4, 2]]
    texts = [torch.randn(4, 8, requires_grad=True) for _ in range(3)]
    labels = torch.zeros(1, 8, 8, dtype=torch.long)
    labels[0, 1, 1] = 2  # disappears in the two coarser nearest-neighbor targets
    actual = ContrastiveLoss()(features, labels, texts, [0], sample_num=1)
    expected = original.ContrastiveLoss(prompt_num=1)(features[:1], labels, texts[:1], [0], sample_num=1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    assert torch.count_nonzero(features[0].grad) > 0
    assert torch.count_nonzero(features[1].grad) == 0


@pytest.mark.parametrize("rows", [0, 3, 5])
def test_reject_incomplete_text_groups(rows):
    features, texts, labels = inputs()
    texts[0] = torch.randn(rows, 8)
    with pytest.raises(ValueError, match="complete ordered groups"):
        ContrastiveLoss()(features, labels, texts, [0])


@pytest.mark.parametrize("value", [-1, 4, 0.5, float("nan")])
def test_reject_invalid_labels(value):
    features, texts, labels = inputs()
    labels = labels.float()
    labels[0, 0, 0] = value
    with pytest.raises(ValueError, match="integer class IDs"):
        ContrastiveLoss()(features, labels, texts, [0])


def test_reject_mismatched_text_dimension_or_scale_count():
    features, texts, labels = inputs()
    with pytest.raises(ValueError, match="three feature/text scales"):
        ContrastiveLoss()(features, labels, texts[:2], [0])
    texts[0] = torch.randn(4, 9)
    with pytest.raises(ValueError, match="dimensions must match"):
        ContrastiveLoss()(features, labels, texts, [0])


def test_dice_loss_and_gradient_unchanged_and_quiet(capsys):
    logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    target = torch.randint(0, 4, (2, 8, 8))
    baseline = logits.detach().clone().requires_grad_()
    actual = DiceLoss(4)(logits, target, softmax=True)
    assert capsys.readouterr().out == ""
    expected = original.DiceLoss(4)(baseline, target, softmax=True)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(logits.grad, baseline.grad, rtol=0, atol=0)


def test_protected_model_and_data_files_unchanged():
    manifest = json.loads(Path(__file__).with_name("fidelity_manifest.json").read_text())
    for name, expected in manifest["protected_files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name


def test_default_configuration():
    tree = ast.parse((ROOT / "train.py").read_text(encoding="utf-8"))
    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            for kw in node.keywords:
                if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                    defaults[node.args[0].value] = kw.value.value
    for key, expected in {"--n_gpu": 1, "--batch_size": 16, "--base_lr": 0.0003333,
                          "--img_size": 128, "--max_epochs": 300,
                          "--start_contrast_epoch": 20, "--region_fusion_start_epoch": 100}.items():
        assert defaults[key] == expected
    assert 'os.environ["CUDA_VISIBLE_DEVICES"]' not in (ROOT / "train.py").read_text()


class ConstantDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        return 20

    def __getitem__(self, index):
        return {**{k: torch.zeros(1, 8, 8) for k in ["image", "image1", "image2"]},
                "label": torch.zeros(8, 8, dtype=torch.long)}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Conv2d(1, 4, 1)

    def forward(self, a, b, c, do_contrast):
        logits = self.head(a)
        return logits, (logits, logits, logits), (None, None, None), None


class RecordingWriter:
    instances = []

    def __init__(self, *args, **kwargs):
        self.images = []
        self.scalars = []
        self.__class__.instances.append(self)

    def add_scalar(self, *args):
        self.scalars.append(args)

    def add_image(self, name, image, step):
        assert torch.isfinite(image).all()
        self.images.append((name, image.detach().clone(), step))

    def close(self):
        pass


def trainer_args(batch=1):
    return SimpleNamespace(base_lr=0.0003333, num_classes=4, batch_size=batch, n_gpu=1,
                           root_path="unused", root_path1="unused", root_path2="unused",
                           list_dir="unused", img_size=128, seed=1234, max_epochs=1,
                           start_contrast_epoch=20, contrast_sample_num=10,
                           contrast_w=0.1, region_fusion_start_epoch=100)


def test_single_image_tensorboard_and_compact_log(tmp_path, monkeypatch, caplog):
    import datasets.dataset_Myops as dataset_module
    monkeypatch.setattr(dataset_module, "Myops_dataset", ConstantDataset)
    monkeypatch.setattr(trainer, "SummaryWriter", RecordingWriter)
    caplog.set_level("INFO")
    # Exercise trainer control flow on CPU; production .cuda() behavior is untouched.
    with patch.object(torch.Tensor, "cuda", lambda self, *a, **k: self):
        trainer.trainer_Myops(trainer_args(), TinyModel(), str(tmp_path))
    writer = RecordingWriter.instances[-1]
    assert len(writer.scalars) == 20 * 3
    assert len(writer.images) == 3
    assert torch.count_nonzero(writer.images[0][1]) == 0
    assert sum("mean_batch_loss" in r.message for r in caplog.records) == 1
    assert not any(r.message.startswith("iteration ") for r in caplog.records)


def test_logging_changes_preserve_optimizer_updates(tmp_path, monkeypatch, capsys):
    import datasets.dataset_Myops as dataset_module
    monkeypatch.setattr(dataset_module, "Myops_dataset", ConstantDataset)
    monkeypatch.setattr(trainer, "SummaryWriter", RecordingWriter)
    monkeypatch.setattr(original, "SummaryWriter", RecordingWriter)
    torch.manual_seed(987)
    initial = TinyModel().state_dict()
    models = []
    for name, fn in [("old", original.trainer_Myops), ("new", trainer.trainer_Myops)]:
        folder = tmp_path / name
        folder.mkdir()
        torch.manual_seed(1234)
        random.seed(1234)
        np.random.seed(1234)
        model = TinyModel()
        model.load_state_dict(initial)
        with patch.object(torch.Tensor, "cuda", lambda self, *a, **k: self):
            fn(trainer_args(batch=2), model, str(folder))
        models.append(model)
    for key, value in models[0].state_dict().items():
        torch.testing.assert_close(value, models[1].state_dict()[key], rtol=0, atol=0)
    capsys.readouterr()


@pytest.mark.skipif(os.environ.get("I_MMSEG_RUN_MODEL_SMOKE") != "1" or not torch.cuda.is_available(),
                    reason="Opt-in production-model CUDA smoke check")
def test_production_model_training_stages():
    from networks.vit_seg_modeling import CONFIGS, VisionTransformer
    batch = int(os.environ.get("I_MMSEG_SMOKE_BATCH_SIZE", "1"))
    torch.manual_seed(1234)
    model = VisionTransformer(copy.deepcopy(CONFIGS["R50-ViT-B_16"]), img_size=128, num_classes=4).cuda().train()
    assert sum(p.numel() for p in model.parameters()) == 84975359
    signature = hashlib.sha256(json.dumps([(n, list(t.shape)) for n, t in model.state_dict().items()]).encode()).hexdigest()
    expected = json.loads(Path(__file__).with_name("fidelity_manifest.json").read_text())["state_dict_shape_sha256"]
    assert signature == expected
    images = [torch.rand(batch, 1, 128, 128, device="cuda") for _ in range(3)]
    target = torch.zeros(batch, 128, 128, dtype=torch.long, device="cuda")
    target[:, :64, 64:] = 1
    target[:, 64:, :64] = 2
    target[:, 64:, 64:] = 3
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003333, weight_decay=0.0001)
    torch.cuda.reset_peak_memory_stats()
    for epoch in [0, 21, 100]:
        optimizer.zero_grad(set_to_none=True)
        out, aux, feature_groups, texts = model(*images, do_contrast=epoch > 20)
        assert out.shape == (batch, 4, 128, 128)
        loss_out = 0.2 * F.cross_entropy(out, target) + 0.8 * DiceLoss(4)(out, target, softmax=True)
        ce_aux = torch.zeros(1, device="cuda")
        dice_aux = torch.zeros(1, device="cuda")
        for pred in aux:
            ce_aux += F.cross_entropy(pred, target)
            dice_aux += DiceLoss(4)(pred, target, softmax=True)
        loss_aux = 0.2 * ce_aux + 0.8 * dice_aux
        contrast = sum(ContrastiveLoss()(f, target, texts, ignore, sample_num=10)
                       for f, ignore in zip(feature_groups, ([2, 3], [0], [0]))) / 3 if epoch > 20 else 0
        loss = loss_out * 0.0 + loss_aux + contrast * 0.1 if epoch < 100 else loss_out + 0.5 * loss_aux + 0.5 * contrast * 0.1
        assert torch.isfinite(loss).all()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        print(f"epoch={epoch}, batch={batch}, loss={loss.item():.6f}, finite_gradients={len(gradients)}")
        del out, aux, feature_groups, texts, loss, loss_out, loss_aux, contrast, ce_aux, dice_aux, gradients, pred
    print(f"GPU={torch.cuda.get_device_name(0)}, torch={torch.__version__}, peak_allocated={torch.cuda.max_memory_allocated()}")
