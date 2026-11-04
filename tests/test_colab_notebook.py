"""Execute local notebook setup/validation with synthetic official-shaped caches."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

import h5py
import nbformat
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def notebook():
    return nbformat.read(ROOT / "I_MMSeg_Colab_Training.ipynb", as_version=4)


def code_containing(marker):
    return next(c.source for c in notebook().cells if c.cell_type == "code" and marker in c.source)


def test_notebook_schema_and_python_syntax():
    nb = notebook()
    nbformat.validate(nb)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"cell_{i}", "exec")
            assert not cell.outputs and cell.execution_count is None


def test_source_zip_verification_and_rerun_protection(tmp_path):
    payload = b"checked source\n"
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as f:
        f.writestr("train.py", payload)
        f.writestr("COLAB_SOURCE_SHA256.json", json.dumps({
            "baseline_commit": "fixture", "files": {"train.py": hashlib.sha256(payload).hexdigest()}}))
    folder = tmp_path / "code"
    before = Path.cwd()
    scope = {"CODE_ZIP": archive, "REPO_DIR": folder, "zipfile": zipfile,
             "json": json, "hashlib": hashlib, "os": os}
    try:
        exec(code_containing("manifest_path ="), scope)
        assert (folder / "train.py").read_bytes() == payload
        (folder / "train.py").write_bytes(b"unrelated user changes")
        with pytest.raises(RuntimeError, match="train.py"):
            exec(code_containing("manifest_path ="), scope)
        assert (folder / "train.py").read_bytes() == b"unrelated user changes"
    finally:
        os.chdir(before)


def test_legacy_cache_preflight_and_canonical_rejection(tmp_path):
    (tmp_path / "list").mkdir()
    for filename in ["train.txt", "test_vol.txt"]:
        shutil.copyfile(ROOT / "list" / filename, tmp_path / "list" / filename)
    shutil.copytree(ROOT / "text_features", tmp_path / "text_features")
    image = np.zeros((128, 128), dtype=np.float32)
    label = np.zeros((128, 128), dtype=np.uint8)
    label[32:64, 32:64] = 2
    label[64:96, 64:96] = 3
    npz = tmp_path / "prototype.npz"
    np.savez_compressed(npz, image=image, label=label, label_order="legacy")
    h5 = tmp_path / "prototype.npy.h5"
    with h5py.File(h5, "w") as f:
        f["image"] = image[:, :, None]
        f["label"] = label[:, :, None]
        f.attrs["label_order"] = "legacy"
    data_root = tmp_path / "data"
    for modality in ["bSSFP", "LGE", "T2w"]:
        for split, manifest, suffix, prototype in [
            ("train_npz", "train.txt", ".npz", npz),
            ("test_vol_h5", "test_vol.txt", ".npy.h5", h5),
        ]:
            folder = data_root / modality / split
            folder.mkdir(parents=True)
            for name in (ROOT / "list" / manifest).read_text().splitlines():
                if name.strip():
                    shutil.copyfile(prototype, folder / (name.strip() + suffix))
    scope = {"REPO_DIR": tmp_path, "DATA_ROOT": data_root, "LABEL_ORDER": "legacy", "torch": torch}
    exec(code_containing("legacy_orders ="), scope)
    assert scope["DATA_VALIDATED"]
    assert np.array_equal(np.load(npz)["label"], label)  # no canonical remapping
    first = (data_root / "bSSFP" / "train_npz" /
             ((ROOT / "list" / "train.txt").read_text().splitlines()[0] + ".npz"))
    first.unlink()
    np.savez_compressed(first, image=image, label=label, label_order="canonical")
    with pytest.raises(ValueError, match="legacy"):
        exec(code_containing("legacy_orders ="), scope)


def test_optional_original_test_adapter_preserves_checkpoint(tmp_path, monkeypatch):
    import runpy
    name = 'TU_pretrain_R50-ViT-B_16_skip3_epo300_bs16_lr0.0003333_128'
    folder = tmp_path / 'weights' / name
    folder.mkdir(parents=True)
    checkpoint = folder / 'epoch_299.pth'
    torch.save({'head.weight': torch.ones(4, 1, 1, 1)}, checkpoint)
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    called = []

    def original_test(path, run_name):
        state = torch.load(checkpoint, weights_only=True)
        assert list(state) == ['module.head.weight']
        assert torch.equal(state['module.head.weight'], torch.ones(4, 1, 1, 1))
        called.append((path, run_name))

    monkeypatch.setattr(runpy, 'run_path', original_test)
    scope = {'RUN_DIR': tmp_path, 'REPO_DIR': ROOT, 'BATCH_SIZE': 16, 'BASE_LR': 0.0003333,
             'IMG_SIZE': 128, 'MAX_EPOCHS': 300, 'SEED': 1234,
             'torch': torch, 'Path': Path, 'sys': sys}
    source = code_containing('RUN_LEGACY_TEST =').replace('RUN_LEGACY_TEST = False', 'RUN_LEGACY_TEST = True')
    argv_before = sys.argv[:]
    exec(source, scope)
    assert called and sys.argv == argv_before
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before
