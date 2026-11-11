import json

import nibabel as nib
import numpy as np
import pytest
import torch

from datasets.myops_plus import EXPORT_ID, check_patients, export_samples, model_labels
from utils import calculate_metric_percase, test_single_volume as evaluate_volume


def test_raw_and_canonical_masks_match_original_prompt_indices():
    raw = np.array([0, 200, 2221, 1220, 500, 600], dtype=np.uint16)
    canonical = np.array([0, 1, 3, 2, 0, 0], dtype=np.uint8)
    for source, is_raw in ((raw, True), (canonical, False)):
        original = source.copy()
        np.testing.assert_array_equal(model_labels(source, raw=is_raw), [0, 1, 2, 3, 0, 0])
        np.testing.assert_array_equal(source, original)
    with pytest.raises(ValueError, match="label codes"):
        model_labels(np.array([2, 3]), raw=True)
    with pytest.raises(ValueError, match="label codes"):
        model_labels(raw, raw=False)


def test_export_train_and_test_read_same_modalities_and_targets(tmp_path):
    patients = [f"Case{i}" for i in (*range(2001, 2036), *range(3001, 3046))]
    splits = {"train": patients[:64], "test_vol": patients[64:]}
    (tmp_path / "splits.json").write_text(json.dumps(dict(dataset_id=EXPORT_ID, patients=splits)))
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(dict(dataset_id=EXPORT_ID, patients=patients)))
    with pytest.raises(ValueError, match="distinct"):
        check_patients(dict(train=splits["train"], test_vol=[patients[0]] + patients[65:]))
    for split, name in ((s, names[0]) for s, names in splits.items()):
        case = tmp_path / "cases" / ("CenterB" if name.startswith("Case2") else "CenterC") / name
        case.mkdir(parents=True)
        (case / "metadata.json").write_text(json.dumps(dict(case=name, split=split)))
        affine = np.diag([-1., 1., 12., 1.])
        for i, suffix in enumerate(("C0", "LGE", "T2", "gd"), 1):
            array = np.full((128, 128, 1), 0.25 * i, np.float32)
            if suffix == "gd":
                array[:] = 0
                array[0, :2, 0] = [2221, 1220]
            nii = nib.Nifti1Image(array, affine)
            nii.header.set_xyzt_units("mm")
            nib.save(nii, case / f"{name}_{suffix}.nii.gz")
        sample = next(export_samples(tmp_path, split))
        for key, value in zip(("image", "image1", "image2"), (.25, .5, .75)):
            np.testing.assert_array_equal(sample[key], value)
        np.testing.assert_array_equal(sample["label"].squeeze()[0, :2], [2, 3])
        if split == "test_vol":
            assert sample["label"].shape == (128, 128, 1)
            np.testing.assert_array_equal(sample["spacing"], [1, 1, 12])
            np.testing.assert_array_equal(sample["affine"], affine)


def test_roi_hd95_and_nifti_export_use_physical_grid_and_scar_ids(tmp_path, monkeypatch):
    class PredictLabels(torch.nn.Module):
        def forward(self, image, *_):
            return torch.nn.functional.one_hot(image[:, 0].long(), 4).permute(0, 3, 1, 2).float()

    first = np.zeros((4, 4, 2), dtype=bool)
    first[2, 2, 0] = True
    assert calculate_metric_percase(first, np.flip(first, axis=2), [1, 1, 12]) == (0., 12.)
    gt = torch.zeros((1, 4, 4, 2), dtype=torch.uint8)
    gt[0, 0, 0, 0], gt[0, 1, 0, 1] = 2, 3
    affine = np.diag([-1., 1., 12., 1.])
    affine[:3, 3] = [52, 75, 24]
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    _, pred = evaluate_volume(gt.float(), gt, gt, gt, PredictLabels(), 4, [4, 4], str(tmp_path),
                              "roi", spacing=[1, 1, 12], roi_affine=affine, return_prediction=True)
    np.testing.assert_array_equal(pred, gt[0])
    for suffix in ("gt", "pred"):
        nii = nib.load(tmp_path / f"roi_{suffix}.nii.gz")
        np.testing.assert_array_equal(nii.affine, affine)
        assert nii.header.get_xyzt_units()[0] == "mm"
        np.testing.assert_array_equal(np.asarray(nii.dataobj), np.array([0, 1, 3, 2])[pred])
