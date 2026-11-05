"""Regression checks for empty-mask scoring and case-level aggregation."""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from utils import (aggregate_case_metrics, calculate_metric_percase,
                   calculate_volume_metrics, evaluation_regions)
import utils as segmentation_utils


def test_both_empty_metrics_are_undefined():
    empty = np.zeros((8, 8, 8), dtype=bool)
    assert np.isnan(calculate_metric_percase(empty.copy(), empty.copy())).all()


@pytest.mark.parametrize("voxel_count", [1, 199, 200, 201, 1000])
@pytest.mark.parametrize("prediction_empty", [True, False])
def test_one_empty_is_zero_dice_regardless_of_size(voxel_count, prediction_empty):
    empty = np.zeros((10, 10, 10), dtype=bool)
    foreground = empty.copy()
    foreground.flat[:voxel_count] = True
    pred, gt = (empty, foreground) if prediction_empty else (foreground, empty)
    dice, hd95 = calculate_metric_percase(pred, gt)
    assert dice == 0.0
    assert np.isnan(hd95)


@pytest.mark.parametrize("scenario,expected", [
    ("perfect", (1.0, 0.0)),
    ("partial", (0.5, 1.0)),
    ("disjoint", (0.0, 2.0)),
])
def test_nonempty_masks_keep_actual_overlap_and_surface_distances(scenario, expected):
    pred = np.zeros((5, 5, 5), dtype=bool)
    pred[2, 2, 1] = True
    gt = np.zeros_like(pred)
    if scenario == "perfect":
        gt[:] = pred
    elif scenario == "partial":
        pred[2, 2, 2] = True
        gt[2, 2, 2:4] = True
    else:
        gt[2, 2, 3] = True
    np.testing.assert_allclose(calculate_metric_percase(pred, gt), expected, rtol=0, atol=0)


def test_average_keeps_missed_case_dice_and_separate_denominators():
    values = [[(0.8, 4.0)], [(0.0, np.nan)], [(np.nan, np.nan)]]
    means, counts = aggregate_case_metrics(values)
    np.testing.assert_allclose(means, [[0.4, 4.0]])
    np.testing.assert_array_equal(counts, [[2, 1]])


def test_all_undefined_regions_stay_nan_without_runtime_warnings():
    with np.errstate(all="raise"):
        means, counts = aggregate_case_metrics([
            [(np.nan, np.nan), (0.0, np.nan)],
            [(np.nan, np.nan), (0.6, 2.0)],
        ])
    assert np.isnan(means[0]).all()
    np.testing.assert_allclose(means[1], [0.3, 2.0])
    np.testing.assert_array_equal(counts, [[0, 0], [2, 1]])


def test_no_cases_is_an_explicit_error():
    with pytest.raises(ValueError, match="at least one case"):
        aggregate_case_metrics([])


def test_infinite_metrics_are_not_silently_averaged():
    with pytest.raises(ValueError, match="Inf"):
        aggregate_case_metrics([[(0.8, np.inf)]])


def test_metric_accepts_readonly_masks_without_binarizing_caller_arrays():
    pred = np.zeros((4, 4, 4), dtype=np.uint8)
    pred[1, 1, 1] = 2
    gt = pred.copy()
    pred.setflags(write=False)
    gt.setflags(write=False)
    assert calculate_metric_percase(pred, gt) == (1.0, 0.0)
    assert pred[1, 1, 1] == gt[1, 1, 1] == 2


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("bad_prediction", [True, False])
def test_invalid_masks_are_reported_instead_of_being_scored(bad, bad_prediction):
    clean = np.zeros((4, 4), dtype=float)
    invalid = clean.copy()
    invalid[0, 0] = bad
    pred, gt = (invalid, clean) if bad_prediction else (clean, invalid)
    with pytest.raises(ValueError, match="NaN or Inf"):
        calculate_metric_percase(pred, gt)


def test_volume_regions_are_independent_of_scoring_order():
    gt = np.zeros((6, 6, 2), dtype=np.uint8)
    gt[1:3, 1:3, 0] = 1
    gt[3:5, 1:3, 0] = 2
    gt[1:3, 3:5, 1] = 3
    pred = np.roll(gt, 1, axis=0)
    before_pred, before_gt = pred.copy(), gt.copy()
    pred.setflags(write=False)
    gt.setflags(write=False)
    regions = evaluation_regions(4)
    assert regions == [(1,), (2,), (3,), (2, 3), (1, 2, 3)]
    actual = calculate_volume_metrics(pred, gt, classes=4)
    reverse = {ids: calculate_metric_percase(np.isin(pred, ids), np.isin(gt, ids))
               for ids in reversed(regions)}
    np.testing.assert_allclose(actual, [reverse[ids] for ids in regions], rtol=0, atol=0)
    np.testing.assert_array_equal(pred, before_pred)
    np.testing.assert_array_equal(gt, before_gt)


@pytest.mark.parametrize("bad", [-1, 4, 1.5, np.nan, np.inf])
def test_volume_rejects_invalid_labels(bad):
    gt = np.zeros((4, 4, 1))
    gt[0, 0, 0] = bad
    with pytest.raises(ValueError, match="class IDs"):
        calculate_volume_metrics(np.zeros_like(gt), gt, classes=4)


def test_mismatched_masks_do_not_broadcast():
    with pytest.raises(ValueError, match="same shape"):
        calculate_metric_percase(np.ones((4, 4)), np.ones((1, 4)))
    with pytest.raises(ValueError, match="same shape"):
        calculate_volume_metrics(np.zeros((4, 4)), np.zeros((1, 4)), classes=4)


def test_volume_pipeline_preserves_region_merges_and_empty_policy(monkeypatch):
    class FixedPrediction(torch.nn.Module):
        def forward(self, *images):
            output = torch.zeros((1, 4, 4, 4))
            output[:, 1, 0, 0] = 10
            output[:, 3, 1, 1] = 10
            return output

    gt = torch.zeros((1, 4, 4, 1))
    gt[0, 0, 0, 0] = 1
    gt[0, 1, 1, 0] = 3
    image = torch.zeros_like(gt)
    original_gt = gt.clone()
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor, *a, **k: tensor)
    values = segmentation_utils.test_single_volume(
        image, image, image, gt, FixedPrediction(), classes=4, patch_size=[4, 4])
    assert len(values) == 5
    assert np.isnan(values[1]).all()  # Scar absent from both masks.
    np.testing.assert_allclose([values[i] for i in (0, 2, 3, 4)], [(1.0, 0.0)] * 4)
    torch.testing.assert_close(gt, original_gt, rtol=0, atol=0)


def test_repeated_volume_evaluation_preserves_inputs_predictions_and_weights(monkeypatch, tmp_path):
    class FixedPrediction(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.score = torch.nn.Parameter(torch.tensor(10.0))

        def forward(self, *images):
            assert not self.training and not torch.is_grad_enabled()
            output = torch.zeros((1, 4, 4, 4))
            for cls in (1, 2, 3):
                output[0, cls, cls - 1, cls - 1] = self.score
            return output

    gt = torch.zeros((1, 4, 4, 1))
    for cls in (1, 2, 3):
        gt[0, cls - 1, cls - 1, 0] = cls
    images = [torch.rand_like(gt) for _ in range(3)]
    before_images = [x.clone() for x in images]
    before_gt = gt.clone()
    model = FixedPrediction()
    before_weights = {k: v.clone() for k, v in model.state_dict().items()}
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor, *a, **k: tensor)
    kwargs = dict(classes=4, patch_size=[4, 4], case="repeat", return_prediction=True)
    first, pred1 = segmentation_utils.test_single_volume(
        *images, gt, model, test_save_path=str(tmp_path), **kwargs)
    second, pred2 = segmentation_utils.test_single_volume(*images, gt, model, **kwargs)
    np.testing.assert_allclose(first, [(1.0, 0.0)] * 5, rtol=0, atol=0)
    np.testing.assert_allclose(second, first, rtol=0, atol=0)
    np.testing.assert_array_equal(pred1, before_gt.squeeze(0).numpy())
    np.testing.assert_array_equal(pred2, pred1)
    torch.testing.assert_close(gt, before_gt, rtol=0, atol=0)
    for image, before in zip(images, before_images):
        torch.testing.assert_close(image, before, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before_weights[key], rtol=0, atol=0)
    for suffix in ("pred", "gt"):
        saved = segmentation_utils.sitk.GetArrayFromImage(
            segmentation_utils.sitk.ReadImage(str(tmp_path / f"repeat_{suffix}.nii.gz")))
        np.testing.assert_array_equal(saved, pred1.transpose(2, 0, 1))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("volume", [True, False])
def test_nonfinite_logits_stop_before_argmax_or_export(monkeypatch, tmp_path, bad, volume):
    class InvalidModel(torch.nn.Module):
        def forward(self, *images):
            return torch.full((1, 4, 4, 4), bad)

    shape = (1, 4, 4, 1) if volume else (1, 4, 4)
    image = torch.zeros(shape)
    gt = torch.zeros(shape)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor, *a, **k: tensor)
    with pytest.raises(ValueError, match="Model output contains NaN or Inf for case 'bad_case'") as error:
        segmentation_utils.test_single_volume(
            image, image, image, gt, InvalidModel(), classes=4, patch_size=[4, 4],
            test_save_path=str(tmp_path), case="bad_case")
    if volume:
        assert "slice 0" in str(error.value)
    assert list(tmp_path.iterdir()) == []


def test_actual_inference_logs_metric_specific_counts(caplog):
    # Load the real function without executing test.py's CLI parser/CUDA setup.
    path = Path(__file__).resolve().parents[1] / "test.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "inference")
    values = [[(0.8, 4.0)] * 5, [(0.0, np.nan)] * 5,
              [(np.nan, np.nan)] * 5, [(0.0, np.nan)] * 5]
    foreground = torch.tensor([[[[1], [2], [3]]]])
    background = torch.zeros_like(foreground)
    labels = [foreground, foreground, background, background]
    predictions = [foreground.squeeze(0).numpy(), background.squeeze(0).numpy(),
                   background.squeeze(0).numpy(), foreground.squeeze(0).numpy()]
    batches = [{"image": None, "image1": None, "image2": None, "label": labels[i],
                "case_name": [str(i)]} for i in range(4)]
    scope = {
        "logging": logging, "aggregate_case_metrics": aggregate_case_metrics,
        "evaluation_regions": evaluation_regions,
        "np": np, "tqdm": lambda iterable: iterable,
        "DataLoader": lambda dataset, **kwargs: dataset,
        "test_single_volume": lambda *a, case, **kw: (values[int(case)], predictions[int(case)]),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    args = SimpleNamespace(Dataset=lambda **kwargs: batches, volume_path="", volume_path1="",
                           volume_path2="", list_dir="", num_classes=4, img_size=128, z_spacing=1)
    caplog.set_level(logging.INFO)
    assert scope["inference"](args, torch.nn.Identity()) == "Testing Finished!"
    assert "Mean class 1 mean_dice 0.266667 mean_hd95 4.000000" in caplog.text
    assert "Mean class 1 valid_dice_cases 3 valid_hd95_cases 1 total_cases 4" in caplog.text
    assert "Mean class 1 both_empty_cases 1 missed_foreground_cases 1 false_positive_foreground_cases 1" in caplog.text
