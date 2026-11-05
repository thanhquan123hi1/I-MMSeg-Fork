"""Evaluate an explicit checkpoint using the repository's test-volume pipeline."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import h5py
import numpy as np
import torch
from medpy import metric

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from networks.vit_seg_modeling import CONFIGS, VisionTransformer
from utils import test_single_volume

REGIONS = {
    "normal_myocardium": [1],
    "scar": [2],
    "edema_only": [3],
    "edema_including_scar": [2, 3],
    "whole_myocardium": [1, 2, 3],
}


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--list-file", type=Path, default=ROOT / "list/test_vol.txt")
    parser.add_argument("--seed", type=int, default=1234, help="Inference random seed")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The original inference pipeline requires CUDA.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pred_dir = args.output_dir / "predictions"
    pred_dir.mkdir()
    names = args.list_file.read_text().split()
    if not names or len(names) != len(set(names)):
        raise ValueError("Test case list must be nonempty and contain no duplicates.")
    train_cases = {n.split("_slice")[0] for n in (ROOT / "list/train.txt").read_text().split()}
    if set(names) & train_cases:
        raise ValueError("Repository train/test case lists overlap.")
    total_slices = 0
    for name in names:
        shape = None
        for modality in ("bSSFP", "LGE", "T2w"):
            with h5py.File(args.data_root / modality / "test_vol_h5" / f"{name}.npy.h5", "r") as data:
                image, label = data["image"][:], data["label"][:]
                if image.shape != label.shape or image.ndim != 3:
                    raise ValueError(f"Invalid image/label shape: {modality}/{name}")
                if shape is not None and shape != image.shape:
                    raise ValueError(f"Modalities are not aligned: {name}")
                if not np.isfinite(image).all() or not np.isin(label, [0, 1, 2, 3]).all():
                    raise ValueError(f"Invalid image values or labels: {modality}/{name}")
                shape = image.shape
        total_slices += shape[2]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = copy.deepcopy(CONFIGS["R50-ViT-B_16"])
    config.n_classes = 4
    config.n_skip = 3
    config.patches.size = (16, 16)
    config.patches.grid = (8, 8)
    model = VisionTransformer(config, img_size=128, num_classes=4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    checkpoint = {k.removeprefix("module."): v for k, v in checkpoint.items()}
    if any(not torch.isfinite(v).all() for v in checkpoint.values() if v.is_floating_point()):
        raise ValueError("Checkpoint contains nonfinite weights.")
    model.load_state_dict(checkpoint, strict=True)
    del checkpoint
    model.cuda().eval()
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "data_root": str(args.data_root.resolve()),
        "test_list": str(args.list_file.resolve()),
        "test_list_sha256": sha256(args.list_file),
        "case_count": len(names), "slice_count": total_slices,
        "train_test_list_overlap": 0,
        "architecture": "R50-ViT-B_16", "input_size": 128,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "strict_checkpoint_loading": True,
        "torch_version": torch.__version__, "device": torch.cuda.get_device_name(0),
        "seed": args.seed, "precision": "float32",
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_hashes": {p: sha256(ROOT / p) for p in ["utils.py", "networks/vit_seg_modeling.py", "text_features/embedding_class_information.pth", "text_features/embedding_MRI_information.pth"]},
        "regions": REGIONS,
        "hd95_units": "voxel (no physical voxel spacing supplied to MedPy)",
        "aggregation": "unweighted mean over non-NaN test cases independently per metric; standard deviation uses ddof=0; all-undefined results are null",
        "repository_empty_rule": "Both empty: Dice=undefined, HD95=undefined. Exactly one empty: Dice=0, HD95=undefined. Undefined values are excluded independently per metric; no voxel-count threshold.",
        "conventional_empty_rule": "Both empty: Dice=1, HD95=0. Exactly one empty: Dice=0, HD95=undefined (excluded from HD95 mean).",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Validated {len(names)} cases / {total_slices} slices; checkpoint loaded strictly.", flush=True)
    rows = []
    start = time.perf_counter()
    fields = ["case", "region", "label_ids", "pred_voxels", "gt_voxels", "repository_dice", "repository_hd95_voxel", "conventional_dice", "conventional_hd95_voxel"]
    with open(args.output_dir / "per_case_metrics.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, name in enumerate(names, 1):
            images = []
            for modality in ("bSSFP", "LGE", "T2w"):
                with h5py.File(args.data_root / modality / "test_vol_h5" / f"{name}.npy.h5", "r") as data:
                    images.append(torch.from_numpy(data["image"][:]).unsqueeze(0))
                    if modality == "bSSFP":
                        label = torch.from_numpy(data["label"][:]).unsqueeze(0)
            with torch.no_grad():
                original_metrics, prediction = test_single_volume(*images, label, model, classes=4, patch_size=[128, 128], test_save_path=str(pred_dir), case=name, z_spacing=1, return_prediction=True)
            # Use in-memory views in the same axis order as the saved NIfTI arrays.
            pred = prediction.transpose(2, 0, 1)
            gt = label.squeeze(0).cpu().detach().numpy().transpose(2, 0, 1)
            for (region, ids), (dice, hd95) in zip(REGIONS.items(), original_metrics):
                p, g = np.isin(pred, ids), np.isin(gt, ids)
                npred, ngt = int(p.sum()), int(g.sum())
                conventional_dice = float(metric.binary.dc(p, g)) if npred + ngt else 1.0
                conventional_hd95 = float(hd95) if npred and ngt else (0.0 if not npred and not ngt else None)
                repository_dice = None if np.isnan(dice) else float(dice)
                repository_hd95 = None if np.isnan(hd95) else float(hd95)
                row = dict(zip(fields, [name, region, "+".join(map(str, ids)), npred, ngt, repository_dice, repository_hd95, conventional_dice, conventional_hd95]))
                rows.append(row)
                writer.writerow(row)
            stream.flush()
            print(f"[{index:02d}/{len(names)}] {name} scar Dice={original_metrics[1][0]:.4f} edema(2+3) Dice={original_metrics[3][0]:.4f} elapsed={time.perf_counter()-start:.1f}s", flush=True)
    summary = []
    for region in REGIONS:
        selected = [r for r in rows if r["region"] == region]
        item = {"region": region, "cases": len(selected)}
        for column in fields[5:]:
            values = [r[column] for r in selected if r[column] is not None]
            item[column + "_mean"] = float(np.mean(values)) if values else None
            item[column + "_std"] = float(np.std(values)) if values else None
            item[column + "_valid_cases"] = len(values)
        item["both_empty_cases"] = sum(r["pred_voxels"] == r["gt_voxels"] == 0 for r in selected)
        item["one_empty_cases"] = sum((r["pred_voxels"] == 0) != (r["gt_voxels"] == 0) for r in selected)
        item["missed_foreground_cases"] = sum(r["gt_voxels"] > 0 and r["pred_voxels"] == 0 for r in selected)
        item["false_positive_foreground_cases"] = sum(r["gt_voxels"] == 0 and r["pred_voxels"] > 0 for r in selected)
        summary.append(item)
    result = {"metadata": metadata, "elapsed_seconds": time.perf_counter()-start, "summary": summary}
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(args.output_dir / "summary.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
