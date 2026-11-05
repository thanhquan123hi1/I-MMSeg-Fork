"""Re-score existing test predictions under two explicit empty-mask policies."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import SimpleITK as sitk
from medpy import metric

REGIONS = {"normal_myocardium": [1], "scar": [2], "edema_only": [3],
           "edema_including_scar": [2, 3], "whole_myocardium": [1, 2, 3]}
POLICIES = ("policy1_both_empty_perfect", "policy2_both_empty_ignored")
TABLE_FIELDS = ["seed", "Scar Dice (%)", "Scar HD95 (voxel)",
                "Edema Dice (%)", "Edema HD95 (voxel)",
                "Myocardium Dice (%)", "Myocardium HD95 (voxel)",
                "Avg. pathology Dice (%)", "Avg. pathology HD95 (voxel)"]


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_csv(path, rows, fields=None):
    with open(path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def apply_policy(pred_count, gt_count, dice, hd95, policy):
    if not pred_count and not gt_count:
        return (1.0, 0.0) if policy == POLICIES[0] else (None, None)
    if not pred_count or not gt_count:
        return 0.0, None
    return dice, hd95


def aggregate(values):
    valid = [v for v in values if v is not None]
    return (float(np.mean(valid)) if valid else None, len(valid))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 1098, 1024])
    args = parser.parse_args()
    # Exercise every consequential policy branch, including symmetric failures.
    for policy in POLICIES:
        assert apply_policy(0, 0, None, None, policy) == ((1.0, 0.0) if policy == POLICIES[0] else (None, None))
        assert apply_policy(0, 10, None, None, policy) == (0.0, None)
        assert apply_policy(10, 0, None, None, policy) == (0.0, None)
        assert apply_policy(10, 10, 0.6, 2.0, policy) == (0.6, 2.0)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows, empty_cases, provenance, checks = [], [], [], []
    reference_split = None
    for seed in args.seeds:
        folder = args.source_dir / f"seed{seed}"
        metadata = json.loads((folder / "metadata.json").read_text())
        assert metadata["seed"] == seed and metadata["strict_checkpoint_loading"]
        assert digest(metadata["checkpoint"]) == metadata["checkpoint_sha256"]
        assert digest(metadata["test_list"]) == metadata["test_list_sha256"]
        names = Path(metadata["test_list"]).read_text().split()
        assert len(names) == len(set(names)) == metadata["case_count"]
        if reference_split is None:
            reference_split = names
        assert reference_split == names
        with open(folder / "per_case_metrics.csv", newline="", encoding="utf-8") as stream:
            old_rows = list(csv.DictReader(stream))
        old = {(r["case"], r["region"]): r for r in old_rows}
        assert len(old) == len(old_rows) == len(names) * len(REGIONS)
        max_error, slices = 0.0, 0
        mask_hashes = {}
        for name in names:
            pred_path = folder / "predictions" / f"{name}_pred.nii.gz"
            gt_path = folder / "predictions" / f"{name}_gt.nii.gz"
            pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path)))
            gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))
            with h5py.File(Path(metadata["data_root"]) / "bSSFP/test_vol_h5" / f"{name}.npy.h5", "r") as source:
                assert np.array_equal(gt, source["label"][:].transpose(2, 0, 1))
            assert pred.shape == gt.shape and np.isin(pred, [0, 1, 2, 3]).all()
            slices += pred.shape[0]
            mask_hashes[name] = {"prediction_sha256": digest(pred_path), "gt_sha256": digest(gt_path)}
            for region, ids in REGIONS.items():
                p, g = np.isin(pred, ids), np.isin(gt, ids)
                npred, ngt = int(p.sum()), int(g.sum())
                status = "both_foreground" if npred and ngt else (
                    "both_empty" if not npred and not ngt else (
                        "missed_foreground" if not npred else "false_positive_foreground"))
                if status == "both_foreground":
                    dice, hd95 = float(metric.binary.dc(p, g)), float(metric.binary.hd95(p, g))
                    assert np.isfinite([dice, hd95]).all()
                else:
                    dice, hd95 = None, None
                    empty_cases.append({"seed": seed, "case": name, "region": region,
                                        "status": status, "gt_voxels": ngt, "pred_voxels": npred})
                previous = old[name, region]
                assert int(previous["gt_voxels"]) == ngt and int(previous["pred_voxels"]) == npred
                old_dice = dice if status == "both_foreground" else float(npred < 200 and ngt <= 200)
                old_hd95 = hd95 if status == "both_foreground" else 0.0
                error = max(abs(old_dice - float(previous["repository_dice"])),
                            abs(old_hd95 - float(previous["repository_hd95_voxel"])))
                assert error < 1e-12, (seed, name, region, error)
                max_error = max(max_error, error)
                for policy in POLICIES:
                    d, h = apply_policy(npred, ngt, dice, hd95, policy)
                    rows.append({"policy": policy, "seed": seed, "case": name, "region": region,
                                 "label_ids": "+".join(map(str, ids)), "status": status,
                                 "gt_voxels": ngt, "pred_voxels": npred, "dice": d, "hd95_voxel": h})
        assert slices == metadata["slice_count"]
        provenance.append({"seed": seed, "source_metadata": metadata, "saved_mask_hashes": mask_hashes})
        checks.append({"seed": seed, "cases": len(names), "slices": slices,
                       "verified_case_region_pairs": len(old), "max_error_vs_previous_metrics": max_error,
                       "gt_matches_source_h5": True})
        print(f"Re-scored seed {seed}: {len(old)} case/region pairs; previous-metric error={max_error}", flush=True)

    summary = []
    for policy in POLICIES:
        for seed in args.seeds:
            for region in REGIONS:
                selected = [r for r in rows if (r["policy"], r["seed"], r["region"]) == (policy, seed, region)]
                dice, nd = aggregate([r["dice"] for r in selected])
                hd, nh = aggregate([r["hd95_voxel"] for r in selected])
                item = {"policy": policy, "seed": seed, "region": region, "total_cases": len(selected),
                        "dice_mean": dice, "dice_percent": None if dice is None else dice * 100,
                        "dice_valid_cases": nd, "hd95_voxel_mean": hd, "hd95_valid_cases": nh}
                for status in ("both_empty", "missed_foreground", "false_positive_foreground", "both_foreground"):
                    item[status + "_cases"] = sum(r["status"] == status for r in selected)
                assert nd == len(selected) - (item["both_empty_cases"] if policy == POLICIES[1] else 0)
                assert nh == item["both_foreground_cases"] + (item["both_empty_cases"] if policy == POLICIES[0] else 0)
                summary.append(item)
    tables = {}
    for policy in POLICIES:
        table = []
        for seed in args.seeds:
            indexed = {r["region"]: r for r in summary if r["policy"] == policy and r["seed"] == seed}
            values = []
            for region in ("scar", "edema_including_scar", "whole_myocardium"):
                values += [indexed[region]["dice_percent"], indexed[region]["hd95_voxel_mean"]]
            values += [(values[0] + values[2]) / 2, (values[1] + values[3]) / 2]
            table.append(dict(zip(TABLE_FIELDS, [seed, *values])))
        table.append({"seed": f"Avg ({len(args.seeds)} seeds)",
                      **{f: float(np.mean([r[f] for r in table])) for f in TABLE_FIELDS[1:]}})
        write_csv(args.output_dir / f"{policy}.csv", table)
        tables[policy] = table
    write_csv(args.output_dir / "per_case_two_policies.csv", rows)
    write_csv(args.output_dir / "summary_with_denominators.csv", summary)
    write_csv(args.output_dir / "empty_cases.csv", empty_cases,
              ["seed", "case", "region", "status", "gt_voxels", "pred_voxels"])
    output = {"source_directory": str(args.source_dir.resolve()), "new_inference_run": False,
              "method": "Recompute Dice and MedPy HD95 from saved predictions; macro mean over valid cases independently per metric.",
              "policy1": "Both empty: Dice=1, HD95=0; one empty: Dice=0, HD95=null; both nonempty: actual metrics.",
              "policy2": "Both empty: Dice=null, HD95=null; one empty: Dice=0, HD95=null; both nonempty: actual metrics.",
              "null_handling": "JSON null and blank CSV fields mean undefined/excluded, never zero-filled.",
              "avg_pathology": "Arithmetic mean of the Scar and Edema(2+3) region means, independently for Dice and HD95.",
              "avg_seeds": "Unweighted arithmetic mean of available checkpoint seeds.",
              "hd95_units": "voxel; no physical voxel spacing supplied", "regions": REGIONS,
              "tables": tables, "summary": summary, "empty_cases": empty_cases, "verification": checks,
              "provenance": provenance, "evaluation_script_sha256": digest(__file__)}
    (args.output_dir / "results.json").write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"tables": tables, "empty_cases": empty_cases}, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
