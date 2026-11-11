"""Verify predictions for all 5 seeds and export table with mean and std."""
import csv
import hashlib
import json
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import SimpleITK as sitk
from medpy import metric

PROJECT = Path(__file__).resolve().parents[1]
FOLDER = PROJECT / "evaluation_results" / "myops380_five_seeds_20261108"
SEEDS = [1234, 1098, 1024, 2026, 6868]
REGIONS = {
    "normal_myocardium": [1],
    "scar": [2],
    "edema_only": [3],
    "edema_including_scar": [2, 3],
    "whole_myocardium": [1, 2, 3],
}
FIELDS = [
    "seed",
    "Scar Dice (%)",
    "Scar HD95",
    "Edema Dice (%)",
    "Edema HD95",
    "Myocardium Dice (%)",
    "Myocardium HD95",
    "Avg. pathology Dice (%)",
    "Avg. pathology HD95",
]


def write_csv(path, rows, fields=None):
    with open(path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    table = []
    region_rows = []
    raw_seed_values = {f: [] for f in FIELDS[1:]}

    for seed in SEEDS:
        seed_dir = FOLDER / f"seed{seed}"
        summary_file = seed_dir / "summary.json"
        assert summary_file.exists(), f"Missing summary.json for seed {seed}"
        result = json.loads(summary_file.read_text(encoding="utf-8"))
        summary = {r["region"]: r for r in result["summary"]}

        # Regions:
        # Scar: region == "scar"
        # Edema: region == "edema_including_scar"
        # Myocardium: region == "whole_myocardium"
        scar_dice = summary["scar"]["repository_dice_mean"] * 100
        scar_hd95 = summary["scar"]["repository_hd95_voxel_mean"]

        edema_dice = summary["edema_including_scar"]["repository_dice_mean"] * 100
        edema_hd95 = summary["edema_including_scar"]["repository_hd95_voxel_mean"]

        myo_dice = summary["whole_myocardium"]["repository_dice_mean"] * 100
        myo_hd95 = summary["whole_myocardium"]["repository_hd95_voxel_mean"]

        avg_path_dice = (scar_dice + edema_dice) / 2
        avg_path_hd95 = (scar_hd95 + edema_hd95) / 2

        row = {
            "seed": str(seed),
            "Scar Dice (%)": scar_dice,
            "Scar HD95": scar_hd95,
            "Edema Dice (%)": edema_dice,
            "Edema HD95": edema_hd95,
            "Myocardium Dice (%)": myo_dice,
            "Myocardium HD95": myo_hd95,
            "Avg. pathology Dice (%)": avg_path_dice,
            "Avg. pathology HD95": avg_path_hd95,
        }
        table.append(row)
        for f in FIELDS[1:]:
            raw_seed_values[f].append(row[f])

        for region, r in summary.items():
            region_rows.append({
                "seed": seed,
                "region": region,
                "Dice (%)": r["repository_dice_mean"] * 100 if r["repository_dice_mean"] is not None else None,
                "HD95": r["repository_hd95_voxel_mean"],
                "valid_Dice_cases": r["repository_dice_valid_cases"],
                "valid_HD95_cases": r["repository_hd95_voxel_valid_cases"],
                "both_empty_cases": r["both_empty_cases"],
                "one_empty_cases": r["one_empty_cases"],
            })

    # Compute statistics across 5 seeds
    mean_row = {"seed": "Mean"}
    std_sample_row = {"seed": "Std (sample, ddof=1)"}
    std_pop_row = {"seed": "Std (pop, ddof=0)"}
    mean_std_sample_row = {"seed": "Mean ± Std"}

    for f in FIELDS[1:]:
        vals = np.array(raw_seed_values[f])
        m = float(np.mean(vals))
        s_sample = float(np.std(vals, ddof=1))
        s_pop = float(np.std(vals, ddof=0))
        mean_row[f] = m
        std_sample_row[f] = s_sample
        std_pop_row[f] = s_pop
        mean_std_sample_row[f] = f"{m:.2f} ± {s_sample:.2f}"

    full_table = list(table) + [mean_row, std_sample_row, mean_std_sample_row]

    write_csv(FOLDER / "results_5seeds.csv", full_table)
    write_csv(FOLDER / "region_metrics_5seeds.csv", region_rows)

    # Format Markdown Table exactly like user's request
    md_lines = [
        "| I-MMSeg | Scar Dice ↑ | Scar HD ↓ | Edema Dice ↑ | Edema HD ↓ | Myocardium Dice ↑ | Myocardium HD ↓ | Avg. pathology Dice ↑ | Avg. pathology HD ↓ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in table:
        md_lines.append(
            f"| **{r['seed']}** | {r['Scar Dice (%)']:.2f} | {r['Scar HD95']:.2f} | "
            f"{r['Edema Dice (%)']:.2f} | {r['Edema HD95']:.2f} | "
            f"{r['Myocardium Dice (%)']:.2f} | {r['Myocardium HD95']:.2f} | "
            f"{r['Avg. pathology Dice (%)']:.2f} | {r['Avg. pathology HD95']:.2f} |"
        )
    # Add Mean ± Std row
    md_lines.append(
        f"| **Avg** | **{mean_row['Scar Dice (%)']:.2f} ± {std_sample_row['Scar Dice (%)']:.2f}** | "
        f"**{mean_row['Scar HD95']:.2f} ± {std_sample_row['Scar HD95']:.2f}** | "
        f"**{mean_row['Edema Dice (%)']:.2f} ± {std_sample_row['Edema Dice (%)']:.2f}** | "
        f"**{mean_row['Edema HD95']:.2f} ± {std_sample_row['Edema HD95']:.2f}** | "
        f"**{mean_row['Myocardium Dice (%)']:.2f} ± {std_sample_row['Myocardium Dice (%)']:.2f}** | "
        f"**{mean_row['Myocardium HD95']:.2f} ± {std_sample_row['Myocardium HD95']:.2f}** | "
        f"**{mean_row['Avg. pathology Dice (%)']:.2f} ± {std_sample_row['Avg. pathology Dice (%)']:.2f}** | "
        f"**{mean_row['Avg. pathology HD95']:.2f} ± {std_sample_row['Avg. pathology HD95']:.2f}** |"
    )

    md_content = "\n".join(md_lines)
    (FOLDER / "results_5seeds.md").write_text(md_content, encoding="utf-8")

    result_data = {
        "seeds": table,
        "mean": {f: mean_row[f] for f in FIELDS[1:]},
        "std_sample_ddof1": {f: std_sample_row[f] for f in FIELDS[1:]},
        "std_pop_ddof0": {f: std_pop_row[f] for f in FIELDS[1:]},
        "markdown_table": md_content,
    }
    (FOLDER / "results_5seeds.json").write_text(json.dumps(result_data, indent=2), encoding="utf-8")
    print(md_content)
    print("\nDetailed JSON summary:")
    print(json.dumps(result_data, indent=2))


if __name__ == "__main__":
    main()
