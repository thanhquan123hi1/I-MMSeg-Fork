"""Read the predicted ROI128 MyoPS++ export or its SCAR cache."""
import json
import re
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

from .profiles import MODALITIES

EXPORT_ID = "MyoPSpp_UNet2D_ROI128_1mm_2026-10-27"
CACHE_ID = "myopspp_unet2d_roi128_1mm"
CANONICAL = "background_normal_edema_scar"
# The permutation is its own inverse. Use only at the cache input and NIfTI output.
SWAP_PATHOLOGY = np.array([0, 1, 3, 2], dtype=np.uint8)
RAW_TO_MODEL = {0: 0, 500: 0, 600: 0, 200: 1, 2221: 2, 1220: 3}
IMAGE_KEYS = ("image", "image1", "image2")


def model_labels(label, *, raw):
    codes, inverse = np.unique(label, return_inverse=True)
    mapping = RAW_TO_MODEL if raw else dict(enumerate(SWAP_PATHOLOGY))
    if not all(code in mapping for code in codes):
        raise ValueError(f"Unexpected MyoPS++ label codes {codes.tolist()}")
    return np.asarray([mapping[code] for code in codes], dtype=np.uint8)[inverse].reshape(label.shape)


def check_patients(patients):
    train, test = patients["train"], patients["test_vol"]
    cohort = train + test
    if (len(train) != 64 or len(test) != 16 or len(set(cohort)) != 80
            or not all(re.fullmatch(r"Case[23]\d{3}", name) for name in cohort)):
        raise ValueError("Expected 64 train and 16 distinct test patients in the MyoPS++ split")


def check_grid(affine, shape):
    if len(shape) != 3 or shape[:2] != (128, 128) or shape[2] < 1:
        raise ValueError(f"Expected a 128x128xD ROI, got {shape}")
    axes = affine[:3, :3]
    spacing = nib.affines.voxel_sizes(affine)
    if (not np.isfinite(affine).all() or not (spacing > 0).all()
            or not np.allclose(spacing[:2], 1, atol=1e-4)
            or not np.allclose(axes.T @ axes, np.diag(spacing**2), atol=1e-3)):
        raise ValueError("Invalid predicted ROI affine or spacing")
    return spacing


def check_images(images, label):
    for image in images:
        if (image.shape != label.shape or not np.isfinite(image).all()
                or image.min() < 0 or image.max() > 1):
            raise ValueError("Expected aligned, finite MyoPS++ images normalized to [0,1]")


def export_samples(root, split):
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    record = json.loads((root / "splits.json").read_text())
    if manifest["dataset_id"] != EXPORT_ID or record["dataset_id"] != EXPORT_ID:
        raise ValueError("Expected the MyoPS++ U-Net2D ROI128 1 mm NIfTI export")
    patients = record["patients"]
    check_patients(patients)
    if set(manifest["patients"]) != set(patients["train"] + patients["test_vol"]):
        raise ValueError("Patient split and ROI export disagree")
    for name in patients[split]:
        center = "CenterB" if name.startswith("Case2") else "CenterC"
        case_dir = root / "cases" / center / name
        metadata = json.loads((case_dir / "metadata.json").read_text())
        if metadata["split"] != split or metadata["case"] != name:
            raise ValueError(f"Inconsistent patient/split metadata for {name}")
        volumes = [nib.load(case_dir / f"{name}_{suffix}.nii.gz") for suffix in ("C0", "LGE", "T2", "gd")]
        reference = volumes[0]
        spacing = check_grid(reference.affine, reference.shape)
        for volume in volumes:
            if (volume.shape != reference.shape or volume.header.get_xyzt_units()[0] != "mm"
                    or not np.allclose(volume.affine, reference.affine, atol=1e-4)):
                raise ValueError(f"Mismatched ROI geometry for {name}")
        images = [v.get_fdata(dtype=np.float32) for v in volumes[:3]]
        label = model_labels(np.asarray(volumes[3].dataobj), raw=True)
        check_images(images, label)
        arrays = dict(zip(IMAGE_KEYS, images), label=label)
        if split == "train":
            for z in range(label.shape[2]):
                sample = {key: array[:, :, z] for key, array in arrays.items()}
                yield dict(sample, case_name=f"{name}_slice{z:03d}")
        else:
            yield dict(arrays, case_name=name, spacing=spacing, affine=reference.affine)


def cache_samples(root, split, metadata):
    if (metadata["dataset_id"] != CACHE_ID or metadata["label_order"] != CANONICAL
            or metadata["class_names"] != ["background", "normal_myocardium", "edema", "scar"]
            or metadata["spacing_unit"] != "mm" or metadata["axis_order"] != "HWD"):
        raise ValueError("Expected a canonical SCAR MyoPS++ ROI128 cache")
    patients = metadata["patients"]
    check_patients(patients)
    train = split == "train"
    names = (root / "lists" / f"{split}.txt").read_text().splitlines()
    expected = ({f"{case}_slice{z:03d}" for case in patients[split]
                 for z in range(metadata["case_shapes"][case][2])} if train else set(patients[split]))
    if len(names) != len(expected) or set(names) != expected:
        raise ValueError(f"Cache manifest disagrees with {split} patients")
    for name in names:
        images = []
        label = affine = None
        for modality in MODALITIES:
            if train:
                archive = np.load(root / modality / "train_npz" / f"{name}.npz", allow_pickle=False)
            else:
                archive = h5py.File(root / modality / "test_vol_h5" / f"{name}.npy.h5", "r")
            with archive as data:
                attributes = data if train else data.attrs
                order = np.asarray(attributes["label_order"]).item()
                if order != CANONICAL:
                    raise ValueError(f"Unexpected label order for {name}/{modality}: {order}")
                images.append(data["image"][:])
                mask, grid = data["label"][:], data["affine"][:]
                if (np.asarray(attributes["spacing_unit"]).item() != "mm"
                        or not np.allclose(data["spacing"][:], nib.affines.voxel_sizes(grid), atol=1e-4)):
                    raise ValueError(f"Inconsistent spacing in mm for {name}/{modality}")
                if label is None:
                    label, affine = mask, grid
                elif not np.array_equal(label, mask) or not np.allclose(affine, grid):
                    raise ValueError(f"Inconsistent modality labels or affine for {name}")
        check_images(images, label)
        spacing = check_grid(affine, (*label.shape, 1) if train else label.shape)
        sample = dict(zip(IMAGE_KEYS, images), label=model_labels(label, raw=False), case_name=name)
        if not train:
            sample.update(spacing=spacing, affine=affine)
        yield sample


class MyoPSPlusDataset(Dataset):
    def __init__(self, root, split, transform=None):
        root = Path(root)
        cache = root / "dataset_metadata.json"
        if cache.is_file():
            samples = cache_samples(root, split, json.loads(cache.read_text()))
        elif (root / "dataset_manifest.json").is_file():
            samples = export_samples(root, split)
        else:
            raise ValueError(f"No MyoPS++ ROI export or SCAR cache manifest at {root}")
        # This ROI cohort fits in RAM. Decompress once per run, outside the epoch loop.
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if self.transform:
            return dict(self.transform(sample), case_name=sample["case_name"])
        return sample
