"""Dataset selection for I-MMSeg's fixed [background, normal, scar, edema] heads."""
import json
from pathlib import Path

from . import dataset_Myops

PROJECT = Path(__file__).resolve().parents[1]
MODALITIES = ("bSSFP", "LGE", "T2w")


def dataset_name(value):
    aliases = {"myops": "Myops", "myops380": "Myops", "myops++": "MyoPS++",
               "myopspp": "MyoPS++", "myopspp_unet2d_roi128_1mm": "MyoPS++"}
    try:
        return aliases[value.lower()]
    except KeyError:
        raise ValueError(f"Unknown dataset {value!r}; choose MyoPS380 or MyoPS++") from None


def default_root(dataset):
    if dataset == "MyoPS++":
        return PROJECT.parent / "Dataset/MyoPSpp_UNet2D_ROI128_1mm"
    bundled = PROJECT / "MyoPS380_dataset/Processed_data"
    local = PROJECT.parent / "Dataset/MyoPS380/Processed_data"
    return local if local.is_dir() and not bundled.is_dir() else bundled


def build_dataset(dataset, data_root, split, list_dir=None, transform=None):
    dataset = dataset_name(dataset)
    root = Path(data_root) if data_root else default_root(dataset)
    if split not in ("train", "test_vol"):
        raise ValueError(f"Unsupported split {split!r}")
    if dataset == "MyoPS++":
        from .myops_plus import MyoPSPlusDataset
        if list_dir is not None:
            raise ValueError("MyoPS++ uses the patient split in its export or cache")
        return MyoPSPlusDataset(root, split, transform)
    metadata = root / "dataset_metadata.json"
    if (root / "dataset_manifest.json").exists() or (
        metadata.exists() and json.loads(metadata.read_text()).get("label_order") != "background_normal_scar_edema"
    ):
        raise ValueError("MyoPS380 expects the legacy cache with scar=2 and edema=3")
    folder = "train_npz" if split == "train" else "test_vol_h5"
    return dataset_Myops.Myops_dataset(*(str(root / m / folder) for m in MODALITIES),
                                      list_dir=list_dir or PROJECT / "list", split=split, transform=transform)
