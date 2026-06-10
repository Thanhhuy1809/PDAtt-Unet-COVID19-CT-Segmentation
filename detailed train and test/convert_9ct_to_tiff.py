#!/usr/bin/env python3
"""
Convert PDAtt 9CT datasets (Dataset1, Dataset2, Dataset3) from NIfTI volumes
to 2D TIFF slices.

Output per slice:
- <case>_<slice>.tif            -> normalized image
- <case>_<slice>_mask.tif       -> infection mask (for Kaggle-style compatibility)
- <case>_<slice>_lung_mask.tif  -> lung mask

You can also use extension .tift with --ext .tift.
The script always saves with TIFF format explicitly.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from PIL import Image


@dataclass
class VolumeTriplet:
    dataset: str
    case_id: str
    image_path: Path
    lung_mask_path: Path
    infection_mask_path: Path


DATASET3_CASE_MAPPING: Sequence[Tuple[str, str]] = [
    ("case009.nii", "coronacases_009.nii"),
    ("coronacases_org_001.nii", "coronacases_001.nii"),
    ("coronacases_org_002.nii", "coronacases_002.nii"),
    ("coronacases_org_003.nii", "coronacases_003.nii"),
    ("coronacases_org_004.nii", "coronacases_004.nii"),
    ("coronacases_org_005.nii", "coronacases_005.nii"),
    ("coronacases_org_006.nii", "coronacases_006.nii"),
    ("coronacases_org_008.nii", "coronacases_008.nii"),
    ("coronacases_org_case007.nii", "coronacases_007.nii"),
    ("coronacases_org_case010.nii", "coronacases_010.nii"),
    ("radiopaedia_org_covid-19-pneumonia-10_85902_1-dcm.nii", "radiopaedia_10_85902_1.nii"),
    ("radiopaedia_org_covid-19-pneumonia-10_85902_3-dcm.nii", "radiopaedia_10_85902_3.nii"),
    ("radiopaedia_org_covid-19-pneumonia-14_85914_0-dcm.nii", "radiopaedia_14_85914_0.nii"),
    ("radiopaedia_org_covid-19-pneumonia-27_86410_0-dcm.nii", "radiopaedia_27_86410_0.nii"),
    ("radiopaedia_org_covid-19-pneumonia-29_86490_1-dcm.nii", "radiopaedia_29_86490_1.nii"),
    ("radiopaedia_org_covid-19-pneumonia-29_86491_1-dcm.nii", "radiopaedia_29_86491_1.nii"),
    ("radiopaedia_org_covid-19-pneumonia-36_86526_0-dcm.nii", "radiopaedia_36_86526_0.nii"),
    ("radiopaedia_org_covid-19-pneumonia-40_86625_0-dcm.nii", "radiopaedia_40_86625_0.nii"),
    ("radiopaedia_org_covid-19-pneumonia-4_85506_1-dcm.nii", "radiopaedia_4_85506_1.nii"),
    ("radiopaedia_org_covid-19-pneumonia-7_85703_0-dcm.nii", "radiopaedia_7_85703_0.nii"),
]


def strip_nifti_suffix(path: Path) -> str:
    lower = path.name.lower()
    if lower.endswith(".nii.gz"):
        return path.name[:-7]
    if lower.endswith(".nii"):
        return path.name[:-4]
    return path.stem


def sorted_nifti_files(files: Sequence[Path]) -> List[Path]:
    def key_fn(p: Path):
        stem = strip_nifti_suffix(p)
        if stem.isdigit():
            return (0, int(stem), stem)
        return (1, stem.lower(), stem)

    return sorted(files, key=key_fn)


def find_matching_nifti(folder: Path, stem: str) -> Optional[Path]:
    for ext in (".nii", ".nii.gz"):
        candidate = folder / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_volume(path: Path) -> np.ndarray:
    vol = nib.load(str(path)).get_fdata(dtype=np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D/4D volume, got {vol.shape} at {path}")
    return vol


def get_slice(vol: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return vol[index, :, :]
    if axis == 1:
        return vol[:, index, :]
    return vol[:, :, index]


def normalize_image_to_uint8(slice_2d: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(slice_2d.astype(np.float32), nan=0.0)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)

    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if high <= low:
        return np.zeros_like(arr, dtype=np.uint8)

    arr = (arr - low) / (high - low)
    arr = np.clip(arr, 0.0, 1.0)
    return np.uint8(np.round(arr * 255.0))


def normalize_mask_to_uint8(slice_2d: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(slice_2d.astype(np.float32), nan=0.0)
    binary = (arr > 0).astype(np.uint8)
    return binary * 255


def rotate_slice(slice_2d: np.ndarray, rotate_k: int) -> np.ndarray:
    k = rotate_k % 4
    if k == 0:
        return slice_2d
    return np.rot90(slice_2d, k=k)


def save_tiff(path: Path, array_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(array_u8)
    img.save(str(path), format="TIFF")


def build_dataset1_triplets(root_9ct: Path) -> List[VolumeTriplet]:
    d1 = root_9ct / "Dataset1"
    image_path = d1 / "rp_im" / "tr_im.nii"
    lung_path = d1 / "tr_lung_mask" / "tr_lungmasks_updated.nii"
    inf_path = d1 / "tr_mask.nii" / "tr_mask.nii"

    if not (image_path.exists() and lung_path.exists() and inf_path.exists()):
        print("[WARN] Dataset1 missing required files.")
        return []

    return [
        VolumeTriplet(
            dataset="Dataset1",
            case_id="tr_im",
            image_path=image_path,
            lung_mask_path=lung_path,
            infection_mask_path=inf_path,
        )
    ]


def build_dataset2_triplets(root_9ct: Path) -> List[VolumeTriplet]:
    d2 = root_9ct / "Dataset2"
    image_dir = d2 / "rp_im"
    lung_dir = d2 / "rp_lung_msk"
    inf_dir = d2 / "rp_msk"

    if not (image_dir.exists() and lung_dir.exists() and inf_dir.exists()):
        print("[WARN] Dataset2 missing one of rp_im/rp_lung_msk/rp_msk.")
        return []

    image_files = sorted_nifti_files([p for p in image_dir.iterdir() if p.is_file()])
    out: List[VolumeTriplet] = []
    for image_path in image_files:
        stem = strip_nifti_suffix(image_path)
        lung_path = find_matching_nifti(lung_dir, stem)
        inf_path = find_matching_nifti(inf_dir, stem)
        if not lung_path or not inf_path:
            print(f"[WARN] Dataset2 case {stem}: missing mask, skipped.")
            continue
        out.append(
            VolumeTriplet(
                dataset="Dataset2",
                case_id=stem,
                image_path=image_path,
                lung_mask_path=lung_path,
                infection_mask_path=inf_path,
            )
        )
    return out


def build_dataset3_triplets(root_9ct: Path) -> List[VolumeTriplet]:
    d3 = root_9ct / "Dataset3"
    ct_dir = d3 / "COVID-19-CT-Seg_20cases"
    lung_dir = d3 / "Lung_Mask"
    inf_dir = d3 / "Infection_Mask"

    if not (ct_dir.exists() and lung_dir.exists() and inf_dir.exists()):
        print("[WARN] Dataset3 missing one of COVID-19-CT-Seg_20cases/Lung_Mask/Infection_Mask.")
        return []

    out: List[VolumeTriplet] = []
    for ct_name, mask_name in DATASET3_CASE_MAPPING:
        image_path = ct_dir / ct_name
        lung_path = lung_dir / mask_name
        inf_path = inf_dir / mask_name

        if not (image_path.exists() and lung_path.exists() and inf_path.exists()):
            print(f"[WARN] Dataset3 case missing: CT={ct_name}, MASK={mask_name}")
            continue

        out.append(
            VolumeTriplet(
                dataset="Dataset3",
                case_id=strip_nifti_suffix(Path(mask_name)),
                image_path=image_path,
                lung_mask_path=lung_path,
                infection_mask_path=inf_path,
            )
        )
    return out


def build_triplets(root_9ct: Path, datasets: Sequence[str]) -> List[VolumeTriplet]:
    selected = set(datasets)
    out: List[VolumeTriplet] = []
    if "1" in selected:
        out.extend(build_dataset1_triplets(root_9ct))
    if "2" in selected:
        out.extend(build_dataset2_triplets(root_9ct))
    if "3" in selected:
        out.extend(build_dataset3_triplets(root_9ct))
    return out


def convert_triplet(
    triplet: VolumeTriplet,
    out_case_dir: Path,
    axis: int,
    rotate_k: int,
    ext: str,
) -> Dict[str, int]:
    image_vol = load_volume(triplet.image_path)
    lung_vol = load_volume(triplet.lung_mask_path)
    inf_vol = load_volume(triplet.infection_mask_path)

    n_slices = min(image_vol.shape[axis], lung_vol.shape[axis], inf_vol.shape[axis])
    if n_slices <= 0:
        return {"slices": 0, "infection_positive": 0}

    positive_inf = 0

    for i in range(n_slices):
        image_sl = get_slice(image_vol, axis, i)
        lung_sl = get_slice(lung_vol, axis, i)
        inf_sl = get_slice(inf_vol, axis, i)

        image_sl = rotate_slice(image_sl, rotate_k)
        lung_sl = rotate_slice(lung_sl, rotate_k)
        inf_sl = rotate_slice(inf_sl, rotate_k)

        image_u8 = normalize_image_to_uint8(image_sl)
        lung_u8 = normalize_mask_to_uint8(lung_sl)
        inf_u8 = normalize_mask_to_uint8(inf_sl)

        if np.any(inf_u8 > 0):
            positive_inf += 1

        base = f"{triplet.case_id}_{i:04d}"
        save_tiff(out_case_dir / f"{base}{ext}", image_u8)
        save_tiff(out_case_dir / f"{base}_mask{ext}", inf_u8)
        save_tiff(out_case_dir / f"{base}_lung_mask{ext}", lung_u8)

    return {"slices": n_slices, "infection_positive": positive_inf}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_in = script_dir.parent / "data" / "9CTscans"
    default_out = script_dir.parent / "data" / "9CTscans_tiff"

    parser = argparse.ArgumentParser(description="Convert Dataset1/2/3 NIfTI files into TIFF slices.")
    parser.add_argument("--input-root", type=Path, default=default_in, help="Path to data/9CTscans.")
    parser.add_argument("--output-root", type=Path, default=default_out, help="Output folder for TIFF files.")
    parser.add_argument("--datasets", nargs="+", choices=["1", "2", "3"], default=["1", "2", "3"], help="Datasets to convert.")
    parser.add_argument("--axis", type=int, choices=[0, 1, 2], default=2, help="Slice axis.")
    parser.add_argument("--rotate-k", type=int, default=1, help="Rotate each slice by k * 90 degrees.")
    parser.add_argument("--ext", type=str, default=".tif", help="Output extension, e.g. .tif, .tiff, .tift.")
    parser.add_argument("--max-cases", type=int, default=0, help="Limit converted cases (0 means all).")
    parser.add_argument("--dry-run", action="store_true", help="Only print what will be converted.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    if not input_root.exists():
        print(f"Input root not found: {input_root}")
        return 1

    ext = args.ext if args.ext.startswith(".") else f".{args.ext}"

    triplets = build_triplets(input_root, args.datasets)
    if args.max_cases > 0:
        triplets = triplets[: args.max_cases]

    if not triplets:
        print("No valid cases found to convert.")
        return 1

    print(f"Found {len(triplets)} cases to convert.")
    for i, t in enumerate(triplets, start=1):
        print(f"{i:03d}. {t.dataset}/{t.case_id}")

    if args.dry_run:
        return 0

    output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "datasets": args.datasets,
        "axis": args.axis,
        "rotate_k": args.rotate_k,
        "ext": ext,
        "cases": [],
        "total_slices": 0,
        "total_infection_positive_slices": 0,
    }

    for idx, triplet in enumerate(triplets, start=1):
        out_case_dir = output_root / triplet.dataset / triplet.case_id
        info = convert_triplet(
            triplet=triplet,
            out_case_dir=out_case_dir,
            axis=args.axis,
            rotate_k=args.rotate_k,
            ext=ext,
        )
        summary["cases"].append(
            {
                "dataset": triplet.dataset,
                "case_id": triplet.case_id,
                "out_dir": str(out_case_dir),
                "slices": info["slices"],
                "infection_positive_slices": info["infection_positive"],
            }
        )
        summary["total_slices"] += info["slices"]
        summary["total_infection_positive_slices"] += info["infection_positive"]
        print(
            f"[{idx}/{len(triplets)}] {triplet.dataset}/{triplet.case_id}: "
            f"{info['slices']} slices"
        )

    summary_path = output_root / "conversion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Conversion done.")
    print(f"Output root: {output_root}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
