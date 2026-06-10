#!/usr/bin/env python3
"""
Browse NIfTI files (.nii / .nii.gz) for PDAtt datasets.

Features:
- Single mode: browse every file independently.
-qqqqqqqqq Paired mode: view 3 synced panels per case (image, lung mask, infection mask)
    across Dataset1, Dataset2, and Dataset3.
- Optional export of all slices to PNG images.

Controls (interactive mode):
- Right / D: next slice
- Left  / A: previous slice
- Up    / W / N: next file or next case
- Down  / S / P: previous file or previous case
- Home: first slice
- End: last slice
- Q or Esc: quit
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def is_nifti_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def find_nifti_files(root: Path, recursive: bool = True) -> List[Path]:
    if recursive:
        files = [p for p in root.rglob("*") if p.is_file() and is_nifti_file(p)]
    else:
        files = [p for p in root.iterdir() if p.is_file() and is_nifti_file(p)]
    return sorted(files)


def strip_nifti_suffix(path: Path) -> str:
    name = path.name
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return name[:-7]
    if lower.endswith(".nii"):
        return name[:-4]
    return path.stem


def sorted_nifti_files(files: Iterable[Path]) -> List[Path]:
    def _key(p: Path):
        stem = strip_nifti_suffix(p)
        if stem.isdigit():
            return (0, int(stem), stem)
        return (1, stem.lower(), stem)

    return sorted(files, key=_key)


def load_volume(path: Path) -> np.ndarray:
    volume = nib.load(str(path)).get_fdata(dtype=np.float32)
    if volume.ndim == 4:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D or 4D NIfTI, got shape {volume.shape} for {path}")
    return volume


def get_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return volume[index, :, :]
    if axis == 1:
        return volume[:, index, :]
    return volume[:, :, index]


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    finite = slice_2d[np.isfinite(slice_2d)]
    if finite.size == 0:
        return np.zeros_like(slice_2d, dtype=np.float32)

    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if high <= low:
        return np.zeros_like(slice_2d, dtype=np.float32)

    out = (slice_2d - low) / (high - low)
    out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)


def normalize_mask(slice_2d: np.ndarray) -> np.ndarray:
    data = np.nan_to_num(slice_2d.astype(np.float32), nan=0.0)
    if data.size == 0:
        return np.zeros_like(data, dtype=np.float32)

    if np.max(data) > 1.0 or np.min(data) < 0.0:
        data = (data > 0).astype(np.float32)
    else:
        data = np.clip(data, 0.0, 1.0)
    return data


def slice_count(volume: np.ndarray, axis: int) -> int:
    return int(volume.shape[axis])


def make_display_slice(volume: np.ndarray, axis: int, index: int, rotate_k: int, is_mask: bool) -> np.ndarray:
    sl = get_slice(volume, axis, index)
    if rotate_k % 4:
        sl = np.rot90(sl, rotate_k % 4)
    if is_mask:
        return normalize_mask(sl)
    return normalize_slice(sl)


def find_matching_nifti(folder: Path, stem: str) -> Optional[Path]:
    for ext in (".nii", ".nii.gz"):
        candidate = folder / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    matches = sorted([p for p in folder.glob(f"{stem}*.nii*") if p.is_file()])
    return matches[0] if matches else None


@dataclass
class CaseTriplet:
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


def build_dataset1_triplets(root_9ct: Path) -> List[CaseTriplet]:
    d1 = root_9ct / "Dataset1"
    image_path = d1 / "rp_im" / "tr_im.nii"
    lung_path = d1 / "tr_lung_mask" / "tr_lungmasks_updated.nii"
    inf_path = d1 / "tr_mask.nii" / "tr_mask.nii"

    if image_path.exists() and lung_path.exists() and inf_path.exists():
        return [
            CaseTriplet(
                dataset="Dataset1",
                case_id="tr_im",
                image_path=image_path,
                lung_mask_path=lung_path,
                infection_mask_path=inf_path,
            )
        ]

    print("[WARN] Dataset1 is incomplete. Expected rp_im/tr_lung_mask/tr_mask.nii structure.")
    return []


def build_dataset2_triplets(root_9ct: Path) -> List[CaseTriplet]:
    d2 = root_9ct / "Dataset2"
    image_dir = d2 / "rp_im"
    lung_dir = d2 / "rp_lung_msk"
    inf_dir = d2 / "rp_msk"

    if not image_dir.exists() or not lung_dir.exists() or not inf_dir.exists():
        print("[WARN] Dataset2 folders missing (rp_im/rp_lung_msk/rp_msk).")
        return []

    image_files = sorted_nifti_files([p for p in image_dir.iterdir() if p.is_file() and is_nifti_file(p)])

    triplets: List[CaseTriplet] = []
    for image_path in image_files:
        stem = strip_nifti_suffix(image_path)
        lung_path = find_matching_nifti(lung_dir, stem)
        inf_path = find_matching_nifti(inf_dir, stem)

        if not lung_path or not inf_path:
            print(f"[WARN] Dataset2 case {stem}: missing mask file, skipped.")
            continue

        triplets.append(
            CaseTriplet(
                dataset="Dataset2",
                case_id=stem,
                image_path=image_path,
                lung_mask_path=lung_path,
                infection_mask_path=inf_path,
            )
        )

    return triplets


def build_dataset3_triplets(root_9ct: Path) -> List[CaseTriplet]:
    d3 = root_9ct / "Dataset3"
    ct_dir = d3 / "COVID-19-CT-Seg_20cases"
    lung_dir = d3 / "Lung_Mask"
    inf_dir = d3 / "Infection_Mask"

    if not ct_dir.exists() or not lung_dir.exists() or not inf_dir.exists():
        print("[WARN] Dataset3 folders missing (COVID-19-CT-Seg_20cases/Lung_Mask/Infection_Mask).")
        return []

    triplets: List[CaseTriplet] = []
    for ct_name, mask_name in DATASET3_CASE_MAPPING:
        image_path = ct_dir / ct_name
        lung_path = lung_dir / mask_name
        inf_path = inf_dir / mask_name

        if not image_path.exists() or not lung_path.exists() or not inf_path.exists():
            print(f"[WARN] Dataset3 case missing: CT={ct_name}, MASK={mask_name}")
            continue

        case_id = strip_nifti_suffix(Path(mask_name))
        triplets.append(
            CaseTriplet(
                dataset="Dataset3",
                case_id=case_id,
                image_path=image_path,
                lung_mask_path=lung_path,
                infection_mask_path=inf_path,
            )
        )

    return triplets


def build_paired_triplets(root_9ct: Path, datasets: Sequence[str]) -> List[CaseTriplet]:
    selected = set(datasets)
    triplets: List[CaseTriplet] = []

    if "1" in selected:
        triplets.extend(build_dataset1_triplets(root_9ct))
    if "2" in selected:
        triplets.extend(build_dataset2_triplets(root_9ct))
    if "3" in selected:
        triplets.extend(build_dataset3_triplets(root_9ct))

    return triplets


class NiftiBrowser:
    def __init__(self, files: Iterable[Path], axis: int = 2, rotate_k: int = 1, start_file: int = 0):
        self.files = list(files)
        if not self.files:
            raise ValueError("No NIfTI files were provided.")

        self.axis = axis
        self.rotate_k = rotate_k % 4
        self.file_idx = max(0, min(start_file, len(self.files) - 1))
        self.slice_idx = 0

        self.volume = None
        self.current_shape = None

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.image_artist = None

        self._load_current_file(reset_slice=True)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _load_current_file(self, reset_slice: bool) -> None:
        self.volume = load_volume(self.files[self.file_idx])
        self.current_shape = self.volume.shape
        n_slices = slice_count(self.volume, self.axis)

        if reset_slice:
            self.slice_idx = 0
        else:
            self.slice_idx = min(self.slice_idx, n_slices - 1)

        self._redraw()

    def _make_display_slice(self) -> np.ndarray:
        return make_display_slice(self.volume, self.axis, self.slice_idx, self.rotate_k, is_mask=False)

    def _title(self) -> str:
        path = self.files[self.file_idx]
        n_slices = slice_count(self.volume, self.axis)
        return (
            f"File {self.file_idx + 1}/{len(self.files)} | {path.name}\n"
            f"Shape: {self.current_shape} | Axis: {self.axis} | Slice: {self.slice_idx + 1}/{n_slices}\n"
            "A/D: prev/next slice | W/S: prev/next file | Q: quit"
        )

    def _redraw(self) -> None:
        display = self._make_display_slice()
        if self.image_artist is None:
            self.image_artist = self.ax.imshow(display, cmap="gray")
            self.ax.axis("off")
        else:
            self.image_artist.set_data(display)

        self.ax.set_title(self._title(), fontsize=10)
        self.fig.canvas.draw_idle()

    def _step_slice(self, step: int) -> None:
        n_slices = slice_count(self.volume, self.axis)
        self.slice_idx = max(0, min(self.slice_idx + step, n_slices - 1))
        self._redraw()

    def _step_file(self, step: int) -> None:
        self.file_idx = max(0, min(self.file_idx + step, len(self.files) - 1))
        self._load_current_file(reset_slice=True)

    def _on_key(self, event) -> None:
        key = (event.key or "").lower()

        if key in {"right", "d"}:
            self._step_slice(+1)
        elif key in {"left", "a"}:
            self._step_slice(-1)
        elif key in {"up", "w", "n"}:
            self._step_file(+1)
        elif key in {"down", "s", "p"}:
            self._step_file(-1)
        elif key == "home":
            self.slice_idx = 0
            self._redraw()
        elif key == "end":
            self.slice_idx = slice_count(self.volume, self.axis) - 1
            self._redraw()
        elif key in {"q", "escape"}:
            plt.close(self.fig)

    def show(self) -> None:
        plt.tight_layout()
        plt.show()


def export_all_slices(files: Iterable[Path], output_dir: Path, axis: int = 2, rotate_k: int = 1) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, nii_file in enumerate(files, start=1):
        volume = load_volume(nii_file)
        n_slices = slice_count(volume, axis)

        stem = nii_file.name
        if stem.lower().endswith(".nii.gz"):
            stem = stem[:-7]
        elif stem.lower().endswith(".nii"):
            stem = stem[:-4]

        volume_dir = output_dir / f"{idx:03d}_{stem}"
        volume_dir.mkdir(parents=True, exist_ok=True)

        for slice_idx in range(n_slices):
            sl = make_display_slice(volume, axis, slice_idx, rotate_k, is_mask=False)
            out_path = volume_dir / f"slice_{slice_idx:04d}.png"
            plt.imsave(out_path, sl, cmap="gray")

        print(f"[Exported] {nii_file} -> {volume_dir} ({n_slices} slices)")


def export_triplet_slices(triplets: Sequence[CaseTriplet], output_dir: Path, axis: int = 2, rotate_k: int = 1) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, case in enumerate(triplets, start=1):
        image_vol = load_volume(case.image_path)
        lung_vol = load_volume(case.lung_mask_path)
        inf_vol = load_volume(case.infection_mask_path)

        n_slices = min(
            slice_count(image_vol, axis),
            slice_count(lung_vol, axis),
            slice_count(inf_vol, axis),
        )
        if n_slices <= 0:
            print(f"[WARN] {case.dataset}/{case.case_id}: no slice to export, skipped.")
            continue

        case_dir = output_dir / f"{idx:03d}_{case.dataset}_{case.case_id}"
        case_dir.mkdir(parents=True, exist_ok=True)

        for s_idx in range(n_slices):
            image_sl = make_display_slice(image_vol, axis, s_idx, rotate_k, is_mask=False)
            lung_sl = make_display_slice(lung_vol, axis, s_idx, rotate_k, is_mask=True)
            inf_sl = make_display_slice(inf_vol, axis, s_idx, rotate_k, is_mask=True)

            plt.imsave(case_dir / f"slice_{s_idx:04d}_image.png", image_sl, cmap="gray")
            plt.imsave(case_dir / f"slice_{s_idx:04d}_lung_mask.png", lung_sl, cmap="gray")
            plt.imsave(case_dir / f"slice_{s_idx:04d}_infection_mask.png", inf_sl, cmap="gray")

        print(
            f"[Exported] {case.dataset}/{case.case_id} -> {case_dir} "
            f"({n_slices} slices x 3 images)"
        )


class TripletBrowser:
    def __init__(self, triplets: Sequence[CaseTriplet], axis: int = 2, rotate_k: int = 1, start_case: int = 0):
        self.triplets = list(triplets)
        if not self.triplets:
            raise ValueError("No paired cases were provided.")

        self.axis = axis
        self.rotate_k = rotate_k % 4
        self.case_idx = max(0, min(start_case, len(self.triplets) - 1))
        self.slice_idx = 0

        self.image_vol: Optional[np.ndarray] = None
        self.lung_vol: Optional[np.ndarray] = None
        self.inf_vol: Optional[np.ndarray] = None
        self.current_n_slices = 0

        self.fig, self.axes = plt.subplots(1, 3, figsize=(14, 5))
        self.image_artists = [None, None, None]

        self._load_current_case(reset_slice=True)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _load_current_case(self, reset_slice: bool) -> None:
        case = self.triplets[self.case_idx]
        self.image_vol = load_volume(case.image_path)
        self.lung_vol = load_volume(case.lung_mask_path)
        self.inf_vol = load_volume(case.infection_mask_path)

        self.current_n_slices = min(
            slice_count(self.image_vol, self.axis),
            slice_count(self.lung_vol, self.axis),
            slice_count(self.inf_vol, self.axis),
        )

        if self.current_n_slices <= 0:
            raise ValueError(f"Case {case.dataset}/{case.case_id} has no valid slices.")

        if reset_slice:
            self.slice_idx = 0
        else:
            self.slice_idx = min(self.slice_idx, self.current_n_slices - 1)

        self._redraw()

    def _redraw(self) -> None:
        case = self.triplets[self.case_idx]

        image_sl = make_display_slice(self.image_vol, self.axis, self.slice_idx, self.rotate_k, is_mask=False)
        lung_sl = make_display_slice(self.lung_vol, self.axis, self.slice_idx, self.rotate_k, is_mask=True)
        inf_sl = make_display_slice(self.inf_vol, self.axis, self.slice_idx, self.rotate_k, is_mask=True)

        displays = [image_sl, lung_sl, inf_sl]
        panel_titles = ["Image (rp_im)", "Lung Mask", "Infection Mask"]

        for i, ax in enumerate(self.axes):
            if self.image_artists[i] is None:
                self.image_artists[i] = ax.imshow(displays[i], cmap="gray", vmin=0.0, vmax=1.0)
                ax.axis("off")
            else:
                self.image_artists[i].set_data(displays[i])
            ax.set_title(panel_titles[i], fontsize=10)

        self.fig.suptitle(
            (
                f"Case {self.case_idx + 1}/{len(self.triplets)} | {case.dataset} | {case.case_id}\n"
                f"Slice: {self.slice_idx + 1}/{self.current_n_slices} | Axis: {self.axis} | "
                f"A/D: prev-next slice | W/S: prev-next case | Q: quit"
            ),
            fontsize=11,
        )
        self.fig.canvas.draw_idle()

    def _step_slice(self, step: int) -> None:
        self.slice_idx = max(0, min(self.slice_idx + step, self.current_n_slices - 1))
        self._redraw()

    def _step_case(self, step: int) -> None:
        self.case_idx = max(0, min(self.case_idx + step, len(self.triplets) - 1))
        self._load_current_case(reset_slice=True)

    def _on_key(self, event) -> None:
        key = (event.key or "").lower()

        if key in {"right", "d"}:
            self._step_slice(+1)
        elif key in {"left", "a"}:
            self._step_slice(-1)
        elif key in {"up", "w", "n"}:
            self._step_case(+1)
        elif key in {"down", "s", "p"}:
            self._step_case(-1)
        elif key == "home":
            self.slice_idx = 0
            self._redraw()
        elif key == "end":
            self.slice_idx = self.current_n_slices - 1
            self._redraw()
        elif key in {"q", "escape"}:
            plt.close(self.fig)

    def show(self) -> None:
        plt.tight_layout()
        plt.show()


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1] / "data" / "9CTscans"

    parser = argparse.ArgumentParser(description="View NIfTI files in single mode or paired Dataset1/2/3 mode.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "paired"],
        default="paired",
        help="single: browse files independently | paired: image+lung+infection synced by case.",
    )
    parser.add_argument("--root", type=Path, default=default_root, help="Folder containing .nii/.nii.gz files.")
    parser.add_argument("--recursive", action="store_true", help="Search files recursively (default: true).")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="Disable recursive search.")
    parser.set_defaults(recursive=True)

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["1", "2", "3"],
        choices=["1", "2", "3"],
        help="Used in paired mode. Select datasets to include (default: 1 2 3).",
    )

    parser.add_argument("--axis", type=int, choices=[0, 1, 2], default=2, help="Slice axis (0/1/2).")
    parser.add_argument("--rotate-k", type=int, default=1, help="Rotate each slice by 90-degree steps.")
    parser.add_argument("--start-file", type=int, default=1, help="1-based starting index (file in single mode, case in paired mode).")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of files (0 means no limit).")

    parser.add_argument("--export-dir", type=Path, default=None, help="Export all slices as PNG under this folder.")
    parser.add_argument("--export-only", action="store_true", help="Only export PNG files, do not open viewer.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    root = args.root.resolve()
    if not root.exists():
        print(f"Root folder does not exist: {root}")
        return 1

    if args.mode == "single":
        files = find_nifti_files(root, recursive=args.recursive)
        if args.max_files and args.max_files > 0:
            files = files[: args.max_files]

        if not files:
            print(f"No .nii/.nii.gz files found under: {root}")
            return 1

        print(f"[Single mode] Found {len(files)} NIfTI files under: {root}")
        for idx, path in enumerate(files, start=1):
            print(f"{idx:03d}. {path}")

        if args.export_dir is not None:
            export_all_slices(files, args.export_dir, axis=args.axis, rotate_k=args.rotate_k)
            print(f"All slices exported to: {args.export_dir.resolve()}")

        if args.export_only:
            return 0

        start_file_zero_based = max(0, args.start_file - 1)
        browser = NiftiBrowser(
            files=files,
            axis=args.axis,
            rotate_k=args.rotate_k,
            start_file=start_file_zero_based,
        )
        browser.show()
        return 0

    triplets = build_paired_triplets(root, args.datasets)
    if args.max_files and args.max_files > 0:
        triplets = triplets[: args.max_files]

    if not triplets:
        print(
            "No paired cases found. In paired mode, root should point to data/9CTscans "
            "and include Dataset1/2/3 folders."
        )
        return 1

    print(f"[Paired mode] Found {len(triplets)} cases from Dataset{','.join(args.datasets)}")
    for idx, case in enumerate(triplets, start=1):
        print(
            f"{idx:03d}. {case.dataset}/{case.case_id} | "
            f"img={case.image_path.name} | lung={case.lung_mask_path.name} | inf={case.infection_mask_path.name}"
        )

    if args.export_dir is not None:
        export_triplet_slices(triplets, args.export_dir, axis=args.axis, rotate_k=args.rotate_k)
        print(f"All triplet slices exported to: {args.export_dir.resolve()}")

    if args.export_only:
        return 0

    start_case_zero_based = max(0, args.start_file - 1)
    browser = TripletBrowser(
        triplets=triplets,
        axis=args.axis,
        rotate_k=args.rotate_k,
        start_case=start_case_zero_based,
    )
    browser.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
