import os
import argparse
import random
from pathlib import Path
from typing import List, Tuple

import torchaudio
from tqdm.auto import tqdm


def collect_audio_files(root_dir: Path, ext: str) -> List[Path]:
    pattern = f"**/*.{ext}"
    files = sorted(root_dir.glob(pattern))
    if not files:
        print(f"No *.{ext} files found under {root_dir} with pattern {pattern}")
    else:
        print(f"Found {len(files)} *.{ext} files under {root_dir}")
        print("Example files:")
        for f in files[:5]:
            print(f"  {f}")
    return files


def split_files(files: List[Path], dev_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
    rng = random.Random(seed)
    files_shuffled = files[:]
    rng.shuffle(files_shuffled)
    n_total = len(files_shuffled)
    n_dev = max(1, int(n_total * dev_ratio))
    dev_files = files_shuffled[:n_dev]
    train_files = files_shuffled[n_dev:]
    return train_files, dev_files


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return
    if mode == "symlink":
        dst.symlink_to(src)
    elif mode == "copy":
        import shutil
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def populate_split(
    files: List[Path],
    split_name: str,
    root_dir: Path,
    out_dir: Path,
    mode: str
) -> None:
    split_root = out_dir / split_name
    ensure_dir(split_root)
    desc = f"{split_name} files"
    for src in tqdm(files, desc=desc):
        rel = src.relative_to(root_dir)
        dst = split_root / rel
        link_or_copy(src, dst, mode)


def compute_hours(files: List[Path]) -> Tuple[float, int]:
    total_seconds = 0.0
    for f in tqdm(files, desc="Computing hours"):
        try:
            info = torchaudio.info(str(f))
            total_seconds += info.num_frames / info.sample_rate
        except Exception as e:
            print(f"Failed to read {f}: {e}")
    hours = total_seconds / 3600.0
    return hours, len(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/bkairatuly/Kazakh_Speech_Corpus_2_unpacked_full/ISSAI_KSC2",
        help="Root directory of original ISSAI_KSC2"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/bkairatuly/KSC2_custom",
        help="Output directory for custom Train/Dev split"
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="flac",
        help="Audio extension to search for (e.g. flac, wav)"
    )
    parser.add_argument(
        "--dev_ratio",
        type=float,
        default=0.05,
        help="Fraction of files to use for Dev"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="symlink",
        choices=["symlink", "copy"],
        help="Whether to symlink or copy audio files"
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    print(f"Root dir: {root_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Dev ratio: {args.dev_ratio}")
    print(f"Mode: {args.mode}")
    print(f"Extension: .{args.ext}")

    files = collect_audio_files(root_dir, args.ext)
    if not files:
        print("No audio files found. Check --root_dir and --ext.")
        return

    train_files, dev_files = split_files(files, args.dev_ratio, args.seed)

    print(f"Train files: {len(train_files)}")
    print(f"Dev files: {len(dev_files)}")

    populate_split(train_files, "Train", root_dir, out_dir, args.mode)
    populate_split(dev_files, "Dev", root_dir, out_dir, args.mode)

    print("Computing hours for Train...")
    train_hours, n_train = compute_hours(train_files)
    print(f"Train: {train_hours:.2f} hours ({n_train} files)")

    print("Computing hours for Dev...")
    dev_hours, n_dev = compute_hours(dev_files)
    print(f"Dev: {dev_hours:.2f} hours ({n_dev} files)")

    print("Done.")


if __name__ == "__main__":
    main()