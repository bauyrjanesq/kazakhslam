import argparse
import os
from pathlib import Path

from tqdm.auto import tqdm


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def move_one_level_up(root_dir: Path, split: str, ext: str) -> None:
    split_root = root_dir / split
    if not split_root.exists():
        print(f"Split {split_root} does not exist, skipping.")
        return

    print(f"\nProcessing split: {split_root}")
    files = sorted(split_root.rglob(f"*.{ext}"))

    if not files:
        print(f"No *.{ext} files found under {split_root}, skipping.")
        return

    print(f"Found {len(files)} *.{ext} files under {split_root}")
    print("Example before reorg:")
    for f in files[:5]:
        print(f"  {f}")

    for src in tqdm(files, desc=f"Reorg {split}"):
        rel = src.relative_to(split_root)
        parts = rel.parts
        if len(parts) < 2:
            # Already at correct level or unexpected layout
            continue

        # Drop the first component ('Train' or 'Test')
        new_rel = Path(*parts[1:])
        dst = split_root / new_rel

        if dst.exists():
            # Already moved or duplicate, skip
            continue

        ensure_dir(dst.parent)
        os.rename(src, dst)

    # Try to remove the now-empty first-level dirs (Train/Test under Dev/Train)
    for first_level in (split_root / "Train", split_root / "Test"):
        if first_level.exists():
            try:
                # Remove only if empty; if not empty, leave it
                first_level.rmdir()
                print(f"Removed empty directory: {first_level}")
            except OSError:
                print(f"Directory not empty, not removed: {first_level}")

    # Show a couple of examples after reorg
    files_after = sorted(split_root.rglob(f"*.{ext}"))
    print(f"Example after reorg in {split_root}:")
    for f in files_after[:5]:
        print(f"  {f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/bkairatuly/KSC2_custom",
        help="Root directory of KSC2_custom (containing Dev and Train)"
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="flac",
        help="Audio extension to process"
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    print(f"Reorganising KSC2 under: {root_dir}")
    print(f"Using extension: .{args.ext}")

    for split in ("Dev", "Train"):
        move_one_level_up(root_dir, split, args.ext)

    print("\nDone reorg.")


if __name__ == "__main__":
    main()