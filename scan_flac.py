import os
from pathlib import Path
from typing import List

import soundfile as sf
from tqdm.auto import tqdm
import shutil


def list_flacs(root: str) -> List[Path]:
    """
    Recursively collect all .flac files under a given root directory.
    """
    root_path = Path(root)
    return sorted(root_path.rglob("*.flac"))


def is_valid_flac(path: Path) -> bool:
    """
    Try to open a FLAC file with soundfile.
    If LibsndfileError or any exception is raised, treat it as corrupted.
    """
    try:
        with sf.SoundFile(str(path), "r") as f:
            _ = f.read(1)
        return True
    except Exception as e:
        # You can print e if you want, but it's enough to mark file as bad
        return False


def copy_good_files(good_files: List[Path], src_root: Path, dst_root: Path) -> None:
    """
    Copy all valid FLACs from src_root to dst_root, preserving directory structure.
    """
    for src in tqdm(good_files, desc="Copying valid FLACs"):
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main():
    # 1) Adjust these two paths to your actual dataset locations
    src_root = Path("/home/bkairatuly/Kazakh_Speech_Corpus_2_unpacked/ISSAI_KSC2/Train")
    dst_root = Path("/home/bkairatuly/Kazakh_Speech_Corpus_2_unpacked/ISSAI_KSC2_Train_clean")

    # 2) List all FLACs
    flacs = list_flacs(str(src_root))
    print(f"Found {len(flacs)} FLAC files under {src_root}")

    good_files = []
    bad_files = []

    # 3) Validate each FLAC with soundfile
    for fpath in tqdm(flacs, desc="Checking FLAC files"):
        if is_valid_flac(fpath):
            good_files.append(fpath)
        else:
            bad_files.append(fpath)

    print(f"Valid FLACs: {len(good_files)}")
    print(f"Corrupted FLACs: {len(bad_files)}")

    # 4) Save a list of bad files for inspection
    bad_list_path = src_root.parent / "bad_flacs.txt"
    with bad_list_path.open("w", encoding="utf-8") as f:
        for bf in bad_files:
            f.write(str(bf) + "\n")
    print(f"List of corrupted files written to: {bad_list_path}")

    # 5) Optionally copy only the valid FLACs to a clean directory
    #    Comment this block out if you don't want a clean copy.
    copy_good_files(good_files, src_root, dst_root)
    print(f"Copied valid FLACs to: {dst_root}")


if __name__ == "__main__":
    main()