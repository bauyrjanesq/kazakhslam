from functools import partial
from multiprocessing.pool import ThreadPool
from omegaconf import DictConfig
import hydra
from tqdm import tqdm
from pathlib import Path
import json
import os
import logging

logger = logging.getLogger(__name__)

from slamkit.tokeniser import tokeniser_factory


def process_jsonl(line, tokeniser, requires_meta, meta_path):
    """
    Process a single JSONL line:
    - load JSON
    - optionally merge metadata
    - turn feature dict into string representation via tokeniser
    - drop raw fields we no longer need
    """
    try:
        cur = json.loads(line)

        # If meta is required, load sidecar JSON and merge into current sample
        if requires_meta:
            meta_file = (
                f"{meta_path}/{Path(cur['file_name']).stem}"
                if meta_path
                else os.path.splitext(cur["file_name"])[0]
            )
            if not os.path.exists(meta_file + ".json"):
                logger.warning(f"{meta_file} does not exist. Skipping")
                return
            with open(meta_file + ".json", "r", encoding="utf-8") as f:
                meta = json.load(f)
            cur.update(meta)

        # Use tokeniser to get string representation of units for training
        cur["audio_repr"] = tokeniser.stringify_representation([cur], mode="train")[0]

        # Remove raw fields that are no longer needed after tokenisation
        cur.pop("units", None)
        cur.pop("duration", None)
        cur.pop("text", None)
        cur.pop("aligned_text", None)
        cur.pop("split_sentence", None)

        return json.dumps(cur)
    except Exception as e:
        logger.warning(f"Failed to process {line}. Error: {e}, skipping")
        return


@hydra.main(config_name="prepare_tokens", config_path="../config", version_base="1.3")
def prepare_tokens(cfg: DictConfig):
    """
    Read feature JSONL, add tokeniser string representation, and write out a new JSONL.

    IMPORTANT CHANGE:
    - cfg.out_path is now treated as a FILE path (e.g. data/kz_tokens_full.jsonl)
    - we only create its PARENT directory, not cfg.out_path itself as a folder
    """
    # Build tokeniser (for Kazakh W2V2 units, cfg.tokeniser.params.load_fe should be false)
    tokeniser = tokeniser_factory(cfg.tokeniser)

    # Treat cfg.out_path as a file path, not a directory
    out_path = cfg.out_path

    # Create only the parent directory, if any (e.g. "data/")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # If somehow out_path already exists as a directory, fail explicitly
    if os.path.isdir(out_path):
        raise RuntimeError(
            f"out_path='{out_path}' is a directory, but a file path is expected."
        )

    # If file already exists, delete it so we start fresh
    if os.path.exists(out_path):
        logger.warning(f"{out_path} already exists. Deleting it!")
        os.remove(out_path)

    logger.info("Starting to prepare tokens")

    requires_meta = cfg.tokeniser.get("requires_meta", False)
    meta_path = cfg.meta_path

    # Process input JSONL and write transformed lines to the single out_path file
    with open(cfg.data_path, "r", encoding="utf-8") as f_in, open(
        out_path, "a+", encoding="utf-8"
    ) as f_out:
        with ThreadPool(cfg.n_threads) as p:
            for jsonl in tqdm(
                p.imap(
                    partial(
                        process_jsonl,
                        tokeniser=tokeniser,
                        requires_meta=requires_meta,
                        meta_path=meta_path,
                    ),
                    f_in,
                )
            ):
                if jsonl:
                    f_out.write(jsonl + "\n")


if __name__ == "__main__":
    prepare_tokens()