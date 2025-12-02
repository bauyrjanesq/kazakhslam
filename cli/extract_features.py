import logging
logger = logging.getLogger(__name__)

import os
import json
import torchaudio
import pickle
from glob import iglob
from tqdm import tqdm
from multiprocessing import Pool
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import hydra
from omegaconf import DictConfig
from typing import Optional, Tuple
from functools import partial

from slamkit.tokeniser import tokeniser_factory


class WavDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        ext: str = "flac",
        cache_path: Optional[str] = None,
        sample_rate: int = 16000,
        torchaudio_backend: Optional[str] = None,
    ):
        self.torchaudio_backend = torchaudio_backend
        self.sample_rate = sample_rate

        save_path = None
        if cache_path is not None:
            os.makedirs(os.path.join(cache_path, "data"), exist_ok=True)
            save_path = f"{cache_path}/data/{data_path.split('/')[-2]}.pkl"
            if os.path.exists(save_path):
                with open(save_path, "rb") as f:
                    self.files = pickle.load(f)
                return

        files = iglob(os.path.join(data_path, f"**/*.{ext}"), recursive=True)

        with Pool() as p:
            raw_entries = list(
                tqdm(
                    p.imap(
                        partial(
                            WavDataset._load_sample_meta,
                            backend=self.torchaudio_backend,
                        ),
                        files,
                    )
                )
            )

        # Filter out files that failed to load metadata
        self.files = [e for e in raw_entries if e is not None]

        # Sort by duration to minimise padding
        self.files = sorted(self.files, key=lambda x: x[1], reverse=True)

        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(self.files, f)

    @staticmethod
    def _load_sample_meta(f_path: str, backend: Optional[str] = None):
        try:
            info = torchaudio.info(f_path, backend=backend)
            return f_path, info.num_frames
        except Exception as e:
            logger.warning(f"Failed to read metadata for {f_path}: {e}")
            return None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        f_name, _ = self.files[idx]
        try:
            data, sr = torchaudio.load(f_name, backend=self.torchaudio_backend)
        except Exception as e:
            logger.warning(f"Failed to load audio {f_name}: {e}. Replacing with silence.")
            # Return a short silent waveform to keep batch shape valid
            data = torch.zeros(1)
            sr = self.sample_rate

        if sr != self.sample_rate:
            data = torchaudio.functional.resample(data, sr, self.sample_rate)
        if data.dim() == 2:
            data = data.mean(dim=0)

        length = len(data)
        if length == 0:
            logger.warning(f"Audio {f_name} has zero length after processing. Replacing with 1-sample silence.")
            data = torch.zeros(1)
            length = 1

        return f_name, data, length

    def skip(self, skip: int):
        self.files = self.files[skip:]

    def take(self, take: int):
        self.files = self.files[:take]


def pad_wav_collate(batch) -> Tuple[list, torch.Tensor, torch.Tensor]:
    f_names, wavs, lens = zip(*batch)
    return f_names, pad_sequence(wavs, batch_first=True, padding_value=0), torch.tensor(lens)


@hydra.main(config_name="extract_features", config_path="../config", version_base="1.3")
def extract_features(cfg: DictConfig):
    """
    Extracts discrete units and durations from audio files using a tokeniser's feature extractor
    and writes them to a JSONL file.
    """
    tokeniser = tokeniser_factory(cfg.tokeniser).to(cfg.device)

    ds = WavDataset(
        data_path=cfg.data_path,
        ext=cfg.ext,
        cache_path=cfg.cache_path,
        sample_rate=cfg.sample_rate,
        torchaudio_backend=cfg.torchaudio_backend,
    )

    if cfg.data_skip is not None:
        ds.skip(cfg.data_skip)
    if cfg.data_take is not None:
        ds.take(cfg.data_take)

    dl = DataLoader(
        ds,
        collate_fn=pad_wav_collate,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    if os.path.exists(cfg.out_path):
        logging.warning(f"{cfg.out_path} already exists. Appending to it.")
    os.makedirs(os.path.dirname(cfg.out_path), exist_ok=True)

    out_file = open(cfg.out_path, "a+")

    for f, w, l in tqdm(dl):
        speech_repr = tokeniser.audio_represent(w.to(cfg.device), l.to(cfg.device))
        out = []
        for cur_f, cur_repr in zip(f, speech_repr):
            cur_repr["file_name"] = cur_f
            out.append(json.dumps(cur_repr) + "\n")
        out_file.writelines(out)

    out_file.close()


if __name__ == "__main__":
    extract_features()