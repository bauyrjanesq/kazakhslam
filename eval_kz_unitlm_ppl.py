import os
import json
import math
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from omegaconf import OmegaConf

from slamkit.tokeniser import tokeniser_factory
from slamkit.model.unit_lm import UnitLM


class JsonlDataset(Dataset):
    def __init__(self, path: str):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class EncodedDataset(Dataset):
    def __init__(self, encoded_samples):
        self.encoded_samples = encoded_samples

    def __len__(self):
        return len(self.encoded_samples)

    def __getitem__(self, idx):
        return self.encoded_samples[idx]


def collate_lm_batch(batch, pad_token_id: int):
    max_len = max(len(sample["input_ids"]) for sample in batch)

    input_ids_list = []
    labels_list = []

    for sample in batch:
        ids = sample["input_ids"]
        labs = sample["labels"]

        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids = torch.cat(
                [
                    ids,
                    torch.full((pad_len,), pad_token_id, dtype=torch.long),
                ]
            )
            labs = torch.cat(
                [
                    labs,
                    torch.full((pad_len,), -100, dtype=torch.long),
                ]
            )

        input_ids_list.append(ids)
        labels_list.append(labs)

    input_ids = torch.stack(input_ids_list, dim=0)
    labels = torch.stack(labels_list, dim=0)

    return {"input_ids": input_ids, "labels": labels}


def prepare_encoded_dataset(tokens_path: str, tokeniser_cfg_path: str, device: str):
    cfg = OmegaConf.load(tokeniser_cfg_path)

    tokeniser = tokeniser_factory(cfg.tokeniser)
    if hasattr(tokeniser, "to"):
        tokeniser.to(device)

    pad_token_id = cfg.tokeniser.params.pad_token_id

    dataset = JsonlDataset(tokens_path)
    encoded_samples = []

    for sample in tqdm(dataset, desc="Encoding samples"):
        encoded = tokeniser.prepare_sample(sample)

        input_ids = encoded["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            ids_tensor = input_ids.clone().detach()
        else:
            ids_tensor = torch.tensor(input_ids, dtype=torch.long)

        if ids_tensor.ndim == 2 and ids_tensor.size(0) == 1:
            ids_tensor = ids_tensor[0]

        labels_tensor = ids_tensor.clone()
        labels_tensor[labels_tensor == pad_token_id] = -100

        encoded_samples.append(
            {
                "input_ids": ids_tensor,
                "labels": labels_tensor,
            }
        )

    return EncodedDataset(encoded_samples), pad_token_id


def compute_perplexity(
    encoded_dataset: Dataset,
    pad_token_id: int,
    checkpoint_path: str,
    batch_size: int,
    device: str,
):
    model = UnitLM.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()

    dataloader = DataLoader(
        encoded_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_lm_batch(batch, pad_token_id),
    )

    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            num_tokens = (labels != -100).sum().item()
            total_nll += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)

    return avg_nll, ppl, total_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Compute perplexity of Kazakh UnitLM on token JSONL."
    )
    parser.add_argument(
        "--tokens_path",
        type=str,
        required=True,
        help="Path to JSONL with audio_repr tokens (e.g. data/kz_tokens_full.jsonl)",
    )
    parser.add_argument(
        "--tokeniser_cfg",
        type=str,
        required=True,
        help="Path to tokeniser config YAML (e.g. config/tokeniser/unit_w2v2_kazakh_500.yaml)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to trained UnitLM checkpoint (e.g. /home/bkairatuly/outputs/kz_unitlm/checkpoint-17625)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use: 'cuda' or 'cpu'",
    )

    args = parser.parse_args()

    encoded_dataset, pad_token_id = prepare_encoded_dataset(
        tokens_path=args.tokens_path,
        tokeniser_cfg_path=args.tokeniser_cfg,
        device=args.device,
    )

    avg_nll, ppl, total_tokens = compute_perplexity(
        encoded_dataset=encoded_dataset,
        pad_token_id=pad_token_id,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        device=args.device,
    )

    print(f"Total tokens: {total_tokens}")
    print(f"Average NLL: {avg_nll:.6f}")
    print(f"Perplexity:  {ppl:.6f}")


if __name__ == "__main__":
    main()