import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import joblib

KSC2_ROOT = "/home/bkairatuly/Kazakh_Speech_Corpus_2_unpacked/ISSAI_KSC2/Train"
OUTPUT_KMEANS_PATH = "/home/bkairatuly/slamkit/data/kmeans/kz_w2v2_ksc2_500.joblib"
MAX_FILES = None
FRAMES_PER_FILE = 200
N_CLUSTERS = 500
TARGET_TOTAL_FRAMES = 1_000_000
SAMPLE_RATE = 16000
BATCH_SIZE = 4


def iter_flac_files(root_dir: str):
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(".flac"):
                yield os.path.join(dirpath, fname)


def load_and_resample_numpy(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray | None:
    try:
        wav, sr = torchaudio.load(path)
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}")
        return None

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.dim() == 2:
        wav = wav.mean(dim=0)

    wav = wav.float().cpu().numpy()
    if wav.size == 0:
        print(f"Warning: empty audio after resample for {path}")
        return None
    return wav


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained("aismlv/wav2vec2-large-xlsr-kazakh")
    model = Wav2Vec2Model.from_pretrained("aismlv/wav2vec2-large-xlsr-kazakh").to(device)
    model.eval()

    flac_files = list(iter_flac_files(KSC2_ROOT))
    if MAX_FILES is not None:
        flac_files = flac_files[:MAX_FILES]

    all_frames = []
    total_frames = 0

    with torch.no_grad():
        for start in tqdm(range(0, len(flac_files), BATCH_SIZE)):
            batch_paths = flac_files[start:start + BATCH_SIZE]
            if not batch_paths:
                break

            batch_wavs = []
            for path in batch_paths:
                wav_np = load_and_resample_numpy(path)
                if wav_np is None:
                    continue
                batch_wavs.append(wav_np)

            if not batch_wavs:
                continue

            inputs = processor(
                batch_wavs,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(device)

            outputs = model(input_values)
            hidden_states = outputs.last_hidden_state.cpu().numpy()

            for h in hidden_states:
                t = h.shape[0]
                n = min(FRAMES_PER_FILE, t)
                if n <= 0:
                    continue
                idx = np.random.choice(t, size=n, replace=False)
                sampled = h[idx]
                all_frames.append(sampled)
                total_frames += n
                if total_frames >= TARGET_TOTAL_FRAMES:
                    break

            if total_frames >= TARGET_TOTAL_FRAMES:
                break

    if not all_frames:
        raise RuntimeError("No features collected from KSC2.")

    X = np.concatenate(all_frames, axis=0)
    print(f"Collected feature matrix shape: {X.shape}")

    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        batch_size=10_000,
        verbose=1,
        random_state=0,
        n_init="auto",
    )
    kmeans.fit(X)

    os.makedirs(os.path.dirname(OUTPUT_KMEANS_PATH), exist_ok=True)
    joblib.dump(kmeans, OUTPUT_KMEANS_PATH)
    print(f"Saved KMeans to: {OUTPUT_KMEANS_PATH}")


if __name__ == "__main__":
    main()