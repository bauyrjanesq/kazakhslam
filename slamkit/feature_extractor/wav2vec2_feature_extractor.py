import torch
import numpy as np
from typing import Optional
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import joblib

from .audio_feature_extractor import AudioFeatureExtractor


class Wav2Vec2FeatureExtractor(AudioFeatureExtractor):
    """
    Wav2Vec2-based discrete unit extractor with KMeans clustering.

    Expected config (from Hydra):

    tokeniser:
      feature_extractor_type: wav2vec2_kazakh
      feature_extractor:
        pretrained_model: aismlv/wav2vec2-large-xlsr-kazakh
        kmeans_path: /home/bkairatuly/slamkit/data/kmeans/kz_w2v2_ksc2_500.joblib
        layer: 11
        num_units: 500
        cache_path: null
        compile: false
        load_config_only: false
    """

    def __init__(
        self,
        pretrained_model: str,
        kmeans_path: str,
        layer: int = 11,
        num_units: int = 500,
        cache_path: Optional[str] = None,
        compile: bool = False,
        load_config_only: bool = False,
        **kwargs,
    ):
        super().__init__()

        # Wav2Vec2 models are trained on 16kHz audio
        self._sample_rate = 16000
        self.layer = layer
        self.num_units = num_units
        self.cache_path = cache_path
        self.compile = compile
        self.load_config_only = load_config_only

        if load_config_only:
            # Only keep metadata, do not load heavy stuff
            self.processor = None
            self.model = None
            self.kmeans = None
            return

        # HF processor + model
        self.processor = Wav2Vec2Processor.from_pretrained(pretrained_model)
        self.model = Wav2Vec2Model.from_pretrained(
            pretrained_model,
            output_hidden_states=True,
        )
        self.model.eval()

        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model.to("cuda")

        # KMeans with `n_clusters == num_units`
        self.kmeans = joblib.load(kmeans_path)
        if hasattr(self.kmeans, "cluster_centers_"):
            assert (
                self.kmeans.cluster_centers_.shape[0] == num_units
            ), f"KMeans has {self.kmeans.cluster_centers_.shape[0]} clusters, but num_units={num_units}"

    # ---- required by AudioFeatureExtractor ----
    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @torch.no_grad()
    def extract(self, wav: torch.Tensor, lens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param wav: (B, T) or (T,) float32 tensor in [-1, 1]
        :param lens: (B,) lengths in samples (not strictly required here)
        :return: (B, L) tensor of discrete unit ids
        """
        if self.model is None or self.processor is None or self.kmeans is None:
            raise RuntimeError(
                "Feature extractor was initialised with load_config_only=True; "
                "cannot call extract()."
            )

        device = self.model.device

        # ---- ensure batch of 1D CPU numpy arrays ----
        if wav.dim() == 1:
            # single example (T,) -> make batch of size 1
            wav_list = [wav.cpu().numpy()]
        elif wav.dim() == 2:
            # batch (B, T)
            wav_list = [w.cpu().numpy() for w in wav]
        else:
            raise ValueError(f"Expected wav with shape (T,) or (B, T), got {wav.shape}")

        # Processor works on CPU numpy / lists
        inputs = self.processor(
            wav_list,
            sampling_rate=self._sample_rate,
            return_tensors="pt",
            padding=True,
        )

        # Now move to model device
        input_values = inputs.input_values.to(device)      # (B, T')
        attention_mask = inputs.attention_mask.to(device)  # (B, T')

        outputs = self.model(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        if self.layer is not None:
            hidden = outputs.hidden_states[self.layer]     # (B, L, D)
        else:
            hidden = outputs.last_hidden_state             # (B, L, D)

        B, L, D = hidden.shape
        feats = hidden.reshape(-1, D).cpu().numpy()        # (B*L, D)

        # KMeans -> discrete unit ids
        unit_ids = self.kmeans.predict(feats)              # (B*L,)
        unit_ids = torch.from_numpy(unit_ids).view(B, L)   # (B, L)

        return unit_ids
    
    def get_unit_duration(self, wav: torch.Tensor, lens: Optional[torch.Tensor] = None):
        """
        For compatibility: return (units, durations).
        Units are frame-level; durations are all 1.
        UnitTokeniser will collapse repeats and sum durations.
        """
        unit_ids = self.extract(wav, lens)                     # (B, L)
        durations = torch.ones_like(unit_ids)                  # dummy durations
        return unit_ids, durations