import torch
from typing import List, Optional, Dict, Union
from abc import ABC, abstractmethod
from omegaconf import DictConfig

from ..feature_extractor.audio_feature_extractor import AudioFeatureExtractor


class AudioTokeniser(ABC, torch.nn.Module):
    text_tokeniser = None

    @abstractmethod
    def audio_represent(self, wav: torch.Tensor, lens: Optional[torch.Tensor] = None) -> List[Dict]:
        pass

    @abstractmethod
    def stringify_representation(self, reps: List[Dict], mode: str = "test") -> List[str]:
        pass

    @abstractmethod
    def string_tokenise(self, audio_repr: List[str], return_tensors: Optional[str] = None) -> dict:
        pass

    @abstractmethod
    def tokenise(self, wav: torch.Tensor, lens: Optional[torch.Tensor] = None) -> dict:
        return self.string_tokenise(self.audio_stringify(wav, lens))

    @abstractmethod
    def build_prompt(
        self,
        wav: torch.Tensor,
        lens: Optional[torch.Tensor] = None,
        output_modality: Optional[str] = None,
    ) -> dict:
        return self.string_tokenise(self.audio_stringify(wav, lens))

    @abstractmethod
    def prepare_sample(self, sample: dict, **tokenise_kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def decode_sample(self, tokens: torch.Tensor, output_modality: str = "SPEECH") -> Union[torch.Tensor, str]:
        pass

    @abstractmethod
    def get_ignore_tokens(self, used_token_modality: str) -> List[int]:
        pass


def _init_feature_extractor(fe_type: str, cfg: DictConfig) -> AudioFeatureExtractor:
    if fe_type == "hubert":
        from ..feature_extractor.hubert_feature_extractor import HubertFeatureExtractor
        return HubertFeatureExtractor(**cfg)
    elif fe_type in ("wav2vec2", "wav2vec2_kazakh"):
        from ..feature_extractor.wav2vec2_feature_extractor import Wav2Vec2FeatureExtractor
        return Wav2Vec2FeatureExtractor(**cfg)
    else:
        raise ValueError(f"Unknown speech tokeniser type: {fe_type}")

def tokeniser_factory(cfg: DictConfig) -> AudioTokeniser:
    num_units = None

    if hasattr(cfg, "feature_extractor") and cfg.feature_extractor is not None:
        if "num_units" in cfg.feature_extractor:
            num_units = cfg.feature_extractor.num_units

    if hasattr(cfg, "params") and cfg.params is not None and getattr(cfg.params, "num_units", None) is not None:
        num_units = cfg.params.num_units

    if num_units is None:
        raise ValueError(
            "num_units is not set. Define tokeniser.feature_extractor.num_units or tokeniser.params.num_units."
        )

    cfg.params.num_units = num_units

    feature_extractor: Optional[AudioFeatureExtractor] = None
    load_fe = getattr(cfg.params, "load_fe", False)

    if load_fe:
        if not hasattr(cfg, "feature_extractor") or cfg.feature_extractor is None:
            raise ValueError("load_fe=True but tokeniser.feature_extractor is missing in config.")
        feature_extractor = _init_feature_extractor(cfg.feature_extractor_type, cfg.feature_extractor)

    if cfg.tokeniser_type == "unit":
        from .unit_tokeniser import UnitTokeniser

        return UnitTokeniser(feature_extractor, **cfg.params)
    elif cfg.tokeniser_type == "interleave":
        from .interleaving_tokeniser import InterleavingTokeniser

        return InterleavingTokeniser(feature_extractor, **cfg.params)
    else:
        raise ValueError(f"Unknown tokeniser type: {cfg.tokeniser_type}")