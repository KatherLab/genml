from abc import ABC, abstractmethod
import torch
from transformers import AutoModel
from ..models.hyenadna import HyenaDNAPreTrainedModel

class EncoderStrategy(ABC):
    @abstractmethod
    def create_model(self, **kwargs) -> torch.nn.Module:
        pass

    @abstractmethod
    def extract_features(self, outputs, cls=False) -> torch.Tensor:
        pass


class DNABERT2EncoderStrategy(EncoderStrategy):
    def __init__(self, pretrained_model_name: str, device: str = 'cpu'):
        self.pretrained_model_name = pretrained_model_name
        self.device = device

    def create_model(self, **kwargs) -> torch.nn.Module:
        return AutoModel.from_pretrained(self.pretrained_model_name, trust_remote_code=True)

    def extract_features(self, outputs, cls=False) -> torch.Tensor:
        if cls:
            return outputs[1] #cls_token as features
        else:
            return outputs[0]


class HyenaDNAEncoderStrategy(EncoderStrategy):
    def __init__(self, pretrained_model_name: str, model_config_path: str, download: bool = False, device: str = 'cpu'):
        self.pretrained_model_name = pretrained_model_name
        self.model_config_path = model_config_path
        self.download = download
        self.device = device

    def create_model(self, **kwargs) -> torch.nn.Module:
        config = kwargs.get('config', None) # in case the config is setup in the encoder.yml for hyenadna
        if self.pretrained_model_name in ['hyenadna-tiny-1k-seqlen',
                                          'hyenadna-small-32k-seqlen',
                                          'hyenadna-medium-160k-seqlen',
                                          'hyenadna-medium-450k-seqlen',
                                          'hyenadna-large-1m-seqlen']:
            return HyenaDNAPreTrainedModel.from_pretrained(
                path=self.model_config_path,
                model_name=self.pretrained_model_name,
                download=self.download,
                config=config,
                device=self.device,
                use_head=False,  # We are only using this for feature extraction
                n_classes=2,  # This won't be used
            )
        else:
            raise ValueError(f"Unsupported pretrained model name: {self.pretrained_model_name}")

    def extract_features(self, outputs, cls=False) -> torch.Tensor:
        if cls:
            return outputs[:, 0, :] #cls_token as features
        else:
            return outputs
