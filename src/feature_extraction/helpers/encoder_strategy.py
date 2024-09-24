from abc import ABC, abstractmethod
import torch
from pathlib import Path
from transformers import AutoModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

from ..f_models.hyenadna import HyenaDNAPreTrainedModel

class EncoderStrategy(ABC):
    @abstractmethod
    def create_model(self, **kwargs) -> torch.nn.Module:
        pass
    
    @abstractmethod
    def extract_features(self, outputs, pooling_type='mean_pooling') -> torch.Tensor:
        pass


class DNABERT2(EncoderStrategy):
    def __init__(self, pretrained_model_name: str, device: str = 'cpu'):
        self.pretrained_model_name = pretrained_model_name
        self.device = device

    def create_model(self, **kwargs) -> torch.nn.Module:
        return AutoModel.from_pretrained(self.pretrained_model_name, trust_remote_code=True)

    def extract_features(self, outputs, pooling_type) -> torch.Tensor:
        hidden_states = outputs[0] # [1, sequence_length, 768]
        print('hidden_states shape:', hidden_states.shape)

        if pooling_type == 'mean_pooling':
            embedding = torch.mean(hidden_states[0], dim=0) #[768]      
        elif pooling_type == 'max_pooling':
            embedding = torch.max(hidden_states[0], dim=0)[0] #[768]
        else: # cls_token
            embedding = outputs[1].squeeze()  # [1, 768]->[768]
            #embedding = hidden_states[:,0,:] # [1, 768]

        print('embedding.shape:', embedding.shape)

        return embedding


class HyenaDNA(EncoderStrategy):
    def __init__(self, pretrained_model_name: str, model_config_path: str, download: bool = False, device: str = 'cpu'):
        self.pretrained_model_name = pretrained_model_name
        self.model_config_path = Path(model_config_path)
        self.download = download
        self.device = device

        self.supported_models = {
            'hyenadna-tiny-1k-seqlen',
            'hyenadna-small-32k-seqlen',
            'hyenadna-medium-160k-seqlen',
            'hyenadna-medium-450k-seqlen',
            'hyenadna-large-1m-seqlen'
        }

    def create_model(self, **kwargs) -> torch.nn.Module:
        if self.pretrained_model_name not in self.supported_models:
            raise ValueError(f"Unsupported pretrained model name: {self.pretrained_model_name}")
        
        config = kwargs.get('config', None) # in case the config is setup in the encoder.yml for hyenadna
        return HyenaDNAPreTrainedModel.from_pretrained(
            path=self.model_config_path,
            model_name=self.pretrained_model_name,
            download=self.download,
            config=config,
            device=self.device,
            use_head=False,  
            n_classes=2,  # This won't be used
        )

    def extract_features(self, outputs, pooling_type) -> torch.Tensor:
        # print('type(outputs):', type(outputs))
        hidden_states = outputs # [1, sequence_length, 256], eg. [1, 65, 256]
        print('hidden_states shape:', hidden_states.shape)

        if pooling_type == 'mean_pooling':
            embedding = torch.mean(hidden_states[0], dim=0) # [256]
        elif pooling_type == 'max_pooling':
            embedding = torch.max(hidden_states[0], dim=0)[0]  # [256]
        else: #'eos' actually
            '''
            https://github.com/ChongWuLab/dna_foundation_benchmark/blob/main/job_scripts/inference_hyena.py
            '''
            embedding = hidden_states[:, hidden_states.shape[1]-1, :].squeeze()  # [1, 256]->[256]
            

        print('embedding.shape:', embedding.shape)

        return embedding


class HyenaDNA2(EncoderStrategy):
    def __init__(self, pretrained_model_name: str, device: str = 'cpu'):
        self.pretrained_model_name = pretrained_model_name
        self.device = device

    def create_model(self, **kwargs) -> torch.nn.Module:
        return AutoModelForCausalLM.from_pretrained(self.pretrained_model_name, trust_remote_code=True)

    def extract_features(self, outputs, pooling_type) -> torch.Tensor:
        # print('type(outputs):', type(outputs))
        hidden_states = outputs # [1, sequence_length, 256], eg. [1, 65, 256]
        print('hidden_states shape:', hidden_states.shape)

        if pooling_type == 'mean_pooling':
            embedding = torch.mean(hidden_states[0], dim=0) # [256]
        elif pooling_type == 'max_pooling':
            embedding = torch.max(hidden_states[0], dim=0)[0]  # [256]
        else: #'eos' actually
            '''
            https://github.com/ChongWuLab/dna_foundation_benchmark/blob/main/job_scripts/inference_hyena.py
            '''
            embedding = hidden_states[:, hidden_states.shape[1]-1, :].squeeze()  # [1, 256]->[256]
            

        print('embedding.shape:', embedding.shape)

        return embedding
    
class NucleotideTransformer(EncoderStrategy):
    pass