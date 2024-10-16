from abc import ABC, abstractmethod
import os
import torch
from pathlib import Path
from transformers import AutoModel, AutoModelForMaskedLM
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from enformer_pytorch import Enformer, from_pretrained

from ..f_models.hyenadna import HyenaDNAPreTrainedModel
from .tokenizer_factory import TokenizerFactory

class EncoderStrategy(ABC):
    def __init__(self, tokenizer_type: str, tokenizer_params: dict, device: str = 'cpu', **kwargs):
        self.device = device
        self.tokenizer = TokenizerFactory.create_tokenizer(tokenizer_type, **tokenizer_params)
        self.model = self.create_model(**kwargs).to(self.device)

    @abstractmethod
    def create_model(self, **kwargs) -> torch.nn.Module:
        """Returns the model to be used for feature extraction."""
        pass

    def encode(self, text: str, feature_type='mean_pooling') -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model is not initialized, please call create_model first.")
        
        # Tokenize the text and move to the specified device
        inputs = self.tokenizer.tokenize(text).to(self.device) 

        # Model inference      
        with torch.no_grad():
            outputs = self.model(inputs)

        # Extract features based on the specified feature_type
        return self.extract_features(outputs, feature_type)
        
    @abstractmethod
    def extract_features(self, outputs, pooling_type, per_mut: bool = False) -> torch.Tensor:
        """Method to extract specific features from the model outputs."""
        pass


class DNABERT2(EncoderStrategy):
    def __init__(self, pretrained_model_name: str, tokenizer_type: str, tokenizer_params: dict, cache_dir: str, download: bool = False, device: str = 'cpu'):
        self.pretrained_model_name = pretrained_model_name
        self.cache_dir = Path(cache_dir)
        self.download = download
        super().__init__(tokenizer_type=tokenizer_type, tokenizer_params=tokenizer_params, device=device)

    def create_model(self, **kwargs) -> torch.nn.Module:
        # check cache_dir
        if self.download:
            print(f"Loading model from pretrained model: {self.pretrained_model_name}")
            return AutoModel.from_pretrained(self.pretrained_model_name, cache_dir=self.cache_dir, trust_remote_code=True)
        else:
            model_path = Path(self.cache_dir) / 'models--zhihan1996--DNABERT-2-117M/snapshots/d064dece8a8b41d9fb8729fbe3435278786931f1'
            print(f"Loading model from cache directory: {model_path}")
            return AutoModel.from_pretrained(model_path, trust_remote_code=True)

    def extract_features(self, outputs, pooling_type, per_mut: bool = False) -> torch.Tensor:
        hidden_states = outputs[0] # [1, sequence_length, 768]
        print('hidden_states shape:', hidden_states.shape)

        if pooling_type == 'mean_pooling':
            # embedding = torch.mean(hidden_states[0], dim=0) #[768]   
            embedding = torch.mean(hidden_states, dim=1) # [1, 768]  
        else: # cls_token
            embedding = outputs[1]  # [1, 768]
            #embedding = hidden_states[:,0,:] # [1, 768]

        print('embedding.shape:', embedding.shape)

        return embedding




class HyenaDNA(EncoderStrategy):
    def __init__(self, pretrained_model_name: str, cache_dir: str, tokenizer_type: str, tokenizer_params: dict, download: bool = False, device: str = 'cpu'):
        self.pretrained_model_name = pretrained_model_name
        self.cache_dir = Path(cache_dir)
        self.download = download

        self.supported_models = {
            'hyenadna-tiny-1k-seqlen',
            'hyenadna-small-32k-seqlen',
            'hyenadna-medium-160k-seqlen',
            'hyenadna-medium-450k-seqlen',
            'hyenadna-large-1m-seqlen'
        }
        # Call the super class constructor to initialize tokenizer and model
        super().__init__(tokenizer_type=tokenizer_type, tokenizer_params=tokenizer_params, device=device)

    def create_model(self, **kwargs) -> torch.nn.Module:
        if self.pretrained_model_name not in self.supported_models:
            raise ValueError(f"Unsupported pretrained model name: {self.pretrained_model_name}")
        
        config = kwargs.get('config', None) # in case the config is setup in the encoder.yml for hyenadna
        return HyenaDNAPreTrainedModel.from_pretrained(
            path=self.cache_dir,
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
            # embedding = torch.mean(hidden_states[0], dim=0) # [256]
            embedding = torch.mean(hidden_states, dim=1) # [1, 256]
        else: #'eos' actually
            '''
            https://github.com/ChongWuLab/dna_foundation_benchmark/blob/main/job_scripts/inference_hyena.py
            '''
            embedding = hidden_states[:, hidden_states.shape[1]-1, :]  # [1, 256]
            

        print('embedding.shape:', embedding.shape)

        return embedding


class HyenaDNA2(EncoderStrategy):
    def __init__(self, pretrained_model_name: str, tokenizer_type: str, tokenizer_params: dict, device: str = 'cpu'):
        self.pretrained_model_name = pretrained_model_name
        super().__init__(tokenizer_type=tokenizer_type, tokenizer_params=tokenizer_params, device=device)

    def create_model(self, **kwargs) -> torch.nn.Module:
        return AutoModelForCausalLM.from_pretrained(self.pretrained_model_name, trust_remote_code=True)

    def extract_features(self, outputs, pooling_type) -> torch.Tensor:
        # print('type(outputs):', type(outputs))
        hidden_states = outputs # [1, sequence_length, 256], eg. [1, 65, 256]
        print('hidden_states shape:', hidden_states.shape)

        if pooling_type == 'mean_pooling':
            # embedding = torch.mean(hidden_states[0], dim=0) # [256]
            embedding = torch.mean(hidden_states, dim=1) # [1, 256]
        else: #'eos' actually
            '''
            https://github.com/ChongWuLab/dna_foundation_benchmark/blob/main/job_scripts/inference_hyena.py
            '''
            embedding = hidden_states[:, hidden_states.shape[1]-1, :]  # [1, 256]
            

        print('embedding.shape:', embedding.shape)

        return embedding
    

class NucleotideTransformer(EncoderStrategy):
    def __init__(self, pretrained_model_name: str, tokenizer_type: str, tokenizer_params: dict, device: str = 'cpu'):
        self.pretrained_model_name = pretrained_model_name
        super().__init__(tokenizer_type=tokenizer_type, tokenizer_params=tokenizer_params, device=device)


    def create_model(self, **kwargs) -> torch.nn.Module:
        return AutoModelForMaskedLM.from_pretrained(self.pretrained_model_name, trust_remote_code=True)

    def encode(self, text: str, feature_type='mean_pooling') -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model is not initialized, please call create_model first.")
        
        # Tokenize the text and move to the specified device
        inputs = self.tokenizer.tokenize(text).to(self.device) 

        #attention_mask = inputs != self.tokenizer.pad_token_id

        # Model inference      
        with torch.no_grad():
            # outputs = self.model(inputs,
            #                      attention_mask=attention_mask,
            #                      encoder_attention_mask=attention_mask,
            #                      output_hidden_states=True)
            outputs = self.model(inputs, output_hidden_states=True)

        # Extract features based on the specified feature_type
        return self.extract_features(outputs, feature_type)
    
    def extract_features(self, outputs, pooling_type) -> torch.Tensor:
        hidden_states = outputs['hidden_states'][-1] # [1, sequence_length, 1024]
        print('hidden_states shape:', hidden_states.shape)

        if pooling_type == 'mean_pooling':
            #attention_mask = torch.unsqueeze(attention_mask, dim=-1)
            #embedding = torch.sum(attention_mask*hidden_states, axis=1)/torch.sum(attention_mask, axis=1)
            embedding = torch.mean(hidden_states, dim=1) # [1, 1024]
        else: # cls_token
            embedding = hidden_states[:,0,:] # [1, 1024]

        print('embedding.shape:', embedding.shape)

        return embedding
    

'''
Problem: triton installation is conflict with DNABERT2
Fix: using another env?
'''
# class EnFormer(EncoderStrategy):
#     def __init__(self, pretrained_model_name: str, tokenizer_type: str, tokenizer_params: dict, cache_dir: str, device: str = 'cpu'):
#         self.pretrained_model_name = pretrained_model_name
#         super().__init__(tokenizer_type=tokenizer_type, tokenizer_params=tokenizer_params, cache_dir=cache_dir, device=device)


#     def create_model(self, **kwargs) -> torch.nn.Module:
#         return from_pretrained('EleutherAI/enformer-official-rough', use_tf_gamma = False, target_length = 1, cache=self.cache)


#     def encode(self, text: str, feature_type='mean_pooling') -> torch.Tensor:
#         if self.model is None:
#             raise RuntimeError("Model is not initialized, please call create_model first.")
        
#         # Tokenize the text and move to the specified device
#         inputs = self.tokenizer.tokenize(text).to(self.device) 

#         # Model inference      
#         with torch.no_grad():
#             outputs = self.model(inputs, return_embeddings = True)

#         # Extract features based on the specified feature_type
#         return self.extract_features(outputs, feature_type)
    
#     def extract_features(self, outputs, pooling_type) -> torch.Tensor:
#         hidden_states = outputs[1] # [1, 3072]????
#         print('hidden_states shape:', hidden_states.shape)

#         if pooling_type == 'mean_pooling':
#             embedding = torch.mean(hidden_states, dim=1) 
#         else: # cls_token
#             embedding = hidden_states[:,0,:]

#         print('embedding.shape:', embedding.shape)

#         return embedding