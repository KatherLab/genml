from abc import ABC, abstractmethod
import os
import torch
from pathlib import Path
import logging
from typing import Optional
from transformers import AutoTokenizer
from transformers import AutoModel, AutoModelForMaskedLM


class EncoderStrategy(ABC):
    def __init__(self, device: str = 'cpu', **kwargs):
        self.device = device

        # Create tokenizer
        self.tokenizer = self.create_tokenizer(**kwargs)
        if self.tokenizer is None:
            logging.error("Tokenizer was not initialized.")
            raise RuntimeError("Tokenizer initialization failed.")

        # Create model
        self.model = self.create_model(**kwargs).to(self.device)
        if self.model is None:
            logging.error("Model was not initialized.")
            raise RuntimeError("Model initialization failed.")
        
    def create_tokenizer(self, **kwargs) -> Optional[AutoTokenizer]:
        if self.download:
            logging.info(f"Downloading tokenizer from {self.pretrained_model_name}...")
            return AutoTokenizer.from_pretrained(self.pretrained_model_name, cache_dir=self.cache_dir, trust_remote_code=True)
        else:
            if not self.checkpoint_path.exists():
                logging.error(f"Tokenizer checkpoint does not exist: {self.checkpoint_path}")
                raise FileNotFoundError(f"Checkpoint path {self.checkpoint_path} not found.")
            logging.info(f"Loading tokenizer from checkpoint: {self.checkpoint_path}")
            return AutoTokenizer.from_pretrained(self.checkpoint_path, trust_remote_code=True)

    def create_model(self, **kwargs) -> torch.nn.Module:
        if self.download:
            logging.info(f"Loading model from pretrained model: {self.pretrained_model_name}")
            return AutoModel.from_pretrained(self.pretrained_model_name, cache_dir=self.cache_dir, trust_remote_code=True)
        else:
            if not self.checkpoint_path.exists():
                logging.error("Model checkpoint does not exist.")
                raise FileNotFoundError(f"Checkpoint path {self.checkpoint_path} not found.")
            logging.info(f"Loading model from cache directory: {self.checkpoint_path}")
            return AutoModel.from_pretrained(self.checkpoint_path, trust_remote_code=True)

    def encode(self, text: str, feature_type='mean_pooling') -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model is not initialized, please call create_model first.")
        
        # # Tokenize the text and move to the specified device
        tok_seq = self.tokenizer(text, add_special_tokens=True, return_tensors='pt')
        inputs = tok_seq['input_ids'].to(self.device)

        # print('inputs.shape:', inputs.shape)

        # Model inference      
        with torch.no_grad():
            outputs = self.model(inputs)

        # Extract features based on the specified feature_type
        return self.extract_feats(outputs, feature_type)

    @abstractmethod
    def extract_feats(self, outputs, pooling_type) -> torch.Tensor:
        """Method to extract specific features from the model outputs."""
        pass


class DNABERT2(EncoderStrategy):
    def __init__(self, pretrained_model_name: str, cache_dir: str, device: str, download: bool = False):
        self.pretrained_model_name = pretrained_model_name
        self.cache_dir = Path(cache_dir)
        self.download = download
        self.checkpoint_path = self.cache_dir / 'models--zhihan1996--DNABERT-2-117M/snapshots/d064dece8a8b41d9fb8729fbe3435278786931f1'
        super().__init__(device=device)

    def extract_feats(self, outputs, pooling_type) -> torch.Tensor:
        hidden_states = outputs[0] # [1, sequence_length, 768]
        # print('hidden_states shape:', hidden_states.shape)

        if pooling_type == 'mean_pooling':
            embedding = torch.mean(hidden_states, dim=1) # [1, 768]
        else: # cls_token
            embedding = outputs[1]  # [1, 768]

        # print('embedding.shape:', embedding.shape)

        return embedding


class HyenaDNA(EncoderStrategy):
    def __init__(self, pretrained_model_name: str, cache_dir: str, device: str, download: bool = False):
        self.pretrained_model_name = pretrained_model_name
        self.cache_dir = Path(cache_dir)
        self.download = download
        self.checkpoint_path = self.cache_dir / 'models--LongSafari--hyenadna-medium-160k-seqlen-hf/snapshots/7ebf71773d22c0ede2cc55cb2be15ee8c289e1ce'
        super().__init__(device=device)

    def create_model(self, **kwargs) -> torch.nn.Module:
        if self.download:
            logging.info(f"Loading model from pretrained model: {self.pretrained_model_name}")
            return AutoModel.from_pretrained(self.pretrained_model_name, cache_dir=self.cache_dir, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        else:
            if not self.checkpoint_path.exists():
                logging.error("Model checkpoint does not exist.")
                raise FileNotFoundError(f"Checkpoint path {self.checkpoint_path} not found.")
            logging.info(f"Loading model from cache directory: {self.checkpoint_path}")
            return AutoModel.from_pretrained(self.checkpoint_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    def extract_feats(self, outputs, pooling_type) -> torch.Tensor:
        hidden_states = outputs[0] # [1, sequence_length, 256]
        # print('hidden_states shape:', hidden_states.shape)

        if pooling_type == 'mean_pooling':
             embedding = torch.mean(hidden_states, dim=1) # [1, 256]
        else: #'eos'
            '''
            https://github.com/ChongWuLab/dna_foundation_benchmark/blob/main/job_scripts/inference_hyena.py
            '''
            embedding = hidden_states[:, hidden_states.shape[1]-1, :]  # [1, 256]

        # print('embedding.shape:', embedding.shape)

        return embedding
    


class NucleotideTransformer(EncoderStrategy):
    def __init__(self, pretrained_model_name: str, cache_dir: str, device: str, download: bool = False):
        self.pretrained_model_name = pretrained_model_name
        self.cache_dir = Path(cache_dir)
        self.download = download
        self.checkpoint_path = self.cache_dir / 'models--InstaDeepAI--nucleotide-transformer-v2-500m-multi-species/snapshots/f1fd7a1df5b19d31b88f11db1ce87caeb1ea4d2a'
        super().__init__(device=device)

    def create_model(self, **kwargs) -> torch.nn.Module:
        if self.download:
            logging.info(f"Loading model from pretrained model: {self.pretrained_model_name}")
            return AutoModelForMaskedLM.from_pretrained(self.pretrained_model_name, cache_dir=self.cache_dir, trust_remote_code=True)
        else:
            if not self.checkpoint_path.exists():
                logging.error("Model checkpoint does not exist.")
                raise FileNotFoundError(f"Checkpoint path {self.checkpoint_path} not found.")
            logging.info(f"Loading model from cache directory: {self.checkpoint_path}")
            return AutoModelForMaskedLM.from_pretrained(self.checkpoint_path, trust_remote_code=True)

    def encode(self, text: str, feature_type='mean_pooling') -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model is not initialized, please call create_model first.")
        
        # Tokenize the text and move to the specified device
        tok_seq = self.tokenizer(text, add_special_tokens=True, return_tensors='pt')
        inputs = tok_seq['input_ids'].to(self.device)

        # self.attention_mask = inputs != self.tokenizer.pad_token_id

        # Model inference      
        with torch.no_grad():
            # outputs = self.model(inputs,
            #                      attention_mask=self.attention_mask,
            #                      encoder_attention_mask=self.attention_mask,
            #                      output_hidden_states=True)
            outputs = self.model(inputs, output_hidden_states=True)

        # Extract features based on the specified feature_type
        return self.extract_feats(outputs, feature_type)
    
    def extract_feats(self, outputs, pooling_type) -> torch.Tensor:
        hidden_states = outputs['hidden_states'][-1] # [1, sequence_length, 1024]
        # print('hidden_states shape:', hidden_states.shape)

        if pooling_type == 'mean_pooling':
            embedding = torch.mean(hidden_states, dim=1) # [1, 1024]
        else:
            raise ValueError(f"Invalid pooling_type: {pooling_type}. Expected 'mean_pooling'.")
        
        # print('embedding.shape:', embedding.shape)

        return embedding
    



class GROVER(EncoderStrategy):
    def __init__(self, pretrained_model_name: str, cache_dir: str, device: str, download: bool = False):
        self.pretrained_model_name = pretrained_model_name
        self.cache_dir = Path(cache_dir)
        self.download = download
        self.checkpoint_path = self.cache_dir / 'models--PoetschLab--GROVER/snapshots/f6ed259a321aacb629cf638a1568c2a40b381cfe'
        super().__init__(device=device)

    def create_model(self, **kwargs) -> torch.nn.Module:
        if self.download:
            logging.info(f"Loading model from pretrained model: {self.pretrained_model_name}")
            return AutoModelForMaskedLM.from_pretrained(self.pretrained_model_name, cache_dir=self.cache_dir, trust_remote_code=True)
        else:
            if not self.checkpoint_path.exists():
                logging.error("Model checkpoint does not exist.")
                raise FileNotFoundError(f"Checkpoint path {self.checkpoint_path} not found.")
            logging.info(f"Loading model from cache directory: {self.checkpoint_path}")
            return AutoModelForMaskedLM.from_pretrained(self.checkpoint_path, trust_remote_code=True)

    def extract_feats(self, outputs, pooling_type) -> torch.Tensor:
        hidden_states = outputs[0] # [1, sequence_length, 609]
        # print('hidden_states shape:', hidden_states.shape)

        if pooling_type == 'mean_pooling':
            embedding = torch.mean(hidden_states, dim=1) # [1, 609]
        else:
            raise ValueError(f"Invalid pooling_type: {pooling_type}. Expected 'mean_pooling'.")

        # print('embedding.shape:', embedding.shape)

        return embedding

