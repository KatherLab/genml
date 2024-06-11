import pandas as pd
import torch
import numpy as np
from typing import Dict, List
from .encoder_factory import EncoderFactory
from .tokenizer_factory import TokenizerFactory

def load_data(file_path: str, columns: list) -> pd.DataFrame:
    return pd.read_csv(file_path, usecols=columns)

def preprocess_data(data: pd.DataFrame, text_column: str) -> Dict[str, list]:
    grouped = data.groupby('Patient_ID')[text_column].apply(list).to_dict()
    return grouped

def feature_extraction(grouped_texts: Dict[str, list], encoder_type: str, encoder_params: dict, tokenizer_type: str, tokenizer_params: dict, device: str) -> Dict[str, torch.Tensor]:
    tokenizer = TokenizerFactory.create_tokenizer(tokenizer_type, **tokenizer_params)
    encoder_strategy = EncoderFactory.create_encoder(encoder_type, device=device, **encoder_params)
    model = encoder_strategy.create_model(**encoder_params).to(device)
    all_features = {}
    for patient_id, texts in grouped_texts.items():
        inputs = tokenizer.tokenize(texts).to(device)
        with torch.no_grad():
            outputs = model(inputs)
        features = encoder_strategy.extract_features(outputs)
        all_features[patient_id] = features
    return all_features

def save_features(all_features: Dict[str, torch.Tensor], output_dir: str):
    for patient_id, features in all_features.items():
        output_path = f"{output_dir}/{patient_id}_features.npy"
        np.save(output_path, features.cpu().numpy())
