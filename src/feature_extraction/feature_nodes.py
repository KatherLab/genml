import pandas as pd
import torch
import os

from typing import Dict, List
from .encoder_factory import EncoderFactory
from .tokenizer_factory import TokenizerFactory


'''def load_data(file_path: str, columns: list) -> pd.DataFrame:
    return pd.read_csv(file_path, usecols=columns)'''
def load_data(file_path: str, columns: list, num_patients: int = None) -> pd.DataFrame:
    data = pd.read_csv(file_path, usecols=columns)
    if num_patients is not None:
        data = data[data['Patient_ID'].isin(data['Patient_ID'].unique()[:num_patients])]
    return data

def preprocess_data(data: pd.DataFrame, text_column: str, chunk_size: int) -> Dict[str, List[str]]:
    grouped_texts = data.groupby('Patient_ID')[text_column].apply(list).to_dict()
    processed_texts = {}

    for patient_id, texts in grouped_texts.items():
        concatenated_chunks = []
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            concatenated_chunks.append(''.join(chunk))
        processed_texts[patient_id] = concatenated_chunks

    return processed_texts

def feature_extraction(grouped_texts: Dict[str, List[str]], encoder_type: str, encoder_params: dict, tokenizer_type: str, tokenizer_params: dict, device: str) -> Dict[str, torch.Tensor]:
    tokenizer = TokenizerFactory.create_tokenizer(tokenizer_type, **tokenizer_params)
    encoder_strategy = EncoderFactory.create_encoder(encoder_type, device, **encoder_params)
    model = encoder_strategy.create_model().to(device)
    
    all_features = {}
    for patient_id, texts in grouped_texts.items():
        features_list = []
        for text in texts:
            inputs = tokenizer.tokenize(text).to(device)
            with torch.no_grad():
                #outputs = model(inputs.unsqueeze(0))  # 增加 batch 维度
                outputs = model(inputs)  # if inputs already include a batch dimension
            features = encoder_strategy.extract_features(outputs)
            features_list.append(features.cpu())
            torch.cuda.empty_cache()  # 清理 GPU 缓存
        all_features[patient_id] = torch.cat(features_list, dim=1)
        torch.cuda.empty_cache()  # 清理 GPU 缓存
    return all_features

def save_features(all_features: Dict[str, torch.Tensor], output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for patient_id, features in all_features.items():
        output_path = os.path.join(output_dir, f"{patient_id}_features.pt")
        torch.save(features, output_path)
