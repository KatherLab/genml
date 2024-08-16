import pandas as pd
import torch
import os

from typing import Dict, List
from .encoder_factory import EncoderFactory
from .tokenizer_factory import TokenizerFactory


def load_data(file_path: str, uni_column: str, num_patients: int = None) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    if num_patients is not None:
        data = data[data[uni_column].isin(data[uni_column].unique()[:num_patients])]
    return data


def preprocess_data(data: pd.DataFrame, text_column: str, uni_column: str, chunk_size: int, sep_token: str = '[SEP]') -> Dict[str, List[str]]:
    ''''
    chunk_size: the num(not length) of alt_sequences in one chunk.
    chunk: a list of alt_sequences/texts with length of chunk_size under one patient.
    concatenated_chunk: concatenated alt_sequences with sep_token for one chunk.
    concatenated_chunks: a list of 'concatenated_chunk's of one patient
    '''
    grouped_texts = data.groupby(uni_column)[text_column].apply(list).to_dict()
    processed_texts = {}

    for patient_id, texts in grouped_texts.items():
        #print('len(texts)', patient_id+'_' +str(len(texts))) # e.g 789 alt_sequences
        concatenated_chunks = [] # list to store chunks per patient
        for i in range(0, len(texts), chunk_size): 
            chunk = texts[i:i + chunk_size]
            chunk = [text + sep_token for text in chunk] # Add sep_token at the end of each text within a chunk
            concatenated_chunk = ''.join(chunk) 
            print('concatenated_chunk', concatenated_chunk)
            concatenated_chunks.append(concatenated_chunk)
        processed_texts[patient_id] = concatenated_chunks
        #print('concatenated_chunks', patient_id+'_' +str(len(concatenated_chunks))) #2 chunks
        print('concatenated_chunks', concatenated_chunks)
    return processed_texts


def save_feature_tensor(features: torch.Tensor, output_dir: str, file_name: str):
    output_path = os.path.join(output_dir, file_name)
    torch.save(features.cpu(), output_path)


def feature_extraction(
        grouped_texts: Dict[str, List[str]], 
        encoder_type: str, 
        encoder_params: dict, 
        tokenizer_type: str, 
        tokenizer_params: dict, 
        device: str, 
        cls: bool, 
        stack_feature: bool, 
        output_dir: str):
    tokenizer = TokenizerFactory.create_tokenizer(tokenizer_type, **tokenizer_params)
    encoder_strategy = EncoderFactory.create_encoder(encoder_type, device, **encoder_params)
    model = encoder_strategy.create_model().to(device)
  

    # Create the dynamic output directory path
    dynamic_output_dir = os.path.join(output_dir, f"{encoder_type}_stack_{stack_feature}")
    if not os.path.exists(dynamic_output_dir):
        os.makedirs(dynamic_output_dir)
    
    for patient_id, texts in grouped_texts.items():
        features_list = []
        for idx, text in enumerate(texts):
            print('text length:', len(text))
            inputs = tokenizer.tokenize(text).to(device)
            with torch.no_grad():
                outputs = model(inputs) # if inputs already include a batch dimension
            features = encoder_strategy.extract_features(outputs, cls)
            features_list.append(features.cpu())
            torch.cuda.empty_cache()  # clean GPU cache
            
            if not stack_feature:
                save_feature_tensor(features, dynamic_output_dir, f"{patient_id}_{idx}.pt")
        
        if stack_feature:
            stacked_features = torch.cat(features_list, dim=1)
            save_feature_tensor(stacked_features, dynamic_output_dir, f"{patient_id}.pt")