import pandas as pd
import torch
import os
from typing import Dict, List
import logging
import time
from datetime import timedelta
from tqdm import tqdm
from pathlib import Path

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
    grouped_chunks: a dict with 'patient_id' as keys, while 'concatenated_chunks' as values
    '''
    grouped_texts = data.groupby(uni_column)[text_column].apply(list).to_dict()
    grouped_chunks = {}

    for patient_id, texts in grouped_texts.items():
        #print('len(texts)', patient_id+'_' +str(len(texts))) # e.g 789 alt_sequences
        concatenated_chunks = [] # list to store chunks per patient
        for i in range(0, len(texts), chunk_size): 
            chunk = texts[i:i + chunk_size]
            chunk = [text + sep_token for text in chunk] # Add sep_token at the end of each text within a chunk
            concatenated_chunk = ''.join(chunk) 
            #print('concatenated_chunk', concatenated_chunk)
            concatenated_chunks.append(concatenated_chunk)
        grouped_chunks[patient_id] = concatenated_chunks
        #print('concatenated_chunks', patient_id+'_' +str(len(concatenated_chunks))) #2 chunks
        #print('concatenated_chunks', concatenated_chunks)
    return grouped_chunks


def save_feature_tensor(features, output_dir, filename):
    output_path = output_dir / f"{filename}"
    torch.save(features, output_path)


def feature_extraction(
        grouped_chunks: Dict[str, List[str]], 
        encoder_type: str, 
        encoder_params: dict, 
        tokenizer_type: str, 
        tokenizer_params: dict, 
        device: str, 
        cls: bool, 
        stack_feature: bool, 
        output_dir: str):
    
    has_gpu = torch.cuda.is_available()
    print(f"GPU is available: {has_gpu}")
    device = torch.device(device) if "cuda" in device and has_gpu else torch.device("cpu")

    tokenizer = TokenizerFactory.create_tokenizer(tokenizer_type, **tokenizer_params)
    encoder_strategy = EncoderFactory.create_encoder(encoder_type, device, **encoder_params)
    model = encoder_strategy.create_model().to(device)
  

    # Create the dynamic output directory path
    dynamic_output_dir = Path(output_dir) / f"{encoder_type}_stack_{stack_feature}_cls_{cls}"
    dynamic_output_dir.mkdir(parents=True, exist_ok=True)

    # Create logfile and set up logging
    logfile_name = f"logfile_{time.strftime('%Y-%m-%d_%H-%M-%S')}_{os.getpid()}.log"
    logdir = dynamic_output_dir / logfile_name
    logging.basicConfig(filename=logdir, level=logging.INFO, format="[%(levelname)s] %(message)s")
    #logging.getLogger().addHandler(logging.StreamHandler())

    logging.info(f"Feature extracting started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Model: {encoder_type}\n")

    total_start_time = time.time()

    # Scan for existing feature files
    logging.info("Scanning for existing feature ...")
    existing_instances = [f.stem for f in dynamic_output_dir.glob("**/*.pt")] if dynamic_output_dir.exists() else []

    instances = list(grouped_chunks.keys())
    grouped_chunks_to_process = {
        patient_id: chunks
        for patient_id, chunks in grouped_chunks.items()
        if patient_id not in existing_instances
    }

    num_total = len(instances)
    num_processed, num_skipped = 0, 0
    error_instances: List[str] = []

    if existing_instances:
        logging.info(f"Skipping {len(existing_instances)} already processed patients out of {num_total} total patients...")

    for patient_id, chunks in tqdm(grouped_chunks_to_process.items(), desc="\nProcessing patients"):
        logging.info(f"\n\n===== Processing patient {patient_id} =====")
        pattern = f"{patient_id}*.pt"
        matching_files = list(dynamic_output_dir.glob(pattern))
    
        if not matching_files: 
            try:
                features_list = []
                for idx, text in enumerate(chunks):
                    #print('text length:', len(text))
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
                
                num_processed += 1

            except Exception as e:
                logging.error(f"Failed extract features, skipping... Error: {e}")
                error_instances.append(patient_id)
                continue
        else:
            logging.info(".pt file for this patinet already exists. Skipping...")
            num_skipped += 1

    logging.info(f"\n\n\n===== End-to-end processing time of {num_total} slides: {str(timedelta(seconds=(time.time() - total_start_time)))} =====")
    logging.info(f"Summary: Processed {num_processed} patients, encountered {len(error_instances)} errors, skipped {num_skipped} slides")
    if error_instances:
        logging.info("The following slides were not processed due to errors:\n  " + "\n  ".join(error_instances))