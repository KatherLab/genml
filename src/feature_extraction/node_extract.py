import pandas as pd
import torch
import os
from typing import Dict, List
import logging
import time
from tqdm import tqdm
from pathlib import Path
import h5py

from .helpers.encoder_factory import EncoderFactory
from .helpers.tokenizer_factory import TokenizerFactory
from .helpers.instance_loaders import PatientLoader


def load_data(file_path: str, uni_column: str, num_patients: int = None) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    if num_patients is not None:
        data = data[data[uni_column].isin(data[uni_column].unique()[:num_patients])]
    return data


def save_feature_h5(features, output_dir, filename):
    """ Save features into an h5 file instead of a .pt file """
    output_path = output_dir / f"{filename}.h5"
    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset('feat', data=features.cpu().numpy())  # Save tensor as numpy array


def feature_extraction(
        file_path: str,
        text_column: str, 
        uni_column: str, 
        chunk_size: int, 
        encoder_type: str, 
        encoder_params: dict, 
        tokenizer_type: str, 
        tokenizer_params: dict, 
        device: str, 
        pooling_type: str, 
        stack_feature: bool, 
        output_dir: str, 
        sep_token: str = "[SEP]", 
        batch_size: int = 5,
        num_patients: int = None):
        
    has_gpu = torch.cuda.is_available()
    print(f"GPU is available: {has_gpu}")
    device = torch.device(device if has_gpu and "cuda" in device else "cpu")   

    # Initialize tokenizer and model
    tokenizer = TokenizerFactory.create_tokenizer(tokenizer_type, **tokenizer_params)
    encoder_strategy = EncoderFactory.create_encoder(encoder_type, device, **encoder_params)
    model = encoder_strategy.create_model().to(device)
  
    # Create the dynamic output directory path
    dynamic_output_dir = Path(output_dir) / f"{encoder_type}_stack_{stack_feature}_{pooling_type}"
    dynamic_output_dir.mkdir(parents=True, exist_ok=True)

    # Create logfile and set up logging
    logfile_name = f"logfile_{time.strftime('%Y-%m-%d_%H-%M-%S')}_{os.getpid()}.log"
    logdir = dynamic_output_dir / logfile_name
    logging.basicConfig(filename=logdir, level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.info(f"Feature extracting started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Model: {encoder_type}\n")

    # Scan for existing feature files
    logging.info("Scanning for existing features ...")
    existing_instances = {f.stem.split('_')[0] for f in dynamic_output_dir.glob("**/*.pt")} if dynamic_output_dir.exists() else []

    # Load data, num_patients is used to limit the data, none as default
    data = load_data(file_path, uni_column, num_patients=num_patients)

    # Initialize the PatientLoader
    patient_loader = PatientLoader(data, text_column, uni_column, chunk_size, sep_token, batch_size)

    num_processed, num_skipped = 0, 0
    error_instances = []

    # Batch processing
    for patient_batch in tqdm(patient_loader, desc="\nProcessing patient batches"):
        for patient_id, chunks in patient_batch.items():
            if patient_id in existing_instances:
                logging.info(f"Skipping already processed patient: {patient_id}")
                num_skipped += 1
                continue

            logging.info(f"Processing patient {patient_id}...")
            try:
                features_list = []
                for idx, text in enumerate(chunks):
                    # tokenization
                    tok_seq = tokenizer.tokenize(text).to(device)
                    #print('tok_seq shape:', tok_seq.shape)

                    # faeture extraction
                    with torch.no_grad():
                        outputs = model(tok_seq)
                    features = encoder_strategy.extract_features(outputs, pooling_type)
                    features_list.append(features.cpu())
                    torch.cuda.empty_cache()

                    if not stack_feature:
                        save_feature_h5(features, dynamic_output_dir, f"{patient_id}_{idx}")
                
                if stack_feature:                       
                    stacked_features = torch.cat(features_list, dim=0)
                    print('stacked_features shape:', stacked_features.shape)
                    save_feature_h5(stacked_features, dynamic_output_dir, f"{patient_id}")

                num_processed += 1

            except Exception as e:
                logging.error(f"Failed to extract features for {patient_id}. Error: {e}")
                error_instances.append(patient_id)

    logging.info(f"\nFeature extraction completed. Processed: {num_processed}, Skipped: {num_skipped}, Errors: {len(error_instances)}")
    if error_instances:
        logging.info(f"Errors encountered for the following patients: {', '.join(error_instances)}")
