import pandas as pd
import torch
import os
from typing import Dict, List
import logging
import time
from tqdm import tqdm
from pathlib import Path
import h5py
from torch.utils.data import DataLoader

from .helpers.encoder_factory import EncoderFactory
from .helpers.data_loader import ChunkedDataset, PatientBatchSampler


def load_data(file_path: str, pat_column: str, num_patients: int = None) -> pd.DataFrame:
    print('num_patients:', num_patients)
    data = pd.read_csv(file_path)
    if num_patients is not None:
        data = data[data[pat_column].isin(data[pat_column].unique()[:num_patients])]
    return data



def save_feature_h5(features, output_dir, filename):
    """ Save features into an h5 file instead of a .pt file """
    features = features.to(torch.float32)  # Convert from BFloat16 to Float32
    output_path = output_dir / f"{filename}.h5"
    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset('feats', data=features.cpu().numpy())  # Save tensor as numpy array


def extract_feats(
        file_path: str,
        mut_column: str, 
        pat_column: str, 
        encoder_type: str, 
        encoder_params: dict, 
        device: str, 
        pooling_type: str, 
        output_dir: str, 
        batch_size: int = 5,
        num_workers: int = 4, 
        num_patients: int = None):
        
    # Check if GPU is available
    has_gpu = torch.cuda.is_available()
    device = torch.device(device if has_gpu and "cuda" in device else "cpu")
    print(f"Using device: {device} (GPU available: {has_gpu})")
  
    # Initialize encoder strategy (which includes tokenizer and model)
    encoder_strategy = EncoderFactory.create_encoder(
        encoder_type=encoder_type, 
        device=device, 
        **encoder_params
    )

    # Create dynamic output directory
    dynamic_output_dir = Path(output_dir) / f"{encoder_type}_{pooling_type}"
    dynamic_output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logfile_name = f"logfile_{time.strftime('%Y-%m-%d_%H-%M-%S')}_{os.getpid()}.log"
    logdir = dynamic_output_dir / logfile_name
    logging.basicConfig(filename=logdir, level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.info(f"Feature extraction started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Model: {encoder_type}\n")

    # Scan for existing feature files
    logging.info("Scanning for existing features...")
    existing_instances = {f.stem for f in dynamic_output_dir.glob("**/*.h5")} if dynamic_output_dir.exists() else set()

    # Load raw data
    raw_data = load_data(file_path, pat_column, num_patients=num_patients)

    # Initialize dataset and sampler
    dataset = ChunkedDataset(data=raw_data, mut_column=mut_column, pat_column=pat_column)
    sampler = PatientBatchSampler(dataset, batch_size=batch_size, drop_last=False)
    data_loader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers)

    num_processed, num_skipped = 0, 0
    error_instances = []
    skipped_patients = set()

    last_patient_id = None
    features_list = []
    skip_current_patient = False  # Flag to skip patient if any error occurs

    # Feature extraction loop with progress bar
    for batch in tqdm(data_loader, desc="Processing batches"):
    
        patient_id, seq_chunks = batch[0], batch[1:]
        patient_id = patient_id[0]  # All IDs in the batch are the same

        # Skip current patient's batch if it's marked to be skipped due to prior error
        if skip_current_patient and patient_id == last_patient_id:
            continue

        # If patient already exists in the output, skip
        if patient_id in existing_instances:
            if patient_id not in skipped_patients:
                logging.info(f"Skipping patient {patient_id} (features already exist)")
                num_skipped += 1
                skipped_patients.add(patient_id)
            continue

        # If this is a new patient, save previous patient's features
        if last_patient_id is not None and patient_id != last_patient_id:
            # Stack features for the last patient and save
            if features_list and not skip_current_patient:
                patient_features = torch.cat(features_list, dim=0)
                save_feature_h5(patient_features, dynamic_output_dir, f"{last_patient_id}")
                logging.info(f"Saved features for patient {last_patient_id}, final extracted features shape {patient_features.shape}")
                num_processed += 1  # Update count only after successful save

            # Reset for new patient
            features_list = []
            skip_current_patient = False  # Reset skip flag for new patient 

        # Process each sequence chunk for the current patient
        try:
            if skip_current_patient:
                continue  # Skip processing if patient is marked for skipping

            # Flatten the list of sequence chunks
            flat_seq_chunks = [seq for chunk in seq_chunks for seq in chunk]  # unpack and flaten seq_chunks

            # Only proceed with feature extraction if no prior errors for this patient
            for seq_chunk in flat_seq_chunks:
                # Extract features for each chunk
                features = encoder_strategy.encode(seq_chunk, feature_type=pooling_type)
                features_list.append(features)

            # Successfully processed chunks, so update last_patient_id
            last_patient_id = patient_id  

        except Exception as e:
            # Log error once per patient and mark to skip all batches for this patient
            if patient_id not in error_instances:
                logging.error(f"Failed to extract features for patient {patient_id}: {e}")
                error_instances.append(patient_id)
            
            # Mark this patient to skip, clear accumulated features
            skip_current_patient = True  
            features_list.clear()  # Use clear() instead of assigning a new list
            last_patient_id = patient_id  # Update last_patient_id to reflect error


    # Save features for the last patient
    if features_list and not skip_current_patient:
        patient_features = torch.cat(features_list, dim=0)
        save_feature_h5(patient_features, dynamic_output_dir, f"{last_patient_id}")
        logging.info(f"Saved features for patient {last_patient_id}, final extracted features shape {patient_features.shape}")
        num_processed += 1  # Update count for the last patient

    logging.info(f"\nFeature extraction completed. Processed: {num_processed}, Skipped: {num_skipped}, Errors: {len(error_instances)}")
    if error_instances:
        logging.info(f"Errors encountered for the following patients: {', '.join(error_instances)}")
