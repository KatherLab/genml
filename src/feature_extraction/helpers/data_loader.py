import pandas as pd
from torch.utils.data import Dataset
from typing import List, Dict
from torch.utils.data import Sampler


class ChunkedDataset(Dataset):
    def __init__(self, data: pd.DataFrame, mut_column: str, pat_column: str, chunk_size: int = 1, sep_token: str = "[SEP]"):
        self.chunk_size = chunk_size
        self.sep_token = sep_token

        # group patients and store the mutations to a list 
        self.grouped_muts = data.groupby(pat_column)[mut_column].apply(list).to_dict()
        self.patients = list(self.grouped_muts.keys())
        self.grouped_seqs = self.chunk_data()

    def chunk_data(self) -> dict:
        grouped_seqs = {}
        for pat, muts in self.grouped_muts.items():
            seq_chunk_ls = []  # store all the seq_chunk for this patient
            for i in range(0, len(muts), self.chunk_size):
                chunk = muts[i:i + self.chunk_size]
                # concatenate mutations by sep_token
                seq_chunk = f" {self.sep_token} ".join(chunk)
                seq_chunk_ls.append(seq_chunk)
            grouped_seqs[pat] = seq_chunk_ls
        return grouped_seqs

    def __len__(self):
        return sum(len(seq_chunks) for seq_chunks in self.grouped_seqs.values())

    def __getitem__(self, idx):
        for pat, seq_chunks in self.grouped_seqs.items():
            if idx < len(seq_chunks):
                return pat, seq_chunks[idx]  # return to patient ID and seq_chunks
            idx -= len(seq_chunks)



class PatientBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # create index list for each patient
        self.indices_per_patient = {}
        idx = 0
        for pat, seq_chunks in dataset.grouped_seqs.items():
            self.indices_per_patient[pat] = list(range(idx, idx + len(seq_chunks)))
            idx += len(seq_chunks)

    def __iter__(self):
        for patient, indices in self.indices_per_patient.items():
            batch = []
            for index in indices:
                batch.append(index)
                # if reaches defined batch_size, create a full batch
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            # if not reaches batch size, don't discard the last batch
            if batch and not self.drop_last:
                yield batch

    def __len__(self):
        total_batches = sum(len(indices) // self.batch_size for indices in self.indices_per_patient.values())
        if not self.drop_last:
            total_batches += sum(len(indices) % self.batch_size > 0 for indices in self.indices_per_patient.values())
        return total_batches
