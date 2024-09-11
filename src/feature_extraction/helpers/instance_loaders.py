import pandas as pd

class PatientLoader:
    def __init__(self, data: pd.DataFrame, text_column: str, uni_column: str, chunk_size: int, sep_token: str = "[SEP]", batch_size: int = 5):
        """
        A loader class to iterate over patients in batches.

        Args:
            data (pd.DataFrame): The input data.
            text_column (str): The column containing the text/Alt_Sequence.
            uni_column (str): The unique identifier column (e.g., Patient_ID).
            chunk_size (int): The number of alt_sequences per chunk.
            sep_token (str): The token to add after each sequence.
            batch_size (int): The number of patients per batch.
        """
        self.batch_size = batch_size
        self.grouped_texts = data.groupby(uni_column)[text_column].apply(list).to_dict()
        self.patient_ids = list(self.grouped_texts.keys())
        self.chunk_size = chunk_size
        self.sep_token = sep_token
        self.total_patients = len(self.patient_ids)
        self.current_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx >= self.total_patients:
            raise StopIteration

        batch = {}
        for _ in range(self.batch_size):
            if self.current_idx >= self.total_patients:
                break
            patient_id = self.patient_ids[self.current_idx]
            texts = self.grouped_texts[patient_id]

            # Create chunks for this patient
            chunks = []
            for i in range(0, len(texts), self.chunk_size):
                chunk = texts[i:i + self.chunk_size]
                chunk = [text + self.sep_token for text in chunk]
                concatenated_chunk = ''.join(chunk)
                chunks.append(concatenated_chunk)

            batch[patient_id] = chunks
            self.current_idx += 1

        return batch
