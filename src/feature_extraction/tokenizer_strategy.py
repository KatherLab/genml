from abc import ABC, abstractmethod
from transformers import AutoTokenizer
from .tokenizers.CharacterTokenizer import CharacterTokenizer
import torch
from typing import List

'''class TokenizationStrategy(ABC):
    @abstractmethod
    def tokenize(self, sequences: list, **kwargs) -> torch.Tensor:
        pass'''

'''class DNABERT2BPE(TokenizationStrategy):
    def __init__(self, pretrained_model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)

    def tokenize(self, texts: list) -> torch.Tensor:
        return self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)["input_ids"]'''


'''class CharacterTokenizerStrategy(TokenizationStrategy):
    def __init__(self, characters=['A', 'C', 'G', 'T', 'N'], model_max_length=512, **kwargs):
        self.tokenizer = CharacterTokenizer(characters=characters, model_max_length=model_max_length, **kwargs)
    
    def tokenize(self, sequences: list, **kwargs) -> torch.Tensor:
        tokenized = [self.tokenizer.convert_tokens_to_ids(self.tokenizer._tokenize(seq)) for seq in sequences]
        return torch.tensor(tokenized)'''


class TokenizationStrategy:
    def tokenize(self, text: str) -> torch.Tensor:
        raise NotImplementedError

class DNABERT2BPE(TokenizationStrategy):
    def __init__(self, pretrained_model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    
    def tokenize(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')
        return encoded.squeeze(0)  # Remove batch dimension

class CharacterTokenizerStrategy(TokenizationStrategy):
    def __init__(self, characters: List[str]):
        self.char_to_id = {char: idx + 1 for idx, char in enumerate(characters)}  # +1 to reserve 0 for padding
        self.id_to_char = {idx + 1: char for idx, char in enumerate(characters)}

    def tokenize(self, text: str) -> torch.Tensor:
        tokenized_text = [self.char_to_id.get(char, 0) for char in text]
        return torch.tensor(tokenized_text, dtype=torch.long)

