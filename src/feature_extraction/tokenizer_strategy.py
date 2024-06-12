from abc import ABC, abstractmethod
from transformers import AutoTokenizer
from .tokenizers.CharacterTokenizer import CharacterTokenizer
import torch
from typing import List

'''class TokenizationStrategy(ABC):
    @abstractmethod
    def tokenize(self, sequences: list, **kwargs) -> torch.Tensor:
        pass'''


class TokenizationStrategy:
    def tokenize(self, text: str) -> torch.Tensor:
        raise NotImplementedError

class DNABERT2BPE(TokenizationStrategy):
    def __init__(self, pretrained_model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    
    #def tokenize(self, text: str) -> torch.Tensor:
     #   encoded = self.tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')
        # return encoded.squeeze(0)  # Remove batch dimension
    #    return encoded['input_ids']  # Keep batch dimension
    def tokenize(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(text, add_special_tokens=True, return_tensors='pt')
        print("Shape of encoded['input_ids']:", encoded['input_ids'].shape)
        return encoded['input_ids']

'''class CharacterTokenizerStrategy(TokenizationStrategy):
    def __init__(self, characters: List[str]):
        self.char_to_id = {char: idx + 1 for idx, char in enumerate(characters)}  # +1 to reserve 0 for padding
        self.id_to_char = {idx + 1: char for idx, char in enumerate(characters)}

    def tokenize(self, text: str) -> torch.Tensor:
        tokenized_text = [self.char_to_id.get(char, 0) for char in text]
        return torch.tensor(tokenized_text, dtype=torch.long)'''

class CharacterTokenizerStrategy(TokenizationStrategy):
    def __init__(self, characters=['A', 'C', 'G', 'T', 'N'], model_max_length=512, **kwargs):
        self.tokenizer = CharacterTokenizer(characters=characters, model_max_length=model_max_length+2, padding=False, **kwargs)
    
    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text using a character-based tokenizer and return tensor of input IDs."""
        encoded_input = self.tokenizer(text, add_special_tokens=False, return_tensors='pt')
        return encoded_input['input_ids']

