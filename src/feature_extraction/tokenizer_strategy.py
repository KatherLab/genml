from transformers import AutoTokenizer
from .tokenizers.CharacterTokenizer import CharacterTokenizer
import torch
from typing import List


class TokenizationStrategy:
    def tokenize(self, text: str) -> torch.Tensor:
        raise NotImplementedError

class DNABERT2BPE(TokenizationStrategy):
    def __init__(self, pretrained_model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    
    def tokenize(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(text, add_special_tokens=True, return_tensors='pt')
        #print("Shape of encoded['input_ids']:", encoded['input_ids'].shape)
        return encoded['input_ids']


class CharacterTokenizerStrategy(TokenizationStrategy):
    def __init__(self, characters=['A', 'C', 'G', 'T', 'N'], model_max_length=512, **kwargs):
        self.tokenizer = CharacterTokenizer(characters=characters, model_max_length=model_max_length+2, padding=False, **kwargs)
    
    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text using a character-based tokenizer and return tensor of input IDs."""
        encoded_input = self.tokenizer(text, add_special_tokens=False, return_tensors='pt')
        return encoded_input['input_ids']

