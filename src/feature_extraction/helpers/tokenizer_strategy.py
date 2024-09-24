from transformers import AutoTokenizer
from ..tokenizers.CharacterTokenizer import CharacterTokenizer
import torch
from typing import List


class TokenizationStrategy:
    def tokenize(self, text: str) -> torch.Tensor:
        raise NotImplementedError


class DNABERT2BPE(TokenizationStrategy):
    def __init__(self, pretrained_model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    
    def tokenize(self, text: str) -> torch.Tensor:
        tok_seq = self.tokenizer(text, add_special_tokens=True, return_tensors='pt')
        tok_seq = tok_seq['input_ids']

        return tok_seq


class CharacterTokenizer(TokenizationStrategy):
    def __init__(self, characters=['A', 'C', 'G', 'T', 'N'], model_max_length=512, **kwargs):
        self.tokenizer = CharacterTokenizer(characters=characters, model_max_length=model_max_length+2, padding=False, **kwargs)
    
    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text using a character-based tokenizer and return tensor of input IDs."""
        tok_seq = self.tokenizer(text, add_special_tokens=False, return_tensors='pt')
        tok_seq = tok_seq["input_ids"] # [1, 65]
        #tok_seq = torch.LongTensor(tok_seq).unsqueeze(0) # unsqueeze for batch dim

        return tok_seq

class CharacterTokenizer2(TokenizationStrategy):
    def __init__(self, pretrained_model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)
    
    def tokenize(self, text: str) -> torch.Tensor:
        tok_seq = self.tokenizer(text, add_special_tokens=True, return_tensors='pt')
        tok_seq = tok_seq['input_ids']

        return tok_seq