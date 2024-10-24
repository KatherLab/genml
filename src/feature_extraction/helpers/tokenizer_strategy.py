import torch
from typing import List

from transformers import AutoTokenizer
from enformer_pytorch import str_to_one_hot

from ..tokenizers.CharacterTokenizer import CharacterTokenizer


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


class HD_CharacterTokenizer(TokenizationStrategy):
    def __init__(self, characters=['A', 'C', 'G', 'T', 'N'], model_max_length=512, **kwargs):
        self.tokenizer = CharacterTokenizer(characters=characters, model_max_length=model_max_length+2, padding=False, **kwargs)
    
    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text using a character-based tokenizer and return tensor of input IDs."""
        tok_seq = self.tokenizer(text, add_special_tokens=False, return_tensors='pt')
        tok_seq = tok_seq["input_ids"] # [1, 65]
        #tok_seq = torch.LongTensor(tok_seq).unsqueeze(0) # unsqueeze for batch dim

        return tok_seq

class HD_CharacterTokenizer2(TokenizationStrategy):
    def __init__(self, pretrained_model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)
    
    def tokenize(self, text: str) -> torch.Tensor:
        tok_seq = self.tokenizer(text, add_special_tokens=True, return_tensors='pt')
        tok_seq = tok_seq['input_ids']

        return tok_seq
    
class NTkmer(TokenizationStrategy):
    def __init__(self, pretrained_model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    
    def tokenize(self, text: str) -> torch.Tensor:
        tok_seq = self.tokenizer(text, add_special_tokens=True, return_tensors='pt')
        tok_seq = tok_seq['input_ids']

        return tok_seq


class EFonehot(TokenizationStrategy):
    def __init__(self, pretrained_model_name: str):
        pass
    def tokenize(self, text: str) -> torch.Tensor:
        tok_seq = str_to_one_hot(text)

        return tok_seq