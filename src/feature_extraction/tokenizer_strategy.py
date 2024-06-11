from abc import ABC, abstractmethod
from transformers import AutoTokenizer
from .tokenizers.CharacterTokenizer import CharacterTokenizer
import torch

class TokenizationStrategy(ABC):
    @abstractmethod
    def tokenize(self, sequences: list, **kwargs) -> torch.Tensor:
        pass

class DNABERT2BPE(TokenizationStrategy):
    def __init__(self, pretrained_model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)

    def tokenize(self, texts: list) -> torch.Tensor:
        return self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)["input_ids"]


class CharacterTokenizerStrategy(TokenizationStrategy):
    def __init__(self, characters=['A', 'C', 'G', 'T', 'N'], model_max_length=512, **kwargs):
        self.tokenizer = CharacterTokenizer(characters=characters, model_max_length=model_max_length, **kwargs)
    
    def tokenize(self, sequences: list, **kwargs) -> torch.Tensor:
        tokenized = [self.tokenizer.convert_tokens_to_ids(self.tokenizer._tokenize(seq)) for seq in sequences]
        return torch.tensor(tokenized)

'''class CharacterTokenizerStrategy(TokenizationStrategy):
    def __init__(self, characters: list, model_max_length: int):
        self.characters = characters
        self.model_max_length = model_max_length
        self.char_to_idx = {char: idx for idx, char in enumerate(characters)}

    def tokenize(self, texts: list) -> torch.Tensor:
        tokenized_texts = []
        for text in texts:
            tokenized_text = [self.char_to_idx.get(char, 0) for char in text[:self.model_max_length]]
            tokenized_texts.append(tokenized_text)
        return torch.nn.utils.rnn.pad_sequence([torch.tensor(t) for t in tokenized_texts], batch_first=True)'''

