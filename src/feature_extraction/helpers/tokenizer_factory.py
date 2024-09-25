from .tokenizer_strategy import TokenizationStrategy, DNABERT2BPE, HD_CharacterTokenizer, HD_CharacterTokenizer2, NTkmer

class TokenizerFactory:
    _strategies = {
        "dnabert2_bpe": DNABERT2BPE,
        "character_tokenizer": HD_CharacterTokenizer,
        "character_tokenizer2": HD_CharacterTokenizer2,
        "ntKmer": NTkmer
    }

    @staticmethod
    def create_tokenizer(tokenizer_type: str, **kwargs) -> TokenizationStrategy:
        strategy_class = TokenizerFactory._strategies.get(tokenizer_type)
        if not strategy_class:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
        return strategy_class(**kwargs)
