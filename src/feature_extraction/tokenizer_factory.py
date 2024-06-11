from .tokenizer_strategy import TokenizationStrategy, DNABERT2BPE, CharacterTokenizerStrategy

class TokenizerFactory:
    _strategies = {
        "dnabert2_bpe": DNABERT2BPE,
        "character_tokenizer": CharacterTokenizerStrategy
    }

    @staticmethod
    def create_tokenizer(tokenizer_type: str, **kwargs) -> TokenizationStrategy:
        strategy_class = TokenizerFactory._strategies.get(tokenizer_type)
        if not strategy_class:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
        return strategy_class(**kwargs)

    @staticmethod
    def get_available_tokenizers(model_name: str) -> list:
        # 假设每个模型对应某些特定的tokenizer
        model_to_tokenizers = {
            "dnabert2": ["dnabert2_bpe"],
            "hyenadna": ["character_tokenizer"]
        }
        return model_to_tokenizers.get(model_name, [])
