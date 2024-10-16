from .encoder_strategy import EncoderStrategy, DNABERT2, HyenaDNA, HyenaDNA2, NucleotideTransformer #, EnFormer

class EncoderFactory:
    _strategies = {
        "dnabert2": DNABERT2,
        "hyenadna": HyenaDNA,
        #"hyenadna2": HyenaDNA2,
        "nt": NucleotideTransformer,
        #"ef": EnFormer
    }

    @staticmethod
    def create_encoder(encoder_type: str, device: str, **kwargs) -> EncoderStrategy:
        strategy_class = EncoderFactory._strategies.get(encoder_type)
        if not strategy_class:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        #return strategy_class(**kwargs)
        return strategy_class(device=device, **kwargs)

