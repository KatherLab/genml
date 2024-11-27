from .encoders import *

class EncoderFactory:
    _strategies = {
        "db2": DNABERT2,
        "hd": HyenaDNA,
        "nt": NucleotideTransformer,
        "gv": GROVER,
    }

    @staticmethod
    def create_encoder(encoder_type: str, device: str, **kwargs) -> EncoderStrategy:
        strategy_class = EncoderFactory._strategies.get(encoder_type)
        if not strategy_class:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

        return strategy_class(device=device, **kwargs)



