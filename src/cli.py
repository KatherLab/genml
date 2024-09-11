import click
import yaml
import logging.config
from pathlib import Path
from .feature_extraction.node_extract import feature_extraction

def setup_logging():
    config_path = Path(__file__).parent.parent / "conf" / "logging.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

def load_config():
    config_path = Path(__file__).parent.parent / "conf" / "config.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_params(file_path):
    config_path = Path(__file__).parent.parent / file_path
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)
    return params

def load_mapping():
    mapping_path = Path(__file__).parent.parent / "conf" / "feature_params" / "mapping.yml"
    with open(mapping_path, "r") as f:
        mapping = yaml.safe_load(f)
    return mapping


@click.group()
def cli():
    pass

@cli.command()
@click.option('--encoder', default=None, help='Specify encoder type (e.g., hyenadna, dnabert2)')
@click.option('--tokenizer', default=None, help='Specify tokenizer type (e.g., character_tokenizer, dnabert2_bpe)')
@click.option('--chunk-size', default=None, type=int, help='Specify the chunk size for concatenating sequences')
def extract_feature(encoder, tokenizer, chunk_size):
    """Run the main pipeline"""
    setup_logging()
    logger = logging.getLogger(__name__)

    config = load_config()
    logger.info(f"Config loaded: {config}")

    encoder_type = encoder if encoder else config["encoder_name"]
    tokenizer_type = tokenizer if tokenizer else config["tokenizer_type"]
    encoder_params = load_params("conf/feature_params/encoder.yml")[encoder_type]
    tokenizer_params = load_params("conf/feature_params/tokenizer.yml")[tokenizer_type]

    chunk_size = chunk_size if chunk_size else config.get("chunk_size", 10)

    logger.info(f"Using encoder: {encoder_type} with params: {encoder_params}")
    logger.info(f"Using tokenizer: {tokenizer_type} with params: {tokenizer_params}")
    logger.info(f"Using sequence chunk size: {chunk_size}")

    stack_feature = config.get("stack_feature", True)
    logger.info(f"Stack feature: {stack_feature}")

    raw_data_path = config["filepath"]
    raw_data_path = Path(raw_data_path)
    uni_column = config["uni_column"]
    text_column = config["text_column"]
    num_patients = config["num_patients"]
    batch_size = config["batch_size"]

    device = config["device"]
    cls = config["cls_token"]

    output_dir = config["output_dir"]
    output_dir = Path(output_dir)

    logger.info("Start feature extraction.")
    feature_extraction(file_path=raw_data_path, 
                       text_column=text_column, 
                       uni_column=uni_column, 
                       chunk_size=chunk_size, 
                       encoder_type=encoder_type, 
                       encoder_params=encoder_params, 
                       tokenizer_type=tokenizer_type, 
                       tokenizer_params=tokenizer_params, 
                       device=device, 
                       cls=cls, 
                       stack_feature=stack_feature, 
                       output_dir=output_dir, 
                       batch_size=batch_size, 
                       num_patients=num_patients)
    logger.info("Feature extraction and saving completed successfully.")


@cli.command()
def list_encoders():
    """List available encoders and their corresponding tokenizers"""
    mapping = load_mapping()
    for encoder, tokenizers in mapping["encoder_tokenizer_mapping"].items():
        click.echo(f"Encoder: {encoder}")
        click.echo(f"  Tokenizers: {', '.join(tokenizers)}")


if __name__ == "__main__":
    cli()
