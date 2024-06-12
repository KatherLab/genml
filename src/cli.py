import click
import yaml
import logging.config
from pathlib import Path
from .feature_extraction.feature_nodes import load_data, preprocess_data, feature_extraction, save_features
from .feature_extraction.tokenizer_factory import TokenizerFactory
from .feature_extraction.encoder_factory import EncoderFactory

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

@click.group()
def cli():
    pass

@cli.command()
@click.option('--encoder', default=None, help='Specify encoder type (e.g., hyenadna, dnabert2)')
@click.option('--tokenizer', default=None, help='Specify tokenizer type (e.g., character_tokenizer, dnabert2_bpe)')
@click.option('--chunk-size', default=None, type=int, help='Specify the chunk size for concatenating sequences')
def run_pipeline(encoder, tokenizer, chunk_size):
    """Run the main pipeline"""
    setup_logging()
    logger = logging.getLogger(__name__)

    config = load_config()
    logger.info(f"Config loaded: {config}")

    encoder_type = encoder if encoder else config["encoder_name"]
    tokenizer_type = tokenizer if tokenizer else config["tokenizer_type"]
    sequence_chunk_size = chunk_size if chunk_size else config.get("sequence_chunk_size", 10)

    encoder_params = load_params("conf/feature_params/encoder.yml")[encoder_type]
    tokenizer_params = load_params("conf/feature_params/tokenizer.yml")[tokenizer_type]

    logger.info(f"Using encoder: {encoder_type} with params: {encoder_params}")
    logger.info(f"Using tokenizer: {tokenizer_type} with params: {tokenizer_params}")
    logger.info(f"Using sequence chunk size: {sequence_chunk_size}")

    # Load data
    raw_data_path = config["raw_data"]["filepath"]
    columns = config["columns"]
    logger.info(f"Loading data from {raw_data_path} with columns {columns}")
    data = load_data(raw_data_path, columns, num_patients=2)
    logger.info("Data loaded successfully.")

    # Preprocess data
    text_column = config["text_column"]
    logger.info(f"Preprocessing data with text column {text_column}")
    grouped_texts = preprocess_data(data, text_column, sequence_chunk_size)
    logger.info("Data preprocessed successfully.")

    # Feature extraction
    device = config["device"]
    logger.info(f"Extracting features using encoder {encoder_type} on device {device}")
    extracted_features = feature_extraction(grouped_texts, encoder_type, encoder_params, tokenizer_type, tokenizer_params, device)
    logger.info("Feature extraction completed successfully.")

    # Save features
    output_dir = config["output_dir"]
    logger.info(f"Saving features to {output_dir}")
    save_features(extracted_features, output_dir)
    logger.info("Features saved successfully.")

@cli.command()
@click.argument("model_name")
def get_tokenizers(model_name):
    """Get supported tokenizers for a given model name"""
    tokenizers = TokenizerFactory.get_available_tokenizers(model_name)
    if tokenizers:
        click.echo(f"Supported tokenizers for {model_name}: {', '.join(tokenizers)}")
    else:
        click.echo(f"No supported tokenizers found for {model_name}")

@cli.command()
def list_encoders():
    """List available encoders"""
    encoders = EncoderFactory.get_available_encoders()
    click.echo(f"Available encoders: {', '.join(encoders)}")


if __name__ == "__main__":
    cli()
