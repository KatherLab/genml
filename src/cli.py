import click
import yaml
import logging.config
from pathlib import Path
from .feature_extraction.feats_extract import extract_feats

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
    mapping_path = Path(__file__).parent.parent / "conf" / "model_params" / "mapping.yml"
    with open(mapping_path, "r") as f:
        mapping = yaml.safe_load(f)
    return mapping


@click.group()
def cli():
    pass

@cli.command()
@click.option('--encoder', default=None, help='Specify encoder type (e.g., hyenadna, dnabert2)')
def extract_feature(encoder):
    """Run the main pipeline"""
    setup_logging()
    logger = logging.getLogger(__name__)

    config = load_config()
    logger.info(f"Config loaded: {config}")

    encoder_type = encoder if encoder else config["encoder_name"]
    encoder_params = load_params("conf/model_params/encoder.yml")[encoder_type]

    logger.info(f"Using encoder: {encoder_type} with params: {encoder_params}")

    raw_data_path = config["filepath"]
    raw_data_path = Path(raw_data_path)
    pat_column = config["pat_column"]
    mut_column = config["mut_column"]
    num_patients = config["num_patients"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]

    device = config["device"]
    pooling_type = config["pooling_type"]

    output_dir = config["output_dir"]
    output_dir = Path(output_dir)

    logger.info("Start feature extraction.")
    extract_feats(file_path=raw_data_path, 
                       mut_column=mut_column, 
                       pat_column=pat_column, 
                       encoder_type=encoder_type, 
                       encoder_params=encoder_params, 
                       device=device, 
                       pooling_type=pooling_type, 
                       output_dir=output_dir, 
                       batch_size=batch_size, 
                       num_workers = num_workers, 
                       num_patients=num_patients)
    logger.info("Feature extraction and saving completed successfully.")


@cli.command()
def list_encoders():
    """List available encoders and their corresponding tokenizers"""
    mapping = load_mapping()
    for encoder, tokenizers in mapping["encoder_tokenizer_mapping"].items():
        click.echo(f"Encoder_Name: {encoder}")
        click.echo(f"Tokenizer and Encoder: {', '.join(tokenizers)}")


if __name__ == "__main__":
    cli()
