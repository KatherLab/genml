# GenML Pipeline

## Overview

GenML Pipeline is a genomic machine learning pipeline designed to preprocess genomic data, tokenize sequences, and extract features using different encoders and tokenizers. The pipeline supports multiple encoders and tokenizers, and can be configured to process data with custom parameters.

## Features

- Load and preprocess genomic data
- Tokenize sequences with configurable tokenizers
- Extract features using different encoders
- Support for custom sequence chunk sizes
- Easy configuration through YAML files

## Installation

1. **Clone the repository:**

   ```sh
   git clone <repository_url>
   cd genml

2. Create and activate a virtual environment:
    python -m venv .venv
    source .venv/bin/activate

3. Install the required dependencies:
    pip install -r requirements.txt

## Usage
Running the Pipeline
Listing Available Encoders and Tokenizers
To list all available encoders and their corresponding tokenizers:
    python -m src list-encoders

To run the pipeline with the specified configurations:
    python -m src run-pipeline --encoder <encoder_name> --tokenizer <tokenizer_name> --chunk-size <chunk_size>
Example:
    python -m src run-pipeline --encoder hyenadna --tokenizer character_tokenizer --chunk-size 500

## Project Structure
genml/
├── conf/
│   ├── config.yml
│   ├── feature_params/
│   │   ├── encoder.yml
│   │   ├── tokenizer.yml
│   │   ├── mapping.yml
│   └── logging.yml
├── data/
│   ├── 01_raw/
│   ├── 04_feature/
│   └── 06_models/
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── feature_extraction/
│   │   ├── __init__.py
│   │   ├── encoder_factory.py
│   │   ├── encoder_strategy.py
│   │   ├── feature_nodes.py
│   │   ├── tokenizer_factory.py
│   │   └── tokenizer_strategy.py
└── requirements.txt




