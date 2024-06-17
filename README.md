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

2. **Create and activate a virtual environment:**

   ```sh
   python -m venv .venv
   source .venv/bin/activate

3. **Install the required dependencies:**

   `pip install -r requirements.txt`

## Usage
Running the Pipeline
Listing Available Encoders and Tokenizers
To list all available encoders and their corresponding tokenizers:
   `python -m src list-encoders`

To run the pipeline with the specified configurations:
   `python -m src run-pipeline --encoder <encoder_name> --tokenizer <tokenizer_name> --chunk-size <chunk_size>`
Example:
   `python -m src run-pipeline --encoder hyenadna --tokenizer character_tokenizer --chunk-size 500`

## Configuration
## Project Structure





