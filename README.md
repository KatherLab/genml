# GenML Pipeline

## Overview

GenML Pipeline is a genomic machine learning pipeline designed to preprocess genomic data, tokenize sequences, and extract features using pretrained foundation models. The pipeline supports multiple encoders and tokenizers, and can be configured to process data with custom parameters.

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
   ```

2. **Create and activate a virtual environment:**

   ```sh
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the required dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. **Listing Available Encoders and Tokenizers** <br>
To list all available encoders and their corresponding tokenizers:
   ```sh
   python -m src list-encoders
   ```

2. **Running the Pipeline** <br>
To run the pipeline with the specified configurations:
   ```sh
   python -m src run-pipeline --encoder <encoder_name> --tokenizer <tokenizer_name> --chunk-size <chunk_size>
   ```
Example:
   ```sh
   python -m src run-pipeline --encoder hyenadna --tokenizer character_tokenizer --chunk-size 500
   ```



## Configuration
## Project Structure
## Contributing
Fork the repository.
Create a new branch (git checkout -b feature-branch).
Commit your changes (git commit -am 'Add new feature').
Push to the branch (git push origin feature-branch).
Create a new Pull Request.





