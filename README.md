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
(Create a new branch firstly.)
1. **Including a new Tokenizer**  <br>
   a. Add a class NewTokenizerStrategy(TokenizationStrategy) in 'src/feature_extraction/tokenizer_strategy.py'  
   b. Register the new Tokeniter to 'src/feature_extraction/tokenizer_factory.py'  
   c. Set up for the new Tokenizer in 'conf/feature_params/tokenizer.yml'  

2. **Including a new Encoder**  <br>
   a. Add a class NewEncoderStrategy(EncoderStrategy) in 'src/feature_extraction/encoder_strategy.py'  
   b. Register the new Encoder to 'src/feature_extraction/encoder_factory.py'  
   c. Set up for the new Encoder in 'conf/feature_params/encoder.yml'  

3. **After validate a pair of Tokenizer and Encoder**  <br>
   Add the mapping to conf/feature_params/mapping.yml

## TODO: 
1. remove the register step for tokenizer and encoder  
2. option to output cls token as embedding







