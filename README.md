# GenML Pipeline

## Overview

GenML Pipeline is a genomic machine learning pipeline designed to preprocess genomic data, tokenize sequences, and extract features using pretrained foundation models. The pipeline supports multiple encoders and tokenizers, and can be configured to process data with custom parameters.

## Features

- Load and preprocess genomic data
   * chunk_size: the num(not length) of alt_sequences in one chunk.  
   * chunk: a list of alt_sequences/texts with length of chunk_size under one patient.  
   * concatenated_chunk: concatenated alt_sequences with sep_token for one chunk.  
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

1. **Configuration** <br>
Go to genml/conf to set the parameter configuration.

2. **Listing Available Encoders and Tokenizers** <br>
To list all available encoders and their corresponding tokenizers(go to genml):
   ```sh
   python -m src list-encoders
   ```

3. **Running the Pipeline** <br>
To run the pipeline with the specified configurations(go to genml):
   ```sh
   python -m src extract-feature
   ```
   Example:
      ```sh
      python -m src extract-feature
      ```



## Configuration
uni_column: the column of patient id  
text_column: the column of mutation sequence  
chunk_size: keep it as 1 now  
encoder_name and tokenizer_type: use the correct pair after listing available encoders and tokenizers  


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

3. **After validating a pair of Tokenizer and Encoder**  <br>
   Add the mapping to conf/feature_params/mapping.yml
  







