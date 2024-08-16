# GenML Pipeline

## Overview

GenML Pipeline is a genomic machine learning pipeline designed to preprocess genomic data, tokenize sequences, and extract features using pretrained foundation models. The pipeline supports multiple encoders and tokenizers, and can be configured to process data with custom parameters.

## Features

- Load and preprocess genomic data
   chunk_size: the num(not length) of alt_sequences in one chunk.  
   chunk: a list of alt_sequences/texts with length of chunk_size under one patient.  
   concatenated_chunk: concatenated alt_sequences with sep_token for one chunk.  
   concatenated_chunks: a list of 'concatenated_chunk's of one patient  
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
![image](https://gitlab.hrz.tu-chemnitz.de/lizh574c--tu-dresden.de/genml/-/blob/master/docs/DFD.png?ref_type=heads)

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

## To-Do List  
- [ ] Remove the register step for tokenizer and encoder  
- [x] Option to output cls token as embedding  
- [ ] Dock container  
- [ ] Include Enformer and nucleotide transformer  







