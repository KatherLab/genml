# GenML Pipeline

## Overview

GenML Pipeline is a genomic machine learning pipeline designed to preprocess genomic data, tokenize sequences, and extract features using pretrained foundation models. The pipeline supports multiple encoders and tokenizers, and can be configured to process data with custom parameters.

## Features

- Load and preprocess genomic data, e.g sequences 
- Tokenize sequences with loaded tokenizers
- Extract features using different encoders/foundation models
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
   or install Anaconda to create env.

   
3. **Install the required dependencies:**

   ```sh
   pip install -r requirements.txt
   ```
   Tips: for DNABERT2, additional env is suggested, and then
   ```sh
   pip install -r requirements_db2.txt
   ```
   and then 
   ```sh
   pip uninstall triton
   ```

## Usage

1. **Listing Available Encoders and Tokenizers** <br>
To list all available encoders and their corresponding tokenizers(go to genml):
   ```sh
   python -m src list-encoders
   ```

2. **Configuration** <br>
Go to genml/conf to set the parameter configuration.  
   - config.yml is for the feature extraction process. 
   - feature_params/encoder.yml is for the foundation models you will use, set the download as True at the first time.



3. **Running the Pipeline** <br>
To run the pipeline with the specified configurations(go to genml):
   ```sh
   python -m src extract-feature
   ```



## Configuration
- pat_column: the column of patient id  
- mut_column: the column of mutation sequence  
- encoder_name: use the correct encoder after listing available encoders and tokenizers  


## Contributing

(Create a new branch firstly.)
1. **Including a new Encoder**  <br>
   a. Add a class NewEncoderStrategy(EncoderStrategy) in 'src/feature_extraction/encoder_strategy.py'  
   b. Register the new Encoder to 'src/feature_extraction/encoder_factory.py'  
   c. Set up for the new Encoder in 'conf/feature_params/encoder.yml'  

3. **After validating an Encoder**  <br>
   Add the mapping to conf/feature_params/mapping.yml
  







