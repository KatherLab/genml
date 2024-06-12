#!/bin/bash
#SBATCH --job-name=run_pipeline_job # MUST NOT CONTAIN WHITESPACES!
#SBATCH --partition=cpu   # Set gpu for titan / europa and cpu for curie / ganymede (the cpu partition still has gpus!) or both separated by comma if it does not matter
#SBATCH --gres=gpu:1      # Request one GPU - delete this line if you don't need a GPU
#SBATCH --time=04:00:00   # Set the maximum run time (e.g. 60 minutes), after this your process will be killed
#SBATCH --mem=64G         # How much RAM do you need?
#SBATCH --cpus-per-task=12 # Number of CPU cores (I think this is the number of ht-threads, so you can select number of physical cores * 2)

# Log files with the program output will automatically be created in the directory where you run this script. (e.g. slurm-1234.txt)

# Activate the Python virtual environment
source /mnt/bulk-curie/lizhang/LiWorkSpace/cancergen/.venv/bin/activate

# Change to the directory containing your Python script
cd /mnt/bulk-curie/lizhang/LiWorkSpace/cancergen/genml

# Run the pipeline with specified parameters
python -m src run-pipeline --encoder hyenadna --tokenizer character_tokenizer --chunk-size 500

# Deactivate the virtual environment
deactivate
