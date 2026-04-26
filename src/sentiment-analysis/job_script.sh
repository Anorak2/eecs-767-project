#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24GB
#SBATCH -t 23:00:00
#SBATCH -J roberta_sentiment
#SBATCH -o slurm-%j.out

[ ! -d ~/.venv ] && python3 -m venv ~/.venv
#spack load cuda@11.7
#spack load py-torch@1.13.1
source ~/.venv/bin/activate
pip install -q "numpy<2" pyarrow pandas "transformers==4.30.0" "accelerate==0.20.3"
pip install -q torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

python3 RobertaClassify.py
