#!/bin/bash
#SBATCH --job-name=lidiff-cuda-new
#SBATCH --partition=gen-part
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=32:00:00
#SBATCH --account=research
#SBATCH --qos=gpu1-32h
#SBATCH --output=/scratch/jacobyoo/logs/lidiff_cuda_new_%j.log

# Environment settings
export PYTHONPATH=$PYTHONPATH:/scratch/jacobyoo/LiDiff
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

module load apptainer

# Run training
apptainer exec --nv \
    --bind /scratch/jacobyoo/LiDiff:/scratch/jacobyoo/LiDiff \
    --bind /scratch/jacobyoo/SemanticKITTI:/scratch/jacobyoo/SemanticKITTI \
    /scratch/jacobyoo/lidiff_v3.sif \
    /usr/local/bin/python3.8-cuda /scratch/jacobyoo/LiDiff/lidiff/train.py \
    -c /scratch/jacobyoo/LiDiff/lidiff/config/config_HPC.yaml
