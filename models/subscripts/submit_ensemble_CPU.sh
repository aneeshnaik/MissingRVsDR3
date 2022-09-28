#!/bin/bash
#SBATCH --partition defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32g
#SBATCH --time=10:00:00
#SBATCH --job-name=BNN-CPU
#SBATCH --output=out-%x-%a.out
#SBATCH --array=0-9

source activate ~/.conda/envs/MissingRVs
cd ${SLURM_SUBMIT_DIR}/..
python3 train_BNN.py ${SLURM_ARRAY_TASK_ID}

