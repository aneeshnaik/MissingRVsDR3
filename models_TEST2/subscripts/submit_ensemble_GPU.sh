#!/bin/bash
#SBATCH --partition voltaq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32g
#SBATCH --time=5-00:00:00
#SBATCH --job-name=TEST2
#SBATCH --output=out-%x-%a.out
#SBATCH --array=0-5
#SBATCH --mail-user=aneesh.naik@nottingham.ac.uk
#SBATCH --mail-type=ALL

module load pytorch-uoneasy/1.9.0-fosscuda-2020b
export CUDA_VISIBLE_DEVICES=`/software/gpucheck/gpuuse.sh`

source activate ~/.conda/envs/MissingRVsGPU
cd ${SLURM_SUBMIT_DIR}/..
python3 train_BNN.py ${SLURM_ARRAY_TASK_ID}
