#!/bin/bash
#
#SBATCH --mail-user=aneesh.naik@roe.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --account=sc094
#SBATCH --job-name=BNN

# change to submission directory
cd $SLURM_SUBMIT_DIR

# pythonpath
export PYTHONPATH="${PYTHONPATH}:/work/sc094/sc094/anaik/Python"

# project environment variables
export MRVDR3DDIR=/work/sc094/sc094/anaik/Data/MissingRVsDR3Data/

# conda env
eval "$(/work/sc094/sc094/anaik/miniconda3/bin/conda shell.bash hook)"
source activate MissingRVs

python predict_6D_test.py 
