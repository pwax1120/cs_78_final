#!/bin/bash -l
#SBATCH --job-name=finetune
#SBATCH --partition=gpu_preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

module load python
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate cs78

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd ~/cs_78_final

nvidia-smi
python model_finetune.py
