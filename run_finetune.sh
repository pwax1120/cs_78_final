#!/bin/bash -l
#SBATCH --job-name=finetune
#SBATCH --partition=gpu_preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --error=logs/%x.%A_%a.err
#SBATCH --array=0-29

module load python
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate cs78

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd ~/cs_78_final
mkdir -p logs

nvidia-smi
python training.py --run_id $SLURM_ARRAY_TASK_ID