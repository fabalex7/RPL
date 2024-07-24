#!/bin/bash -l
#
#SBATCH --gres=gpu:v100:1                  # Resource requirements, job runtime, other options 
#SBATCH --partition=v100                 #All #SBATCH lines have to follow uninterrupted
#SBATCH --time=24:00:00
#SBATCH --job-name=train_rpl
#SBATCH --export=NONE              # do not export environment from submitting shell
                                    # first non-empty non-comment line ends SBATCH options
unset SLURM_EXPORT_ENV             # enable export of environment from this script to srun

echo TEST

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

module load python                 # Setup job environment (load modules, stage data, ...)

conda activate rpl

srun python rpl.code/main.py           # Execute parallel application
