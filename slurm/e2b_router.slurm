#!/bin/bash

#SBATCH --partition=cpu
#SBATCH --mem=16g
#SBATCH --cpus-per-task=16
#SBATCH --output=/fsx/open-r1/logs/e2b_router/%x-%j.out
#SBATCH --error=/fsx/open-r1/logs/e2b_router/%x-%j.err
#SBATCH --requeue
#SBATCH --time=5-00:00:00

echo "Starting job"
set -x -e

source ~/.bashrc
cd ~/Projects/open-r1
source .venv/bin/activate

srun python scripts/e2b_router.py