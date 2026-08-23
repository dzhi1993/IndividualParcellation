#!/bin/bash
#SBATCH --partition=bigmem
#SBATCH --job-name=hbp_rest_retrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=600G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-4
#SBATCH --output=/PHShome/dz010/Slurm/output/hbp_rest_retrain.%A_%a.out
#SBATCH --error=/PHShome/dz010/Slurm/output/hbp_rest_retrain.%A_%a.err

set -euo pipefail

module load conda
conda activate my_env1

export PYTHONPATH=/PHShome/dz010/Workspace:/PHShome/dz010/Workspace/Functional_Fusion:/PHShome/dz010/Workspace/HierarchBayesParcel:/PHShome/dz010/Workspace/FusionModel:${PYTHONPATH:-}
export INDIVPAR_DEVICE=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd /PHShome/dz010/Workspace/IndividualParcellation

sizes=(80 120 200 300 362)
subject_list="HCP${sizes[$SLURM_ARRAY_TASK_ID]}_rest_retrain"

python -u -c \
    "from train_group import rerun_rest_group_fit; rerun_rest_group_fit(num_parcel=[17], smoothing_levels=[6], subj_list='${subject_list}')"
