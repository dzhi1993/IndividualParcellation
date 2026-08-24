#!/bin/bash
#SBATCH --partition=bigmem
#SBATCH --job-name=hbp_rest_chunks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-9%3
#SBATCH --output=/PHShome/dz010/Slurm/output/hbp_rest_chunks.%A_%a.out
#SBATCH --error=/PHShome/dz010/Slurm/output/hbp_rest_chunks.%A_%a.err

# Default: HCP120, 10 chunks x 5 repetitions = 50 repetitions.
# Override with: --export=ALL,HCP_SIZE=200,REPEATS_PER_JOB=5

set -eo pipefail

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate my_env1

export PYTHONPATH=/PHShome/dz010/Workspace:/PHShome/dz010/Workspace/Functional_Fusion:/PHShome/dz010/Workspace/HierarchBayesParcel:/PHShome/dz010/Workspace/FusionModel:${PYTHONPATH:-}
export INDIVPAR_DEVICE=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd /PHShome/dz010/Workspace/IndividualParcellation

HCP_SIZE=${HCP_SIZE:-120}
REPEATS_PER_JOB=${REPEATS_PER_JOB:-5}
CHUNK_ID=${SLURM_ARRAY_TASK_ID}
SEED=$((HCP_SIZE * 100 + CHUNK_ID))
SUBJECT_LIST="HCP${HCP_SIZE}_rest_retrain"

echo "Subject list: ${SUBJECT_LIST}; chunk: ${CHUNK_ID}; repetitions: ${REPEATS_PER_JOB}; seed: ${SEED}"

python -u -c 'import os, sys, torch as pt; pt.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"])); from train_group import run_hcp_rest_group_fit; print(f"PyTorch CPU threads: {pt.get_num_threads()}"); run_hcp_rest_group_fit(num_parcel=[17], smoothing_levels=[6], subj_list=sys.argv[1], repeats=int(sys.argv[2]), seed=int(sys.argv[3]), output_suffix=f"_chunk-{int(sys.argv[4]):02d}")' \
    "${SUBJECT_LIST}" "${REPEATS_PER_JOB}" "${SEED}" "${CHUNK_ID}"
