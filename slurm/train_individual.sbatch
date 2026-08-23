#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --job-name=hbp-individual
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Run the repository's reproducible individual-parcellation example on CPU.
# PHS/Eris2 CPU:
#   sbatch slurm/train_individual.sbatch
# PHS GPU (matching the earlier individual job's resource convention):
#   sbatch --partition=Medium --gpus=1 --export=ALL,INDIVPAR_DEVICE=cuda slurm/train_individual.sbatch
# Another installation/environment can additionally export PROJECT_ROOT,
# INDIVPAR_DATA_ROOT, INDIVPAR_MODEL_DIR, and PYTHON_EXECUTABLE.

set -euo pipefail

SCRIPT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/PHShome/dz010/Workspace/IndividualParcellation}"
if [[ ! -d "${PROJECT_ROOT}" ]]; then
    PROJECT_ROOT="${SCRIPT_PROJECT_ROOT}"
fi
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(dirname "${PROJECT_ROOT}")}"

if [[ -n "${VIRTUAL_ENV_PATH:-}" ]]; then
    source "${VIRTUAL_ENV_PATH}/bin/activate"
elif [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
    if command -v module >/dev/null 2>&1; then
        module load conda >/dev/null 2>&1 || true
    fi
    if ! command -v conda >/dev/null 2>&1; then
        CONDA_SETUP="${CONDA_SETUP:-/apps/software-2.12/Anaconda3/5.1.0/etc/profile.d/conda.sh}"
        [[ -r "${CONDA_SETUP}" ]] && source "${CONDA_SETUP}"
    fi
    command -v conda >/dev/null 2>&1 || {
        echo "Conda was not found; set PYTHON_EXECUTABLE or VIRTUAL_ENV_PATH." >&2
        exit 1
    }
    conda activate "${CONDA_ENV:-my_env1}"
fi

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"
command -v "${PYTHON_EXECUTABLE}" >/dev/null 2>&1 || {
    echo "Python executable not found: ${PYTHON_EXECUTABLE}" >&2
    exit 1
}

export INDIVPAR_DATA_ROOT="${INDIVPAR_DATA_ROOT:-/data/tge}"
export INDIVPAR_MODEL_DIR="${INDIVPAR_MODEL_DIR:-${INDIVPAR_DATA_ROOT}/dzhi/Indiv_par/Models}"
export FUNCTIONAL_FUSION_DIR="${FUNCTIONAL_FUSION_DIR:-${INDIVPAR_DATA_ROOT}/Tian/UKBB_full/imaging}"
export INDIVPAR_DEVICE="${INDIVPAR_DEVICE:-cpu}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONPATH="${WORKSPACE_ROOT}:${WORKSPACE_ROOT}/Functional_Fusion:${WORKSPACE_ROOT}/HierarchBayesParcel:${WORKSPACE_ROOT}/FusionModel:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_ROOT}"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Host: $(hostname)"
echo "Project: ${PROJECT_ROOT}"
echo "Git commit: $(git rev-parse --verify HEAD 2>/dev/null || echo unavailable)"
echo "Python: $(command -v "${PYTHON_EXECUTABLE}")"
echo "Device: ${INDIVPAR_DEVICE}"

"${PYTHON_EXECUTABLE}" -u train_individual.py
