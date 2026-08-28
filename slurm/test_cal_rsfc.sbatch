#!/usr/bin/env bash
#SBATCH --partition=short
#SBATCH --job-name=rsfc-benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/PHShome/dz010/Slurm/output/hbp_rsfc_benchmark.%A_%a.out
#SBATCH --error=/PHShome/dz010/Slurm/output/hbp_rsfc_benchmark.%A_%a.err

set -eo pipefail

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate my_env1

individual_parcellation_dir="/PHShome/dz010/Workspace/IndividualParcellation"
functional_fusion_dir="/PHShome/dz010/Workspace/Functional_Fusion"
script_path="${functional_fusion_dir}/scripts/test_cal_rsfc.py"

hcp_dir="/data/tge/Tian/HCP_img"
atlas_dir="/data/tge/Tian/UKBB_full/imaging/Atlases"
subject_list="/subj_list/HCP40_training_KONG2019.tsv"

profile_dir="${individual_parcellation_dir}/results/additional_experiments/speed_test/${SLURM_ARRAY_JOB_ID}"
mkdir -p "${profile_dir}"

if [[ ! -f "${script_path}" ]]; then
    echo "Script not found: ${script_path}" >&2
    exit 1
fi

if [[ ! -d "${atlas_dir}" ]]; then
    echo "Atlas directory not found: ${atlas_dir}" >&2
    exit 1
fi

if [[ ! -f "${hcp_dir}${subject_list}" ]]; then
    echo "Subject list not found: ${hcp_dir}${subject_list}" >&2
    exit 1
fi

# Keep the CPU configuration identical across benchmark repetitions.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

printf -v repetition "%02d" "${SLURM_ARRAY_TASK_ID}"

cd "${functional_fusion_dir}"

for session_id in ses-rest1 ses-rest2; do
    profile_file="${profile_dir}/${session_id}_repeat-${repetition}.tsv"

    echo "Starting ${session_id}, repetition ${repetition}"
    echo "Profile: ${profile_file}"

    srun --ntasks=1 \
         --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
         python -u "${script_path}" \
         --hcp-dir "${hcp_dir}" \
         --atlas-dir "${atlas_dir}" \
         --subj-list "${subject_list}" \
         --type Ico642Run \
         --space fs32k \
         --ses-id "${session_id}" \
         --threshold 0.1 \
         --sample-interval 0.1 \
         --profile-out "${profile_file}"

    echo "Completed ${session_id}, repetition ${repetition}"
done
