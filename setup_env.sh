#!/usr/bin/env bash
# Source this file after activating the local or HPC Python environment.
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
lab_repo_dir=$(dirname "$project_dir")
export PYTHONPATH="$project_dir:$lab_repo_dir/HierarchBayesParcel:$lab_repo_dir/Functional_Fusion:$lab_repo_dir:${PYTHONPATH:-}"
unset project_dir lab_repo_dir
