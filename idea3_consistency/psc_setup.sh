#!/bin/bash
# One-time PSC setup for idea3 (runs on a login node; no GPU SUs).
set -e
cd $PROJECT
export PIP_CACHE_DIR=$PROJECT/.pip_cache HF_HOME=$PROJECT/hf_cache CONDA_PKGS_DIRS=$PROJECT/.conda_pkgs
module load anaconda3/2024.10-1
[ -d envs/idea3 ] || conda create -y -p $PROJECT/envs/idea3 python=3.11
source activate $PROJECT/envs/idea3
# same versions as the Babel env that produced the dev runs
pip install "vllm==0.29.0" "transformers==5.17.0" "pandas==3.0.6" "pyarrow==25.0.1" \
    "pillow==12.3.0" "matplotlib==3.11.2" "huggingface_hub[cli]"
python - <<'PY'
import os
from huggingface_hub import hf_hub_download, snapshot_download
hf_hub_download(repo_id="RunsenXu/MMSI-Bench", filename="MMSI_Bench.parquet",
                repo_type="dataset", local_dir=os.environ["PROJECT"] + "/data")
snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct")
PY
python -c "import vllm, torch; print('SETUP_OK vllm', vllm.__version__, 'torch', torch.__version__)"
