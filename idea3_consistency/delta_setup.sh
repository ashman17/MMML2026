#!/bin/bash
# One-time NCSA Delta setup for idea3 (runs on a login node; no GPU hours).
set -e
BASE=/projects/bjjb/$USER
cd $BASE
export PIP_CACHE_DIR=$BASE/.pip_cache HF_HOME=$BASE/hf_cache CONDA_PKGS_DIRS=$BASE/.conda_pkgs
module load miniforge3-python
[ -d envs/idea3 ] || conda create -y -p $BASE/envs/idea3 python=3.11
source activate $BASE/envs/idea3
# same versions as the Babel env that produced the dev runs
pip install "vllm==0.29.0" "transformers==5.17.0" "pandas==3.0.6" "pyarrow==25.0.1" \
    "pillow==12.3.0" "matplotlib==3.11.2" "huggingface_hub[cli]"
python - <<'PY'
import os
from huggingface_hub import hf_hub_download, snapshot_download
base = "/projects/bjjb/" + os.environ["USER"]
hf_hub_download(repo_id="RunsenXu/MMSI-Bench", filename="MMSI_Bench.parquet",
                repo_type="dataset", local_dir=base + "/data")
snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct")
PY
python -c "import vllm, torch; print('SETUP_OK vllm', vllm.__version__, 'torch', torch.__version__)"
