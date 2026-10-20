#! /bin/bash
set -eux -o pipefail

VENVDIR=/tmp/run_timm_venv
CURRENT_DIR=$(realpath $(dirname $0))
CODEGEN_DIR=$(realpath ${CURRENT_DIR}/../../)
BUILD_DIR=${BUILD_DIR:-${CODEGEN_DIR}/build}

if [[ ! -d ${VENVDIR} ]]; then
    python3 -m venv --system-site-packages ${VENVDIR}
    source ${VENVDIR}/bin/activate
    # pip's own --retries only covers establishing the connection, so a read
    # timeout part-way through a download propagates straight out with no
    # "WARNING: Retrying" line. That is how this dies: 18 seconds in, on a 50 KB
    # .metadata file.
    for _timm_attempt in 1 2 3; do
        if pip3 install --timeout 30 --retries 5 timm==1.0.14 huggingface-hub==0.28.1; then
            break
        fi
        if [ "${_timm_attempt}" = 3 ]; then
            echo "error: pip3 install timm failed after 3 attempts" >&2
            exit 1
        fi
        echo "=== pip3 install timm failed (attempt ${_timm_attempt}/3), retrying in 15s ==="
        sleep 15
    done
else
    source ${VENVDIR}/bin/activate
fi

source "${BUILD_DIR}/codegen_pythonpath.sh"

# Set Hugging Face cache directory to avoid filling up the home directory
HF_HOME=${HF_HOME:-"/tmp/huggingface"} \
    exec python3 ${CURRENT_DIR}/run_timm.py "$@"
