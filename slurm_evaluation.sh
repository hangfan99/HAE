#!/usr/bin/env bash
set -euo pipefail

GPUS=${GPUS:-1}
CPUS=${CPUS:-4}
PARTITION=${PARTITION:-earth-e2e-p}
QUOTATYPE=${QUOTATYPE:-spot}
JOB_NAME=${JOB_NAME:-hae_wrmse_eval_2017}
CFG=${CFG:-configs/ae_kl_hybrid_1024_16_full.yaml}
CKPT=${CKPT:-output/AE_KL_hybrid_1024_16_full/best.pth}
YEAR=${YEAR:-2017}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-0}
MAX_BATCHES=${MAX_BATCHES:-0}
OUTDIR=${OUTDIR:-eval_outputs/wrmse_2017}
SAVE_NPY=${SAVE_NPY:-1}
NPY_PATH=${NPY_PATH:-}

mkdir -p out
SUBMIT_LOG="./out/submit_${JOB_NAME}.out"

CMD=(
  python -u test.py
  --cfg "${CFG}"
  --ckpt "${CKPT}"
  --year "${YEAR}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --max_batches "${MAX_BATCHES}"
  --outdir "${OUTDIR}"
)

if [[ "${SAVE_NPY}" == "1" ]]; then
  CMD+=(--save_npy)
fi
if [[ -n "${NPY_PATH}" ]]; then
  CMD+=(--npy_path "${NPY_PATH}")
fi

if srun -p "${PARTITION}" --quotatype="${QUOTATYPE}" \
  --job-name="${JOB_NAME}" \
  --ntasks=1 \
  --cpus-per-task="${CPUS}" \
  -N 1 \
  --gres="gpu:${GPUS}" \
  --async \
  --kill-on-bad-exit=1 \
  -o "./out/eval_wrmse_%j.out" \
  "${CMD[@]}" > "${SUBMIT_LOG}" 2>&1; then
  echo "Submitted ${JOB_NAME}. Submit log: ${SUBMIT_LOG}; eval log: ./out/eval_wrmse_<jobid>.out"
else
  echo "Submit failed. See ${SUBMIT_LOG}" >&2
  tail -40 "${SUBMIT_LOG}" >&2 || true
  exit 1
fi
