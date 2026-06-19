#!/usr/bin/env bash
set -euo pipefail

PARTITION=${PARTITION:-earth-e2e-p}
QUOTATYPE=${QUOTATYPE:-spot}
JOB_NAME=${JOB_NAME:-hae_ae_eval_nb}
GPUS=${GPUS:-1}
CPUS=${CPUS:-4}
NOTEBOOK=${NOTEBOOK:-notebooks/evaluate_ae_case.ipynb}
CFG_PATH=${CFG_PATH:-configs/ae_kl_552_16_evit_full.yaml}
CKPT_PATH=${CKPT_PATH:-}
SPLIT=${SPLIT:-valid}
MAX_BATCHES=${MAX_BATCHES:-2}
BATCH_SIZE=${BATCH_SIZE:-2}
DATA_NUM_WORKERS=${DATA_NUM_WORKERS:-0}
EVAL_OUTDIR=${EVAL_OUTDIR:-eval_outputs}
REPO_ROOT=${REPO_ROOT:-$(pwd)}

mkdir -p "${REPO_ROOT}/out" "${REPO_ROOT}/${EVAL_OUTDIR}"

export CFG_PATH CKPT_PATH SPLIT MAX_BATCHES BATCH_SIZE DATA_NUM_WORKERS EVAL_OUTDIR REPO_ROOT

echo "Notebook: ${NOTEBOOK}"
echo "Config: ${CFG_PATH}"
echo "Checkpoint: ${CKPT_PATH:-<auto latest best/final/snapshot>}"
echo "Split=${SPLIT}, MAX_BATCHES=${MAX_BATCHES}, BATCH_SIZE=${BATCH_SIZE}, DATA_NUM_WORKERS=${DATA_NUM_WORKERS}"
echo "Outdir: ${EVAL_OUTDIR}"
echo "Repo root: ${REPO_ROOT}"
echo "Partition=${PARTITION}, Quota=${QUOTATYPE}, GPUS=${GPUS}, CPUS=${CPUS}"

INNER_CMD="cd '${REPO_ROOT}' && export REPO_ROOT='${REPO_ROOT}' CFG_PATH='${CFG_PATH}' CKPT_PATH='${CKPT_PATH}' SPLIT='${SPLIT}' MAX_BATCHES='${MAX_BATCHES}' BATCH_SIZE='${BATCH_SIZE}' DATA_NUM_WORKERS='${DATA_NUM_WORKERS}' EVAL_OUTDIR='${EVAL_OUTDIR}' && python -m jupyter nbconvert --to notebook --execute '${NOTEBOOK}' --output-dir '${EVAL_OUTDIR}' --output 'evaluate_ae_case_executed' --ExecutePreprocessor.timeout=-1"

srun -p "${PARTITION}" --quotatype="${QUOTATYPE}" \
  --job-name="${JOB_NAME}" \
  -N 1 \
  --ntasks=1 \
  --cpus-per-task="${CPUS}" \
  --gres="gpu:${GPUS}" \
  --async \
  --kill-on-bad-exit=1 \
  -o "${REPO_ROOT}/out/eval_ae_%j.out" \
  bash -lc "${INNER_CMD}"
