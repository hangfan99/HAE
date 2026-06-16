#!/usr/bin/env bash
set -euo pipefail

GPUS=${GPUS:-2}
NODE_NUM=${NODE_NUM:-1}
CPUS=${CPUS:-4}
PARTITION=${PARTITION:-earth-e2e-p}
QUOTATYPE=${QUOTATYPE:-spot}
JOB_NAME=${JOB_NAME:-hae_ae_kl}
CFG=${CFG:-configs/ae_kl_smoke.yaml}
OUTDIR=${OUTDIR:-output}

SINGLE_GPUS=$((GPUS / NODE_NUM))
if (( SINGLE_GPUS < 1 )); then
  echo "Invalid GPU setting: GPUS=${GPUS}, NODE_NUM=${NODE_NUM}" >&2
  exit 1
fi

PORT=$(( (RANDOM << 15 | RANDOM) % 49152 + 10000 ))
echo "Using port: ${PORT}"
echo "Config: ${CFG}"

mkdir -p out

srun -p "${PARTITION}" --quotatype="${QUOTATYPE}" \
  --job-name="${JOB_NAME}" \
  --ntasks-per-node="${SINGLE_GPUS}" \
  --cpus-per-task=1 \
  -N "${NODE_NUM}" \
  --gres="gpu:${SINGLE_GPUS}" \
  --async \
  --kill-on-bad-exit=1 \
  -o "./out/train_%j.out" \
  python -u train.py \
    --cfg "${CFG}" \
    --outdir "${OUTDIR}" \
    --init_method "tcp://127.0.0.1:${PORT}" \
    --world_size "${GPUS}" \
    --per_cpus "${CPUS}"
