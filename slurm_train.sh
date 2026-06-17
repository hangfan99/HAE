#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Usage examples
# -----------------------------
# 1) default submit (2 GPU, spot)
#    bash slurm_train.sh
#
# 2) override to reserved
#    QUOTATYPE=reserved bash slurm_train.sh
#
# 3) override config / gpus / cpus
#    CFG=configs/ae_kl_552_16_evit_smoke.yaml GPUS=1 CPUS=4 bash slurm_train.sh

GPUS=${GPUS:-2}
NODE_NUM=${NODE_NUM:-1}
CPUS=${CPUS:-4}
PARTITION=${PARTITION:-earth-e2e-p}
QUOTATYPE=${QUOTATYPE:-reserved}
JOB_NAME=${JOB_NAME:-hae_ae_552_16_evit}
CFG=${CFG:-configs/ae_kl_552_16_evit_smoke.yaml}
OUTDIR=${OUTDIR:-output}

SINGLE_GPUS=$((GPUS / NODE_NUM))
if (( SINGLE_GPUS < 1 )); then
  echo "Invalid GPU setting: GPUS=${GPUS}, NODE_NUM=${NODE_NUM}" >&2
  exit 1
fi

PORT=$(( (RANDOM << 15 | RANDOM) % 49152 + 10000 ))

echo "Using port: ${PORT}"
echo "Partition: ${PARTITION}, Quota: ${QUOTATYPE}"
echo "GPUS=${GPUS}, NODE_NUM=${NODE_NUM}, SINGLE_GPUS=${SINGLE_GPUS}, CPUS=${CPUS}"
echo "Config: ${CFG}"
echo "Outdir: ${OUTDIR}"

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
