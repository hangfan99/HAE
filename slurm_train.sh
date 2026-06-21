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
#    CFG=configs/ae_kl_hybrid_hier2_34_full.yaml GPUS=4 CPUS=4 bash slurm_train.sh

GPUS=${GPUS:-4}
NODE_NUM=${NODE_NUM:-1}
CPUS=${CPUS:-4}
PARTITION=${PARTITION:-earth-e2e-p}
QUOTATYPE=${QUOTATYPE:-reserved}
JOB_NAME=${JOB_NAME:-hae_ae_hybrid_hier2_34}
CFG=${CFG:-configs/ae_kl_hybrid_hier2_34_full.yaml}
OUTDIR=${OUTDIR:-output}

SINGLE_GPUS=$((GPUS / NODE_NUM))
if (( SINGLE_GPUS < 1 )); then
  echo "Invalid GPU setting: GPUS=${GPUS}, NODE_NUM=${NODE_NUM}" >&2
  exit 1
fi

PORT=$(( (RANDOM << 15 | RANDOM) % 49152 + 10000 ))

mkdir -p out
SUBMIT_LOG="./out/submit_${JOB_NAME}.out"

if srun -p "${PARTITION}" --quotatype="${QUOTATYPE}" \
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
    --per_cpus "${CPUS}" > "${SUBMIT_LOG}" 2>&1; then
  echo "Submitted ${JOB_NAME}. Submit log: ${SUBMIT_LOG}; training log: ./out/train_<jobid>.out"
else
  echo "Submit failed. See ${SUBMIT_LOG}" >&2
  tail -40 "${SUBMIT_LOG}" >&2 || true
  exit 1
fi
