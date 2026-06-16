#!/usr/bin/env bash
set -euo pipefail

python train.py \
  --cfg configs/ae_kl.yaml \
  --outdir output \
  --world_size 1 \
  --per_cpus 4
