#!/bin/bash

gpus=4
node_num=1
single_gpus=`expr $gpus / $node_num`
cpus=4

partition=earth-e2e-p
quotatype=reserved
job_name=hae
# yaml=./configs/ae_kl_hybrid_1024_16_full_patch_nores.yaml
yaml=./configs/ae_kl_hybrid_hier2_34_full.yaml
outdir=output

# yaml=./configs/ae_kl_hybrid_1024_16_full.yaml
# job_name=hae_ae_hybrid_1024_16_full

while true
do
  PORT=$((((RANDOM<<15)|RANDOM)%49152 + 10000))
  break
done
echo $PORT

mkdir -p ./out

srun -p $partition --quotatype=$quotatype --job-name=$job_name \
  --ntasks-per-node=$single_gpus --cpus-per-task=1 -N $node_num \
  -o ./out/train_%j.out --gres=gpu:$single_gpus --async --kill-on-bad-exit=1 \
  python -u train.py \
  --cfg $yaml \
  --outdir $outdir \
  --init_method tcp://127.0.0.1:$PORT \
  --per_cpus $cpus \
  --world_size $gpus

sleep 2
rm -f batchscript-*
