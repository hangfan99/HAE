import argparse
import os
import yaml
import torch

from utils import misc as dist_utils
from utils.builder import ConfigBuilder
from trainers.ae_kl_trainer import AEKLTrainer
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/ae_kl.yaml")
    parser.add_argument("--outdir", type=str, default="output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_dir", type=str, default="")
    parser.add_argument("--snapshot_path", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--per_cpus", type=int, default=4)
    parser.add_argument("--init_method", type=str, default="tcp://127.0.0.1:19111")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.world_size > 1:
        dist_utils.init_distributed_mode(args)
    else:
        args.rank = 0
        args.local_rank = 0
        args.distributed = False
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)

    cfg = load_config(args.cfg)
    dist_utils.setup_seed(args.seed + args.rank)

    builder = ConfigBuilder(cfg)
    trainer = AEKLTrainer(cfg, args, builder)

    if args.rank == 0:
        os.makedirs(trainer.run_dir, exist_ok=True)
        with open(os.path.join(trainer.run_dir, "config_resolved.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    try:
        trainer.train()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
