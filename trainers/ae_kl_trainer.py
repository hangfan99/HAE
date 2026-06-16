import os
import time
import math
import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from model.AE_2D_v2 import Mix_loss
from utils import misc as dist_utils
from utils.logger import get_logger


STD_LAYER = [
    5.610453475051704, 4.798220612223473, 21.32010786700973, 1336.2115992274876,
    3755.2810557402927, 4357.588191568988, 5253.301115477269, 5540.73074484052,
    5405.73040397736, 5020.194961603476, 4104.233456672573, 3299.702929930327,
    2629.7201995715513, 2060.9872289877453, 1399.3410970050247, 1187.5419349409494,
    1098.9952409939283, 1.1555282996146702e-07, 4.2315237954921815e-07,
    3.1627283344500357e-06, 2.093742795871515e-05, 7.02963683704546e-05,
    0.00016131853114827985, 0.00048331132466880735, 0.001023028433607086,
    0.0016946778969914426, 0.0024928432426471183, 0.004184742037434761,
    0.005201345241925773, 0.00611814321149996, 11.557361639969054,
    11.884088705628045, 15.407016747306344, 17.286773058038722,
    17.720698660431694, 17.078782531259524, 14.509924979003983,
    12.215305549952125, 10.503871726997783, 9.286354460633103,
    8.179197305830433, 7.93264239491015, 6.126056325796786, 8.417864770061094,
    8.178248048405905, 9.998695230009567, 11.896325029659364, 13.360381609448558,
    13.474533447403218, 11.44656476066317, 9.321096224035244, 7.835396470389893,
    6.858187372121642, 6.186618416862026, 6.345356147017278, 5.23175612906023,
    9.495652698988557, 13.738672642636256, 9.090666595626503, 5.933385737657316,
    7.389004707914384, 10.212310312072752, 12.773099916244078,
    13.459313552230206, 13.858620163486986, 15.021590351519892,
    16.00275340237577, 16.88523210573196, 18.59201174892538,
]


class AEKLTrainer:
    def __init__(self, cfg, args, builder):
        self.cfg = cfg
        self.args = args
        self.builder = builder

        self.rank = args.rank
        self.local_rank = args.local_rank
        self.distributed = args.world_size > 1
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")

        self.run_dir = self._build_run_dir()
        self.logger = get_logger("train_ae_kl", self.run_dir, self.rank, filename="train.log", resume=args.resume)
        self.writer = SummaryWriter(log_dir=os.path.join(self.run_dir, "tb")) if self.rank == 0 else None

        self.model = self.builder.build_model().to(self.device)
        self.model = dist_utils.DistributedParallel_Model(
            self.model, self.local_rank, find_unused_parameters=True
        )

        self.criterion = Mix_loss(
            kl_weight=cfg["loss"].get("kl_weight", 2e-5),
            KL=cfg["loss"].get("enable_kl", True),
        ).to(self.device)

        self.optimizer = self.builder.build_optimizer(self.model)
        self.train_loader, _, self.train_sampler = self.builder.build_dataloader(
            "train", args.per_cpus, self.distributed
        )

        total_steps = len(self.train_loader) * cfg["trainer"]["epochs"]
        self.scheduler = self._build_scheduler(self.optimizer, total_steps)

        self.best_loss = float("inf")
        self.global_step = 0
        self.start_epoch = 0

        if args.resume:
            self._resume()

    def _build_run_dir(self):
        exp_name = self.cfg["experiment"]["name"]
        ts = time.strftime("%Y%m%d_%H%M%S")
        if self.args.resume and self.args.resume_dir:
            run_dir = self.args.resume_dir
        else:
            run_dir = os.path.join(self.args.outdir, exp_name, f"world_size{self.args.world_size}-{ts}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def _build_scheduler(self, optimizer, total_steps):
        warmup_ratio = self.cfg["scheduler"].get("warmup_ratio", 0.05)
        cycles = self.cfg["scheduler"].get("num_cycles", 0.5)
        warmup_steps = max(1, int(total_steps * warmup_ratio))

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * cycles * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _model_state(self):
        return self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self._model_state(),
            "SCHEDULER_STATE": self.scheduler.state_dict(),
            "EPOCHS_RUN": epoch,
            "BEST_LOSS": self.best_loss,
            "GLOBAL_STEP": self.global_step,
        }
        if self.cfg["trainer"].get("save_optimizer_in_snapshot", False):
            snapshot["OPTIMIZER_STATE"] = self.optimizer.state_dict()
        torch.save(snapshot, os.path.join(self.run_dir, "snapshot_latest.pth"))

    def _resume(self):
        snapshot_path = self.args.snapshot_path or os.path.join(self.run_dir, "snapshot_latest.pth")
        if not os.path.exists(snapshot_path):
            self.logger.info("No snapshot found at %s, train from scratch.", snapshot_path)
            return

        map_location = {"cuda:0": f"cuda:{self.local_rank}"}
        snapshot = torch.load(snapshot_path, map_location=map_location)
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(snapshot["MODEL_STATE"])
        else:
            self.model.load_state_dict(snapshot["MODEL_STATE"])
        if "OPTIMIZER_STATE" in snapshot:
            self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])
        self.start_epoch = int(snapshot["EPOCHS_RUN"]) + 1
        self.best_loss = float(snapshot.get("BEST_LOSS", float("inf")))
        self.global_step = int(snapshot.get("GLOBAL_STEP", 0))
        self.logger.info("Resume from %s at epoch %d", snapshot_path, self.start_epoch)

    def _rebuild_train_loader(self):
        self.train_loader, _, self.train_sampler = self.builder.build_dataloader(
            "train", self.args.per_cpus, self.distributed
        )

    def _log_train(self, epoch, step, loss):
        if self.rank != 0:
            return
        msg = (
            f"Epoch [{epoch + 1}/{self.cfg['trainer']['epochs']}] "
            f"Step [{step + 1}/{len(self.train_loader)}] "
            f"loss={loss:.6f} lr={self.scheduler.get_last_lr()[0]:.6e}"
        )
        self.logger.info(msg)
        if self.writer is not None:
            self.writer.add_scalar("train/loss", loss, self.global_step)
            self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)

    @torch.no_grad()
    def _validate(self):
        self.model.eval()

        valid_loader, _, _ = self.builder.build_dataloader(
            "valid", self.cfg["trainer"].get("valid_num_workers", 4), False
        )

        std = torch.tensor(STD_LAYER, dtype=torch.float32, device=self.device).view(1, -1, 1, 1)

        data_num = len(valid_loader.dataset)
        mix_err = 0.0
        z_re_loss = q_re_loss = t_re_loss = u_re_loss = v_re_loss = 0.0

        for x in valid_loader:
            x = x.to(self.device, non_blocking=True)
            batch = x.size(0)

            x_recon, _ = self.model(x)
            mix_err += F.l1_loss(x_recon, x).item() * batch

            x_denorm = x * std
            x_recon_denorm = x_recon * std

            z_re_loss += F.l1_loss(x_recon_denorm[:, 4:17], x_denorm[:, 4:17]).item() * batch
            q_re_loss += F.l1_loss(x_recon_denorm[:, 17:30], x_denorm[:, 17:30]).item() * batch
            u_re_loss += F.l1_loss(x_recon_denorm[:, 30:43], x_denorm[:, 30:43]).item() * batch
            v_re_loss += F.l1_loss(x_recon_denorm[:, 43:56], x_denorm[:, 43:56]).item() * batch
            t_re_loss += F.l1_loss(x_recon_denorm[:, 56:69], x_denorm[:, 56:69]).item() * batch

        metrics = {
            "mix_err": mix_err / data_num,
            "z_err": z_re_loss / data_num,
            "q_err": q_re_loss / data_num,
            "u_err": u_re_loss / data_num,
            "v_err": v_re_loss / data_num,
            "t_err": t_re_loss / data_num,
        }

        if self.rank == 0:
            self.logger.info(
                "Valid mix=%.6f z=%.6f q=%.6f t=%.6f u=%.6f v=%.6f",
                metrics["mix_err"], metrics["z_err"], metrics["q_err"],
                metrics["t_err"], metrics["u_err"], metrics["v_err"],
            )
            if self.writer is not None:
                for k, v in metrics.items():
                    self.writer.add_scalar(f"valid/{k}", v, self.global_step)

        return metrics

    def train(self):
        epochs = self.cfg["trainer"]["epochs"]
        log_interval = self.cfg["trainer"].get("log_interval", 50)
        grad_clip = self.cfg["trainer"].get("grad_clip", 1.0)
        refresh_loader = self.cfg["trainer"].get("refresh_dataloader_each_epoch", True)

        for epoch in range(self.start_epoch, epochs):
            if refresh_loader and epoch > self.start_epoch:
                self._rebuild_train_loader()

            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            self.model.train()

            for step, x in enumerate(self.train_loader):
                x = x.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)

                x_recon, posterior = self.model(x)
                loss = self.criterion(x_recon, x, posterior)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
                self.optimizer.step()
                self.scheduler.step()

                if (step + 1) % log_interval == 0:
                    self._log_train(epoch, step, loss.item())
                self.global_step += 1

            if self.rank == 0:
                metrics = self._validate()
                if metrics["mix_err"] < self.best_loss:
                    self.best_loss = metrics["mix_err"]
                    if self.cfg["trainer"].get("save_best_checkpoint", True):
                        torch.save(self._model_state(), os.path.join(self.run_dir, "best.pth"))
                        self.logger.info("Save best checkpoint, mix_err=%.6f", self.best_loss)
                    else:
                        self.logger.info("Best metric updated to %.6f (best checkpoint disabled)", self.best_loss)

                self._save_snapshot(epoch)

            if self.distributed:
                torch.distributed.barrier(device_ids=[self.local_rank])

        if self.writer is not None:
            self.writer.close()
