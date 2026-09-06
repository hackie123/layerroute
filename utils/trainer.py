"""
utils/trainer.py
================
Fine-tunes LoRA adapters + routers via standard LM loss.

Trainable: LoRA A/B matrices + router linears (~3.65M params)
Frozen   : All original Qwen weights (494M)

CSV logs:
    training_log.csv  — step, loss, ppl, layers_run, skip_pct, lr, elapsed_s
    eval_log.csv      — step, val_loss, val_ppl, val_layers_run, val_skip_pct
    gate_log.csv      — step, gate_0 .. gate_23 (soft gate values per layer)
    run_summary.csv   — one row per run
"""

import os, csv, time, json, datetime
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.gated_qwen import GatedQwenLoRA
from utils.config import SystemConfig


class CSVLogger:
    def __init__(self, path, fieldnames):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path; self.fieldnames = fieldnames
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def write(self, row):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(
                {k: row.get(k, "") for k in self.fieldnames}
            )


class Trainer:

    def __init__(self, model: GatedQwenLoRA, cfg: SystemConfig):
        self.model  = model
        self.cfg    = cfg
        self.device = next(p for p in model.parameters()).device

        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(trainable, lr=cfg.training.lr,
                                     weight_decay=0.01, betas=(0.9, 0.999))
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=cfg.training.max_steps, eta_min=1e-6
        )

        # Ensure routers are on same device as model
        model.routers = model.routers.to(self.device)
        self.step          = 0
        self.best_val_loss = float("inf")
        self.run_id        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = cfg.training.output_dir
        os.makedirs(out, exist_ok=True)

        n_layers = 24
        self.train_log = CSVLogger(os.path.join(out, "training_log.csv"), [
            "step","loss","ppl","layers_run","skip_pct","gate_reg","lr","elapsed_s"
        ])
        self.eval_log = CSVLogger(os.path.join(out, "eval_log.csv"), [
            "step","val_loss","val_ppl","val_layers_run","val_skip_pct"
        ])
        self.gate_log = CSVLogger(os.path.join(out, "gate_log.csv"),
            ["step"] + [f"gate_{i}" for i in range(n_layers)]
        )
        self.summary_log = CSVLogger(os.path.join(out, "run_summary.csv"), [
            "run_id","best_val_loss","final_step","total_time_s","config_json"
        ])

    def train(self, train_loader, val_loader=None):
        cfg     = self.cfg.training
        n_accum = cfg.grad_accum

        print(f"\n{'='*55}")
        print(f"  CONFGATE v6 — LoRA + Hard-Gated Skip Connections")
        print(f"{'='*55}")
        print(f"  Trainable params : {self.model.trainable_params():,}")
        print(f"  Total params     : {self.model.total_params():,}")
        print(f"  LoRA rank        : {self.cfg.lora.r}")
        print(f"  Grad accum       : {n_accum} (effective batch={cfg.batch_size*n_accum})")
        print(f"  Max steps        : {cfg.max_steps}")
        print(f"  LR               : {cfg.lr}")
        print(f"{'='*55}\n")

        self.model.train()
        running = {"loss":0.0, "layers_run":0.0, "skip_pct":0.0}
        t_start = time.perf_counter()
        train_iter = iter(train_loader)
        self.optimizer.zero_grad()

        while self.step < cfg.max_steps:
            try: batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            ids        = batch["input_ids"].to(self.device)
            labels     = batch["labels"].to(self.device)
            provenance = batch["provenance"].to(self.device)  # [B], 1=tool_call, 0=planning


            if cfg.ablation in ("bce_naive", "bce_corrected"):
                # BCE ablation (paper Table 2, "LayerRoute-BCE"): replaces the
                # gate-regularization term with BCE against dataset-provenance
                # labels. Not implemented in the original codebase (verified
                # absent everywhere); reconstructed here from the paper's
                # one-line description, which is genuinely ambiguous about
                # label direction -- both readings are implemented so the
                # ambiguity can be tested directly rather than guessed at:
                #
                #   bce_naive:     target = provenance directly (tool_call=1
                #                   trains gate toward OPEN=1/run). This is
                #                   the most literal reading of "tool_call=1,
                #                   planning=0" as the BCE target, but note
                #                   it pushes AGAINST the paper's own goal --
                #                   tool_call is supposed to SKIP MORE (gate
                #                   toward 0), not run more.
                #   bce_corrected: target = 1 - provenance (tool_call=1 trains
                #                   gate toward CLOSED=0/skip -- the direction
                #                   actually consistent with "tool_call skips
                #                   more"). Tests whether the naive reading's
                #                   inverted direction explains why the
                #                   paper's own BCE ablation underperformed
                #                   (5.1% differential vs. 12.91% for the
                #                   implicit objective).
                logits, soft_gates, gate_stats = self.model._forward_layers(ids)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                lm_loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, logits.shape[-1]),
                    shift_labels.view(-1), ignore_index=-100,
                )

                bce_target = provenance if cfg.ablation == "bce_naive" else (1.0 - provenance)
                per_layer_soft = gate_stats["soft_gates_per_sample"]  # [L] list of [B]
                bce_losses = [
                    torch.nn.functional.binary_cross_entropy(sg, bce_target)
                    for sg in per_layer_soft
                ]
                bce_loss = torch.stack(bce_losses).mean()
                loss = lm_loss + self.cfg.router.gate_reg_weight * bce_loss
            else:
                _, loss, gate_stats = self.model(ids, labels=labels)
            (loss / n_accum).backward()

            running["loss"]       += loss.item()
            running["layers_run"] += gate_stats["layers_run"]
            running["skip_pct"]   += gate_stats["skip_pct"]

            if (self.step + 1) % n_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    cfg.grad_clip
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            self.step += 1

            if self.step % cfg.log_every == 0:
                n       = cfg.log_every
                elapsed = time.perf_counter() - t_start
                lr_now  = self.scheduler.get_last_lr()[0]
                avg_loss     = running["loss"] / n
                avg_layers   = running["layers_run"] / n
                avg_skip_pct = running["skip_pct"] / n
                ppl          = min(torch.exp(torch.tensor(avg_loss)).item(), 9999)

                print(
                    f"  step {self.step:5d}/{cfg.max_steps} | "
                    f"loss={avg_loss:.4f} | ppl={ppl:.1f} | "
                    f"layers_run={avg_layers:.1f}/24 | "
                    f"skip={avg_skip_pct:.1f}% | "
                    f"lr={lr_now:.2e} | {elapsed:.0f}s"
                )

                self.train_log.write({
                    "step"      : self.step,
                    "loss"      : round(avg_loss, 6),
                    "ppl"       : round(ppl, 2),
                    "layers_run": round(avg_layers, 2),
                    "skip_pct"  : round(avg_skip_pct, 2),
                    "lr"        : round(lr_now, 8),
                    "elapsed_s" : round(elapsed, 2),
                })

                # Log gate values
                gvals = self.model.routers.gate_values()
                self.gate_log.write({
                    "step": self.step,
                    **{f"gate_{i}": round(gvals[i], 4) for i in range(len(gvals))}
                })

                running = {"loss":0.0, "layers_run":0.0, "skip_pct":0.0}

            if self.step % cfg.eval_every == 0:
                if val_loader:
                    vm = self._validate(val_loader)
                    print(
                        f"  [Val] step={self.step} "
                        f"loss={vm['val_loss']:.4f} ppl={vm['val_ppl']:.1f} "
                        f"layers_run={vm['val_layers_run']:.1f} "
                        f"skip={vm['val_skip_pct']:.1f}%"
                    )
                    self.eval_log.write({"step": self.step, **vm})
                    if vm["val_loss"] < self.best_val_loss:
                        self.best_val_loss = vm["val_loss"]
                        self.model.save_adapters(
                            os.path.join(cfg.output_dir, "best_adapters.pt")
                        )
                self.model.save_adapters(
                    os.path.join(cfg.output_dir, f"adapters_{self.step}.pt")
                )
                self.model.train()

        total_time = time.perf_counter() - t_start
        print(f"\n  Training complete. Best val loss: {self.best_val_loss:.4f}")
        print(f"  Total time: {total_time:.0f}s")

        # Final gate analysis
        print("\n  Final gate values (lower = more often skipped):")
        gvals = self.model.routers.gate_values()
        for i, g in enumerate(gvals):
            bar = "█" * int(g * 20)
            print(f"  Layer {i:2d}: {g:.4f}  {bar}")

        self.summary_log.write({
            "run_id"       : self.run_id,
            "best_val_loss": round(self.best_val_loss, 6),
            "final_step"   : self.step,
            "total_time_s" : round(total_time, 1),
            "config_json"  : json.dumps({
                "lora_r"    : self.cfg.lora.r,
                "lr"        : cfg.lr,
                "max_steps" : cfg.max_steps,
                "batch_size": cfg.batch_size,
                "grad_accum": cfg.grad_accum,
            })
        })
        print(f"  Logs → {cfg.output_dir}/")

    def _validate(self, val_loader, max_batches=30):
        self.model.eval()
        totals = {"val_loss":0.0, "val_layers_run":0.0, "val_skip_pct":0.0}
        n = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= max_batches: break
                ids        = batch["input_ids"].to(self.device)
                labels     = batch["labels"].to(self.device)
                provenance = batch["provenance"].to(self.device)

                if self.cfg.training.ablation in ("bce_naive", "bce_corrected"):
                    logits, soft_gates, gs = self.model._forward_layers(ids)
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    lm_loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, logits.shape[-1]),
                        shift_labels.view(-1), ignore_index=-100,
                    )
                    bce_target = provenance if self.cfg.training.ablation == "bce_naive" else (1.0 - provenance)
                    per_layer_soft = gs["soft_gates_per_sample"]
                    bce_loss = torch.stack([
                        torch.nn.functional.binary_cross_entropy(sg, bce_target)
                        for sg in per_layer_soft
                    ]).mean()
                    loss = lm_loss + self.cfg.router.gate_reg_weight * bce_loss
                else:
                    _, loss, gs = self.model(ids, labels=labels)
                totals["val_loss"]       += loss.item()
                totals["val_layers_run"] += gs["layers_run"]
                totals["val_skip_pct"]   += gs["skip_pct"]
                n += 1
        avg = {k: round(v/max(n,1), 4) for k, v in totals.items()}
        avg["val_ppl"] = round(
            min(torch.exp(torch.tensor(avg["val_loss"])).item(), 9999), 2
        )
        return avg
