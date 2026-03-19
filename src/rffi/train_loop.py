from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rffi.config import AppConfig
from rffi.models.jrffp_sc_plus import JRFFPSCPlus
from rffi.utils.env import RuntimeEnv


@dataclass
class EvalResult:
    closed_set_accuracy: float
    open_set_auc: float
    open_set_eer: float
    open_set_overall_accuracy: float


class TrainerEngine:
    def __init__(self, cfg: AppConfig, env: RuntimeEnv):
        self.cfg = cfg
        self.env = env
        self.device = env.device
        self.criterion = nn.CrossEntropyLoss()

    def _autocast_ctx(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=self.env.amp_dtype)
        return torch.autocast(device_type="cpu", enabled=False)

    def train_classifier(
        self,
        model: JRFFPSCPlus,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> JRFFPSCPlus:
        model = model.to(self.device)

        if self.cfg.train.compile_model and self.device.type == "cuda":
            model = torch.compile(model)  # type: ignore[assignment]

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
        )
        scaler_enabled = self.device.type == "cuda" and self.env.amp_dtype == torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

        global_step = 0
        model.train()
        for epoch in range(self.cfg.train.epochs):
            pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{self.cfg.train.epochs}", leave=False)
            optimizer.zero_grad(set_to_none=True)
            for i, (x, y) in enumerate(pbar):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                if self.cfg.train.channels_last and x.ndim == 4 and self.device.type == "cuda":
                    x = x.to(memory_format=torch.channels_last)

                with self._autocast_ctx():
                    logits, _ = model(x)
                    loss = self.criterion(logits, y) / self.cfg.train.grad_accum_steps

                if scaler_enabled:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (i + 1) % self.cfg.train.grad_accum_steps == 0:
                    if scaler_enabled:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                global_step += 1
                if global_step % self.cfg.train.log_every == 0:
                    pbar.set_postfix(loss=f"{loss.item() * self.cfg.train.grad_accum_steps:.4f}")

            if val_loader is not None:
                val_acc = self.closed_set_accuracy(model, val_loader)
                print(f"[epoch={epoch+1}] val_closed_acc={val_acc:.4f}")

        return model

    @torch.no_grad()
    def closed_set_accuracy(self, model: JRFFPSCPlus, loader: DataLoader) -> float:
        model.eval()
        total = 0
        correct = 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            if self.cfg.train.channels_last and x.ndim == 4 and self.device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            with self._autocast_ctx():
                logits, _ = model(x)
            pred = torch.argmax(logits, dim=1)
            total += y.numel()
            correct += (pred == y).sum().item()
        model.train()
        return correct / max(total, 1)

    @torch.no_grad()
    def build_prototypes(
        self,
        model: JRFFPSCPlus,
        enroll_loader: DataLoader,
    ) -> dict[int, torch.Tensor]:
        model.eval()
        features: dict[int, list[torch.Tensor]] = {}
        for x, y in enroll_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            if self.cfg.train.channels_last and x.ndim == 4 and self.device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            with self._autocast_ctx():
                _, emb = model(x)
            for vec, label in zip(emb, y):
                key = int(label.item())
                features.setdefault(key, []).append(vec.detach().float().cpu())

        prototypes: dict[int, torch.Tensor] = {}
        for label, vecs in features.items():
            stacked = torch.stack(vecs, dim=0)
            proto = stacked.mean(dim=0)
            proto = torch.nn.functional.normalize(proto, dim=0)
            prototypes[label] = proto
        model.train()
        return prototypes

    @staticmethod
    def _eer_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> float:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fnr = 1.0 - tpr
        idx = np.nanargmin(np.abs(fpr - fnr))
        return float((fpr[idx] + fnr[idx]) / 2.0)

    @torch.no_grad()
    def collect_fused_scores(
        self,
        model: JRFFPSCPlus,
        loader: DataLoader,
        prototypes: dict[int, torch.Tensor],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not prototypes:
            raise ValueError("Prototype dictionary is empty")

        proto_labels = torch.tensor(sorted(prototypes.keys()), dtype=torch.long)
        proto_matrix = torch.stack([prototypes[int(k)] for k in proto_labels.tolist()], dim=0).to(self.device)

        all_scores: list[float] = []
        all_known_flags: list[int] = []
        all_closed_correct: list[int] = []
        all_open_correct: list[int] = []

        model.eval()
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            if self.cfg.train.channels_last and x.ndim == 4 and self.device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)

            with self._autocast_ctx():
                logits, emb = model(x)

            probs = torch.softmax(logits.float(), dim=1)
            topk_vals, topk_idx = torch.topk(probs, k=min(self.cfg.model.top_k, probs.shape[1]), dim=1)

            emb = torch.nn.functional.normalize(emb.float(), dim=1)
            for i in range(emb.shape[0]):
                cand_labels = topk_idx[i]
                cand_probs = topk_vals[i].clamp_min(1e-8)

                dists = []
                for c, p in zip(cand_labels, cand_probs):
                    c_int = int(c.item())
                    if c_int not in prototypes:
                        continue
                    proto = prototypes[c_int].to(self.device)
                    d = torch.norm(emb[i] - proto, p=2)
                    fused = d - self.cfg.model.score_alpha * torch.log(p)
                    dists.append((float(fused.item()), c_int))

                if not dists:
                    # If top-k classes are absent from prototype set, force reject.
                    all_scores.append(1e9)
                    all_known_flags.append(0)
                    all_closed_correct.append(0)
                    all_open_correct.append(0)
                    continue

                best_score, pred_label = min(dists, key=lambda x_: x_[0])
                gt = int(y[i].item())

                is_known = int(gt in prototypes)
                closed_ok = int(is_known and pred_label == gt)

                all_scores.append(best_score)
                all_known_flags.append(is_known)
                all_closed_correct.append(closed_ok)
                all_open_correct.append(0)

        model.train()
        return (
            np.asarray(all_scores, dtype=np.float64),
            np.asarray(all_known_flags, dtype=np.int64),
            np.asarray(all_closed_correct, dtype=np.int64),
            np.asarray(all_open_correct, dtype=np.int64),
        )

    @staticmethod
    def calibrate_threshold_from_known(scores_known: np.ndarray, target_frr: float = 0.05) -> float:
        q = 1.0 - target_frr
        q = float(min(max(q, 0.5), 0.999))
        return float(np.quantile(scores_known, q=q))

    def evaluate_open_set(
        self,
        scores: np.ndarray,
        known_flags: np.ndarray,
        closed_correct: np.ndarray,
        threshold: float,
    ) -> EvalResult:
        if scores.shape[0] == 0:
            raise ValueError("No scores to evaluate")

        # Smaller fused score means more likely legitimate.
        y_true_known = known_flags.astype(np.int64)
        y_score_known = -scores

        if len(np.unique(y_true_known)) < 2:
            auc = 0.5
            eer = 0.5
        else:
            auc = float(roc_auc_score(y_true_known, y_score_known))
            eer = self._eer_from_scores(y_true_known, y_score_known)

        accepted = scores <= threshold
        closed_set_acc = float(closed_correct.sum() / max(y_true_known.sum(), 1))

        overall_correct = 0
        for is_known, is_acc, is_accepted in zip(y_true_known, closed_correct, accepted):
            if is_known == 1:
                overall_correct += int(is_accepted and is_acc == 1)
            else:
                overall_correct += int(not is_accepted)
        overall_acc = float(overall_correct / scores.shape[0])

        return EvalResult(
            closed_set_accuracy=closed_set_acc,
            open_set_auc=auc,
            open_set_eer=eer,
            open_set_overall_accuracy=overall_acc,
        )

    @staticmethod
    def save_checkpoint(model: JRFFPSCPlus, output_dir: str | Path, name: str) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ckpt_path = out / f"{name}.pt"
        torch.save(model.state_dict(), ckpt_path)
        return ckpt_path
