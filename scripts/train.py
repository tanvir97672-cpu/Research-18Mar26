from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rffi.config import AppConfig, load_config
from rffi.data.iq_dataset import LoRaIQDataset, SampleRecord, discover_samples
from rffi.models.jrffp_sc_plus import JRFFPSCPlus
from rffi.train_loop import TrainerEngine
from rffi.utils.env import pick_device
from rffi.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train open-set LoRa RFFI model (L4-optimized)")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--dry-run", action="store_true", help="Run data/model sanity checks only")
    p.add_argument("--disable-wandb", action="store_true")
    return p.parse_args()


def split_known_unknown(records: list[SampleRecord], num_known: int) -> tuple[list[SampleRecord], list[SampleRecord], list[int]]:
    labels = sorted({r.label for r in records})
    if len(labels) < num_known + 1:
        raise ValueError(
            f"Need at least {num_known + 1} device labels for open-set, found {len(labels)}"
        )
    known_labels = labels[:num_known]
    known_set = set(known_labels)
    known = [r for r in records if r.label in known_set]
    unknown = [r for r in records if r.label not in known_set]
    if not known or not unknown:
        raise ValueError("Known/unknown split failed; check labels and num_classes")
    return known, unknown, known_labels


def stratified_partition(records: list[SampleRecord], seed: int) -> tuple[list[SampleRecord], list[SampleRecord], list[SampleRecord], list[SampleRecord]]:
    rng = random.Random(seed)
    by_label: dict[int, list[SampleRecord]] = {}
    for r in records:
        by_label.setdefault(r.label, []).append(r)

    train: list[SampleRecord] = []
    enroll: list[SampleRecord] = []
    calib: list[SampleRecord] = []
    test_known: list[SampleRecord] = []

    for label, bucket in by_label.items():
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = max(1, int(0.7 * n))
        n_enroll = max(1, int(0.1 * n))
        n_calib = max(1, int(0.1 * n))
        cut1 = min(n_train, n)
        cut2 = min(cut1 + n_enroll, n)
        cut3 = min(cut2 + n_calib, n)

        train.extend(bucket[:cut1])
        enroll.extend(bucket[cut1:cut2])
        calib.extend(bucket[cut2:cut3])
        test_known.extend(bucket[cut3:])

        if len(test_known) == 0:
            # Keep at least one test sample per class when class is tiny.
            test_known.append(enroll.pop())

    return train, enroll, calib, test_known


def make_loader(dataset: LoRaIQDataset, cfg: AppConfig, shuffle: bool) -> DataLoader:
    kwargs = {
        "batch_size": cfg.train.batch_size,
        "shuffle": shuffle,
        "num_workers": cfg.train.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    if cfg.train.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs)


def maybe_init_wandb(cfg: AppConfig, disabled: bool):
    if disabled or not cfg.runtime.use_wandb:
        return None
    try:
        import wandb
    except Exception as ex:
        print(f"wandb import failed: {ex}; continuing without wandb")
        return None

    run = wandb.init(
        project=cfg.runtime.wandb_project,
        entity=cfg.runtime.wandb_entity or None,
        name=cfg.runtime.run_name,
        config={
            "data": vars(cfg.data),
            "train": vars(cfg.train),
            "model": vars(cfg.model),
            "runtime": vars(cfg.runtime),
        },
    )
    return run


def in_channels(representation: str) -> int:
    if representation in {"stft", "fft"}:
        return 1
    if representation == "iq":
        return 2
    raise ValueError(f"Unsupported representation {representation}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg.train.seed, deterministic=cfg.train.deterministic)
    env = pick_device(cfg.train.amp_dtype)
    print(f"device={env.device} amp_dtype={env.amp_dtype}")

    try:
        records = discover_samples(cfg.data)
    except Exception as ex:
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "dataset_discovery": "failed",
                        "reason": str(ex),
                        "action": "mount dataset and rerun",
                    },
                    indent=2,
                )
            )
            return
        raise
    known, unknown, known_labels = split_known_unknown(records, cfg.model.num_classes)
    train_rec, enroll_rec, calib_rec, test_known_rec = stratified_partition(known, cfg.train.seed)
    test_rec = test_known_rec + unknown

    if args.dry_run:
        print(
            json.dumps(
                {
                    "total_samples": len(records),
                    "known_samples": len(known),
                    "unknown_samples": len(unknown),
                    "train": len(train_rec),
                    "enroll": len(enroll_rec),
                    "calib": len(calib_rec),
                    "test_known": len(test_known_rec),
                    "test_total": len(test_rec),
                    "known_labels": known_labels,
                },
                indent=2,
            )
        )
        ds = LoRaIQDataset(train_rec[: min(4, len(train_rec))], cfg.data)
        x, y = ds[0]
        print(f"sample_shape={tuple(x.shape)} sample_label={y}")
        return

    train_ds = LoRaIQDataset(train_rec, cfg.data)
    enroll_ds = LoRaIQDataset(enroll_rec, cfg.data)
    calib_ds = LoRaIQDataset(calib_rec, cfg.data)
    test_ds = LoRaIQDataset(test_rec, cfg.data)

    train_loader = make_loader(train_ds, cfg, shuffle=True)
    enroll_loader = make_loader(enroll_ds, cfg, shuffle=False)
    calib_loader = make_loader(calib_ds, cfg, shuffle=False)
    test_loader = make_loader(test_ds, cfg, shuffle=False)

    model = JRFFPSCPlus(
        in_ch=in_channels(cfg.data.representation),
        num_classes=cfg.model.num_classes,
        embedding_dim=cfg.model.embedding_dim,
        width_mult=cfg.model.width_mult,
    )

    engine = TrainerEngine(cfg, env)
    run = maybe_init_wandb(cfg, args.disable_wandb)

    model = engine.train_classifier(model, train_loader)
    ckpt = engine.save_checkpoint(model, cfg.runtime.output_dir, cfg.runtime.run_name)
    print(f"saved_checkpoint={ckpt}")

    prototypes = engine.build_prototypes(model, enroll_loader)

    calib_scores, calib_known_flags, _, _ = engine.collect_fused_scores(model, calib_loader, prototypes)
    known_mask = calib_known_flags == 1
    calib_threshold = engine.calibrate_threshold_from_known(calib_scores[known_mask], target_frr=0.05)

    test_scores, test_known_flags, test_closed_correct, _ = engine.collect_fused_scores(
        model, test_loader, prototypes
    )
    result = engine.evaluate_open_set(
        scores=test_scores,
        known_flags=test_known_flags,
        closed_correct=test_closed_correct,
        threshold=calib_threshold,
    )

    result_payload = {
        "closed_set_accuracy": result.closed_set_accuracy,
        "open_set_auc": result.open_set_auc,
        "open_set_eer": result.open_set_eer,
        "open_set_overall_accuracy": result.open_set_overall_accuracy,
        "threshold": calib_threshold,
    }
    print(json.dumps(result_payload, indent=2))

    out_metrics = Path(cfg.runtime.output_dir) / f"{cfg.runtime.run_name}_metrics.json"
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")

    if run is not None:
        import wandb

        wandb.log(result_payload)
        wandb.finish()


if __name__ == "__main__":
    main()
