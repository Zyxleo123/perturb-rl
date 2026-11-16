#!/usr/bin/env python
"""Retrain GEARS under a split that actually holds perturbations out.

Needed because the only published checkpoint uses split='no_test'. See the
header of jobs/22_retrain_oracle.sbatch for the reasoning.

Saves <out-dir>/{model.pt,config.pkl} in the layout GEARS.load_pretrained wants,
so scripts/oracle_probe.py can read it unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import traceback


def log(*a):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[{time.strftime('%H:%M:%S')} rss={rss:.1f}GB]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gears-dir", required=True)
    ap.add_argument("--dataset", default="norman")
    ap.add_argument("--split", default="simulation")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--test-batch-size", type=int, default=128)
    ap.add_argument("--hidden-size", type=int, default=64)
    ap.add_argument("--train-gene-set-size", type=float, default=0.75)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.artifacts, exist_ok=True)
    status = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "epochs": args.epochs,
        "out_dir": args.out_dir,
        "status": "started",
    }
    status_path = os.path.join(
        args.artifacts, f"retrain_status_{args.dataset}_{args.split}_seed{args.seed}.json"
    )

    def dump():
        with open(status_path, "w") as f:
            json.dump(status, f, indent=2)

    dump()
    try:
        import torch
        from gears import GEARS, PertData

        log("torch", torch.__version__, "cuda:", torch.cuda.is_available())
        if not torch.cuda.is_available():
            args.device = "cpu"
            log("WARNING: no CUDA; CPU training will be very slow")

        # PertData writes into its root, so point it at the data dir that already
        # holds the extracted dataset; it reuses what is there.
        pert_data = PertData(args.gears_dir)
        pert_data.load(data_path=os.path.join(args.gears_dir, args.dataset))
        log("adata:", pert_data.adata.shape)

        log(f"prepare_split(split={args.split}, seed={args.seed}, "
            f"train_gene_set_size={args.train_gene_set_size})")
        pert_data.prepare_split(
            split=args.split, seed=args.seed, train_gene_set_size=args.train_gene_set_size
        )
        s2c = pert_data.set2conditions or {}
        status["split_sizes"] = {k: len(v) for k, v in s2c.items()}
        log("split sizes:", status["split_sizes"])

        log("get_dataloader (builds cell graphs; this is the memory-hungry step)")
        pert_data.get_dataloader(
            batch_size=args.batch_size, test_batch_size=args.test_batch_size
        )

        model = GEARS(pert_data, device=args.device, weight_bias_track=False,
                      proj_name="gears", exp_name=f"{args.dataset}_{args.split}_{args.seed}")
        model.model_initialize(hidden_size=args.hidden_size)
        log(f"training {args.epochs} epochs")
        t0 = time.time()
        model.train(epochs=args.epochs)
        status["train_seconds"] = round(time.time() - t0, 1)
        log(f"training done in {status['train_seconds']}s")

        model.save_model(args.out_dir)
        log("saved to", args.out_dir, "->", sorted(os.listdir(args.out_dir)))

        try:
            res = model.evaluate(model.dataloader["test_loader"])
            status["gears_internal_eval"] = {
                k: (float(v) if hasattr(v, "__float__") else str(v))
                for k, v in (res[1] if isinstance(res, tuple) else res).items()
            }
            log("GEARS internal eval:", status["gears_internal_eval"])
        except Exception as e:
            log("internal eval skipped:", type(e).__name__, e)
            status["gears_internal_eval"] = f"skipped: {type(e).__name__}: {e}"

        status["status"] = "ok"
    except Exception as e:
        status["status"] = f"error: {type(e).__name__}: {e}"
        status["traceback"] = traceback.format_exc()
        log("ERROR:", status["status"])
        traceback.print_exc()
        dump()
        return 1

    dump()
    log("wrote", status_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
