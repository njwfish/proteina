"""Check the released checkpoint, native CIF loader, gradients and generation.

This is an execution check, not a training run or a protein-quality assay.
Run from the repository root with DATA_PATH set to the prepared asset root.
"""

import argparse
import json
import os
from pathlib import Path
import time

import hydra
import lightning as L
from omegaconf import OmegaConf
import torch

from proteinfoundation.datasets.pdb_data import PDBDataset, PDBLightningDataModule
from proteinfoundation.proteinflow.proteina import Proteina
from proteinfoundation.utils.dense_padding_data_loader import DensePaddingDataLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cif", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    L.seed_everything(42)
    if torch.__version__.split("+")[0] != "2.4.1" or torch.version.cuda != "12.1":
        raise RuntimeError("This check requires native PyTorch 2.4.1 with CUDA 12.1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.steps < 1:
        raise ValueError("Sampling steps must be positive")
    config_dir = Path(__file__).resolve().parents[1] / "configs/experiment_config"
    with hydra.initialize_config_dir(str(config_dir), version_base="1.3"):
        cfg = hydra.compose(config_name="training_small60m")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = checkpoint["hyper_parameters"]["cfg_exp"]
    for key in ("model", "loss", "training", "ema"):
        actual = OmegaConf.to_container(OmegaConf.select(cfg, key), resolve=True)
        expected = OmegaConf.to_container(OmegaConf.select(saved, key), resolve=True)
        if actual != expected:
            raise ValueError(f"Config differs from the released checkpoint: {key}")
    for key in ("opt.lr", "seed", "force_precision_f32", "opt.accumulate_grad_batches"):
        if OmegaConf.select(cfg, key) != OmegaConf.select(saved, key):
            raise ValueError(f"Config differs from the released checkpoint: {key}")
    model = Proteina(cfg, store_dir=str(args.output))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    report = {
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "device": args.device, "parameters": model.nparams,
        "checkpoint_step": checkpoint["global_step"], "strict_load": True,
    }
    del checkpoint
    model.to(args.device)
    print(json.dumps(report), flush=True)

    # Use the public parser and its usual atom ordering and padded collator.
    data_dir = args.output / "native_data"
    raw = data_dir / "raw"
    raw.mkdir(parents=True)
    (raw / args.cif.name).symlink_to(args.cif.resolve())
    dm = PDBLightningDataModule(data_dir=str(data_dir), format="cif", num_workers=0)
    dm.prepare_data()
    if not (data_dir / "processed" / f"{args.cif.stem}.pt").exists():
        raise RuntimeError("The native CIF parser did not produce a structure")
    ds = PDBDataset(pdb_codes=[args.cif.stem], data_dir=str(data_dir),
                    file_names=[args.cif.stem], format="cif")
    batch = next(iter(DensePaddingDataLoader(ds, batch_size=1))).to(args.device)
    batch.cath_code = [[]]  # Unannotated smoke sample; native null conditioning.
    model.train()
    loss = model.training_step(batch, -1)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not torch.isfinite(loss) or not grads or not all(torch.isfinite(g).all() for g in grads):
        raise RuntimeError("Native real-structure loss or gradients are not finite")
    report.update(real_cif=args.cif.name, residues=int(batch["mask"].sum()),
                  native_loss=float(loss.detach()), finite_gradients=True)
    model.zero_grad(set_to_none=True)

    # Exercise the released per-device batch and maximum D_FS length with padding.
    mask = torch.arange(256, device=args.device)[None, :] < torch.tensor(
        [256, 240, 128, 96, 64], device=args.device)[:, None]
    x_clean = model.fm._mask_and_zero_com(torch.randn(5, 256, 3, device=args.device), mask)
    t = model.sample_t((5,))
    x_ref = model.fm.sample_reference(256, shape=(5,), device=args.device, mask=mask)
    x_t = model.fm.interpolate(x_ref, x_clean, t, mask)
    inputs = {"x_t": x_t, "t": t, "mask": mask}
    start = time.monotonic()
    with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == "cuda"):
        with torch.no_grad():
            inputs["x_sc"] = model.predict_clean(inputs)[0]
        prediction, _ = model.predict_clean(inputs)
        full_loss = model.compute_fm_loss(x_clean, prediction, x_t, t, mask, log_prefix=None).mean()
    full_loss.backward()
    if args.device == "cuda":
        torch.cuda.synchronize()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not torch.isfinite(full_loss) or not grads or not all(torch.isfinite(g).all() for g in grads):
        raise RuntimeError("Full-width padded loss or gradients are not finite")
    if torch.count_nonzero(prediction[~mask]):
        raise RuntimeError("Predicted padding is nonzero")
    report.update(full_width_batch=5, full_width_length=256,
                  full_width_loss=float(full_loss.detach()),
                  full_width_seconds=time.monotonic() - start)
    model.zero_grad(set_to_none=True)

    model.eval()
    with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == "cuda"):
        samples = model.generate(
            nsamples=2, n=100, dt=1 / args.steps, self_cond=True, cath_code=None,
            schedule_mode="log", schedule_p=2.0, sampling_mode="sc",
            sc_scale_noise=0.45, sc_scale_score=1.0, gt_mode="1/t",
            gt_p=1.0, gt_clamp_val=None,
        )
    if not torch.isfinite(samples).all():
        raise RuntimeError("Native generated coordinates are not finite")
    torch.save(samples.cpu(), args.output / "samples.pt")
    report.update(sampling_steps=args.steps, generated_shape=list(samples.shape),
                  generated_com_max=float(samples.mean(dim=-2).abs().max()))
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
