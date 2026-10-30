"""Prepare the complete D_FS corpus with Proteina's native parser on CPU workers."""

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import tempfile

import hydra
from omegaconf import OmegaConf
import torch


def prepare_chunk(arguments):
    config, identifiers = arguments
    torch.set_num_threads(1)
    data = hydra.utils.instantiate(config)
    destination = data.processed_dir
    completed = 0
    # Publish only fully written native graphs; interrupted writes stay temporary.
    with tempfile.TemporaryDirectory(prefix=".parse-", dir=destination) as staging:
        data.processed_dir = Path(staging)
        for i, identifier in enumerate(identifiers):
            if (destination / f"{identifier}.pt").exists():
                continue
            name = data._load_and_process_pdb((i, identifier))
            if name is None:
                raise RuntimeError(f"Native parser failed for {identifier}")
            (data.processed_dir / name).replace(destination / name)
            completed += 1
            if completed % 2000 == 0:
                print(f"Worker {os.getpid()}: {completed} structures prepared", flush=True)
    return completed


@hydra.main(config_path="../configs/datasets_config/afdb", config_name="d_FS_small60m", version_base="1.3")
def main(config):
    torch.set_num_threads(1)
    data = hydra.utils.instantiate(config.datamodule)
    identifiers = data.validate_raw_data()
    workers = max(1, int(config.datamodule.num_workers))
    specification = OmegaConf.to_container(config.datamodule, resolve=True)
    chunks = [(specification, identifiers[i::workers]) for i in range(workers)]
    print(f"Preparing {len(identifiers)} indexed structures with {workers} CPU workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        completed = sum(pool.map(prepare_chunk, chunks))

    # The native preparation writes its usual CSV and enforces the complete cohort.
    data.prepare_data()
    data.setup("fit")
    batch = next(iter(data.train_dataloader()))
    if batch.coords.shape[0] != data.batch_size or not torch.isfinite(batch.coords).all():
        raise RuntimeError("The native training loader did not produce a finite full batch")
    report = {
        "indexed_structures": len(identifiers), "newly_prepared": completed,
        "splits": {name: len(frame) for name, frame in data.dfs_splits.items()},
        "native_batch_shape": list(batch.coords.shape),
        "fold_labels_per_sample": [len(labels) for labels in batch.cath_code],
    }
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
