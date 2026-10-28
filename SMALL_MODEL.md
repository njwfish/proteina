# Released 60M baseline and adaptive training

This checkout extends NVIDIA's native Proteina stack at upstream commit
`a44b407daf6a5358e43cd68907f3e3f1cbc65fdc`. It uses the native backbone,
conditioning, optimizer, sampler and evaluation. The old 30M adapter is not used.

## Which small model

The public `ca_af3_60M_notri.yaml` differs from the released
`proteina_v1.3_DFS_60M_notri.ckpt`. The configuration saved inside that
checkpoint defines this baseline:

| Setting | Released checkpoint |
|---|---|
| Trainable backbone parameters | 58,926,644 |
| Token width / layers / attention heads | 512 / 12 / 12 |
| Pair width / registers | 196 / 10 |
| Pair updates / triangle layers | Disabled |
| Self-conditioning / fold-conditioning | Enabled / enabled |
| Time distribution | 98% Beta(1.9, 1), 2% Uniform(0, 1) |
| Parameterization / objective | Native velocity output and flow-matching loss |
| Optimizer | Adam, learning rate 0.0001, gradient norm clip 1 |
| Precision / EMA | BF16 mixed / 0.999 every update |
| Per-device / global batch | 5 / 80 |
| Reached checkpoint step | 1,300,000 |

`training_small60m.yaml` uses the saved architecture, conditioning, loss, EMA
and validation samplers, with a matched stopping point of 1,300,000 updates.
W&B is disabled; native checkpointing stays enabled. The upstream model,
objective and sampler modules have no scientific changes.

The checkpoint used an unreleased FoldComp datamodule named `genie2` in
`afdb_fromraw`. Its only saved datamodule state is an iteration counter. The
public code cannot recover its exact train/validation split or file ordering.
This setup uses the public native CIF parser and random 98%/1.9%/0.1% splitter.
It is a matched public implementation with this explicit data-backend difference.

## Environment and assets

Create the isolated environment from `environment.yaml`, then install this
checkout with `python -m pip install --no-deps -e .`. CUDA 12.1 and the CUDA
PyTorch mutex are explicit because the unconstrained upstream recipe can
select CPU PyTorch alongside CUDA PyG extensions. NumPy and pandas are solved
at their upstream pinned versions before pip installs the remaining packages.
Both Lightning namespaces used by the native code are pinned to 2.5.0.
Use `PYTHONNOUSERSITE=1` to keep unrelated user-site packages out of this runtime.
Biotite stays at 0.41.0 but uses its PyPI wheel: the current conda build requires
NumPy 1.26.4, which conflicts with the upstream NumPy 1.23.5 pin.
The three PyTorch packages are qualified to the official `pytorch` channel;
Conda-forge's alternate 2.4.1.post300 build uses a different CUDA/C++ ABI and cannot
load the native PyG extensions. PyG 2.6.1 also selects its matching CUDA 12.1 build.

Set `DATA_PATH` to a remote data directory. Acquire the official D_FS index
and prepare the released checkpoint and evaluation assets:

```bash
python -m script_utils.prepare_small --data-root "$DATA_PATH" --index /path/to/d_FS_index.txt
python -m script_utils.download_dfs --data-root "$DATA_PATH" --gcloud /path/to/gcloud
python -m script_utils.prepare_dfs_labels --data-root "$DATA_PATH" --ted /path/to/ted_365m.domain_summary.cath.globularity.taxid.tsv.gz
```

The D_FS index contains 588,318 exact AFDB-v4 identifiers. The downloader uses
the authenticated original-v4 GCS bucket and CIF files, which the native parser
supports. It does not substitute current AFDB versions. The full raw corpus
and successfully processed native structures are required by the training
datamodule; incomplete downloads cannot silently become the training set.

Obtain annotations from the [TED release](https://zenodo.org/records/13908086).
The filter retains every original domain record for indexed proteins. Native
`TEDLabelTransform` parses CATH labels and treats unannotated proteins as usual.
The supplied inference label-distribution asset is not a per-protein annotation
table and cannot replace these annotations.

The preparer also caches native CA ProteinMPNN v_48_020 weights and Hugging Face
ESMFold. ProteinMPNN gets an ignored symlink at its native source location;
the weights and model cache stay under `DATA_PATH`. Before evaluation, export
`CACHE_DIR="$DATA_PATH/huggingface"` and `HF_HUB_CACHE="$CACHE_DIR"`. Set
`PYTHON_EXEC` to the active environment's Python so the native ProteinMPNN
subprocess uses the same runtime.

## Execution checks and commands

The execution check strictly loads all released weights, compares scientific
configuration with the checkpoint, parses a real indexed CIF through the
public loader, differentiates the native loss, checks the full 5-by-256 padded
backbone with self-conditioning and BF16, then generates two 100-residue
structures with the native sampler. It never updates the released weights.

```bash
python -m script_utils.check_small \
  --checkpoint "$DATA_PATH/checkpoints/proteina_v1.3_DFS_60M_notri.ckpt" \
  --cif "$DATA_PATH/d_FS/raw/AF-A0A1M3JS45-F1-model_v4.cif" \
  --output /path/to/new-check-directory
```

Default generation uses the native 400-step logarithmic grid, power 2,
stochastic noise scale 0.45 and self-conditioning. `--steps` and `--device cpu`
are diagnostic overrides, not the baseline quality protocol. The check writes
tensors and `results.json`; passing establishes execution compatibility, not
designability or improved generative quality.

After full data preparation, use the native entry points:

```bash
python proteinfoundation/train.py --config_name training_small60m
python proteinfoundation/inference.py --config_name inference_small60m
python proteinfoundation/inference.py --config_name inference_small60m_fid
```

Launch training with 16 GPU processes for global batch 80. The saved hardware
layout is two nodes of eight GPUs; four nodes of four GPUs preserve the global
batch using `hardware.ngpus_per_node_=4 hardware.nnodes_=4`. Use the cluster's
native Lightning/Slurm launch convention. `--single` is only for debugging and
changes the global batch. Hydra overrides pass through both entry points.
Inference refuses to erase an existing output directory; move retained outputs
before reusing a config name.

The designability config retains upstream lengths 50/100/150/200/250 and
100 samples per length, with eight ProteinMPNN sequences per structure. The
FPSD/fold-distribution config retains 5,000 structures, lengths 60 through 255,
and the supplied native GearNet reference features.

## Adaptive integration

Keep the same transformer, time embedding, units, masks, rotation augmentation,
self-conditioning, fold-conditioning, Adam, EMA and evaluation. Adaptivity
belongs at three connected seams:

1. Replace Gaussian interpolation in `ModelTrainerBase.training_step` with draws
   from a specified displacement bridge.
2. Learn that bridge's conditional mean and local covariance targets. For an
   infinitesimal model, add its moment output to the existing transformer trunk.
3. Step with the compatible adaptive generator. The Gaussian score conversion
   in `R3NFlowMatcher` cannot be reused for another law by changing noise scale.

Native time is noise-at-zero to clean-at-one. Our bridge time uses the opposite
endpoint convention, so the adapter must use `tau = 1 - t` explicitly while
retaining the native model's physical time input and sampled time distribution.

Atom displacement norms are a natural candidate because common rotations
preserve them. However, substituting a 3D norm into the scalar Bessel sampler
does not give a centered atom-level vector bridge. Centering couples atoms,
and the radial generator changes with dimension. That law and its covariance
target remain to be derived and checked. Scalar Helmert-chart bridges are an
available historical control, but adapt residue contrasts rather than atoms.

No adaptive training config is enabled until draws, targets and sampling steps
agree. Protein coordinates do not use the cell-specific zero/BCE head. The
focused [geometry note](../../research/proteina-adaptive-geometry.md) owns the
mathematical alternatives and their limitations.
