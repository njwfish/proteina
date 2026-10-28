"""Acquire the exact indexed AFDB-v4 CIF files using an authenticated gcloud."""

import argparse
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--gcloud", default="gcloud")
    args = parser.parse_args()
    root = args.data_root / "d_FS"
    ids = (root / "d_FS_index.txt").read_text().splitlines()
    if len(ids) != 588318 or len(set(ids)) != len(ids):
        raise ValueError("Expected all 588,318 unique D_FS identifiers")
    raw = root / "raw"
    raw.mkdir(exist_ok=True)
    present = {p.stem for p in raw.glob("*.cif")}
    missing = [name for name in ids if name not in present]
    print(f"D_FS: {len(ids) - len(missing)} present, {len(missing)} to acquire", flush=True)
    if not missing:
        return
    with tempfile.TemporaryFile(mode="w+") as urls:
        for name in missing:
            urls.write(f"gs://public-datasets-deepmind-alphafold-v4/{name}.cif\n")
        urls.seek(0)
        subprocess.run(
            [args.gcloud, "storage", "cp", "--quiet", "--no-user-output-enabled",
             "--no-clobber", "--read-paths-from-stdin", str(raw)],
            stdin=urls, check=True,
        )


if __name__ == "__main__":
    main()
