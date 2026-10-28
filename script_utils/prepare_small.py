"""Download the released small checkpoint and the upstream evaluation assets."""

import argparse
from pathlib import Path
import shutil
import urllib.request
import zipfile


def download(url, destination, size=None):
    if destination.exists() and (size is None or destination.stat().st_size == size):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {destination.name}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as source, partial.open("wb") as output:
        shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
    if size is not None and partial.stat().st_size != size:
        raise IOError(f"Incomplete download: {partial}")
    partial.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True,
                        help="The official d_FS_index.txt already acquired from NVIDIA")
    args = parser.parse_args()
    ids = args.index.read_text().splitlines()
    if len(ids) != 588318 or len(set(ids)) != len(ids):
        raise ValueError("Expected all 588,318 unique D_FS identifiers")
    root = args.data_root.resolve()
    ngc = "https://api.ngc.nvidia.com/v2/resources/nvidia/clara"
    checkpoint = "proteina_v1.3_DFS_60M_notri.ckpt"
    download(f"{ngc}/proteina_v1.3_dfs_60m_notri/versions/1.0/files/{checkpoint}",
             root / "checkpoints" / checkpoint, 707719009)
    archive = root / "proteina_additional_files.zip"
    download(f"{ngc}/proteina_additional_files/versions/1.0/files/{archive.name}",
             archive, 208879609)
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            parts = Path(member.filename).parts
            if (member.is_dir() or parts[0] != "proteina_additional_files"
                    or parts[-1] == ".DS_Store"):
                continue
            destination = root.joinpath(*parts[1:]).resolve()
            if not destination.is_relative_to(root):
                raise ValueError("Archive member escapes the data directory")
            if destination.exists() and destination.stat().st_size == member.file_size:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_suffix(destination.suffix + ".part")
            with source.open(member) as data, partial.open("wb") as output:
                shutil.copyfileobj(data, output)
            partial.replace(destination)
    (root / "d_FS").mkdir(exist_ok=True)
    index = root / "d_FS/d_FS_index.txt"
    if args.index.resolve() != index:
        shutil.copyfile(args.index, index)

    weights = root / "proteinmpnn/ca_model_weights/v_48_020.pt"
    download("https://github.com/dauparas/ProteinMPNN/raw/"
             "8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/ca_model_weights/v_48_020.pt",
             weights)
    native_weights = Path(__file__).resolve().parents[1] / "ProteinMPNN/ca_model_weights/v_48_020.pt"
    native_weights.parent.mkdir(parents=True, exist_ok=True)
    if native_weights.exists() or native_weights.is_symlink():
        if native_weights.resolve() != weights:
            raise FileExistsError(f"Preserve the existing ProteinMPNN weights: {native_weights}")
    else:
        native_weights.symlink_to(weights)

    from huggingface_hub import snapshot_download

    snapshot_download(
        "facebook/esmfold_v1", cache_dir=str(root / "huggingface"),
        allow_patterns=["*.json", "vocab.txt", "pytorch_model.bin"],
    )
    print(f"Released checkpoint, native evaluation assets and index ready at {root}")


if __name__ == "__main__":
    main()
