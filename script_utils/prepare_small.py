"""Download the released small checkpoint and the upstream evaluation assets."""

import argparse
from pathlib import Path
import shutil
import urllib.request
import zipfile


def download(resource, name, size, destination):
    if destination.exists() and destination.stat().st_size == size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://api.ngc.nvidia.com/v2/resources/nvidia/clara/"
        f"{resource}/versions/1.0/files/{name}"
    )
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {name}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as source, partial.open("wb") as output:
        shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
    if partial.stat().st_size != size:
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
    checkpoint = "proteina_v1.3_DFS_60M_notri.ckpt"
    download("proteina_v1.3_dfs_60m_notri", checkpoint, 707719009,
             root / "checkpoints" / checkpoint)
    archive = root / "proteina_additional_files.zip"
    download("proteina_additional_files", archive.name, 208879609, archive)
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            parts = Path(member.filename).parts
            if (member.is_dir() or parts[0] != "proteina_additional_files"
                    or parts[-1] == ".DS_Store"):
                continue
            destination = root.joinpath(*parts[1:]).resolve()
            if not destination.is_relative_to(root):
                raise ValueError("Archive member escapes the data directory")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as data, destination.open("wb") as output:
                shutil.copyfileobj(data, output)
    (root / "d_FS").mkdir(exist_ok=True)
    index = root / "d_FS/d_FS_index.txt"
    if args.index.resolve() != index:
        shutil.copyfile(args.index, index)
    print(f"Released checkpoint, evaluation assets and index ready at {root}")


if __name__ == "__main__":
    main()
