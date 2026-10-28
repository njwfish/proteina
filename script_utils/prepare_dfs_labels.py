"""Select complete TED domain records for D_FS before the native label transform."""

import argparse
import gzip
from pathlib import Path


def select_labels(source, output, identifiers):
    count = 0
    for line in source:
        sample = "_".join(line.split("\t", 1)[0].split("_")[:-1])
        if sample in identifiers:
            output.write(line)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--ted", type=Path, required=True,
                        help="Official ted_365m.domain_summary.cath.globularity.taxid.tsv.gz")
    args = parser.parse_args()
    root = args.data_root / "d_FS"
    ids = (root / "d_FS_index.txt").read_text().splitlines()
    if len(ids) != 588318 or len(set(ids)) != len(ids):
        raise ValueError("Expected all 588,318 unique D_FS identifiers")
    destination = root / "ted_dfs.tsv"
    if destination.exists():
        raise FileExistsError(f"Retain the existing annotation file: {destination}")
    partial = destination.with_suffix(".tsv.part")
    with gzip.open(args.ted, "rt") as source, partial.open("w") as output:
        count = select_labels(source, output, set(ids))
    if not count:
        raise ValueError("No D_FS identifiers matched the official TED annotations")
    partial.replace(destination)
    print(f"Selected {count} complete domain records; native TEDLabelTransform parses their labels")


if __name__ == "__main__":
    main()
