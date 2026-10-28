"""The public Proteina loader restricted to the complete released D_FS index."""

import csv

from proteinfoundation.datasets.pdb_data import PDBLightningDataModule


class DFSDataModule(PDBLightningDataModule):
    def prepare_data(self):
        index = (self.data_dir / "d_FS_index.txt").read_text().splitlines()
        expected = set(index)
        if len(index) != 588318 or len(expected) != len(index):
            raise ValueError("Expected the complete 588,318-entry D_FS index")
        observed = {p.stem for p in self.raw_dir.glob("*." + self.format)}
        missing, extra = expected - observed, observed - expected
        if missing or extra:
            raise ValueError(
                f"D_FS raw corpus is incomplete: {len(missing)} missing, "
                f"{len(extra)} unexpected structures"
            )
        super().prepare_data()
        with (self.data_dir / f"{self.data_dir.name}.csv").open() as source:
            rows = list(csv.DictReader(source))
        if len(rows) != len(expected) or {row["id"] for row in rows} != expected:
            raise ValueError("The cached native dataset CSV does not match the D_FS index")
        processed = {p.stem for p in self.processed_dir.glob("*.pt")}
        missing = expected - processed
        if missing:
            raise ValueError(f"Native preprocessing failed for {len(missing)} D_FS structures")
