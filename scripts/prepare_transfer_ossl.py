"""Build the larger OSSL source pool used by the local-transfer experiment."""

import argparse
from pathlib import Path

from npk_spectra.data import prepare_transfer_source


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--project-root", type=Path, default=Path.cwd())
parser.add_argument("--pool-size", type=int, default=10000)
parser.add_argument("--candidates", type=int, default=25000)
args = parser.parse_args()
dataset = prepare_transfer_source(args.project_root.resolve(), pool_size=args.pool_size, candidate_count=args.candidates)
print(f"OSSL transfer pool siap: {dataset.n_samples} sampel × {dataset.n_features} kanal")
