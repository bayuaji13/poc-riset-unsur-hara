"""Append transfer benchmark results to a safe copy of the local workbook."""

import argparse
from pathlib import Path

from npk_spectra.export import export_transfer_workbook


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--workbook", type=Path, default=Path("NPK_Filled_Soil_Data_v2.xlsx"))
parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
parser.add_argument("--output", type=Path)
args = parser.parse_args()
print(export_transfer_workbook(args.workbook, args.artifacts, args.output))
