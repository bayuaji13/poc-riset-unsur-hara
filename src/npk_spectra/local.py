"""Load the paired Central Java Shimadzu spectra and laboratory workbook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .config import MIR_GRID
from .dataset import SpectralDataset


SPECTRUM_PATTERN = re.compile(r"^2-(?P<code>GB|PT|RB)\s+(?P<number>\d+)\.txt$", re.I)
LAB_CODE_PATTERN = re.compile(r"\b(?P<code>GB|PT|RB)\s*(?P<number>\d+)\b", re.I)
LOCATION_NAMES = {"GB": "Grobogan", "PT": "Pati", "RB": "Rembang"}
TARGET_COLUMNS = {"N": "N (%)", "P": "P-Tersedia (ppm)", "K": "K-Tersedia (me/100g)"}
_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


@dataclass(frozen=True, slots=True)
class LocalLoadReport:
    matched: int
    label_only: int
    spectrum_only: int


def _xlsx_rows(path: Path) -> list[list[str]]:
    """Read the simple, single-sheet supplied workbook without hidden converters."""
    with zipfile.ZipFile(path) as archive:
        strings_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = ["".join(node.text or "" for node in item.iterfind(".//m:t", _NS)) for item in strings_root.findall("m:si", _NS)]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows: list[list[str]] = []
    for row in sheet.findall(".//m:sheetData/m:row", _NS):
        values: list[str] = []
        for cell in row.findall("m:c", _NS):
            value = cell.findtext("m:v", default="", namespaces=_NS)
            values.append(shared[int(value)] if cell.get("t") == "s" and value else value)
        rows.append(values)
    return rows


def _read_shimadzu(path: Path) -> np.ndarray:
    values: list[tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.startswith("##"):
            continue
        fields = line.split()
        if len(fields) >= 2:
            values.append((float(fields[0]), float(fields[1])))
    raw = np.asarray(values, dtype=float)
    if raw.shape[0] < 21 or np.any(raw[:, 1] <= 0):
        raise ValueError(f"Spektrum Shimadzu tidak valid: {path}")
    # Exports are ascending, while np.interp requires an ascending source grid.
    return -np.log10(np.interp(MIR_GRID, raw[:, 0], raw[:, 1]) / 100.0)


def load_local_dataset(workbook: Path | str, spectra_root: Path | str) -> tuple[SpectralDataset, LocalLoadReport]:
    workbook, spectra_root = Path(workbook), Path(spectra_root)
    rows = _xlsx_rows(workbook)
    if not rows:
        raise ValueError("Workbook lokal kosong.")
    header = rows[0]
    records = pd.DataFrame(rows[1:], columns=header)
    indexed: dict[str, pd.Series] = {}
    for _, row in records.iterrows():
        match = LAB_CODE_PATTERN.search(str(row["Komoditas pada lahan"]))
        if match:
            key = f"{match.group('code').upper()}-{int(match.group('number')):02d}"
            if key in indexed:
                raise ValueError(f"Kode laboratorium duplikat: {key}")
            indexed[key] = row

    paired: list[tuple[dict[str, object], np.ndarray]] = []
    seen: set[str] = set()
    spectrum_only = 0
    for path in sorted(spectra_root.glob("*/*.txt")):
        match = SPECTRUM_PATTERN.match(path.name)
        if not match:
            continue
        code, number = match.group("code").upper(), int(match.group("number"))
        sample_id = f"{code}-{number:02d}"
        if sample_id not in indexed:
            spectrum_only += 1
            continue
        row = indexed[sample_id]
        expected = LOCATION_NAMES[code]
        if str(row["Kabupaten"]).strip() != expected:
            raise ValueError(f"Lokasi workbook tidak cocok untuk {sample_id}: {row['Kabupaten']}")
        metadata = {
            "sample_id": sample_id, "group_id": expected, "location_code": code,
            "source_path": str(path), "workbook_row": int(row["No."]),
            **{target: float(row[column]) for target, column in TARGET_COLUMNS.items()},
        }
        paired.append((metadata, _read_shimadzu(path)))
        seen.add(sample_id)
    if not paired:
        raise FileNotFoundError(f"Tidak ada spektrum yang cocok di {spectra_root}")
    paired.sort(key=lambda item: str(item[0]["sample_id"]))
    frame = pd.DataFrame(item[0] for item in paired).reset_index(drop=True)
    matrix = np.vstack([item[1] for item in paired])
    report = LocalLoadReport(len(frame), len(records) - len(seen), spectrum_only)
    return SpectralDataset(frame, matrix, MIR_GRID.copy()), report
