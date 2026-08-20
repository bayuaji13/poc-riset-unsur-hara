"""Scientific definitions used by the proof-of-concept."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


OSSL_VERSION = "v1.2"
OSSL_LAB_URL = "https://storage.googleapis.com/soilspec4gg-public/ossl_soillab_L1_v1.2.csv.gz"
OSSL_MIR_URL = "https://storage.googleapis.com/soilspec4gg-public/ossl_mir_L0_v1.2.csv.gz"
JOIN_COLUMNS = ("dataset.code_ascii_txt", "id.layer_uuid_txt")
MIR_GRID = np.arange(600.0, 4000.0 + 1.0, 2.0)
MIR_COLUMNS = tuple(f"scan_mir.{int(value)}_abs" for value in MIR_GRID)
BUDGETS = (60, 120, 180, 240, 300, 1000)
MAX_BUDGET = max(BUDGETS)
RANDOM_STATE = 42


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """One laboratory property, including method and unit semantics."""

    code: str
    column: str
    label: str
    method: str
    unit: str
    transform: str = "log1p"
    color: str = "#1E40AF"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


DEFAULT_TARGETS = (
    TargetSpec(
        code="N",
        column="n.tot_usda.a623_w.pct",
        label="Nitrogen total",
        method="USDA a623 (NCS/dry combustion)",
        unit="% berat",
        color="#168657",
    ),
    TargetSpec(
        code="P",
        column="p.ext_usda.a274_mg.kg",
        label="Fosfor tersedia",
        method="USDA a274 (Olsen NaHCO3)",
        unit="mg/kg",
        color="#C06C00",
    ),
    TargetSpec(
        code="K",
        column="k.ext_usda.a725_cmolc.kg",
        label="Kalium dapat ditukar",
        method="USDA a725 (NH4OAc pH 7)",
        unit="cmolc/kg",
        color="#6D48C7",
    ),
)


def target_by_code(code: str) -> TargetSpec:
    for target in DEFAULT_TARGETS:
        if target.code == code.upper():
            return target
    raise KeyError(f"Target tidak dikenal: {code}")
