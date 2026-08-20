from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from npk_spectra.benchmark import run_benchmark  # noqa: E402


if __name__ == "__main__":
    _, _, summary = run_benchmark(PROJECT_ROOT)
    print(summary.to_string(index=False))

