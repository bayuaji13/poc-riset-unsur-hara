from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from npk_spectra.data import prepare_dataset  # noqa: E402


if __name__ == "__main__":
    dataset = prepare_dataset(PROJECT_ROOT)
    print(f"Dataset siap: {dataset.n_samples} sampel × {dataset.n_features} kanal")

