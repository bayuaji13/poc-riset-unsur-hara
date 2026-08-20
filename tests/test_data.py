import numpy as np
import pandas as pd

from npk_spectra.config import BUDGETS, MAX_BUDGET
from npk_spectra.data import diversity_order, extend_diversity_order, stable_priority


def test_stable_priority_is_reproducible():
    assert stable_priority("KSSL", "abc") == stable_priority("KSSL", "abc")
    assert stable_priority("KSSL", "abc") != stable_priority("KSSL", "def")


def test_learning_curve_includes_1000_samples():
    assert BUDGETS == (60, 120, 180, 240, 300, 1000)
    assert MAX_BUDGET == 1000


def test_diversity_order_is_unique_and_exact():
    rng = np.random.default_rng(4)
    metadata = pd.DataFrame({"N": rng.random(20), "P": rng.random(20), "K": rng.random(20)})
    spectra = rng.normal(size=(20, 31))
    order = diversity_order(metadata, spectra, max_samples=12)
    assert len(order) == 12
    assert len(set(order.tolist())) == 12


def test_diversity_extension_preserves_initial_order():
    rng = np.random.default_rng(9)
    metadata = pd.DataFrame({"N": rng.random(20), "P": rng.random(20), "K": rng.random(20)})
    spectra = rng.normal(size=(20, 31))
    initial = np.asarray([7, 2, 14])
    order = extend_diversity_order(metadata, spectra, initial, max_samples=10)
    assert order[:3].tolist() == initial.tolist()
    assert len(set(order.tolist())) == 10
