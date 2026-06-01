import numpy as np
import pytest

from forestry_ts.model import wape


def test_wape_perfect_forecast() -> None:
    actual = np.array([1.0, 2.0, 3.0])
    assert wape(actual, actual) == 0.0


def test_wape_nonzero_error() -> None:
    actual = np.array([10.0, 10.0])
    forecast = np.array([8.0, 12.0])
    assert wape(actual, forecast) == pytest.approx(0.2)
