"""Independent metric audit must reject corruption, including exact-zero cases."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_winograd import audit_fields, scalar_metrics


def test_metrics_expose_finite_and_identical_cases_without_infinite_psnr():
    reference = np.asarray([0., 1.], dtype=np.float32)
    exact = scalar_metrics(reference, reference)
    assert exact["identical"] and exact["psnr_unit_range"] is None
    actual = scalar_metrics(np.asarray([0., 0.]), reference)
    assert actual["rmse"] == pytest.approx(np.sqrt(.5))
    assert actual["relative_mse"] == 1
    assert actual["psnr_unit_range"] == pytest.approx(10 * np.log10(2))
    audit_fields(actual, actual)
    with pytest.raises(ValueError, match="rmse"):
        audit_fields(actual, {**actual, "rmse": actual["rmse"] + 1e-3})
    with pytest.raises(ValueError, match="identical"):
        audit_fields(exact, {**exact, "identical": False})
