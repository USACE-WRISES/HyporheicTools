# tests/integration/test_basic_case.py
#An integration‑style test stub that does not run the model by default (it uses dry_run=True). 
# Enable full runs manually when you want to test end‑to‑end.
from __future__ import annotations

from pathlib import Path
import os
import pytest

from HyporheicTools.core.run_from_yaml import run_from_yaml


def _example_yaml() -> Path:
    repo = Path(__file__).resolve().parents[2]
    return repo / "examples" / "basic_case" / "input" / "inputs.yaml"


@pytest.mark.skipif(not _example_yaml().exists(), reason="example data not present")
def test_dry_run_loads_config_only():
    outputs = run_from_yaml(_example_yaml(), dry_run=True)
    assert isinstance(outputs, dict)
    assert outputs == {}  # dry_run returns no outputs


@pytest.mark.slow
@pytest.mark.skipif(os.getenv("RUN_SLOW", "0") != "1", reason="Set RUN_SLOW=1 to execute full model run")
@pytest.mark.skipif(not _example_yaml().exists(), reason="example data not present")
def test_full_run_optional():
    # WARNING: runs MF6/MP7 and may take several minutes with real data.
    outputs = run_from_yaml(_example_yaml(), dry_run=False, make_figures=False)
    # If the run succeeded, you should at least have pathlines/points shapefiles
    assert outputs.get("pathlines_fc") or outputs.get("points_fc")
