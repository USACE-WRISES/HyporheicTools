# tests/unit/test_inputs.py
#Lightweight unit test to confirm path resolution is relative to the YAML’s folder 
# and workspace creation doesn’t throw. Skips if your example data isn’t present.
from __future__ import annotations

from pathlib import Path
import pytest

from HyporheicTools.inputs import load, Settings


def _example_yaml() -> Path:
    repo = Path(__file__).resolve().parents[2]
    return repo / "examples" / "basic_case" / "input" / "inputs.yaml"


@pytest.mark.skipif(not _example_yaml().exists(), reason="example YAML not found")
def test_paths_resolve_relative_to_yaml():
    yaml_path = _example_yaml()
    cfg = load(yaml_path)
    # Key paths should be absolute and exist (if data present)
    assert cfg.ground_water_domain_shapefile.is_absolute()
    assert cfg.left_boundary_floodplain.is_absolute()
    assert cfg.right_boundary_floodplain.is_absolute()
    assert cfg.projection_file.is_absolute()
    assert cfg.output_directory.is_absolute()

    # Workspace should be creatable
    cfg.setup_workspace(clean=False)
    assert Path(cfg.gwf_ws).exists()
    assert Path(cfg.mp7_ws).exists()
