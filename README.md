# HyporheicTools

YAML-driven workflow to build and run MODFLOW 6 (MF6) and MODPATH 7 (MP7) models for hyporheic analysis from geospatial inputs. The CLI consumes a single `inputs.yaml`, prepares rasters/vectors, constructs a grid and `idomain`, assigns boundary conditions, runs MF6/MP7 via FloPy, and exports pathlines, summaries, and publication-ready plots.

## Features

- YAML configuration for all inputs, parameters, and run flags
- Clean CLI runner with table-form config preview
- Reusable, side-effect-free helper modules for rasters, grids, and boundaries
- End-to-end orchestration: preprocessing → MF6/MP7 runs → post-processing/plots
- Example dataset and structured outputs for quick validation
- Uses local MF6/MP7 binaries (or auto-downloads via FloPy)

## Project Structure

```
HypFlopyArc/
├─ README.md
├─ bin/
│  └─ modflow/
│     ├─ mf6.exe, mp7.exe, libmf6.dll, gridgen.exe, …
├─ examples/
│  └─ basic_case/
│     ├─ input/
│     │  ├─ BlendedTerrain.LoweredTerrain.tif
│     │  ├─ HighResOrtho_23Oct2014_Clip.tif[.ovr/.aux.xml/.tfw]
│     │  ├─ GWDomain.shp/.shx/.dbf/.prj
│     │  ├─ L_FPL.shp, R_FPL.shp
│     │  └─ 102739_TX_central.prj
│     └─ output/
│        ├─ model/            # MF6/MP7 workspaces, arrays, exports
│        └─ summary/          # figures, CSVs, shapefiles
└─ src/
   └─ HyporheicTools/
      ├─ __init__.py
      ├─ inputs.yaml           # user-editable configuration
      ├─ inputs.py             # Pydantic Settings + loaders/utilities
      ├─ cli/
      │  └─ main.py            # CLI entrypoint; orchestrates full workflow
      └─ functions/
         ├─ __init__.py        # lazy submodule loader
         ├─ path_utils.py      # PATH/binaries, project root, config printer
         ├─ raster_utils.py    # raster/grid helpers (pure functions)
         ├─ model_utils.py     # domain/grid/boundary helpers (pure functions)
         ├─ my_utils.py        # business logic for steps 1–7/8
         └─ common_imports_for_main.py
```

## Quick Start

## Load the Environment

If you are using ArcGIS Pro's Python environment, follow these steps to activate it:

1. Ensure the ArcGIS Pro Conda environment is accessible (one-time needed only):
  ```
  conda config --append envs_dirs C:\Users\gtmen\AppData\Local\ESRI\conda\envs
  ```

2. Activate the ArcGIS Pro environment (replace `arcgispro-py3-GTM` with the actual environment name if different):
  ```
  conda activate arcgispro-py3-GTM
  ```

3. Verify the environment is active and ready:
  ```
  python --version
  ```
  Ensure the Python version matches the one configured for ArcGIS Pro.

---

4) Create an environment (Conda recommended for the Geo stack)
```
conda create -n hyporheic python=3.11 -y
conda activate hyporheic
# Geo stack (use conda-forge for prebuilt wheels)
conda install -c conda-forge numpy pandas matplotlib scipy shapely rasterio geopandas pyproj -y
# FloPy and extras
pip install flopy tabulate seaborn modflow-devtools contextily
```

5) Ensure MF6/MP7 binaries are available
- Prefer: use the provided `bin/modflow` folder (already in repo)
- Or allow the CLI to download via FloPy into `bin/modflow`

6) Edit `src/HyporheicTools/inputs.yaml`
- Point to your rasters/shapefiles, projection `.prj`, and output directory
- Tweak grid, hydraulics, and run flags

7) Run the model (from repo root)
```
python -m HyporheicTools.cli.main --yaml src/HyporheicTools/inputs.yaml
```
Run main.py
Run the unit tests

## 5) Testing and Debugging

You have two options for testing and debugging:

### Option 1: Run the CLI
- Execute the CLI directly to test the full workflow:
  ```
  python -m HyporheicTools.cli.main --yaml src/HyporheicTools/inputs.yaml
  ```
- This runs the model using the `inputs.yaml` configuration file and outputs results to the specified directory.

### Option 2: Run Unit Tests
- Use the "Testing" section in VS Code to run unit tests for the `basic_case` example.
- The `launch.json` and `settings.json` are preconfigured to:
  - Run `pytest` for unit tests.
  - Execute `main.py` in the CLI, passing `inputs.yaml` from the `basic_case` folder as `stdin`.

Both options help ensure the tool is functioning as expected.

## CLI Usage

```
python -m HyporheicTools.cli.main [--yaml PATH] [--yaml-stdin]
```

- `--yaml PATH`: path to a YAML file (default: `src/HyporheicTools/inputs.yaml`)
- `--yaml-stdin`: read YAML content from STDIN (Windows PowerShell example):
  ```
  Get-Content .\my_config.yaml -Raw | python -m HyporheicTools.cli.main --yaml-stdin
  ```

Environment overrides (optional): `WRITE`, `RUN`, `PLOT`, `PLOT_SHOW`, `PLOT_SAVE` can be set to truthy values (`1,true,yes,on`) to override the YAML flags.

## Configuration (inputs.yaml)

Key sections (example snippet):
```yaml
# Spatial inputs
water_surface_elevation_raster: ./examples/basic_case/input/WSE.tif
terrain_elevation_raster:       ./examples/basic_case/input/BlendedTerrain.LoweredTerrain.tif
ground_water_domain_shapefile:  ./examples/basic_case/input/GWDomain.shp
left_boundary_floodplain:       ./examples/basic_case/input/L_FPL.shp
right_boundary_floodplain:      ./examples/basic_case/input/R_FPL.shp
projection_file:                ./examples/basic_case/input/102739_TX_central.prj
output_directory:               ./examples/basic_case/output
aerial_raster:                  ./examples/basic_case/input/HighResOrtho_23Oct2014_Clip.tif

# Executables
md6_exe_path: ./bin/modflow/mf6.exe
md7_exe_path: ./bin/modflow/mp7.exe

# Simulation metadata
sim_name: Hyporheic
length_units: feet   # feet | meters
time_units: days     # days  | seconds

# Grid / domain
cell_size_x: 10.0
cell_size_y: 10.0
gw_mod_depth: 20.0
z: 0.5                # layer thickness

# Hydraulics
kh: 10.0
kv: 1.0
gw_offset: 0.1        # WSE → GW table offset
porosity: 0.1

# Stress period / time stepping
nstp: 1
perlen: 1.0
tsmult: 1.0

# Run Flags
write: true
run: true
plot: true
plot_show: true
plot_save: true
```

Notes
- Relative paths are resolved relative to the YAML file’s folder.
- On first run, the CLI prints a formatted configuration table for verification.

## What It Does

- Preprocess
  - Load project CRS from `.prj`
  - Reproject terrain raster to target CRS, crop water-surface raster
  - Load/reproject groundwater domain and boundary vectors
- Grid/Domain
  - Build grid extents from raster transform and cell size
  - Compute `top`, `tops`, `botm`, `nlay`, `idomain`
  - Generate grid polygons and classify boundary cells (left/right/up/down)
- Boundary/Heads
  - Sample WSE at grid points; compute boundary groundwater elevations; interpolate along boundaries
  - Assemble CHD from river cells + boundaries (with dedupe/priority rules)
- MF6/MP7
  - Build and (optionally) run MF6 model and MP7 pathlines via FloPy
- Post-Processing
  - Export shapefiles/CSVs/GeoTIFFs
  - Produce summary figures (Forward direction) in `output/summary`

## Outputs

- `{output_directory}/model/`
  - `gwf_workspace/`, `mp7_workspace/`, `arrays/*.bin`, exports
- `{output_directory}/summary/`
  - Figures: e.g., `Forward_*` PNGs (head overlays, pathlines, distributions)
  - Tables: e.g., `Forward_points_table.csv`, pathline stats
  - Shapefiles: start/end points, pathlines, debug/QA layers

The included `examples/basic_case/output` demonstrates the expected structure.

## Modules Overview

- `cli/main.py`: Top-level runner. Sets paths, ensures MF binaries available, loads YAML, prints config, orchestrates steps, and triggers MF6/MP7 + plotting.
- `inputs.py`: Pydantic `Settings` with:
  - Path resolution validator (relative→absolute)
  - `setup_workspace()`, `setup_projection()`, `setup_terrain()` and related prep
  - Environment flag overrides for write/run/plot/show/save
- `functions/`
  - `my_utils.py`: Business logic for steps 1–7/8 (preprocess → model run → exports)
  - `raster_utils.py`: Pure helpers for raster I/O, masking, interpolation, and grid centers
  - `model_utils.py`: Pure helpers for grid polygons, `idomain`, boundary cell classification, sampling
  - `path_utils.py`: PATH/binaries helpers, MODFLOW download, and config table printer
  - `common_imports_for_main.py`: Shared scientific stack imports and optional installer

## Troubleshooting

- Geo stack installation (Shapely/Rasterio/GeoPandas):
  - Prefer `conda-forge` on Windows/macOS for prebuilt wheels
- MF6/MP7 not found:
  - Ensure `bin/modflow` binaries exist or let the CLI downloader populate them
  - Confirm `md6_exe_path`/`md7_exe_path` in YAML if using custom locations
- CRS issues:
  - Verify `projection_file` is a valid `.prj` and matches your inputs

## Acknowledgements

- FloPy for Pythonic MODFLOW/MP orchestration
- The GeoPython stack: NumPy, SciPy, Pandas, GeoPandas, Shapely, Rasterio, PyProj, Matplotlib, Seaborn

## License

No license file found.
