# -*- coding: utf-8 -*-
# Hyporheic_Toolbox.pyt — ArcGIS Pro Python toolbox

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import arcpy

# --- Make the package importable when running from a repo checkout ---
_PYT_DIR = Path(__file__).resolve().parent
for parent in _PYT_DIR.parents:
    src_candidate = parent / "src"
    if (src_candidate / "HyporheicTools").exists():
        if str(src_candidate) not in sys.path:
            sys.path.insert(0, str(src_candidate))
        break

# now regular imports work:
try:
    from HyporheicTools.core.run_from_yaml import run_from_yaml  # noqa: F401  (import checked later too)
except Exception:
    # We'll import again inside execute() and surface a helpful message if it fails there.
    pass


class Toolbox(object):
    def __init__(self):
        self.label = "Hyporheic Tools"
        self.alias = "hyporheic"
        self.description = "Build and run MODFLOW 6 + MODPATH 7 hyporheic models."
        self.tools = [RunModel]


class RunModel(object):
    def __init__(self):
        self.label = "Run Hyporheic Model (MODFLOW 6 + MODPATH 7)"
        self.description = (
            "Run the hyporheic workflow from a YAML or from parameters; "
            "adds key outputs (pathlines/points + hydraulic head layers) to the active map."
        )
        self.canRunInBackground = False

        # will hold a mapping like {'inputs_yaml': 0, 'out_folder': 1, ...}
        self._pidx: Dict[str, int] = {}

    # --------------------------
    # Small helpers for safe index access
    # --------------------------
    def _ensure_index_map(self, params) -> None:
        """Build a robust name->index map if missing or stale."""
        if not self._pidx or len(self._pidx) != len(params):
            pidx = {}
            for i, p in enumerate(params):
                try:
                    if p.name:
                        pidx[p.name] = i
                except Exception:
                    # fall back: don't crash if a param is odd
                    pass
            self._pidx = pidx

    def _i(self, params, name: str) -> int:
        """Get the index of a parameter by its .name, raising KeyError if missing."""
        self._ensure_index_map(params)
        if name not in self._pidx:
            raise KeyError(f"Parameter named '{name}' not found.")
        return self._pidx[name]

    def _get(self, params, name: str) -> arcpy.Parameter:
        return params[self._i(params, name)]

    # --------------------------
    # Licensing helpers (Pro)
    # --------------------------
    def _get_pro_license_level(self) -> str:
        try:
            info = arcpy.GetInstallInfo()
            for key in ("LicenseLevel", "License Level", "Edition"):
                v = info.get(key)
                if v:
                    v = str(v)
                    low = v.lower()
                    if low in ("arcview", "arceditor", "arcinfo"):
                        return {"arcview": "Basic", "arceditor": "Standard", "arcinfo": "Advanced"}[low]
                    return v
        except Exception:
            pass
        try:
            prod = arcpy.ProductInfo()
            return {"ArcView": "Basic", "ArcEditor": "Standard", "ArcInfo": "Advanced"}.get(prod, "Unknown")
        except Exception:
            return "Unknown"

    def _extension_available(self, name: str) -> tuple[bool, str]:
        try:
            status = arcpy.CheckExtension(name)
            return (status == "Available", status)
        except Exception as e:
            return (False, f"Error: {e}")

    def _log_licensing_summary(self, messages):
        lvl = self._get_pro_license_level()
        messages.AddMessage(f"ArcGIS Pro license level detected: {lvl}")
        for ext in ("Spatial", "3D", "ImageAnalyst"):
            ok, status = self._extension_available(ext)
            messages.AddMessage(f" Extension '{ext}': {status}")
        return lvl

    def _dedupe_and_move_to_group(self, m, lyr_obj, parent_group):
        if not lyr_obj:
            return
        try:
            if parent_group:
                m.addLayerToGroup(parent_group, lyr_obj)
            ds = getattr(lyr_obj, "dataSource", None)
            if ds:
                nds = os.path.normpath(ds)
                for lyr in m.listLayers():
                    if lyr is lyr_obj or lyr.isGroupLayer:
                        continue
                    try:
                        ds2 = getattr(lyr, "dataSource", None)
                        if ds2 and os.path.normpath(ds2) == nds:
                            long_name = getattr(lyr, "longName", lyr.name) or ""
                            if parent_group and (parent_group.name not in long_name):
                                m.removeLayer(lyr)
                    except Exception:
                        continue
        except Exception:
            pass

    # --------------------------
    # Utilities (no package imports here)
    # --------------------------
    def _param(self, display, name, dtype, required=False, direction="Input", category=None):
        p = arcpy.Parameter(
            displayName=display,
            name=name,
            datatype=dtype,
            parameterType="Required" if required else "Optional",
            direction=direction,
        )
        if category:
            try:
                p.category = category
            except Exception:
                pass
        return p

    def _active_map_sr(self) -> Optional[arcpy.SpatialReference]:
        try:
            aprx = arcpy.mp.ArcGISProject("CURRENT")
            m = aprx.activeMap
            return m.spatialReference if m else None
        except Exception:
            return None

    def _coerce_to_dataset_path(self, param: arcpy.Parameter) -> Optional[str]:
        try:
            v = param.value
            if v is not None:
                desc = arcpy.Describe(v)
                cp = getattr(desc, "catalogPath", None)
                if cp:
                    return str(Path(cp).resolve())
        except Exception:
            pass
        vtxt = getattr(param, "valueAsText", None)
        if vtxt:
            try:
                if os.path.exists(vtxt):
                    return str(Path(vtxt).resolve())
            except Exception:
                pass
            return vtxt
        return None

    def _packaged_modflow_bin(self) -> Path:
        pkg_root = Path(__file__).resolve().parents[2]
        return pkg_root / "bin" / "modflow"

    def _write_prj(self, sr: arcpy.SpatialReference, out_dir: Path, name: str) -> Path:
        prj_path = out_dir / name
        wkt = sr.exportToString()
        prj_path.write_text(wkt, encoding="utf-8")
        return prj_path

    def _resolve_projection_file_path(self, params, out_dir: Path) -> Optional[Path]:
        prj_param = self._get(params, "projection_file")
        sr_param = self._get(params, "target_spatial_reference")
        use_map = bool(self._get(params, "use_map_crs").value)

        prj_path_text = prj_param.valueAsText
        if prj_path_text:
            return Path(prj_path_text).resolve()

        if use_map:
            sr = self._active_map_sr()
            if sr:
                return self._write_prj(sr, out_dir, "map_projection.prj")

        try:
            sr = sr_param.value
            if sr:
                return self._write_prj(sr, out_dir, "selected_projection.prj")
        except Exception:
            pass
        return None

    # --------------------------
    # Boundary profile parsing/validation
    # --------------------------
    @staticmethod
    def _parse_gradient_profile(text: str) -> List[Tuple[float, float]]:
        """
        Parse a space-delimited series of 'fraction,gradient' pairs into a list of (f, g) floats.
        Example: '0,0.01 0.5,0.05 1,0.1'
        """
        pairs: List[Tuple[float, float]] = []
        if text is None:
            return pairs
        s = str(text).strip()
        if not s:
            return pairs
        tokens = [tok for tok in s.replace(";", " ").split() if tok.strip()]
        for tok in tokens:
            if "," not in tok:
                raise ValueError(f"Token '{tok}' is missing a comma. Use 'fraction,gradient' pairs.")
            a, b = tok.split(",", 1)
            try:
                f = float(a.strip()); g = float(b.strip())
            except Exception:
                raise ValueError(f"Could not parse numbers in pair '{tok}'.")
            pairs.append((f, g))
        return pairs

    @staticmethod
    def _validate_profile_pairs(pairs: List[Tuple[float, float]]) -> Tuple[bool, str]:
        """
        Validate that:
          - at least 2 pairs
          - all fractions within [0,1]
          - fractions include 0 and 1 (within tolerance)
        Returns (ok, message). message is empty when ok=True, otherwise an error/warning.
        """
        if not pairs or len(pairs) < 2:
            return False, "Provide at least two 'fraction,gradient' pairs (e.g., '0,0.01 1,0.1')."
        fracs = [p[0] for p in pairs]
        if any((f < -1e-9 or f > 1 + 1e-9) for f in fracs):
            return False, "Fractions must be within [0, 1]."
        has0 = any(abs(f - 0.0) <= 1e-9 for f in fracs)
        has1 = any(abs(f - 1.0) <= 1e-9 for f in fracs)
        if not (has0 and has1):
            return False, "Profile must include fractions at 0 and 1."
        sorted_ok = all(fracs[i] <= fracs[i + 1] + 1e-12 for i in range(len(fracs) - 1))
        if not sorted_ok:
            return True, "Fractions are not monotonic; they will be sorted before use."
        return True, ""

    # --------------------------
    # Parameter definitions
    # --------------------------
    def getParameterInfo(self):
        # --- Top-level controls ---
        p_yaml = self._param("Inputs YAML (optional)", "inputs_yaml", "DEFile", required=False)
        try:
            p_yaml.filter.list = ["yml", "yaml"]
        except Exception:
            pass

        p_out = self._param("Output Folder", "out_folder", "DEFolder", required=True)
        p_make_fig = self._param("Create Additional Results Figures in Folder", "make_figures", "GPBoolean", required=False)
        p_make_fig.value = False
        p_clean = self._param("Clean output folder before run", "clean_first", "GPBoolean", required=False)
        p_clean.value = True

        # --- Paths (drag from Contents OR browse) ---
        cat_paths = "Input Paths"
        p_wse = self._param("Water-surface raster (layer or file)", "water_surface_elevation_raster",
                            ["GPRasterLayer", "DERasterDataset", "DEFile"], category=cat_paths)
        p_terr = self._param("Terrain raster (layer or file)", "terrain_elevation_raster",
                             ["GPRasterLayer", "DERasterDataset", "DEFile"], category=cat_paths)
        p_dom = self._param("GW domain (polygon layer or feature class)", "ground_water_domain_shapefile",
                            ["GPFeatureLayer", "DEFeatureClass"], category=cat_paths)
        p_left = self._param("Left floodplain boundary (line layer or feature class)", "left_boundary_floodplain",
                             ["GPFeatureLayer", "DEFeatureClass"], category=cat_paths)
        p_right = self._param("Right floodplain boundary (line layer or feature class)", "right_boundary_floodplain",
                              ["GPFeatureLayer", "DEFeatureClass"], category=cat_paths)
        p_prj = self._param("Projection override (.prj file, optional)", "projection_file", "DEFile", category=cat_paths)
        try:
            p_prj.filter.list = ["prj"]
        except Exception:
            pass
        p_sr = self._param("Target projection (optional)", "target_spatial_reference", "GPSpatialReference",
                           category=cat_paths)
        try:
            amsr = self._active_map_sr()
            if amsr:
                p_sr.value = amsr
        except Exception:
            pass
        p_use_map = self._param("Use active map projection (default)", "use_map_crs", "GPBoolean", category=cat_paths)
        p_use_map.value = True
        p_aerial = self._param("Aerial raster (layer or file, optional)", "aerial_raster",
                               ["GPRasterLayer", "DERasterDataset", "DEFile"], category=cat_paths)

        # --- Executables (advanced) ---
        cat_exec = "Advanced — Executables (override)"
        p_mf6 = self._param("MODFLOW 6 exe (mf6.exe)", "md6_exe_path", "DEFile", category=cat_exec)
        p_mp7 = self._param("MODPATH 7 exe (mp7.exe)", "md7_exe_path", "DEFile", category=cat_exec)

        # --- Simulation metadata ---
        cat_sim = "Simulation"
        p_sim = self._param("Simulation name", "sim_name", "GPString", category=cat_sim); p_sim.value = "Hyporheic"
        p_len = self._param("Length units", "length_units", "GPString", category=cat_sim)
        p_len.filter.list = ["feet", "meters"]; p_len.value = "feet"
        p_time = self._param("Time units", "time_units", "GPString", category=cat_sim)
        p_time.filter.list = ["days", "seconds"]; p_time.value = "days"

        # --- Grid / domain ---
        cat_grid = "Grid / Domain"
        p_dx = self._param("Cell size X", "cell_size_x", "GPDouble", category=cat_grid); p_dx.value = 10.0
        p_dy = self._param("Cell size Y", "cell_size_y", "GPDouble", category=cat_grid); p_dy.value = 10.0
        p_depth = self._param("Model depth (ft/m)", "gw_mod_depth", "GPDouble", category=cat_grid); p_depth.value = 20.0
        p_dz = self._param("Layer thickness (z)", "z", "GPDouble", category=cat_grid); p_dz.value = 0.5

        # --- Hydraulic ---
        cat_hyd = "Hydraulic"
        p_kh = self._param("Horizontal K", "kh", "GPDouble", category=cat_hyd); p_kh.value = 10.0
        p_kv = self._param("Vertical K", "kv", "GPDouble", category=cat_hyd); p_kv.value = 1.0
        p_por = self._param("Porosity (0–0.6)", "porosity", "GPDouble", category=cat_hyd); p_por.value = 0.3

        # --- Boundary Conditions ---
        cat_bc = "Boundary Conditions"
        p_bc_mode = self._param("Boundary condition mode", "boundary_condition_mode", "GPString", category=cat_bc)
        p_bc_mode.filter.list = ["4 Corner Gradients", "Spatially Varying Gradient"]
        p_bc_mode.value = "4 Corner Gradients"

        # Profiles for spatially varying mode
        p_left_prof = self._param(
            "Left boundary gradient profile (e.g., 0,0.01 0.5,0.05 1,0.1)",
            "left_boundary_gradient_profile",
            "GPString",
            category=cat_bc
        )
        p_right_prof = self._param(
            "Right boundary gradient profile (e.g., 0,0.01 0.5,0.05 1,0.1)",
            "right_boundary_gradient_profile",
            "GPString",
            category=cat_bc
        )

        # Four corner gradients (dimensionless L/L)
        p_ul_grad = self._param("Upstream‑Left gradient (L/L)", "upstream_left_fpl_gw_gradient",
                                "GPDouble", category=cat_bc); p_ul_grad.value = 0.01
        p_ur_grad = self._param("Upstream‑Right gradient (L/L)", "upstream_right_fpl_gw_gradient",
                                "GPDouble", category=cat_bc); p_ur_grad.value = 0.01
        p_dl_grad = self._param("Downstream‑Left gradient (L/L)", "downstream_left_fpl_gw_gradient",
                                "GPDouble", category=cat_bc); p_dl_grad.value = 0.01
        p_dr_grad = self._param("Downstream‑Right gradient (L/L)", "downstream_right_fpl_gw_gradient",
                                "GPDouble", category=cat_bc); p_dr_grad.value = 0.01

        # --- Stress period / timestep ---
        cat_time = "Stress Period / Timestep"
        p_nstp = self._param("nstp", "nstp", "GPLong", category=cat_time); p_nstp.value = 1
        p_perlen = self._param("perlen", "perlen", "GPDouble", category=cat_time); p_perlen.value = 1.0
        p_tsmult = self._param("tsmult", "tsmult", "GPDouble", category=cat_time); p_tsmult.value = 1.0

        # --- Run toggles (visible) ---
        cat_run = "Run Options"
        p_write = self._param("Write MF inputs", "write", "GPBoolean", category=cat_run); p_write.value = True
        p_run = self._param("Run MF6/MP7", "run", "GPBoolean", category=cat_run); p_run.value = True
        p_plot = self._param("Generate summary tables", "plot", "GPBoolean", category=cat_run); p_plot.value = True
        p_plot_show = self._param("Plot show (interactive)", "plot_show", "GPBoolean", category=cat_run); p_plot_show.value = False
        p_plot_save = self._param("Plot save (png)", "plot_save", "GPBoolean", category=cat_run); p_plot_save.value = True
        p_make_fig.category = cat_run
        p_clean.category = cat_run

        # --- Contour options ---
        p_build_contours = self._param(
            "Build head contours and add to map",
            "build_contours_in_driver",
            "GPBoolean",
            category=cat_run
        ); p_build_contours.value = False
        p_max_layers = self._param(
            "Max contour layers (leave blank = all)",
            "max_layers",
            "GPLong",
            category=cat_run
        )
        p_contour_interval = self._param(
            "Contour Interval",
            "contour_interval",
            "GPDouble",
            category=cat_run
        ); p_contour_interval.value = 0.5

        # --- Derived outputs (hidden) ---
        p_lines_out = self._param("Pathlines (output, 2D)", "pathlines_fc", ["DEShapeFile", "DEFeatureClass"], direction="Output")
        p_lines_out.parameterType = "Derived"; p_lines_out.enabled = False
        try:
            p_lines_out.hidden = True
        except Exception:
            pass
        p_points_out = self._param("Points (output)", "points_fc", ["DEShapeFile", "DEFeatureClass"], direction="Output")
        p_points_out.parameterType = "Derived"; p_points_out.enabled = False
        try:
            p_points_out.hidden = True
        except Exception:
            pass

        # --- Optional KH polygon zones ---
        p_khpoly = self._param("Use KH polygon (zones)", "kh_polygon", "GPBoolean", category=cat_hyd); p_khpoly.value = False
        p_khpoly_shp = self._param(
            "KH polygon (polygon layer / feature class / shapefile)",
            "kh_polygon_shapefile",
            ["GPFeatureLayer", "DEFeatureClass", "DEShapefile"],
            category=cat_hyd
        )
        try:
            p_khpoly_shp.filter.list = ["Polygon"]
        except Exception:
            pass

        # --- 3D derived outputs ---
        p_lines3d_out = self._param("Pathlines (output, 3D Z)", "pathlines_3d_fc", ["DEShapeFile", "DEFeatureClass"], direction="Output")
        p_lines3d_out.parameterType = "Derived"; p_lines3d_out.enabled = False
        try:
            p_lines3d_out.hidden = True
        except Exception:
            pass
        p_lines3d_full_out = self._param("Pathlines (output, 3D Z, Full)", "pathlines_3d_full_fc", ["DEShapeFile", "DEFeatureClass"], direction="Output")
        p_lines3d_full_out.parameterType = "Derived"; p_lines3d_full_out.enabled = False
        try:
            p_lines3d_full_out.hidden = True
        except Exception:
            pass

        params = [
            p_yaml, p_out, p_make_fig, p_clean,
            p_wse, p_terr, p_dom, p_left, p_right, p_prj, p_sr, p_use_map, p_aerial,
            p_mf6, p_mp7,
            p_sim, p_len, p_time,
            p_dx, p_dy, p_depth, p_dz,
            p_kh, p_kv, p_por,
            p_bc_mode, p_left_prof, p_right_prof,
            p_ul_grad, p_ur_grad, p_dl_grad, p_dr_grad,
            p_nstp, p_perlen, p_tsmult,
            p_write, p_run, p_plot, p_plot_show, p_plot_save,
            p_build_contours, p_max_layers, p_contour_interval,
            p_lines_out, p_points_out,
            p_khpoly, p_khpoly_shp,
            p_lines3d_out, p_lines3d_full_out
        ]
        self._ensure_index_map(params)
        return params

    def _toggle_form_mode(self, params, using_yaml: bool):
        """Enable/disable form sections based on 'using_yaml'. Advanced options are hidden/disabled by default."""

        # Non-YAML inputs (from first path to last stress toggle)
        start = self._i(params, "water_surface_elevation_raster")
        end = self._i(params, "tsmult")
        for i in range(start, end + 1):
            params[i].enabled = (not using_yaml)

        # Run toggles are always enabled
        for nm in ("write", "run", "plot", "make_figures", "clean_first", "build_contours_in_driver", "max_layers", "contour_interval"):
            self._get(params, nm).enabled = True

        # Advanced toggles
        for nm in ("md6_exe_path", "md7_exe_path", "plot_show", "plot_save"):
            self._get(params, nm).enabled = False

        # Spatial reference control: disable "Target projection" if "Use active map" is checked
        if not using_yaml:
            use_map = bool(self._get(params, "use_map_crs").value)
            self._get(params, "target_spatial_reference").enabled = (not use_map)

        # KH polygon enable/disable
        self._get(params, "kh_polygon").enabled = True
        self._get(params, "kh_polygon_shapefile").enabled = bool(self._get(params, "kh_polygon").value)

        # Boundary condition mode toggling
        if not using_yaml:
            mode_txt = (self._get(params, "boundary_condition_mode").valueAsText or "4 Corner Gradients").strip()
            spatial_mode = (mode_txt.lower().startswith("spatial"))
            # Profiles only in spatial mode
            self._get(params, "left_boundary_gradient_profile").enabled = spatial_mode
            self._get(params, "right_boundary_gradient_profile").enabled = spatial_mode
            # Four-corner gradients only in non-spatial mode
            for nm in (
                "upstream_left_fpl_gw_gradient",
                "upstream_right_fpl_gw_gradient",
                "downstream_left_fpl_gw_gradient",
                "downstream_right_fpl_gw_gradient",
            ):
                self._get(params, nm).enabled = (not spatial_mode)

    def updateParameters(self, parameters):
        using_yaml = bool(self._get(parameters, "inputs_yaml").value)
        self._toggle_form_mode(parameters, using_yaml)

        tgt = self._get(parameters, "target_spatial_reference")
        if (not using_yaml) and tgt.enabled and (tgt.value is None):
            amsr = self._active_map_sr()
            if amsr:
                tgt.value = amsr
        return

    def updateMessages(self, parameters):
        # YAML existence
        yaml_param = self._get(parameters, "inputs_yaml")
        yaml_path = yaml_param.valueAsText
        if yaml_path and not os.path.exists(yaml_path):
            yaml_param.setErrorMessage("YAML path does not exist.")

        # Output folder write test
        out_p = self._get(parameters, "out_folder")
        out_dir = out_p.valueAsText
        if not out_dir:
            out_p.setErrorMessage("Please choose an output folder.")
        else:
            try:
                test = Path(out_dir) / "_writecheck.tmp"
                with open(test, "w") as f:
                    f.write("ok")
                test.unlink()
            except Exception:
                out_p.setErrorMessage("Cannot write to the output folder. Pick another location.")

        using_yaml = bool(yaml_param.value)

        # Required datasets when not using YAML
        if not using_yaml:
            must_exist = [
                (self._get(parameters, "water_surface_elevation_raster"), "Water-surface raster"),
                (self._get(parameters, "terrain_elevation_raster"), "Terrain raster"),
                (self._get(parameters, "ground_water_domain_shapefile"), "GW domain"),
                (self._get(parameters, "left_boundary_floodplain"), "Left boundary"),
                (self._get(parameters, "right_boundary_floodplain"), "Right boundary"),
            ]
            for p, label in must_exist:
                if not p.value:
                    p.setErrorMessage(f"{label} is required when not using a YAML.")
                else:
                    vtxt = p.valueAsText
                    if vtxt and os.path.exists(vtxt) is False:
                        try:
                            arcpy.Describe(vtxt)
                        except Exception:
                            p.setErrorMessage(f"{label} does not exist: {vtxt}")

            # CRS choices check
            use_map = bool(self._get(parameters, "use_map_crs").value)
            has_prj = bool(self._get(parameters, "projection_file").valueAsText)
            has_sr = bool(self._get(parameters, "target_spatial_reference").value)
            if not (has_prj or use_map or has_sr):
                self._get(parameters, "projection_file").setErrorMessage(
                    "Choose one: provide a .prj, enable 'Use active map projection', or pick a Target projection."
                )

            # Boundary condition validations
            mode_txt = (self._get(parameters, "boundary_condition_mode").valueAsText or "4 Corner Gradients").strip()
            spatial_mode = (mode_txt.lower().startswith("spatial"))

            if spatial_mode:
                lp = self._get(parameters, "left_boundary_gradient_profile")
                rp = self._get(parameters, "right_boundary_gradient_profile")
                for p, side in ((lp, "Left"), (rp, "Right")):
                    txt = p.valueAsText
                    if not txt or not txt.strip():
                        p.setErrorMessage(f"{side} boundary gradient profile is required (e.g., '0,0.01 1,0.1').")
                        continue
                    try:
                        pairs = self._parse_gradient_profile(txt)
                    except ValueError as e:
                        p.setErrorMessage(str(e))
                        continue
                    ok, msg = self._validate_profile_pairs(pairs)
                    if not ok:
                        p.setErrorMessage(msg)
                    elif msg:
                        p.setWarningMessage(msg)
            else:
                for nm, label in (
                    ("upstream_left_fpl_gw_gradient", "Upstream‑Left gradient"),
                    ("upstream_right_fpl_gw_gradient", "Upstream‑Right gradient"),
                    ("downstream_left_fpl_gw_gradient", "Downstream‑Left gradient"),
                    ("downstream_right_fpl_gw_gradient", "Downstream‑Right gradient"),
                ):
                    pv = self._get(parameters, nm)
                    try:
                        _ = float(pv.value) if pv.value is not None else None
                    except Exception:
                        pv.setErrorMessage(f"{label} must be a number (L/L).")

        # Porosity sanity
        por_p = self._get(parameters, "porosity")
        por_val = por_p.value
        try:
            if por_val is not None and (float(por_val) < 0.0 or float(por_val) > 0.6):
                por_p.setWarningMessage("Porosity is typically within 0.0–0.6.")
        except Exception:
            pass

        # KH polygon validation
        try:
            if bool(self._get(parameters, "kh_polygon").value):
                shp_param = self._get(parameters, "kh_polygon_shapefile")
                val_obj = shp_param.value
                val_txt = shp_param.valueAsText
                if not val_obj and not val_txt:
                    shp_param.setErrorMessage("Provide a KH polygon shapefile/feature class/layer.")
                else:
                    try:
                        d = arcpy.Describe(val_obj if val_obj is not None else val_txt)
                        st = getattr(d, "shapeType", None)
                        if st and str(st).lower() != "polygon":
                            shp_param.setErrorMessage(
                                f"KH polygon must be a Polygon; got {st}."
                            )
                        if val_obj is None and val_txt and not os.path.exists(val_txt):
                            pass
                    except Exception:
                        if val_txt and os.path.exists(val_txt) is False:
                            shp_param.setErrorMessage(f"KH polygon dataset does not exist: {val_txt}")
                        else:
                            shp_param.setErrorMessage("Could not read KH polygon dataset.")
        except Exception:
            pass

    def _build_yaml_from_params(self, params, projection_prj_path: Optional[Path]) -> dict:
        def _norm(name: str):
            return self._coerce_to_dataset_path(self._get(params, name))

        cfg = {
            # required/paths
            "water_surface_elevation_raster": _norm("water_surface_elevation_raster"),
            "terrain_elevation_raster": _norm("terrain_elevation_raster"),
            "ground_water_domain_shapefile": _norm("ground_water_domain_shapefile"),
            "left_boundary_floodplain": _norm("left_boundary_floodplain"),
            "right_boundary_floodplain": _norm("right_boundary_floodplain"),
            "output_directory": str(Path(self._get(params, "out_folder").valueAsText).resolve())
            if self._get(params, "out_folder").valueAsText else None,
            "aerial_raster": _norm("aerial_raster"),

            # exe overrides
            "md6_exe_path": self._get(params, "md6_exe_path").valueAsText or None,
            "md7_exe_path": self._get(params, "md7_exe_path").valueAsText or None,

            # sim
            "sim_name": self._get(params, "sim_name").valueAsText or "Hyporheic",
            "length_units": self._get(params, "length_units").valueAsText or "feet",
            "time_units": self._get(params, "time_units").valueAsText or "days",

            # grid
            "cell_size_x": float(self._get(params, "cell_size_x").value) if self._get(params, "cell_size_x").value is not None else 10.0,
            "cell_size_y": float(self._get(params, "cell_size_y").value) if self._get(params, "cell_size_y").value is not None else 10.0,
            "gw_mod_depth": float(self._get(params, "gw_mod_depth").value) if self._get(params, "gw_mod_depth").value is not None else 20.0,
            "z": float(self._get(params, "z").value) if self._get(params, "z").value is not None else 0.5,

            # hydraulic
            "kh": float(self._get(params, "kh").value) if self._get(params, "kh").value is not None else 10.0,
            "kv": float(self._get(params, "kv").value) if self._get(params, "kv").value is not None else 1.0,
            "porosity": float(self._get(params, "porosity").value) if self._get(params, "porosity").value is not None else 0.3,

            # stress
            "nstp": int(self._get(params, "nstp").value) if self._get(params, "nstp").value is not None else 1,
            "perlen": float(self._get(params, "perlen").value) if self._get(params, "perlen").value is not None else 1.0,
            "tsmult": float(self._get(params, "tsmult").value) if self._get(params, "tsmult").value is not None else 1.0,

            # run toggles
            "write": bool(self._get(params, "write").value),
            "run": bool(self._get(params, "run").value),
            "plot": bool(self._get(params, "plot").value),
        }

        # Advanced toggles
        plot_show_p = self._get(params, "plot_show")
        plot_save_p = self._get(params, "plot_save")
        cfg["plot_show"] = bool(plot_show_p.value) if plot_show_p.enabled else False
        cfg["plot_save"] = bool(plot_save_p.value) if plot_save_p.enabled else True

        # Contour options
        try:
            cfg["build_contours_in_driver"] = bool(self._get(params, "build_contours_in_driver").value)
        except Exception:
            pass
        try:
            ml = self._get(params, "max_layers").value
            if ml is not None:
                ml_int = int(ml)
                if ml_int > 0:
                    cfg["max_layers"] = ml_int
        except Exception:
            pass
        try:
            ci = self._get(params, "contour_interval").value
            if ci is not None:
                cfg["contour_interval"] = float(ci)
        except Exception:
            pass

        # Projection file resolution
        prj_explicit = self._get(params, "projection_file").valueAsText
        if prj_explicit:
            cfg["projection_file"] = str(Path(prj_explicit).resolve())
        elif projection_prj_path:
            cfg["projection_file"] = str(projection_prj_path.resolve())

        # If no explicit exe overrides, but packaged bin exists, point to it
        if not cfg.get("md6_exe_path") and not cfg.get("md7_exe_path"):
            pkg_bin = self._packaged_modflow_bin()
            if pkg_bin.exists():
                cfg["modflow_bin_dir"] = str(pkg_bin.resolve())

        # Back-compat / dual keys
        if cfg.get("md6_exe_path"):
            cfg["mf6_exe_path"] = cfg["md6_exe_path"]
        if cfg.get("md7_exe_path"):
            cfg["mp7_exe_path"] = cfg["md7_exe_path"]

        # KH polygon options
        try:
            cfg["kh_polygon"] = bool(self._get(params, "kh_polygon").value)
            if cfg["kh_polygon"]:
                shp = self._coerce_to_dataset_path(self._get(params, "kh_polygon_shapefile"))
                if shp:
                    cfg["kh_polygon_shapefile"] = shp
        except Exception:
            pass

        # ---- Boundary-condition mode & values ----
        mode_txt = (self._get(params, "boundary_condition_mode").valueAsText or "4 Corner Gradients").strip()
        if mode_txt.lower().startswith("spatial"):
            cfg["boundary_condition_mode"] = "spatially_varying_gradient"
            lp = self._get(params, "left_boundary_gradient_profile").valueAsText
            rp = self._get(params, "right_boundary_gradient_profile").valueAsText
            if lp:
                cfg["left_boundary_gradient_profile"] = lp.strip()
            if rp:
                cfg["right_boundary_gradient_profile"] = rp.strip()
        else:
            cfg["boundary_condition_mode"] = "4_corner_gradients"
            cfg["upstream_left_fpl_gw_gradient"] = float(self._get(params, "upstream_left_fpl_gw_gradient").value) \
                if self._get(params, "upstream_left_fpl_gw_gradient").value is not None else 0.01
            cfg["upstream_right_fpl_gw_gradient"] = float(self._get(params, "upstream_right_fpl_gw_gradient").value) \
                if self._get(params, "upstream_right_fpl_gw_gradient").value is not None else 0.01
            cfg["downstream_left_fpl_gw_gradient"] = float(self._get(params, "downstream_left_fpl_gw_gradient").value) \
                if self._get(params, "downstream_left_fpl_gw_gradient").value is not None else 0.01
            cfg["downstream_right_fpl_gw_gradient"] = float(self._get(params, "downstream_right_fpl_gw_gradient").value) \
                if self._get(params, "downstream_right_fpl_gw_gradient").value is not None else 0.01

        # Clean out Nones
        return {k: v for k, v in cfg.items() if v is not None}

    def _add_with_symbology(self, m, dataset_path: str, symbology_name: str | None = None, *,
                             desired_name: str | None = None, parent_group=None):
        if not dataset_path:
            return None
        prev_add = arcpy.env.addOutputsToMap
        arcpy.env.addOutputsToMap = False
        try:
            lyr_obj = None
            try:
                d = arcpy.Describe(dataset_path)
                dt = getattr(d, "dataType", "") or getattr(d, "datasetType", "")
            except Exception:
                d = None; dt = ""

            try:
                if dt in ("FeatureClass", "ShapeFile", "FeatureDataset") or dataset_path.lower().endswith(".shp"):
                    out_name = desired_name or Path(dataset_path).stem
                    lyr_obj = arcpy.management.MakeFeatureLayer(dataset_path, out_name).getOutput(0)
                elif dt in ("MosaicDataset",) or (".gdb" in dataset_path and "mosaic" in dataset_path.lower()):
                    out_name = desired_name or "Mosaic"
                    lyr_obj = arcpy.management.MakeMosaicLayer(dataset_path, out_name).getOutput(0)
                else:
                    out_name = desired_name or Path(dataset_path).stem
                    try:
                        lyr_obj = arcpy.management.MakeRasterLayer(dataset_path, out_name).getOutput(0)
                    except Exception:
                        lyr_obj = m.addDataFromPath(dataset_path)
            except Exception:
                lyr_obj = m.addDataFromPath(dataset_path)

            if lyr_obj and desired_name:
                try:
                    lyr_obj.name = desired_name
                except Exception:
                    pass

            if lyr_obj and symbology_name:
                sym_dir = Path(__file__).resolve().parent.parent / "symbology"
                sym_file = sym_dir / symbology_name
                if sym_file.exists():
                    try:
                        arcpy.management.ApplySymbologyFromLayer(lyr_obj, str(sym_file))
                    except Exception:
                        pass

            if parent_group and lyr_obj:
                self._dedupe_and_move_to_group(m, lyr_obj, parent_group)
            return lyr_obj
        finally:
            try:
                arcpy.env.addOutputsToMap = prev_add
            except Exception:
                pass

    # --------------------------
    # Execute
    # --------------------------
    def execute(self, params, messages):
        # Import inside execute so the dialog opens even if deps missing
        try:
            from HyporheicTools.core.run_from_yaml import run_from_yaml  # noqa: F401
            try:
                from HyporheicTools.functions import path_utils as pu
            except Exception:
                pu = None
        except ModuleNotFoundError:
            mod = sys.modules.get("HyporheicTools")
            mod_file = getattr(mod, "__file__", "") if mod else ""
            if mod_file.endswith(".pyt"):
                messages.AddErrorMessage(
                    "Import failed because your toolbox file is named 'HyporheicTools.pyt'. "
                    "Please rename to 'Hyporheic_Toolbox.pyt' and try again."
                )
            else:
                messages.AddErrorMessage("Could not import the 'HyporheicTools' package.")
            raise

        arcpy.env.overwriteOutput = True
        arcpy.env.addOutputsToMap = True

        self._log_licensing_summary(messages)

        def log(msg: str) -> None:
            try:
                messages.AddMessage(str(msg))
            except Exception:
                print(str(msg))

        yaml_path = self._get(params, "inputs_yaml").valueAsText
        out_folder = self._get(params, "out_folder").valueAsText or arcpy.env.scratchFolder
        make_fig = bool(self._get(params, "make_figures").value)
        clean_first = bool(self._get(params, "clean_first").value)

        out_dir = Path(out_folder)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Use packaged exes (toolbox ships with executables)
        try:
            from HyporheicTools.functions import path_utils as pu2  # ensure local symbol
            pkg_bin = self._packaged_modflow_bin()
            if pkg_bin.exists() and (pu or pu2):
                (pu or pu2).add_modflow_executables(pkg_bin)
                log(f"Using packaged MODFLOW bin: {pkg_bin}")
        except Exception:
            pass

        # Clean outputs if requested
        if clean_first:
            arcpy.SetProgressor("default", "Cleaning output folder …")
            for item in out_dir.iterdir():
                try:
                    if item.is_file() or item.is_symlink():
                        item.unlink()
                    elif item.is_dir():
                        import shutil
                        shutil.rmtree(item)
                except Exception:
                    pass

        # Resolve YAML or build one from form params
        if yaml_path:
            runtime_yaml = Path(yaml_path).resolve()
            log(f"Using Inputs YAML: {runtime_yaml}")
        else:
            prj_path = self._resolve_projection_file_path(params, out_dir)
            import yaml
            cfg_map = self._build_yaml_from_params(params, prj_path)
            cfg_map["output_directory"] = str(out_dir.resolve())
            ts = time.strftime("%Y%m%d_%H%M%S")
            runtime_yaml = out_dir / f"inputs_runtime_{ts}.yaml"
            with open(runtime_yaml, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg_map, f, sort_keys=False, default_flow_style=False)
            log(f"Wrote runtime YAML → {runtime_yaml}")


        # Run the workflow (run_from_yaml now ALWAYS adds products to a dynamic group)
        arcpy.SetProgressor("default", "Running hyporheic workflow …")
        outputs = run_from_yaml(
            yaml_path=runtime_yaml,
            out_folder=out_dir,
            log=log,
            dry_run=False,
            make_figures=make_fig,
        )

        # Derived outputs for chaining
        pathlines_fc = outputs.get("pathlines_fc")
        points_fc = outputs.get("points_fc")
        pathlines_3d = outputs.get("pathlines_fc_3d")
        pathlines_3d_full = outputs.get("pathlines_fc_3d_full")

        if pathlines_fc:
            self._get(params, "pathlines_fc").value = pathlines_fc
        if points_fc:
            self._get(params, "points_fc").value = points_fc
        if pathlines_3d:
            self._get(params, "pathlines_3d_fc").value = pathlines_3d
        if pathlines_3d_full:
            self._get(params, "pathlines_3d_full_fc").value = pathlines_3d_full

        # Inform about the created group
        grp_name = outputs.get("group_name")
        if grp_name:
            messages.AddMessage(f"Results added to map group: {grp_name}")

        arcpy.AddMessage("✓ Done.")
        arcpy.ResetProgressor()
