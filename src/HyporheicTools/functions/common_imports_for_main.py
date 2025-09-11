# filepath: .../common_imports.py
# --------------------------------------------------------------------
"""
Shared imports for Jupyter notebooks.

Usage in a notebook
-------------------
from common_imports import *          # ← brings in np, pd, flopy, etc.

If you edit this file while a notebook is running, reload it once:

import importlib, common_imports
importlib.reload(common_imports)
"""
# -------------------- Standard library -----------------------------
import logging
from pathlib import Path                         # exported as Path

# -------------------- Third-party libraries -------------------------
import shutil
import os
import ssl
import numpy as np                               # exported as np
import pandas as pd                              # exported as pd
import matplotlib.pyplot as plt                  # exported as plt
import flopy                                     # exported as flopy
import geopandas as gpd                          # exported as gpd
import rasterio
# import alphashape                                  # exported as rasterio
from shapely.geometry import box, Point, Polygon, LineString
from scipy.interpolate import griddata           # exported as griddata
from pyproj import Transformer
from flopy.utils import ZoneBudget
from flopy.utils.binaryfile import CellBudgetFile

# Utility re-exports (optional)
from pprint import pformat                       # exported as pformat
# from modflow_devtools.misc import get_env, timed # exported as get_env, timed

#rasterio imports
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from rasterio.plot import show
from rasterio.transform import from_bounds
from rasterio.transform import rowcol
from rasterio.mask import mask

# -------------------- Runtime tweaks --------------------------------
# Fix Windows event-loop quirk so async libraries (e.g., rasterio) behave:
import asyncio
if asyncio.get_event_loop_policy().__class__.__name__ == "ProactorEventLoopPolicy":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# -------------------- Public API -----------------------------------
__all__ = [
    # core analysis stack
    "shutil", "os","np", "pd", "plt",
    # hydro-model libraries
    "flopy", "gpd", "rasterio", 
    # geometry helpers
    "box", "Point", "Polygon", "LineString",
    # misc utilities
    "Path", "griddata", "pformat", 
    #"get_env", "timed",

    # GIS utilities
    "calculate_default_transform","reproject","Resampling", "from_bounds", "rowcol", "mask", "show", "CRS",
    # helper functions (none exported here)
]
# --------------------------------------------------------------------

# Optional: a tiny CLI that installs missing packages.
# Executed **only** if you run the file directly (`python common_imports.py`),
# never when a notebook imports it – so Pyright/Pylance see a pure import-only
# module.
if __name__ == "__main__":
    import subprocess, sys

    required = [
        "flopy", "matplotlib", "numpy", "geopandas", "pandas",
        "rasterio", "pyproj", "shapely", "scipy", "modflow_devtools",
    ]
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            logging.info("Installing %s …", pkg)
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

if __name__ == "Hyporheic_Workflow_Merged":
    import subprocess, sys

    required = [
        "flopy", "matplotlib", "numpy", "geopandas", "pandas",
        "rasterio", "pyproj", "shapely", "scipy", "modflow_devtools",
    ]
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            logging.info("Installing %s …", pkg)
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])



# # filepath: /c:/Users/u4eeevmq/Documents/Python/HyporheicFloPy/VQuintana/common_imports.py
# #-----------------------Importing Libraries-----------------------#
# import subprocess
# import sys
# import logging

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def install(package):
#     logger.info(f"Installing package: {package}")
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# # List of required packages
# required_packages = [
#     "flopy",
#     "matplotlib",
#     "numpy",
#     "geopandas",
#     "pandas",
#     "rasterio",
#     "pyproj",
#     "shutil",
#     "random",
#     "pathlib",
#     "shapely",
#     "scipy",
#     "modflow_devtools",
#     "pickleshare",
#     "warnings"
# ]

# # Install required packages
# for package in required_packages:
#     try:
#         __import__(package)
#     except ImportError:
#         install(package)

# # Importing Libraries
# import os
# import flopy
# import pathlib
# import matplotlib.pyplot as plt
# import numpy as np
# import geopandas as gpd
# import pandas as pd
# import rasterio
# import pyproj
# import shutil
# import random
# import scipy
# import pathlib as pl
# import papermill as pm
# from pathlib import Path
# from rasterio.crs import CRS
# from rasterio.plot import show
# from rasterio.warp import calculate_default_transform, reproject, Resampling
# from rasterio.transform import from_bounds
# from rasterio.transform import rowcol
# from rasterio.mask import mask
# from shapely.geometry import box, Point, Polygon, LineString
# from flopy.utils.binaryfile import HeadFile
# from scipy.interpolate import griddata
# from pprint import pformat
# from flopy.plot.styles import styles
# from matplotlib.lines import Line2D
# from flopy.mf6 import MFSimulation
# from matplotlib import cbook, cm
# from matplotlib.colors import LightSource
# from modflow_devtools.misc import get_env, timed
# import jupyter_book

# # Runtime Settings
# import asyncio
# if asyncio.get_event_loop_policy().__class__.__name__ == "ProactorEventLoopPolicy":
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
