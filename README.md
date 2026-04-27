# M3C2 Tool — Point Cloud Change Detection

**CHECKPOINT 14** | Python GUI for M3C2-based surface change detection from airborne and UAV LiDAR point clouds.

Developed as part of a master's thesis on landslide monitoring at STU SvF Bratislava, in collaboration with ŠGÚDŠ.

---

## Overview

This tool provides a complete graphical workflow for computing **M3C2 (Multiscale Model to Model Cloud Comparison)** distances between two point clouds, with full export of rasters, DEMs, quiver maps, profiles, wind roses, volume estimates, and PDF reports — all from a single Python file.

**CHECKPOINT 14** adds full **batch processing** support: run dozens of parameter-study sessions unattended, with results aggregated automatically into `batch_stats.csv` and an analysis-ready Excel workbook (`batch_stats_analyza.xlsx`).

> Core algorithm powered by [py4dgeo](https://github.com/3dgeo-heidelberg/py4dgeo).  
> Optional PDAL engine available as an alternative M3C2 backend.

---

## Features

### Core Analysis
| Feature | Description |
|---|---|
| M3C2 distance computation | Via py4dgeo (default) or PDAL |
| Corepoint subsampling | Spatial minimum-distance subsampling or use all points |
| Significance testing | LoD-based (95% / 99% / custom), manual σ, or registration-error mode |
| Normal estimation | Configurable multi-scale normal radii |
| Max depth clipping | Optional D-max parameter to clip outlier distances |

### Exports
| Output | Format | Notes |
|---|---|---|
| Classified point cloud | `.las` / `.laz` | M3C2_DISTANCE + optional fields: Nz, Nxy, DX, DY, DZ, LOD, SPREAD1/2, NORMAL_SCALE |
| Results table | `.txt` | Tab-separated, all corepoints |
| Raster grids | `.tif` (GeoTIFF) | Any combination of 11 fields; IDW / TIN / Nearest / Kriging interpolation |
| DEM | `.tif` (GeoTIFF) | Z values of reference cloud |
| Quiver normal | `.png` / `.html` / `.gpkg` | Normal-direction displacement vectors |
| Quiver motion | `.png` / `.html` / `.gpkg` | DX·DY horizontal motion vectors |
| Wind rose | 4× `.png` | All / Significant / Positive / Negative change directions |
| Profile cuts | `.png` / `.gpkg` / `.txt` | Cross-section distance plots; ref cloud = blue, comp = red |
| Volume estimate | Scalar (in PDF report) | DoD-based IDW grid integration |
| Distance histogram | `.png` | Significant / all changes |
| PDF analysis report | `.pdf` | Full statistics, system info, cloud metadata, profile map |
| Batch stats | `batch_stats.csv` | One row per run — appended automatically |
| Batch Excel | `batch_stats_analyza.xlsx` | Auto-populated from CSV via template |

### Batch Processing (New in CHECKPOINT 14)
- Save any parameter configuration as a **JSON session file**
- Select multiple JSON files → **Batch** button → runs all sequentially, unattended
- Each job creates its own output subfolder
- All runs append to a shared `batch_stats.csv`
- After every run, `batch_stats_analyza.xlsx` is regenerated from the Excel template
- Designed for M3C2 **parameter sensitivity studies** (e.g. varying cylinder diameter `d`)

### Interactive Profile Picker
- Opens a hillshaded terrain view rendered from the reference cloud
- Click to define profile lines interactively
- Adjustable grid resolution slider
- Undo / Clear / Done controls
- Lines saved as coordinates for profile cut export

### UI & Architecture
- Single-file design for portability (no installation, just `python m3c2_checkpoint14.py`)
- Modern card-based tkinter GUI with scrollable layout
- Thread-safe multithreaded pipeline (worker thread + `root.after()` marshalling)
- Matplotlib locked to Agg backend to prevent TkAgg/thread crashes
- Dual progress bars (current step + overall)
- Session save / load (JSON)

---

## Requirements

### Python
Python 3.9 or newer recommended. Tested on Python 3.11 / 3.12.

### Required packages
```
py4dgeo
laspy[lazrs]
numpy
scipy
matplotlib
```

### Optional packages (enable additional outputs)
| Package | Enables |
|---|---|
| `rasterio` | GeoTIFF raster and DEM export |
| `pyproj` | CRS assignment on rasters |
| `geopandas` + `shapely` | GeoPackage export (quiver, profiles) |
| `reportlab` | PDF report generation |
| `openpyxl` | Excel batch stats auto-populate |
| `pykrige` | Kriging interpolation on rasters |
| `psutil` | System memory info in PDF report |
| `pdal` | Alternative M3C2 engine |

### Install all at once
```bash
pip install py4dgeo "laspy[lazrs]" numpy scipy matplotlib rasterio pyproj geopandas shapely reportlab openpyxl pykrige psutil
```

> **Note:** `pdal` must be installed separately via conda or a system package manager — it is not available via pip on all platforms.

---

## Installation & Launch

1. Clone or download the repository:
```bash
git clone https://github.com/Kostka22/Geology_M3C2-from-point-cloud.git
cd Geology_M3C2-from-point-cloud
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place the Excel template **next to the script** (required for batch Excel export):
```
m3c2_checkpoint14.py
m3c2_batch_analyza_template.xlsx   ← must be in the same folder
```

4. Launch:
```bash
python m3c2_checkpoint14.py
```

---

## Quickstart — Single Run

1. **Input / Output Files** — browse for Reference cloud (LAS/LAZ), Comparison cloud (LAS/LAZ), and output path
2. **M3C2 Parameters** — set cylinder diameter `d`, normal estimation radii, corepoint spacing
3. **Output Options** — choose which exports to enable
4. Click **Run M3C2**

Results are written next to the output `.laz` file.

---

## Quickstart — Batch / Parameter Study

> See [BATCH_WORKFLOW.md](BATCH_WORKFLOW.md) for the full step-by-step guide.

**Short version:**
1. Configure parameters for one run → **Save Session** → save as e.g. `d0.5.json`
2. Repeat for each parameter variant → `d1.0.json`, `d2.0.json`, ...
3. Enable **Export batch stats (batch_stats.csv + batch_stats_analyza.xlsx)** in Output Options
4. Click **Batch** → select all JSON files → confirm → wait
5. Open `batch_stats_analyza.xlsx` — charts and knee-point analysis are ready

---

## M3C2 Parameters — Quick Reference

| Parameter | Typical range | Notes |
|---|---|---|
| Cylinder diameter `d` | 0.5 – 8.0 m | Main sensitivity parameter; start with 2–4× point spacing |
| Normal radii | `0.5,1.0,2.0` | Comma-separated list; smallest = local normals, largest = smooth normals |
| Core spacing | 0.1 – 1.0 m | Spatial subsampling of corepoints; 0 = use all points |
| Max depth | 0 (disabled) | Clips M3C2 distances beyond this value |
| Registration error | 0.0 – 0.1 m | Instrument/registration accuracy, used in LoD formula |
| Confidence level | 0.95 | 95% standard; 99% available |

---

## Output Files Reference

After a single run with output path `D:/data/result.laz`, typical outputs:

```
result.laz                    ← M3C2 classified point cloud
result.txt                    ← tab-separated results table (optional)
result_report.pdf             ← full PDF analysis report
result_histogram.png          ← distance distribution histogram
result_raster_M3C2_DISTANCE.tif
result_raster_LOD.tif
result_dem.tif
result_quiver_normal.png / .html / .gpkg
result_quiver_motion.png / .html / .gpkg
result_windrose_all.png
result_windrose_sig.png
result_windrose_pos.png
result_windrose_neg.png
result_profile_LINE1.png / .gpkg
batch_stats.csv               ← appended if batch CSV enabled
batch_stats_analyza.xlsx      ← auto-regenerated from template
```

---

## LAS Extra Fields

Optional fields written to the output point cloud:

| Field | Description |
|---|---|
| `M3C2_DISTANCE` | Signed change distance (always written) |
| `Nz` | Normal vector Z component |
| `Nxy` | Normal vector XY magnitude |
| `DX` / `DY` / `DZ` | Displacement vector components |
| `LOD` | Level of Detection threshold at corepoint |
| `SPREAD1` / `SPREAD2` | Spatial spread in reference / comparison cloud |
| `NORMAL_SCALE` | Normal estimation scale used |
| `SIGNIFICANT` | Binary: 1 = significant change, 0 = within noise |

---

## Batch Stats CSV — Column Reference

`batch_stats.csv` columns (one row per run):

| Column | Description |
|---|---|
| `run_name` | Base name of the output file |
| `timestamp` | ISO timestamp of the run |
| `d` | Cylinder diameter [m] |
| `normal_radii` | Normal estimation radii (string) |
| `core_spacing` | Corepoint spacing [m] |
| `reg_error` | Registration error [m] |
| `sigma_mode` | Uncertainty mode used |
| `sigma1` / `sigma2` | Manual σ values (if used) |
| `LoD95` | Median LoD at 95% confidence [m] |
| `total_cp` | Total corepoints |
| `valid_cp` | Valid (non-NaN) corepoints |
| `nan_cp` | NaN corepoints |
| `sig_count` | Significant change count |
| `sig_pct` | Significant change percentage |
| `mean_dist` | Mean M3C2 distance [m] |
| `median_dist` | Median M3C2 distance [m] |
| `std_dist` | Std. deviation of distances [m] |
| `min_dist` / `max_dist` | Range of distances [m] |
| `rms_dist` | RMS of distances [m] |
| `nz_mean` / `nz_min` | Normal Z component stats |

---

## Project Structure

```
Geology_M3C2-from-point-cloud/
├── m3c2_checkpoint14.py              ← main script (single-file GUI)
├── m3c2_batch_analyza_template.xlsx  ← Excel template for batch analysis
├── requirements.txt
├── README.md
├── CHANGELOG.md
├── BATCH_WORKFLOW.md
├── LICENSE
└── THIRD_PARTY_LICENSES.md
```

---

## Study Localities (Thesis Context)

| Locality | Datasets | Purpose |
|---|---|---|
| Kykulá (Machnáč) | LLS MB1 vs MB2 | Primary landslide monitoring |
| Veľká Čausa | MB1, MB2, UAV, TLS | Multi-sensor comparison |

Data source: ÚGKK SR national ALS cycles (MB1/MB2), ŠGÚDŠ cooperation.

---

## License

MIT License — see [LICENSE](LICENSE).

Third-party licenses: [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)

---

## Author

**Samuel Lebó**  
Geodesy and Cartography, STU SvF Bratislava  

Supervisor: Ing. Tibor Lieskovský, PhD.  
In cooperation with: ŠGÚDŠ (Štátny geologický ústav Dionýza Štúra)
