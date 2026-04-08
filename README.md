# M3C2 Tool — CHECKPOINT 13

**Author:** Samuel Lebó , Bratislava  
**Language:** Python 3.11  
**Repository:** [github.com/Kostka22/Geology_M3C2-from-point-cloud](https://github.com/Kostka22/Geology_M3C2-from-point-cloud)

---

A desktop GUI application for detecting and quantifying surface changes between two registered 3-D point clouds using the **M3C2 algorithm** (Lague, Brodu & Leroux, 2013). Designed for field surveyors — single file, no configuration, copy and run.

---

## Table of Contents

1. [What this tool does](#1-what-this-tool-does)
2. [Installation](#2-installation)
3. [How to run](#3-how-to-run)
4. [GUI walkthrough](#4-gui-walkthrough)
5. [Core pipeline — step by step](#5-core-pipeline--step-by-step)
6. [M3C2 algorithm explained](#6-m3c2-algorithm-explained)
7. [Subsampling](#7-subsampling)
8. [Significance testing](#8-significance-testing)
9. [Normal vectors and displacement components](#9-normal-vectors-and-displacement-components)
10. [Raster export](#10-raster-export)
11. [DEM export](#11-dem-export)
12. [Quiver export](#12-quiver-export)
13. [Profile cut export](#13-profile-cut-export)
14. [Wind rose export](#14-wind-rose-export)
15. [Volume change estimation](#15-volume-change-estimation)
16. [Distance histogram](#16-distance-histogram)
17. [PDF report](#17-pdf-report)
18. [Session save and load](#18-session-save-and-load)
19. [PDAL engine](#19-pdal-engine)
20. [Multithreading](#20-multithreading)
21. [Output files reference](#21-output-files-reference)
22. [Tested datasets](#22-tested-datasets)
23. [References](#23-references)

---

## 1. What this tool does

Given two point clouds of the same area captured at different times, the tool:

- Computes a signed 3-D distance at every corepoint using M3C2
- Flags changes that exceed the statistical noise threshold (Level of Detection)
- Exports results as a LAS/LAZ file with scalar fields
- Optionally produces GeoTIFF rasters, quiver plots, cross-section profiles, wind roses, a volume estimate, and a full PDF analysis report

It handles any surface orientation — flat terrain, steep slopes, vertical walls, and overhangs — because M3C2 measures distance along the local surface normal rather than purely vertically.

---

## 2. Installation

### Requirements

- Python 3.10 or newer
- Windows, Linux, or macOS

### Install dependencies

```bash
pip install py4dgeo laspy[laszip] numpy scipy matplotlib plotly geopandas shapely rasterio pyproj pykrige psutil reportlab
```

| Package | Purpose |
|---------|---------|
| `py4dgeo` | M3C2 core algorithm |
| `laspy[laszip]` | LAS/LAZ file read/write — the `[laszip]` extra enables compressed LAZ |
| `numpy` | All array computation |
| `scipy` | KD-trees, interpolation, statistics |
| `matplotlib` | Profile PNGs, quiver PNGs, wind roses, histogram, overview map |
| `plotly` | Interactive quiver HTML |
| `geopandas` + `shapely` | GPKG export, clip mask loading |
| `rasterio` | GeoTIFF read/write and clip by mask |
| `pyproj` | Bundled proj.db — fixes EPSG lookup conflicts with PostGIS |
| `pykrige` | Kriging interpolation (only needed if you select Kriging) |
| `psutil` | CPU/RAM info in the PDF report |
| `reportlab` | PDF report generation |

### Optional — PDAL engine (for large datasets >1M corepoints)

```bash
conda install -c conda-forge pdal python-pdal
```

PDAL uses OpenMP multi-core parallelism for the M3C2 computation itself, which py4dgeo cannot do.

---

## 3. How to run

```bash
python m3c2_checkpoint13.py
```

No arguments needed. The GUI opens immediately.

---

## 4. GUI walkthrough

The window is a single scrollable form divided into sections:

| Section | What you fill in |
|---------|-----------------|
| **Input / Output Files** | Reference LAZ, comparison LAZ, output LAZ path |
| **M3C2 Parameters** | Cylinder radius, normal radii, corepoint spacing, confidence level, sigma |
| **Output Options** | TXT export, PDF report, histogram, LAS scalar fields to include |
| **Raster Export** | Which scalar fields to write as GeoTIFF, resolution, interpolation method |
| **DEM Export** | Z-value raster of the reference cloud |
| **Quiver Export** | Arrow-field plots of displacement direction |
| **Volume Change** | DoD and M3C2 surface-corrected volume estimation |
| **Wind Rose** | Polar histogram of displacement direction and magnitude |
| **Profile Cut** | Cross-section PNGs along interactively picked or loaded lines |
| **Progress bars** | Current task and overall pipeline progress |
| **Session save/load** | Persist and restore all settings as a JSON file |
| **Run button** | Launches the pipeline in a background thread |

### Guess Parameters button

Reads up to 5,000 random points from the reference cloud, computes the mean nearest-neighbour distance `d_avg`, and fills in:

- Cylinder radius = 2 × d_avg
- Normal radii = 3×, 5×, 8× d_avg
- Corepoint spacing = 2 × d_avg

These follow the recommendations in Lague et al. (2013).

---

## 5. Core pipeline — step by step

When you click **Run**, all GUI values are read on the main thread and passed as plain Python scalars to a background worker thread. The worker runs the following steps:

```
Step 1 — Load both clouds into py4dgeo epoch objects
Step 2 — Subsample the reference cloud to corepoints
Step 3 — Run M3C2 to get distances and uncertainties
Step 4 — Compute independent PCA normals for DX/DY/DZ decomposition
Step 5 — Apply max depth filter (set outliers to NaN)
Step 6 — Compute significance (LoD) and flag significant changes
Step 7 — Compute Nz, Nxy, DX, DY, DZ from normals and distances
Step 8 — Save LAS/LAZ output with all scalar fields
Step 9 — Compute and print statistics
        — Export rasters, quiver, profiles, wind roses, histogram, volume, PDF
```

The GUI never freezes because all heavy work runs in the worker thread. Progress bars update via `root.after()` which safely marshals calls back to the main thread.

---

## 6. M3C2 algorithm explained

### What it measures

At each corepoint, M3C2 measures how far the surface moved between epoch 1 (reference) and epoch 2 (comparison) along the local surface normal. This is the **signed M3C2 distance** `d`.

- Positive `d` → surface moved away from the reference (deposition, growth)
- Negative `d` → surface moved toward the reference (erosion, loss)

### Step 1 — Normal estimation

For each corepoint, the tool collects all neighbouring points within each radius in the **Normal radii** list. For each radius it builds a 3×3 covariance matrix:

```
C = (1/N) × Σ (pᵢ − μ)(pᵢ − μ)ᵀ
```

The eigenvector corresponding to the **smallest eigenvalue** of C is the surface normal. This works because in a planar neighbourhood the smallest variance direction is perpendicular to the surface.

py4dgeo automatically selects the scale (radius) that minimises the ratio `λ_min / (λ_min + λ_mid + λ_max)` — the scale where the neighbourhood is most planar.

### Step 2 — Projection cylinder

A cylinder of radius `r` is projected along the normal at each corepoint. All points from both clouds that fall inside the cylinder are collected:

```
C₁(p) = { x ∈ Cloud₁ : perpendicular distance from x to normal axis ≤ r }
C₂(p) = { x ∈ Cloud₂ : perpendicular distance from x to normal axis ≤ r }
```

### Step 3 — Distance measurement

The signed distance along the normal between the mean positions of the two sub-clouds:

```
d(p) = (μ₂(p) − μ₁(p)) · n̂
```

where `μᵢ(p)` is the mean XYZ position of cloud i inside the cylinder and `n̂` is the unit normal vector.

### Step 4 — Level of Detection

The LoD combines point cloud roughness (spread inside the cylinder) and registration error:

```
LoD(p) = t × √( σ₁²(p)/N₁ + σ₂²(p)/N₂ )
```

where `σᵢ(p)` is the standard deviation of point positions inside the cylinder (SPREAD1/SPREAD2 from py4dgeo) and `t` is the t-critical value for the chosen confidence level:

| Confidence | t-critical |
|-----------|-----------|
| 90% | 1.645 |
| 95% | 1.960 |
| 99% | 2.576 |

### What py4dgeo does

`py4dgeo.M3C2` handles steps 1-4 internally in C++. The tool calls:

```python
m3c2 = py4dgeo.M3C2(epochs=(epoch1, epoch2),
                    corepoints=corepoints,
                    cyl_radius=cyl_radius,
                    normal_radii=normal_radii)
distances, uncertainties = m3c2.run()
```

`distances` is a float64 array of shape (N,) — one value per corepoint.  
`uncertainties` is a dict with keys `lodetection`, `spread1`, `spread2`.

---

## 7. Subsampling

### Why subsample

Running M3C2 on every raw point in the reference cloud would be redundant — neighbouring raw points at 5 cm spacing give nearly identical results. Subsampling to a corepoint grid reduces computation time by the square of the spacing ratio without meaningful loss of spatial information.

### How it works — greedy KD-tree

```python
def subsample_by_distance(points, min_dist):
    tree = cKDTree(points)
    mask = np.ones(len(points), dtype=bool)  # all points initially available
    selected = []
    for i, p in enumerate(points):
        if mask[i]:                           # this point not yet rejected
            selected.append(i)
            neighbours = tree.query_ball_point(p, r=min_dist)
            mask[neighbours] = False          # reject everything within min_dist
    return points[selected]
```

The result is a set of corepoints where no two are closer than `min_dist`. The algorithm sweeps points in index order — it is deterministic and maximises spatial coverage.

### Use all points option

If **Use all points as corepoints** is ticked, subsampling is skipped and every reference cloud point becomes a corepoint. Useful for very small or sparse clouds. Not recommended for dense ALS data (millions of points).

---

## 8. Significance testing

A change is only flagged as significant if its absolute distance exceeds the Level of Detection:

```
SIGNIFICANT(p) = 1   if |d(p)| > LoD(p)
               = 0   otherwise
```

### Automatic sigma (default)

The tool reads SPREAD1 and SPREAD2 from py4dgeo and computes a single global LoD:

```python
sigma1 = mean(spread1)
sigma2 = mean(spread2)
detection_limit = t_critical × sqrt(sigma1² + sigma2²)
```

### Manual sigma

If you know the registration accuracy and instrument noise from your survey, you can enter sigma1 and sigma2 directly. This is more accurate than the automatic estimate for well-characterised instruments.

### Max depth filter

Before significance testing, distances greater than `max_depth` are set to NaN. This removes physically impossible values caused by shadows, flying points, or gross misregistration. Setting max depth to 0 disables the filter.

---

## 9. Normal vectors and displacement components

The tool computes a second set of normals independently of py4dgeo using PCA on k=10 nearest neighbours in the reference cloud. These are used for the DX/DY/DZ decomposition:

```
DX = d × nₓ        East component  [m]
DY = d × n_y       North component [m]
DZ = d × n_z       Vertical component [m]
```

Two orientation scalars:

```
Nz  = |n_z|              1.0 = horizontal surface, 0.0 = vertical wall
Nxy = √(nₓ² + n_y²)     complement of Nz;  Nz² + Nxy² = 1
```

These fields are written as extra dimensions to the output LAS/LAZ file and can be exported as GeoTIFFs. In QGIS, DX and DY can be visualised as arrow symbology using the GPKG quiver output.

---

## 10. Raster export

Interpolates any scalar field from corepoints to a regular GeoTIFF grid. Four interpolation methods are available:

### IDW — Inverse Distance Weighting

```
ẑ(x) = Σᵢ wᵢ zᵢ / Σᵢ wᵢ     where wᵢ = 1 / dᵢ²
```

Uses k=12 nearest neighbours. Exact at data points, smooth elsewhere. Best general-purpose choice. Can underestimate peaks and troughs because it averages.

### TIN — Triangulated Irregular Network

Linear interpolation on a Delaunay triangulation (`scipy.interpolate.griddata` with `method='linear'`). Preserves break lines. Cells outside the convex hull of corepoints are set to NaN.

### Nearest

Each grid cell gets the value of its nearest corepoint. No smoothing. Fastest method. Good for categorical fields like SIGNIFICANT.

### Kriging

Ordinary kriging with a linear variogram via `pykrige.OrdinaryKriging`. Gives the Best Linear Unbiased Predictor but memory usage is O(n²) — only practical for n < 50,000 corepoints.

### CRS tagging

The tool sets the coordinate reference system on the output GeoTIFF using the EPSG code you enter. It temporarily overrides the `PROJ_DATA` environment variable to point to pyproj's bundled `proj.db`, bypassing any stale PostgreSQL/PostGIS installation that may be on your system PATH:

```python
os.environ['PROJ_DATA'] = pyproj.datadir.get_data_dir()
crs = rasterio.crs.CRS.from_epsg(int(epsg))
# restored in finally block
```

---

## 11. DEM export

Interpolates the Z coordinate of corepoints (not M3C2 distances) to a GeoTIFF. Uses the same interpolation methods as the raster export. The result is a Digital Elevation Model of the reference cloud at the resolution you specify.

---

## 12. Quiver export

Aggregates corepoints into a regular grid and draws displacement arrows. Two modes:

### Normal-direction mode

The arrow at each grid cell points in the horizontal projection of the surface normal, scaled by the mean M3C2 distance in that cell:

```
u = (mean_DX / horizontal_magnitude) × mean|d|
v = (mean_DY / horizontal_magnitude) × mean|d|
```

The arrow direction shows surface orientation. On flat terrain where Nxy ≈ 0 the arrows are very short even when DZ is large.

### Motion vector mode

The arrow represents the actual horizontal displacement vector:

```
u = mean(DX)    for all corepoints in the cell
v = mean(DY)
```

This directly shows where the surface moved horizontally.

### Output formats

Each mode produces up to three files:

- **PNG** — static matplotlib figure with dark background, RdBu_r colour scale
- **HTML** — interactive Plotly figure; hover over arrows for exact values
- **GPKG** — GeoPackage LineString layer loadable in QGIS; use Arrow symbology on the geometry to render vectors

The grid cell size is set independently of the corepoint spacing. Cells with fewer than 3 contributing corepoints are discarded.

---

## 13. Profile cut export

Produces cross-section PNGs along user-defined lines through the point cloud.

### Defining profile lines

Two options:

1. **Interactive picker** — opens a hillshaded terrain view of the reference cloud. Click two points to define each line. Lines are shown on the terrain in real time.
2. **TXT file** — one vertex per line in Y X (Northing Easting) format, groups separated by blank lines.

Both sources can be combined — TXT lines are appended to any interactively picked lines.

### Hillshade computation

The terrain viewer uses the standard analytical hillshade formula:

```
HS = cos(z_sun)·cos(slope) + sin(z_sun)·sin(slope)·cos(az_sun − aspect)
```

with solar elevation 45° and azimuth 315° (NW light). Slope and aspect are derived from the Z-grid using Sobel gradients. The result is blended with the terrain colormap at a 40/60 ratio so colour is visible even in shadow.

### Profile projection

Each raw cloud point is projected onto the nearest foot on the polyline:

```
t = clamp( (p − A)·(B − A) / |B − A|², 0, 1 )
foot = A + t × (B − A)
along = cumulative_length_to_A + t × |B − A|
perp  = |p − foot|
```

Points with `perp > corridor_width / 2` are excluded.

### Output PNG layout

- **Upper panel** — reference cloud (blue) and comparison cloud (red) projected into the corridor. Up to 500,000 random points per cloud are used for speed.
- **Lower panel** — M3C2 distance vs distance along the profile. Grey = not significant. Coloured (RdBu_r) = significant. Dashed ±LoD lines with a shaded noise band.

---

## 14. Wind rose export

Four polar histogram PNGs are produced when wind rose export is enabled:

| File | Content |
|------|---------|
| `_windrose.png` | All valid points — grey = not significant, coloured by 3-D magnitude |
| `_windrose_sig.png` | All significant changes — all bars coloured by magnitude |
| `_windrose_pos.png` | Significant positive only (DZ > 0) — gain / deposition |
| `_windrose_neg.png` | Significant negative only (DZ < 0) — loss / erosion |

### Azimuth computation

DX and DY give the horizontal displacement direction. This is converted from mathematical angle (0°=East, counter-clockwise) to geographic bearing (0°=North, clockwise):

```
math_angle = arctan2(DY, DX)
bearing    = (90° − math_angle) mod 360°
```

Points are binned into N equal sectors (default 16, each 22.5°).

### Vertical angle

```
vert_angle = arctan2(|DZ|, √(DX² + DY²))    ∈ [0°, 90°]
```

0° = purely horizontal displacement, 90° = purely vertical. Displayed as a mirrored semicircle.

### Magnitude class boundaries

Class boundaries are computed from the significant-point magnitude distribution using quartiles (P25, P50, P75), rounded to 2 significant figures for clean legend labels. The minimum class starts at 0 for all-points mode and at the minimum significant magnitude for significant-only modes.

Colour scheme: green → yellow → orange → red by increasing magnitude.

---

## 15. Volume change estimation

Four methods are computed using two independent approaches. All results appear in the PDF report with percentage differences between them.

### Methods A and B — M3C2 surface-corrected

Each corepoint represents a patch of the actual surface, not a horizontal cell. The patch area is corrected for slope using the Nz component:

```
surface_area(p) = cell_size² / Nz(p)
```

where Nz = cos(slope). A horizontal surface (Nz=1) needs no correction. A 45° slope (Nz=0.707) has 41% more true surface area than its horizontal footprint. Surfaces steeper than ~84° (Nz < 0.10) are capped to avoid instability.

```
Method A:  V = Σ d(p) × cell_size² / Nz(p)    all valid points
Method B:  V = Σ d(p) × cell_size² / Nz(p)    significant only
```

### Methods C and D — DoD (Difference of DEMs)

Both raw clouds are read from the original LAZ files, thinned to one point per grid cell (keeping the highest Z per cell), then interpolated to a regular grid:

```
dZ[i,j] = Z₂[i,j] − Z₁[i,j]
Method C:  V = Σ dZ × cell_area    all non-NaN cells
Method D:  V = Σ dZ × cell_area    only cells where |dZ| > LoD
```

Method C is equivalent to CloudCompare's **Compute 2.5D Volume**. Method D applies the same noise threshold as M3C2 significance testing.

### Why the thinning step matters for speed

The raw clouds can have 2-3 million points each. At 0.5 m resolution the grid has ~500k cells. Without thinning, IDW would query 12 nearest neighbours from 3 million source points for every single grid cell — extremely slow. After thinning to one point per cell, the source point count drops to roughly the same as the cell count, making interpolation 5-10× faster with no meaningful accuracy loss.

### Percent difference

```
Δ% = |net_A − net_B| / ((|net_A| + |net_B|) / 2) × 100
```

A small Δ% means most volume change is real signal. A large Δ% between all-cells and significant-cells results indicates sub-LoD noise accumulation.

---

## 16. Distance histogram

A publication-ready histogram of the M3C2 distance distribution.

### Bin width — Freedman–Diaconis rule

```
bin_width = 2 × IQR × n^(−1/3)
```

where IQR is the interquartile range of the clipped distance distribution. This rule is robust to outliers and adapts to the actual data spread. The display range is clipped to the 0.5th–99.5th percentile and extended to at least ±2.5× LoD so the threshold lines are always visible.

### Layer composition

Three histogram layers in z-order:

1. **Grey** — all valid points (background)
2. **Blue** — significant negative distances (erosion/loss)
3. **Red** — significant positive distances (deposition/gain)

Dashed ±LoD lines and a shaded noise band are overlaid.

---

## 17. PDF report

An 11-section A4 PDF built with reportlab Platypus. All pages share a header and footer added by an `onPage` callback.

| Section | Content |
|---------|---------|
| Cover | Title, author, date, processing time, version, GitHub |
| 1. System info | OS, Python, CPU cores/frequency, RAM, GPU (via psutil + nvidia-smi) |
| 2. Input data | File names, sizes, point counts, XYZ extents for both clouds |
| 3. Parameters | All M3C2 and export settings including engine (py4dgeo/PDAL) |
| 4. Algorithm | Mathematical explanation with formulas and t-critical table |
| 5. Statistics | Overview counts, distance stats, magnitude distribution, normal components, surface change classification |
| 6. Histogram | Embedded `_histogram.png` |
| 7. Volume change | Four-method table with percentage differences |
| 8. Profile map | Top-down overview with north arrow and auto scale bar |
| 9. Profile cross-sections | Embedded profile PNGs |
| 10. Wind roses | All four wind rose PNGs |
| 11. Output files | Table of all output files with sizes and descriptions |

The profile overview map is generated fresh inside the report function, saved to a temp PNG, embedded, then deleted. It is not written as a standalone output file.

---

## 18. Session save and load

All GUI values can be saved to a JSON file and restored later. This means you never re-enter parameters for a locality you have processed before.

```
💾 Save session → saves current settings to a .json file
📂 Load session → restores all settings from a .json file
```

Suggested workflow: keep one JSON file per locality — `settings_kykula.json`, `settings_velka_causa.json`, etc.

When loading, if the session was saved with PDAL engine but PDAL is not installed on the current machine, the tool silently falls back to py4dgeo without error.

---

## 19. PDAL engine

An alternative computation engine for large datasets.

### When to use it

| Dataset size | Recommended engine |
|-------------|-------------------|
| < 500k corepoints | py4dgeo (default) |
| 500k – 5M corepoints | Either; PDAL significantly faster |
| > 5M corepoints | PDAL strongly recommended |

### How it works

1. Builds a PDAL JSON pipeline with `readers.las` → `filters.m3c2` → `writers.las`
2. Calls `pdal pipeline` as a subprocess — PDAL runs natively in C++ with OpenMP using all CPU cores
3. Reads the output LAZ file back into numpy
4. Runs a small py4dgeo sample (50,000 points) to estimate sigma1/sigma2 for the significance test, since PDAL does not return per-point spread values
5. Computes normals, significance, DX/DY/DZ — all downstream exports are identical to the py4dgeo path

### Installation

```bash
conda install -c conda-forge pdal python-pdal
```

The tool detects PDAL at startup by running `pdal --version`. If PDAL is not found, the PDAL radio button is greyed out and the version hint shows red install instructions.

---

## 20. Multithreading

A checkbox under the Guess Parameters button enables multithreading:

```
☑ Allow multithreading  (N threads, 1 left for OS)
```

Where N = `cpu_count − 1`. When unchecked, all operations run single-threaded.

### What is actually parallelised

| Operation | How |
|-----------|-----|
| Multiple raster field exports | `ThreadPoolExecutor` — each TIF in its own thread |
| Multiple profile PNGs | `ThreadPoolExecutor` — each PNG in its own thread |
| Volume DoD interpolation | 2 threads — z1_grid and z2_grid computed simultaneously |
| KD-tree queries (IDW, normals) | `workers=N` parameter in scipy's cKDTree.query |

### What cannot be parallelised from Python

The py4dgeo M3C2 `.run()` call is a single blocking C++ function. Python threads cannot distribute it across cores. This is why the M3C2 step runs at low CPU utilisation regardless of the thread setting. Use the PDAL engine for true multi-core M3C2.

---

## 21. Output files reference

| File | Description |
|------|-------------|
| `<stem>.laz` | Corepoint cloud with all M3C2 scalar fields |
| `<stem>_results.txt` | Tab-delimited text of all scalar fields (optional) |
| `<stem>_M3C2_DISTANCE.tif` | GeoTIFF — signed M3C2 distance |
| `<stem>_SIGNIFICANT.tif` | GeoTIFF — significance flag (1/0) |
| `<stem>_Nz.tif` | GeoTIFF — normal Z component |
| `<stem>_Nxy.tif` | GeoTIFF — normal XY magnitude |
| `<stem>_DX.tif` | GeoTIFF — East displacement component |
| `<stem>_DY.tif` | GeoTIFF — North displacement component |
| `<stem>_DZ.tif` | GeoTIFF — Vertical displacement component |
| `<stem>_LOD.tif` | GeoTIFF — Level of Detection per point |
| `<stem>_SPREAD1.tif` | GeoTIFF — Epoch 1 roughness |
| `<stem>_SPREAD2.tif` | GeoTIFF — Epoch 2 roughness |
| `<stem>_DEM.tif` | GeoTIFF — Digital Elevation Model from reference cloud |
| `<stem>_quiver_normal.png` | Quiver PNG — normal-direction arrows |
| `<stem>_quiver_normal.html` | Quiver interactive HTML — normal-direction |
| `<stem>_quiver_normal.gpkg` | Quiver GeoPackage — normal-direction arrows for QGIS |
| `<stem>_quiver_motion.png` | Quiver PNG — horizontal displacement vectors |
| `<stem>_quiver_motion.html` | Quiver interactive HTML — motion vectors |
| `<stem>_quiver_motion.gpkg` | Quiver GeoPackage — motion vectors for QGIS |
| `<stem>_profile_01.png` | Profile cross-section PNG (one per line) |
| `<stem>_profiles.gpkg` | Profile line geometries — loadable in QGIS |
| `<stem>_profiles.txt` | Profile line trajectories — reloadable into the tool |
| `<stem>_windrose.png` | Wind rose — all points |
| `<stem>_windrose_sig.png` | Wind rose — all significant changes |
| `<stem>_windrose_pos.png` | Wind rose — positive changes (gain/deposition) |
| `<stem>_windrose_neg.png` | Wind rose — negative changes (loss/erosion) |
| `<stem>_histogram.png` | M3C2 distance histogram |
| `<stem>_report.pdf` | Full PDF analysis report |

### LAS/LAZ scalar fields

| Field | Type | Description |
|-------|------|-------------|
| `M3C2_DISTANCE` | float32 | Signed distance along surface normal. Positive = surface moved away from reference. |
| `SIGNIFICANT` | int8 | 1 = change exceeds LoD, 0 = below noise threshold |
| `Nz` | float32 | Normal Z: 1.0 = horizontal surface, 0.0 = vertical wall |
| `Nxy` | float32 | Normal XY magnitude: complement of Nz |
| `DX` | float32 | East component of displacement in metres |
| `DY` | float32 | North component of displacement in metres |
| `DZ` | float32 | Vertical component of displacement in metres |
| `LOD` | float32 | Level of Detection from py4dgeo in metres |
| `SPREAD1` | float32 | Epoch 1 roughness inside the projection cylinder |
| `SPREAD2` | float32 | Epoch 2 roughness inside the projection cylinder |

---

## 22. Tested datasets

| Locality | Reference | Comparison | Corepoints | LoD |
|----------|-----------|------------|-----------|-----|
| Kykulá | MB1_vyrez.laz (2.06M pts) | MB2_vyrez.laz (3.36M pts) | 794,162 | 0.3577 m |
| Veľká Čausa | TLS epoch | UAV epoch | — | — |
| Staromestská ulica | — | — | — | — |

Test data coordinate system: S-JTSK / Krovak East North (EPSG:8353)

---

## 23. References

**Lague, D., Brodu, N., Leroux, J.** (2013). Accurate 3D comparison of complex topography with terrestrial laser scanner: Application to the Rangitikei canyon (N-Z). *ISPRS Journal of Photogrammetry and Remote Sensing*, 82, 10–26.  
https://doi.org/10.1016/j.isprsjprs.2013.04.009

**py4dgeo documentation** — https://py4dgeo.readthedocs.io/

**PDAL documentation** — https://pdal.io/

**rasterio documentation** — https://rasterio.readthedocs.io/

---

*M3C2 Tool · CHECKPOINT 13 · Samuel Lebó · GEOsys s.r.o.*
