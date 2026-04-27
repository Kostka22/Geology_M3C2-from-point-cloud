# Third-Party Licenses

This project uses the following open-source packages. Their licenses are reproduced below for compliance.

---

## py4dgeo

**License:** MIT  
**Source:** https://github.com/3dgeo-heidelberg/py4dgeo  
**Used for:** M3C2 distance computation (primary engine)

```
Copyright (c) 2021 3DGeo Research Group, Heidelberg University
MIT License — see https://github.com/3dgeo-heidelberg/py4dgeo/blob/main/LICENSE
```

---

## laspy

**License:** BSD 2-Clause  
**Source:** https://github.com/laspy/laspy  
**Used for:** LAS/LAZ file reading and writing

---

## NumPy

**License:** BSD 3-Clause  
**Source:** https://numpy.org  
**Used for:** Array operations throughout

---

## SciPy

**License:** BSD 3-Clause  
**Source:** https://scipy.org  
**Used for:** Spatial indexing (cKDTree), interpolation (griddata, NearestNDInterpolator), statistics

---

## Matplotlib

**License:** PSF-based (BSD-compatible)  
**Source:** https://matplotlib.org  
**Used for:** Profile plots, wind roses, histograms, quiver maps

---

## rasterio

**License:** BSD 3-Clause  
**Source:** https://github.com/rasterio/rasterio  
**Used for:** GeoTIFF raster and DEM export

---

## pyproj

**License:** MIT  
**Source:** https://github.com/pyproj4/pyproj  
**Used for:** CRS assignment on exported rasters

---

## GeoPandas

**License:** BSD 3-Clause  
**Source:** https://geopandas.org  
**Used for:** GeoPackage export of quiver vectors and profile lines

---

## Shapely

**License:** BSD 3-Clause  
**Source:** https://shapely.readthedocs.io  
**Used for:** Geometry construction for GeoPackage export

---

## ReportLab

**License:** BSD  
**Source:** https://www.reportlab.com/opensource/  
**Used for:** PDF report generation

---

## openpyxl

**License:** MIT  
**Source:** https://openpyxl.readthedocs.io  
**Used for:** Excel batch stats auto-populate (`batch_stats_analyza.xlsx`)

---

## PyKrige

**License:** BSD 3-Clause  
**Source:** https://github.com/GeoStat-Framework/PyKrige  
**Used for:** Ordinary kriging interpolation (optional raster method)

---

## psutil

**License:** BSD 3-Clause  
**Source:** https://github.com/giampaolo/psutil  
**Used for:** System memory info in PDF reports

---

## PDAL (optional)

**License:** BSD  
**Source:** https://pdal.io  
**Used for:** Alternative M3C2 computation engine (optional, CLI-based)

---

*All packages are used unmodified as external dependencies. No source code from these packages is included in this repository.*
