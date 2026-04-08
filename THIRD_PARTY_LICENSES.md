# Third-Party Licenses

This project uses the following open-source libraries.
All are distributed under permissive MIT or BSD licenses.
No GPL or copyleft licenses are present.

---

## py4dgeo

**License:** MIT  
**Copyright:** py4dgeo Development Core Team, 3DGeo Research Group Heidelberg  
**Source:** https://github.com/3dgeo-heidelberg/py4dgeo  
**Used for:** M3C2 core algorithm, KD-tree construction, LAS/LAZ I/O

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.

---

## laspy

**License:** BSD-2-Clause  
**Copyright:** Grant Brown and contributors  
**Source:** https://github.com/laspy/laspy  
**Used for:** LAS/LAZ file reading and writing for volume estimation and cloud info

> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
> 1. Redistributions of source code must retain the above copyright notice,
>    this list of conditions and the following disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright notice,
>    this list of conditions and the following disclaimer in the documentation
>    and/or other materials provided with the distribution.

---

## NumPy

**License:** BSD-3-Clause  
**Copyright:** NumPy Developers  
**Source:** https://github.com/numpy/numpy  
**Used for:** All array computation throughout the pipeline

> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
> 1. Redistributions of source code must retain the above copyright notice,
>    this list of conditions and the following disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright notice,
>    this list of conditions and the following disclaimer in the documentation
>    and/or other materials provided with the distribution.
> 3. Neither the name of the copyright holder nor the names of its contributors
>    may be used to endorse or promote products derived from this software
>    without specific prior written permission.

---

## SciPy

**License:** BSD-3-Clause  
**Copyright:** SciPy Developers  
**Source:** https://github.com/scipy/scipy  
**Used for:** KD-tree queries (cKDTree), grid interpolation (griddata),
statistical functions (stats.norm.ppf)

Same BSD-3-Clause terms as NumPy above.

---

## Matplotlib

**License:** BSD-style (PSF-compatible)  
**Copyright:** Matplotlib Development Team  
**Source:** https://github.com/matplotlib/matplotlib  
**Used for:** Profile PNGs, quiver PNGs, wind rose PNGs, histogram PNG,
profile overview map PNG

> Matplotlib only uses BSD compatible code, and its license is based on the
> PSF license. See the full license at:
> https://matplotlib.org/stable/devel/license.html

---

## Plotly

**License:** MIT  
**Copyright:** 2016–2024 Plotly Technologies Inc.  
**Source:** https://github.com/plotly/plotly.py  
**Used for:** Interactive quiver HTML export

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.

---

## GeoPandas

**License:** BSD-3-Clause  
**Copyright:** GeoPandas Developers  
**Source:** https://github.com/geopandas/geopandas  
**Used for:** GPKG export of quiver arrows and profile lines, clip mask loading

Same BSD-3-Clause terms as NumPy above.

---

## Shapely

**License:** BSD-3-Clause  
**Copyright:** Sean Gillies and contributors  
**Source:** https://github.com/shapely/shapely  
**Used for:** LineString geometry creation for GPKG exports

Same BSD-3-Clause terms as NumPy above.

---

## Rasterio

**License:** BSD-3-Clause  
**Copyright:** Mapbox and contributors  
**Source:** https://github.com/rasterio/rasterio  
**Used for:** GeoTIFF read/write, CRS tagging, clip by vector mask

Same BSD-3-Clause terms as NumPy above.

---

## pyproj

**License:** MIT  
**Copyright:** Jeff Whitaker and contributors  
**Source:** https://github.com/pyproj4/pyproj  
**Used for:** Bundled proj.db for EPSG coordinate reference system lookup;
avoids conflicts with stale PostgreSQL/PostGIS PROJ installations

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction.

---

## pykrige

**License:** BSD-3-Clause  
**Copyright:** Benjamin Murphy and contributors  
**Source:** https://github.com/GeoStat-Framework/pykrige  
**Used for:** Ordinary Kriging interpolation (optional, only when Kriging
method is selected for raster export)

Same BSD-3-Clause terms as NumPy above.

---

## psutil

**License:** BSD-3-Clause  
**Copyright:** Giampaolo Rodolà and contributors  
**Source:** https://github.com/giampaolo/psutil  
**Used for:** CPU core count, CPU frequency, RAM total/available — reported
in the PDF analysis report system info section

Same BSD-3-Clause terms as NumPy above.

---

## ReportLab

**License:** BSD  
**Copyright:** 2000–2025 ReportLab Inc.  
**Source:** https://www.reportlab.com / https://pypi.org/project/reportlab  
**Used for:** PDF report generation (Platypus framework)

> BSD license — see license.txt for full details.
> Copyright (c) 2000–2025, ReportLab Inc.
> All rights reserved.
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the BSD license conditions are met.

---

## PDAL (optional)

**License:** BSD  
**Copyright:** PDAL Contributors  
**Source:** https://github.com/PDAL/PDAL  
**Used for:** Optional alternative computation engine for the M3C2 algorithm;
provides OpenMP multi-core parallelism for large datasets (>1M corepoints).
PDAL is called as an external subprocess — it is not bundled with this tool
and must be installed separately via conda.

> PDAL is licensed under the BSD license.
> See https://pdal.io/en/stable/copyright.html for full details.

---

## Summary

| Library | License | Commercial use | Modify | Distribute | Copyleft |
|---------|---------|---------------|--------|------------|----------|
| py4dgeo | MIT | ✓ | ✓ | ✓ | ✗ |
| laspy | BSD-2 | ✓ | ✓ | ✓ | ✗ |
| numpy | BSD-3 | ✓ | ✓ | ✓ | ✗ |
| scipy | BSD-3 | ✓ | ✓ | ✓ | ✗ |
| matplotlib | BSD | ✓ | ✓ | ✓ | ✗ |
| plotly | MIT | ✓ | ✓ | ✓ | ✗ |
| geopandas | BSD-3 | ✓ | ✓ | ✓ | ✗ |
| shapely | BSD-3 | ✓ | ✓ | ✓ | ✗ |
| rasterio | BSD-3 | ✓ | ✓ | ✓ | ✗ |
| pyproj | MIT | ✓ | ✓ | ✓ | ✗ |
| pykrige | BSD-3 | ✓ | ✓ | ✓ | ✗ |
| psutil | BSD-3 | ✓ | ✓ | ✓ | ✗ |
| reportlab | BSD | ✓ | ✓ | ✓ | ✗ |
| PDAL | BSD | ✓ | ✓ | ✓ | ✗ |

No library imposes any copyleft (GPL-style) requirements.
This tool may therefore be distributed under the MIT License without restriction.
