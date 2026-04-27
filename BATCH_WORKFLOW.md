# Batch Processing & Parameter Study Workflow

This guide describes the full workflow for running M3C2 parameter sensitivity studies using the **Batch** feature introduced in CHECKPOINT 14.

---

## What Is Batch Processing?

The batch feature lets you run multiple M3C2 configurations unattended — for example, testing different cylinder diameters `d` or normal radii across the same pair of point clouds. Results from all runs are aggregated into:

- `batch_stats.csv` — one row per run, appended automatically
- `batch_stats_analyza.xlsx` — auto-populated from the Excel template, with charts and knee-point analysis

---

## Step-by-Step

### Step 1 — Set Up the Excel Template

Place `m3c2_batch_analyza_template.xlsx` **in the same folder** as `m3c2_checkpoint14.py`. This is required for the Excel auto-populate feature to work.

```
m3c2_checkpoint14.py
m3c2_batch_analyza_template.xlsx   ← same folder!
```

If the template is missing, the tool will still write `batch_stats.csv` — only the Excel step is skipped.

---

### Step 2 — Enable Batch CSV Export

In the **Output Options** section, tick:

> ☑ Export batch stats (batch_stats.csv + batch_stats_analyza.xlsx)

This checkbox is saved into each JSON session, so it applies to every batch job.

---

### Step 3 — Configure and Save Sessions

For each parameter variant you want to test:

1. Set all parameters in the GUI (input files, `d`, normal radii, etc.)
2. Click **Save Session** → save as a descriptive name, e.g.:
   - `kykula_d0.5.json`
   - `kykula_d1.0.json`
   - `kykula_d2.0.json`
   - `kykula_d4.0.json`

> **Tip:** Keep the input LAS files and output folder consistent across sessions — only vary the M3C2 parameter(s) you are studying.

---

### Step 4 — Run Batch

1. Click the **Batch** button
2. Select all the JSON session files you saved
3. Confirm the job list in the dialog
4. The tool processes all jobs sequentially in a background thread

For each job, the tool:
- Loads settings from the JSON
- Creates a subfolder named after the JSON file (e.g. `kykula_d0.5/`) next to the JSON
- Runs the full M3C2 pipeline
- Appends one row to `batch_stats.csv` in the folder of the first JSON file
- Regenerates `batch_stats_analyza.xlsx` from the template after each run
- Copies the JSON into the output subfolder for traceability

---

### Step 5 — Review Results

After all jobs complete:

```
/your/session/folder/
├── kykula_d0.5/
│   ├── kykula_d0.5.json     ← copied for traceability
│   ├── vystup.laz
│   ├── vystup_report.pdf
│   └── ...
├── kykula_d1.0/
│   └── ...
├── batch_stats.csv           ← aggregated stats (all runs)
└── batch_stats_analyza.xlsx  ← auto-populated Excel workbook
```

Open `batch_stats_analyza.xlsx` and navigate to the sheets:

| Sheet | Content |
|---|---|
| **RAW_DATA** | Raw imported stats — one row per run |
| **Grafy** | Charts: significant % vs `d`, LoD vs `d`, etc. |
| **Analyza** | Knee-point analysis: recommended `d` based on σ₁ vs reg_error |
| **Navod** | Usage instructions (Slovak) |

---

## Excel Template — Sheet Details

### RAW_DATA

Data is written here automatically by the tool. Rows start at row 4 (rows 1–3 are the header). The template supports up to **100 runs** (rows 4–103). If you have more, the tool will warn and truncate.

Columns match `batch_stats.csv` exactly — see [README.md](README.md#batch-stats-csv--column-reference) for the full column list.

### Grafy (Charts)

Charts are linked to `RAW_DATA!C4:C53` (the `d` column) and corresponding metric columns. Charts update automatically when data is present. If you run more than 50 sessions, extend the chart data ranges manually.

### Analyza (Knee-Point Analysis)

Set the number of runs in cell `B4` to match how many rows you have in RAW_DATA. The analysis computes:

- **Knee point** — the first `d` value where σ₁ exceeds `reg_error`
- Recommended `d`, σ₁, σ₂, LoD₉₅, Sig.%, Valid CP at that point

> **Note:** The summary formulas in `C34:C39` are array formulas. If Excel shows `#VALUE!`, confirm them with **Ctrl+Shift+Enter** instead of just Enter.

---

## Manual CSV Import (Fallback)

If the Excel auto-populate fails (e.g. template not found, openpyxl not installed), you can populate manually:

1. Open `batch_stats_analyza.xlsx`
2. Go to sheet **RAW_DATA**
3. Open `batch_stats.csv` in Notepad or Excel
4. Copy all data rows (skip the header if it's already in RAW_DATA row 3)
5. Paste into cell **A4** using **Paste Special → Values only** (Ctrl+Shift+V → Values)
6. Go to **Grafy** — charts update automatically

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `batch_stats_analyza.xlsx` not generated | Check that `m3c2_batch_analyza_template.xlsx` is in the same folder as the `.py` file |
| Excel open error | Close `batch_stats_analyza.xlsx` in Excel/LibreOffice before running batch |
| `openpyxl` not installed | `pip install openpyxl` — CSV is still written without it |
| Batch job fails mid-run | The tool logs the error and continues with remaining jobs. Check the console output. |
| Charts show no data | Verify row count in Analyza sheet matches actual data rows in RAW_DATA |

---

## Example: d-Parameter Study

Recommended session naming and `d` values for a typical sensitivity study:

| JSON filename | d [m] | Purpose |
|---|---|---|
| `study_d0.5.json` | 0.5 | Very fine — may produce many NaNs |
| `study_d1.0.json` | 1.0 | Fine |
| `study_d2.0.json` | 2.0 | Medium — often optimal for national ALS |
| `study_d3.0.json` | 3.0 | Coarse |
| `study_d4.0.json` | 4.0 | Very coarse — resolves NaN areas from small d |
| `study_d6.0.json` | 6.0 | Over-smoothed |

After the batch run, the knee-point analysis in `batch_stats_analyza.xlsx` will identify the optimal `d` as the first value where σ₁ > `reg_error`.
