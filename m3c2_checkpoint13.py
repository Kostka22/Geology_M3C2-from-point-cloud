import os
import numpy as np
import py4dgeo
import laspy
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from scipy.spatial import cKDTree
from scipy import stats
import threading

# ── Lock matplotlib to the non-interactive Agg backend BEFORE anything
#    else can import it.  If matplotlib is initialised with TkAgg first
#    (which happens when ProfilePicker imports FigureCanvasTkAgg), any
#    subsequent matplotlib.use("Agg") call in the worker thread silently
#    fails and all figure creation in that thread still uses TkAgg —
#    allocating tk.Variable / tk.PhotoImage objects in the wrong thread,
#    which causes the Tcl_AsyncDelete crash on exit.
#    Setting it here, at import time, before *any* matplotlib symbol is
#    touched, guarantees worker-thread figures always use Agg.
import matplotlib
matplotlib.use("Agg")          # must be before any other matplotlib import
import matplotlib.pyplot as _mpl_plt   # pre-initialise in main thread
import matplotlib.colors as _mpl_colors
import matplotlib.cm as _mpl_cm
del _mpl_plt, _mpl_colors, _mpl_cm    # don't pollute the namespace

# ── Thread-safe UI helpers ────────────────────────────────────────────
# Pipeline functions run in a worker thread; all tkinter calls must be
# marshalled back to the main thread via root.after().
_root_ref = []   # filled in after root = tk.Tk()

def _ui_error(title, msg):
    """Show an error dialog safely from any thread."""
    if _root_ref:
        _root_ref[0].after(0, lambda t=title, m=msg: messagebox.showerror(t, m))
    else:
        messagebox.showerror(title, msg)

def _ui_info(title, msg):
    """Show an info dialog safely from any thread."""
    if _root_ref:
        _root_ref[0].after(0, lambda t=title, m=msg: messagebox.showinfo(t, m))
    else:
        messagebox.showinfo(title, msg)

# -----------------------------
# Progress Bar Manager
# -----------------------------
class ProgressManager:
    """
    All public methods are safe to call from any thread.
    Widget updates are marshalled to the main thread via root.after().
    """
    def __init__(self, current_bar, current_label, overall_bar, overall_label):
        self.current_bar   = current_bar
        self.current_label = current_label
        self.overall_bar   = overall_bar
        self.overall_label = overall_label
        self.total_steps   = 9
        self.current_step  = 0

    def _schedule(self, fn):
        """Run fn() on the main thread. Safe to call from any thread."""
        if _root_ref:
            _root_ref[0].after(0, fn)
        else:
            fn()   # fallback: already on main thread (startup)

    def update_current(self, value, text):
        v, t = value, text
        def _do():
            self.current_bar['value'] = v
            self.current_label.config(text=t)
        self._schedule(_do)

    def update_overall(self, step_name):
        self.current_step += 1
        pct  = (self.current_step / self.total_steps) * 100
        step = self.current_step
        name = step_name
        total = self.total_steps
        def _do():
            self.overall_bar['value'] = pct
            self.overall_label.config(
                text=f"Overall Progress: {name} ({step}/{total})")
        self._schedule(_do)

    def reset(self):
        self.current_step = 0
        def _do():
            self.current_bar['value']  = 0
            self.overall_bar['value']  = 0
            self.current_label.config(text="Ready")
            self.overall_label.config(text="Overall Progress: Ready")
        self._schedule(_do)




# ═══════════════════════════════════════════════════════════════════════
# Interactive Profile Picker  —  hillshaded terrain raster + profile lines
# ═══════════════════════════════════════════════════════════════════════
class ProfilePicker:
    """
    Bins the reference cloud to a Z-grid, computes an analytical hillshade
    (light from NW, 45°), blends it with the terrain colormap, and displays
    the result as an imshow — identical to what QGIS does with a DEM.
    Edges and ridges are immediately visible.

    Slider controls grid resolution (coarser = faster, finer = more detail).
    Left-click: first point → second point → line saved.
    Done button always visible (packed before canvas).
    """

    def __init__(self, parent, las_file1, las_file2, existing_lines=None):
        self.parent     = parent
        self.las_file1  = las_file1
        self.lines      = list(existing_lines or [])
        self.pending_pt = None
        self.pts_xy     = None   # raw (N,2) XY — kept for resolution rebin
        self.pts_z      = None   # raw (N,)  Z
        self._pa        = None   # pending-point artist
        self._res_var   = tk.DoubleVar(value=1.0)   # grid resolution [m]
        self._img_cache = {}     # res → (rgba, extent)

        # ── Window ──────────────────────────────────────────────────
        self.win = tk.Toplevel(parent)
        self.win.title(
            "Profile Picker  —  hillshaded terrain  |  click two points per line")
        self.win.geometry("1200x880")
        self.win.minsize(800, 640)
        self.win.grab_set()
        self.win.protocol("WM_DELETE_WINDOW", self._cancel)

        # ── Status bar (top) ────────────────────────────────────────
        self.status_var = tk.StringVar(value="Loading reference cloud…")
        tk.Label(self.win, textvariable=self.status_var,
                 fg="#1565c0", font=("Arial", 9, "bold"), anchor="w",
                 bg="#e8f0fb", relief="flat", padx=8, pady=4
                 ).pack(side="top", fill="x")

        # ══ Bottom bar packed FIRST — guaranteed never obscured ══
        bot = tk.Frame(self.win, bg="#f0f0f0", bd=1, relief="raised", pady=5)
        bot.pack(side="bottom", fill="x")

        # Right: Done  Cancel
        self.btn_done = tk.Button(
            bot, text="✓   Done — use these lines",
            command=self._done, state="disabled",
            font=("Arial", 10, "bold"), relief="flat",
            bg="#2e7d32", fg="white", padx=18, pady=7,
            activebackground="#1b5e20", activeforeground="white",
            cursor="hand2")
        self.btn_done.pack(side="right", padx=12, pady=2)

        tk.Button(bot, text="✕  Cancel", command=self._cancel,
                  font=("Arial", 9), relief="flat",
                  bg="#e0e0e0", padx=10, pady=5
                  ).pack(side="right", padx=4, pady=2)

        # Left: count  undo  clear  |  resolution slider
        self.lbl_count = tk.Label(bot, text="Lines: 0",
                                   font=("Arial", 9, "bold"), fg="#333333",
                                   bg="#f0f0f0")
        self.lbl_count.pack(side="left", padx=(12, 8), pady=2)

        self.btn_undo = tk.Button(bot, text="↩ Undo",
                                   command=self._undo, state="disabled",
                                   font=("Arial", 9), relief="flat",
                                   bg="#e0e0e0", padx=7, pady=5)
        self.btn_undo.pack(side="left", padx=3, pady=2)

        self.btn_clear = tk.Button(bot, text="✕ Clear all",
                                    command=self._clear, state="disabled",
                                    font=("Arial", 9), relief="flat",
                                    bg="#e0e0e0", padx=7, pady=5)
        self.btn_clear.pack(side="left", padx=3, pady=2)

        tk.Frame(bot, width=2, bg="#cccccc").pack(
            side="left", fill="y", padx=10, pady=4)

        tk.Label(bot, text="Grid res [m]:", font=("Arial", 9),
                 bg="#f0f0f0", fg="#444444").pack(side="left", padx=(0, 4))
        self.lbl_res = tk.Label(bot, text="1.0 m",
                                 font=("Arial", 9, "bold"),
                                 bg="#f0f0f0", fg="#1565c0", width=6)
        self.lbl_res.pack(side="left", padx=(0, 4))
        tk.Scale(bot, from_=0.25, to=5.0, resolution=0.25,
                 orient="horizontal", length=180, showvalue=False,
                 variable=self._res_var, command=self._on_res_change,
                 bg="#f0f0f0", highlightthickness=0,
                 troughcolor="#cccccc", activebackground="#1565c0"
                 ).pack(side="left", pady=2)
        tk.Label(bot, text="← finer    coarser →",
                 font=("Arial", 7), fg="#888888", bg="#f0f0f0"
                 ).pack(side="left", padx=(4, 0))

        # ── Canvas + toolbar fill remaining space ────────────────────
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
        import matplotlib.pyplot as plt
        self._plt = plt

        cf = tk.Frame(self.win)
        cf.pack(side="top", fill="both", expand=True, padx=4, pady=(2, 0))

        self.fig, self.ax = plt.subplots(figsize=(11, 7.5))
        self.fig.patch.set_facecolor('#1a1a1a')
        self.ax.set_facecolor('#111111')
        self.ax.set_aspect('equal')
        self._style_ax()

        self.canvas = FigureCanvasTkAgg(self.fig, master=cf)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        tbf = tk.Frame(cf, bg="#2a2a2a")
        tbf.pack(fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas, tbf)
        self.toolbar.update()

        self._cid   = self.canvas.mpl_connect('button_press_event', self._on_click)
        self.result = None

        threading.Thread(target=self._load_cloud, daemon=True).start()

    # ── Core raster computation ─────────────────────────────────────

    @staticmethod
    def _hillshade_rgba(x, y, z, res, azimuth=315.0, altitude=45.0):
        """
        Bin XYZ to a grid at *res* metres, compute analytical hillshade,
        blend with terrain colormap, return (rgba uint8, extent).

        Algorithm (identical to QGIS / GDAL hillshade):
          1. Build mean-Z grid
          2. np.gradient(z_grid, res) → dz/dx, dz/dy
          3. Compute slope and aspect
          4. hs = sin(alt)*cos(slope) + cos(alt)*sin(slope)*cos(az - aspect)
          5. Multiply terrain-colourmap RGB by hillshade factor
        """
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        nx = max(2, int(np.ceil((x_max - x_min) / res)) + 1)
        ny = max(2, int(np.ceil((y_max - y_min) / res)) + 1)

        xi = np.clip(((x - x_min) / res).astype(int), 0, nx - 1)
        yi = np.clip(((y - y_min) / res).astype(int), 0, ny - 1)

        z_sum   = np.zeros((ny, nx), dtype=np.float64)
        z_cnt   = np.zeros((ny, nx), dtype=np.int32)
        np.add.at(z_sum, (yi, xi), z)
        np.add.at(z_cnt, (yi, xi), 1)

        valid   = z_cnt > 0
        z_grid  = np.where(valid, z_sum / np.where(z_cnt > 0, z_cnt, 1),
                           np.nan).astype(np.float32)

        # Fill NaN holes with nearest valid neighbour (simple 2-pass)
        if np.isnan(z_grid).any():
            from scipy.ndimage import distance_transform_edt
            nan_mask = np.isnan(z_grid)
            idx = distance_transform_edt(nan_mask,
                                         return_distances=False,
                                         return_indices=True)
            z_grid = z_grid[idx[0], idx[1]]

        # ── Hillshade ───────────────────────────────────────────────
        dzdx, dzdy = np.gradient(z_grid, res, res)
        slope  = np.arctan(np.hypot(dzdx, dzdy))
        aspect = np.arctan2(-dzdy, dzdx)   # GDAL convention

        az_rad  = np.radians(360.0 - azimuth + 90.0)
        alt_rad = np.radians(altitude)

        hs = (np.sin(alt_rad) * np.cos(slope)
              + np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
        hs = np.clip(hs, 0.0, 1.0)

        # ── Terrain colour × hillshade ───────────────────────────────
        z_lo = float(np.percentile(z_grid, 2))
        z_hi = float(np.percentile(z_grid, 98))
        if z_hi <= z_lo:
            z_hi = z_lo + 1.0

        norm     = mcolors.Normalize(vmin=z_lo, vmax=z_hi)
        cmap     = plt.cm.terrain
        rgb      = cmap(norm(z_grid))[:, :, :3].astype(np.float32)

        # Blend: darken terrain by hillshade (0=dark, 1=full colour)
        # blend = terrain * (0.4 + 0.6 * hs)  — keeps colour even in shadow
        blend_hs = (0.4 + 0.6 * hs)[:, :, np.newaxis]
        rgb_out  = np.clip(rgb * blend_hs, 0.0, 1.0)

        alpha    = valid.astype(np.float32)
        rgba     = (np.dstack([rgb_out, alpha[:, :, np.newaxis]])
                    * 255).astype(np.uint8)

        extent = (x_min - res * 0.5, x_max + res * 0.5,
                  y_min - res * 0.5, y_max + res * 0.5)
        return rgba, extent, z_lo, z_hi

    # ── Load / plot ─────────────────────────────────────────────────

    def _load_cloud(self):
        try:
            self.status_var.set("Loading reference cloud…")
            las   = laspy.read(self.las_file1)
            x_all = np.asarray(las.x, dtype=np.float32)
            y_all = np.asarray(las.y, dtype=np.float32)
            z_all = np.asarray(las.z, dtype=np.float32)
            n = len(x_all)
            self.status_var.set(
                f"Building hillshade raster  ({n:,} points)…")
            self.pts_xy = np.column_stack([x_all, y_all])
            self.pts_z  = z_all
            self.win.after(0, self._plot_cloud)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.win.after(0, lambda: self.status_var.set(f"Error: {e}"))

    def _get_raster(self, res):
        """Return cached (rgba, extent, z_lo, z_hi) for this resolution."""
        if res not in self._img_cache:
            self.status_var.set(f"Rasterising at {res} m…")
            result = self._hillshade_rgba(
                self.pts_xy[:, 0], self.pts_xy[:, 1], self.pts_z, res)
            self._img_cache[res] = result
        return self._img_cache[res]

    def _plot_cloud(self, keep_view=False):
        """Full redraw. If keep_view=True, restores zoom level after draw."""
        if self.pts_xy is None:
            return

        try:
            xlim = self.ax.get_xlim() if keep_view else None
            ylim = self.ax.get_ylim() if keep_view else None
            had  = keep_view and self.ax.has_data()
        except Exception:
            had = False

        res = float(self._res_var.get())
        rgba, extent, z_lo, z_hi = self._get_raster(res)

        self.ax.cla()
        self._style_ax()
        self.ax.set_facecolor('#111111')

        # Main hillshaded image
        self.ax.imshow(rgba, extent=extent, origin='lower',
                       aspect='equal', interpolation='bilinear',
                       zorder=2)

        # Invisible scatter just for a mappable → colorbar
        import matplotlib.colors as mcolors
        import matplotlib.cm as mcm
        norm = mcolors.Normalize(vmin=z_lo, vmax=z_hi)
        sm   = mcm.ScalarMappable(cmap='terrain', norm=norm)
        sm.set_array([])
        for _a in self.fig.get_axes()[1:]:
            _a.remove()
        cb = self.fig.colorbar(sm, ax=self.ax,
                               fraction=0.025, pad=0.01, aspect=35)
        cb.set_label("Z  [m]", color='#aaaaaa', fontsize=8)
        cb.ax.tick_params(colors='#aaaaaa', labelsize=7)
        cb.outline.set_edgecolor('#555555')

        if had and xlim and xlim[0] != xlim[1]:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)

        for i, ln in enumerate(self.lines):
            self._draw_line_on_ax(ln, i + 1)

        self.canvas.draw()
        ny, nx = rgba.shape[:2]
        self.status_var.set(
            f"Hillshade  {nx}×{ny} cells @ {res} m  "
            "·  Left-click: pt 1 → pt 2 = line  "
            "·  Deactivate pan/zoom before clicking")
        self._refresh_buttons()

    def _on_res_change(self, _=None):
        res = float(self._res_var.get())
        self.lbl_res.config(text=f"{res:.2f} m")
        if self.pts_xy is not None:
            # Recompute in background, then redraw
            def _worker():
                self._get_raster(res)
                self.win.after(0, lambda: self._plot_cloud(keep_view=True))
            threading.Thread(target=_worker, daemon=True).start()

    # ── Axis style ──────────────────────────────────────────────────

    def _style_ax(self):
        self.ax.tick_params(colors='#aaaaaa', labelsize=8)
        self.ax.set_xlabel("X  [m]", color='#aaaaaa', fontsize=9)
        self.ax.set_ylabel("Y  [m]", color='#aaaaaa', fontsize=9)
        for sp in self.ax.spines.values():
            sp.set_edgecolor('#555555')

    # ── Line drawing ────────────────────────────────────────────────

    def _draw_line_on_ax(self, line, idx):
        x0, y0 = float(line[0, 0]), float(line[0, 1])
        x1, y1 = float(line[1, 0]), float(line[1, 1])
        length  = float(np.linalg.norm(line[1] - line[0]))
        self.ax.plot([x0, x1], [y0, y1], '-',
                     color='#FF4400', linewidth=2.4, zorder=9,
                     solid_capstyle='round')
        self.ax.plot([x0, x1], [y0, y1], 'o',
                     color='#FF4400', markersize=7, zorder=10,
                     markeredgecolor='white', markeredgewidth=0.8)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        self.ax.annotate(
            f' P{idx}  {length:.1f} m',
            xy=(mx, my), fontsize=9, color='white', fontweight='bold',
            zorder=11,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#CC3300',
                      alpha=0.88, edgecolor='none'))

    # ── Click handler ───────────────────────────────────────────────

    def _on_click(self, event):
        if self.toolbar and self.toolbar.mode:
            return
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self.pts_xy is None:
            return

        x, y = event.xdata, event.ydata

        if self.pending_pt is None:
            self.pending_pt = (x, y)
            self._pa = self.ax.plot(
                x, y, 'o', color='white', markersize=10, zorder=12,
                markeredgecolor='#FF4400', markeredgewidth=2.0)[0]
            self.canvas.draw_idle()
            self.status_var.set(
                f"Point 1: ({x:.2f}, {y:.2f})  —  now click second point")
        else:
            x0, y0 = self.pending_pt
            ln = np.array([[x0, y0], [x, y]], dtype=np.float64)
            self.lines.append(ln)
            if self._pa is not None:
                self._pa.remove()
                self._pa = None
            self.pending_pt = None
            self._draw_line_on_ax(ln, len(self.lines))
            self.canvas.draw_idle()
            length = float(np.linalg.norm(ln[1] - ln[0]))
            self.status_var.set(
                f"Line {len(self.lines)} added  ({length:.1f} m)  "
                "—  click to add another,  or click  ✓ Done")
            self._refresh_buttons()

    # ── Buttons ─────────────────────────────────────────────────────

    def _refresh_buttons(self):
        n  = len(self.lines)
        st = "normal" if n > 0 else "disabled"
        self.lbl_count.config(text=f"Lines: {n}")
        self.btn_undo.config(state=st)
        self.btn_clear.config(state=st)
        self.btn_done.config(state=st)

    def _undo(self):
        if not self.lines:
            return
        if self.pending_pt is not None:
            if self._pa:
                self._pa.remove()
                self._pa = None
            self.pending_pt = None
        self.lines.pop()
        self._plot_cloud(keep_view=True)

    def _clear(self):
        if not messagebox.askyesno("Clear all",
                "Remove all defined profile lines?", parent=self.win):
            return
        self.lines.clear()
        self.pending_pt = None
        self._pa = None
        self._plot_cloud(keep_view=True)

    def _done(self):
        if self.pending_pt is not None:
            if not messagebox.askyesno(
                    "Unfinished line",
                    "First point set but no second point yet.\n"
                    "Discard it and finish?", parent=self.win):
                return
        self.result = list(self.lines)
        self.canvas.mpl_disconnect(self._cid)
        self._plt.close(self.fig)
        self.win.grab_release()
        self.win.destroy()

    def _cancel(self):
        self.result = None
        self.canvas.mpl_disconnect(self._cid)
        self._plt.close(self.fig)
        self.win.grab_release()
        self.win.destroy()

    def wait(self):
        self.parent.wait_window(self.win)
        return self.result


# -----------------------------
# Subsampling
# -----------------------------
def subsample_by_distance(points, min_dist, progress_mgr=None):
    if progress_mgr:
        progress_mgr.update_current(0, "Subsampling: Building KD-Tree...")
    selected = []
    tree = cKDTree(points)
    mask = np.ones(len(points), dtype=bool)
    total_points = len(points)
    for i, p in enumerate(points):
        if mask[i]:
            selected.append(i)
            idx = tree.query_ball_point(p, r=min_dist)
            mask[idx] = False
        if progress_mgr and i % 1000 == 0:
            progress_mgr.update_current((i / total_points) * 100,
                                        f"Subsampling: {i}/{total_points} points processed")
    if progress_mgr:
        progress_mgr.update_current(100, f"Subsampling: Complete - {len(selected)} corepoints selected")
    return points[selected]


# -----------------------------
# Significance
# -----------------------------
def calculate_significance_improved(distances, uncertainties=None, confidence_level=0.95,
                                    sigma1_manual=None, sigma2_manual=None, use_manual_sigma=False,
                                    progress_mgr=None):
    if progress_mgr:
        progress_mgr.update_current(0, "Calculating significance...")
    if confidence_level == 0.95:
        t_critical = 1.96
    elif confidence_level == 0.99:
        t_critical = 2.576
    elif confidence_level == 0.90:
        t_critical = 1.645
    else:
        t_critical = stats.t.ppf((1 + confidence_level) / 2, df=1000)
    if progress_mgr:
        progress_mgr.update_current(33, "Calculating detection limits...")
    if use_manual_sigma and sigma1_manual is not None and sigma2_manual is not None:
        sigma1 = sigma1_manual
        sigma2 = sigma2_manual
        detection_limit = t_critical * np.sqrt(sigma1**2 + sigma2**2)
        print(f"Manuálne sigma: σ1={sigma1:.4f} m, σ2={sigma2:.4f} m")
    else:
        if uncertainties is not None:
            spread1 = uncertainties['spread1']
            spread2 = uncertainties['spread2']
            sigma1 = np.nanmean(spread1)
            sigma2 = np.nanmean(spread2)
            detection_limit = t_critical * np.sqrt(sigma1**2 + sigma2**2)
            print(f"Automatický výpočet sigma: σ1={sigma1:.4f} m, σ2={sigma2:.4f} m")
        else:
            lod = uncertainties['lodetection'] if uncertainties is not None else 0.1
            mean_lod = np.nanmean(lod) if isinstance(lod, np.ndarray) else lod
            detection_limit = t_critical * mean_lod
            sigma1 = sigma2 = mean_lod / t_critical * np.sqrt(2)
            print(f"LoD-based výpočet: LoD={mean_lod:.4f} m")
    if progress_mgr:
        progress_mgr.update_current(66, "Identifying significant changes...")
    print(f"Detekčný limit ({confidence_level*100}%): {detection_limit:.4f} m")
    significant = np.abs(distances) > detection_limit
    if progress_mgr:
        progress_mgr.update_current(100, "Significance calculation complete")
    return significant, detection_limit, sigma1, sigma2


# -----------------------------
# Max depth filter
# -----------------------------
def apply_max_depth_filter(distances, max_depth, progress_mgr=None):
    if progress_mgr:
        progress_mgr.update_current(0, "Applying max depth filter...")
    if max_depth <= 0:
        return distances, np.ones_like(distances, dtype=bool)
    valid_mask = np.abs(distances) <= max_depth
    filtered_distances = distances.astype(np.float64).copy()
    filtered_distances[~valid_mask] = np.nan
    print(f"Max depth filter: {np.sum(valid_mask)}/{len(valid_mask)} bodov zostalo po filtri")
    if progress_mgr:
        progress_mgr.update_current(100, f"Max depth filter: {np.sum(valid_mask)} points retained")
    return filtered_distances, valid_mask


# -----------------------------
# Normals via PCA
# -----------------------------
def calculate_normals(corepoints, ref_points, k_neighbors=10, progress_mgr=None):
    if progress_mgr:
        progress_mgr.update_current(0, "Building KD-Tree for normals...")
    print("Vypočítavam normály...")
    tree = cKDTree(ref_points)
    k = min(k_neighbors, len(ref_points))
    if progress_mgr:
        progress_mgr.update_current(20, "Querying neighbours (batch)...")
    _, indices = tree.query(corepoints, k=k, workers=-1)
    if progress_mgr:
        progress_mgr.update_current(60, "Computing normals (vectorized PCA)...")
    neighbors = ref_points[indices]
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', centered, centered) / max(k - 1, 1)
    _, eigenvectors = np.linalg.eigh(cov)
    normals = eigenvectors[:, :, 0].astype(np.float32)
    if progress_mgr:
        progress_mgr.update_current(100, "Normals computation complete")
    print("Normály vypočítané.")
    return normals


# -----------------------------
# Nz / Nxy
# -----------------------------
def calculate_normal_angles(normals, progress_mgr=None):
    if progress_mgr:
        progress_mgr.update_current(0, "Computing Nz / Nxy components...")
    norm_magnitude = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_magnitude[norm_magnitude == 0] = 1.0
    normals_normalized = normals / norm_magnitude
    nz  = np.abs(normals_normalized[:, 2])
    nxy = np.sqrt(normals_normalized[:, 0]**2 + normals_normalized[:, 1]**2)
    if progress_mgr:
        progress_mgr.update_current(100, "Nz / Nxy computation complete")
    print(f"Nz  – min={nz.min():.3f}  max={nz.max():.3f}  mean={nz.mean():.3f}")
    print(f"Nxy – min={nxy.min():.3f}  max={nxy.max():.3f}  mean={nxy.mean():.3f}")
    return nz, nxy


# -----------------------------
# DX / DY / DZ
# -----------------------------
def calculate_normal_components(distances, normals, progress_mgr=None):
    if progress_mgr:
        progress_mgr.update_current(0, "Computing DX / DY / DZ...")
    norm_magnitude = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_magnitude[norm_magnitude == 0] = 1.0
    n = normals / norm_magnitude
    dx = distances * n[:, 0]
    dy = distances * n[:, 1]
    dz = distances * n[:, 2]
    if progress_mgr:
        progress_mgr.update_current(100, "DX / DY / DZ complete")
    return dx, dy, dz


# ═══════════════════════════════════════════════════════════════════════
# Quiver export  —  two modes
# ═══════════════════════════════════════════════════════════════════════
def export_quiver(corepoints, distances, normals, significant, output_file,
                  grid_size=1.0, export_png=True, export_html=True,
                  export_gpkg=True, progress_mgr=None,
                  mode='normal', dx=None, dy=None):
    if mode not in ('normal', 'motion'):
        raise ValueError(f"mode must be 'normal' or 'motion', got: {mode!r}")

    mode_label  = "Normal-direction quiver" if mode == 'normal' else "Motion vector (DX · DY)"
    mode_suffix = "_quiver_normal"           if mode == 'normal' else "_quiver_motion"

    if progress_mgr:
        progress_mgr.update_current(0, f"Quiver [{mode}]: binning onto grid...")

    norm_mag = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_mag[norm_mag == 0] = 1.0
    n_norm = normals / norm_mag

    xs   = corepoints[:, 0]
    ys   = corepoints[:, 1]
    dist = np.asarray(distances, dtype=np.float64)

    valid = ~np.isnan(dist)
    xs, ys, dist = xs[valid], ys[valid], dist[valid]
    n_norm = n_norm[valid]

    if mode == 'motion':
        if dx is None or dy is None:
            _ui_error("Chyba", "Motion-vector quiver requires DX/DY arrays.")
            return []
        dx_v = np.asarray(dx, dtype=np.float64)[valid]
        dy_v = np.asarray(dy, dtype=np.float64)[valid]

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cols    = int(np.ceil((x_max - x_min) / grid_size)) + 1
    col_idx = np.floor((xs - x_min) / grid_size).astype(int)
    row_idx = np.floor((ys - y_min) / grid_size).astype(int)
    cell_key = row_idx * cols + col_idx

    sort_order  = np.argsort(cell_key, kind="stable")
    sorted_keys = cell_key[sort_order]
    sorted_dist = dist[sort_order]
    sorted_nx   = n_norm[sort_order, 0]
    sorted_ny   = n_norm[sort_order, 1]

    unique_keys, first_idx, n_points = np.unique(
        sorted_keys, return_index=True, return_counts=True)

    mean_d  = np.add.reduceat(sorted_dist, first_idx) / n_points
    mean_nx = np.add.reduceat(sorted_nx,   first_idx) / n_points
    mean_ny = np.add.reduceat(sorted_ny,   first_idx) / n_points

    if mode == 'motion':
        sorted_dx = dx_v[sort_order]
        sorted_dy = dy_v[sort_order]
        mean_dx   = np.add.reduceat(sorted_dx, first_idx) / n_points
        mean_dy   = np.add.reduceat(sorted_dy, first_idx) / n_points

    if progress_mgr:
        progress_mgr.update_current(20, f"Quiver [{mode}]: {len(unique_keys)} grid cells...")

    cell_rows = unique_keys // cols
    cell_cols = unique_keys  % cols
    gx = x_min + (cell_cols + 0.5) * grid_size
    gy = y_min + (cell_rows + 0.5) * grid_size

    if mode == 'normal':
        horiz_mag = np.sqrt(mean_nx**2 + mean_ny**2)
        safe = horiz_mag > 1e-9
        nx_n = np.where(safe, mean_nx / horiz_mag, 0.0)
        ny_n = np.where(safe, mean_ny / horiz_mag, 0.0)
        u = nx_n * np.abs(mean_d)
        v = ny_n * np.abs(mean_d)
    else:
        u = mean_dx
        v = mean_dy

    c  = mean_d
    gx = np.array(gx); gy = np.array(gy)
    u  = np.array(u);  v  = np.array(v); c = np.array(c)

    arrow_mag  = np.sqrt(u**2 + v**2)
    max_mag    = arrow_mag.max() if arrow_mag.max() > 0 else 1.0
    plot_scale = (0.9 * grid_size) / max_mag
    u_plot = u * plot_scale
    v_plot = v * plot_scale

    abs_max = float(np.nanpercentile(np.abs(c), 98))
    if abs_max == 0:
        abs_max = 1.0
    n_arrows = len(gx)
    saved_paths = []

    if export_png:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            from matplotlib.cm import ScalarMappable
            if progress_mgr:
                progress_mgr.update_current(40, f"Quiver [{mode}]: rendering PNG...")
            fig, ax = plt.subplots(figsize=(14, 11), dpi=150)
            ax.set_aspect("equal")
            ax.set_facecolor("#1a1a1a"); fig.patch.set_facecolor("#1a1a1a")
            norm_c = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
            cmap   = plt.cm.RdBu_r
            ax.quiver(gx, gy, u_plot, v_plot, color=cmap(norm_c(c)),
                      angles="xy", scale_units="xy", scale=1,
                      width=0.0015, headwidth=5, headlength=6, alpha=0.92)
            sm = ScalarMappable(cmap=cmap, norm=norm_c); sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("Mean M3C2 Distance [m]", color="white", fontsize=11)
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
            subtitle = ("Direction = normal XY projection  |  Length & colour = mean M3C2"
                        if mode == 'normal' else
                        "Direction & length = true horizontal displacement (DX, DY)  |  colour = mean M3C2")
            ax.set_title(f"M3C2 {mode_label}  (grid={grid_size} m, {n_arrows} cells)\n{subtitle}",
                         color="white", fontsize=10, pad=12)
            ax.set_xlabel("X [m]", color="white"); ax.set_ylabel("Y [m]", color="white")
            ax.tick_params(colors="white")
            for sp in ax.spines.values(): sp.set_edgecolor("#444444")
            png_path = (output_file.replace(".las", f"{mode_suffix}.png")
                                   .replace(".laz", f"{mode_suffix}.png"))
            fig.savefig(png_path, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"Quiver PNG [{mode}]: {png_path}")
            saved_paths.append(png_path)
        except ImportError:
            _ui_error("Chyba", "Chýba matplotlib.\npip install matplotlib")

    if export_html:
        try:
            import plotly.graph_objects as go
            if progress_mgr:
                progress_mgr.update_current(65, f"Quiver [{mode}]: building HTML...")
            head_x = gx + u_plot; head_y = gy + v_plot
            seg_x = np.empty(n_arrows * 3); seg_y = np.empty(n_arrows * 3)
            seg_x[0::3]=gx;  seg_x[1::3]=head_x; seg_x[2::3]=np.nan
            seg_y[0::3]=gy;  seg_y[1::3]=head_y; seg_y[2::3]=np.nan
            fig_html = go.Figure()
            fig_html.add_trace(go.Scatter(
                x=seg_x, y=seg_y, mode="lines",
                line=dict(color="rgba(200,200,200,0.35)", width=1),
                hoverinfo="skip", showlegend=False, name="shafts"))
            hover_extra = ("Azimuth: %{customdata[3]:.1f}°<extra></extra>"
                           if mode == 'normal' else
                           "DX: %{customdata[4]:.4f} m<br>DY: %{customdata[5]:.4f} m<extra></extra>")
            cd_cols = [c, gx, gy, np.degrees(np.arctan2(v_plot, u_plot))]
            if mode == 'motion':
                cd_cols += [u, v]
            fig_html.add_trace(go.Scatter(
                x=head_x, y=head_y, mode="markers",
                marker=dict(symbol="arrow", size=9,
                            angle=np.degrees(np.arctan2(v_plot, u_plot)),
                            color=c, colorscale="RdBu_r", reversescale=False,
                            cmin=-abs_max, cmax=abs_max,
                            colorbar=dict(
                                title=dict(text="Mean M3C2<br>Distance [m]", font=dict(color="white")),
                                tickfont=dict(color="white"), thickness=16, len=0.75),
                            line=dict(width=0)),
                customdata=np.stack(cd_cols, axis=-1),
                hovertemplate=("Cell: (%{customdata[1]:.2f}, %{customdata[2]:.2f})<br>"
                               "Mean M3C2: %{customdata[0]:.4f} m<br>" + hover_extra),
                showlegend=False, name="heads"))
            subtitle_html = ("Direction = normal XY projection<br><sup>Length & colour = mean M3C2 distance</sup>"
                             if mode == 'normal' else
                             "Direction & length = true horizontal displacement (DX, DY)<br><sup>Colour = mean M3C2 distance</sup>")
            fig_html.update_layout(
                title=dict(text=f"M3C2 {mode_label} — grid={grid_size} m | {n_arrows} cells<br>{subtitle_html}",
                           font=dict(color="white", size=13)),
                xaxis=dict(title="X [m]", scaleanchor="y", scaleratio=1,
                           color="white", gridcolor="#2a2a2a", zerolinecolor="#444444"),
                yaxis=dict(title="Y [m]", color="white",
                           gridcolor="#2a2a2a", zerolinecolor="#444444"),
                paper_bgcolor="#1a1a1a", plot_bgcolor="#111111",
                font=dict(color="white"), width=1300, height=950,
                margin=dict(l=70, r=70, t=90, b=60))
            html_path = (output_file.replace(".las", f"{mode_suffix}.html")
                                    .replace(".laz", f"{mode_suffix}.html"))
            fig_html.write_html(html_path, include_plotlyjs="cdn")
            print(f"Quiver HTML [{mode}]: {html_path}")
            saved_paths.append(html_path)
        except ImportError:
            _ui_error("Chyba", "Chýba plotly.\npip install plotly")

    if export_gpkg:
        try:
            import geopandas as gpd
            from shapely.geometry import LineString
            if progress_mgr:
                progress_mgr.update_current(85, f"Quiver [{mode}]: writing GPKG...")
            geometries = [
                LineString([(float(gx[i]), float(gy[i])),
                            (float(gx[i]) + float(u_plot[i]),
                             float(gy[i]) + float(v_plot[i]))])
                for i in range(len(gx))
            ]
            records = []
            for i in range(len(gx)):
                r = {"mean_m3c2": float(c[i]), "abs_m3c2": float(abs(c[i])),
                     "cell_x": float(gx[i]), "cell_y": float(gy[i]),
                     "azimuth_deg": float(np.degrees(np.arctan2(float(v[i]), float(u[i])))),
                     "n_points": int(n_points[i]), "quiver_mode": mode}
                if mode == 'motion':
                    r["dx_mean"]   = float(u[i])
                    r["dy_mean"]   = float(v[i])
                    r["horiz_mag"] = float(arrow_mag[i])
                records.append(r)
            gdf = gpd.GeoDataFrame(records, geometry=geometries)
            gpkg_path = (output_file.replace(".las", f"{mode_suffix}.gpkg")
                                    .replace(".laz", f"{mode_suffix}.gpkg"))
            gdf.to_file(gpkg_path, driver="GPKG", layer=f"m3c2_{mode}")
            print(f"Quiver GPKG [{mode}]: {gpkg_path}  ({len(gdf)} arrows)")
            saved_paths.append(gpkg_path)
        except ImportError:
            _ui_error("Chyba", "Chýba geopandas/shapely.\npip install geopandas shapely")

    if progress_mgr:
        progress_mgr.update_current(100, f"Quiver [{mode}] export complete")
    return saved_paths


# -----------------------------
# Save LAS/LAZ + optional TXT
# -----------------------------
def save_results_simplified(corepoints, distances, nz, nxy, significant,
                            output_file, save_txt=False, progress_mgr=None,
                            uncertainties=None, dx=None, dy=None, dz=None,
                            las_fields=None):
    if las_fields is None:
        las_fields = {"Nz","Nxy","DX","DY","DZ","LOD","SPREAD1","SPREAD2"}
    n = len(corepoints)
    if progress_mgr:
        progress_mgr.update_current(0, "Creating LAS file...")
    print("Ukladám výsledky do LAS/LAZ...")
    core_las = laspy.LasData(laspy.LasHeader(point_format=3, version="1.4"))
    core_las.x = corepoints[:, 0]; core_las.y = corepoints[:, 1]; core_las.z = corepoints[:, 2]
    if progress_mgr:
        progress_mgr.update_current(25, "Adding fields...")
    for name, arr, dtype in [("M3C2_DISTANCE", distances, np.float32),
                               ("SIGNIFICANT",   significant, np.int8)]:
        core_las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
        core_las[name] = arr.astype(dtype)
    txt_cols = [corepoints[:, 0], corepoints[:, 1], corepoints[:, 2],
                distances.astype(np.float32), significant.astype(np.float32)]
    txt_hdrs = ["X", "Y", "Z", "M3C2_DISTANCE", "SIGNIFICANT"]
    optional_fields = []
    if "Nz"  in las_fields: optional_fields.append(("Nz",  nz,  np.float32))
    if "Nxy" in las_fields: optional_fields.append(("Nxy", nxy, np.float32))
    if "DX"  in las_fields and dx is not None: optional_fields.append(("DX", dx, np.float32))
    if "DY"  in las_fields and dy is not None: optional_fields.append(("DY", dy, np.float32))
    if "DZ"  in las_fields and dz is not None: optional_fields.append(("DZ", dz, np.float32))
    for name, arr, dtype in optional_fields:
        core_las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
        core_las[name] = arr.astype(dtype)
        txt_cols.append(arr.astype(dtype)); txt_hdrs.append(name)
    if progress_mgr:
        progress_mgr.update_current(50, "Adding LOD / SPREAD1 / SPREAD2...")
    for py4d_key, col_name in [("lodetection","LOD"),("spread1","SPREAD1"),("spread2","SPREAD2")]:
        if col_name not in las_fields:
            continue
        arr = None
        if uncertainties is not None:
            try:
                raw = uncertainties[py4d_key]
                arr = np.asarray(raw, dtype=np.float32).ravel()
                if len(arr) != n: arr = None
            except Exception: arr = None
        if arr is None:
            arr = np.full(n, np.nan, dtype=np.float32)
        core_las.add_extra_dim(laspy.ExtraBytesParams(name=col_name, type=np.float32))
        core_las[col_name] = arr
        txt_cols.append(arr); txt_hdrs.append(col_name)
    if progress_mgr:
        progress_mgr.update_current(70, "Writing LAS/LAZ file...")
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    try:
        core_las.write(output_file, do_compress=output_file.lower().endswith(".laz"))
        print(f"Uložený: {output_file}")
    except Exception as e:
        _ui_error("Chyba", f"Chyba pri ukladaní LAS: {e}")
        return False
    if save_txt:
        if progress_mgr:
            progress_mgr.update_current(85, "Exporting TXT...")
        txt_file = output_file.replace(".las","_results.txt").replace(".laz","_results.txt")
        try:
            np.savetxt(txt_file, np.column_stack(txt_cols), delimiter="\t",
                       header="\t".join(txt_hdrs), comments="", fmt="%.6f")
            print(f"TXT: {txt_file}")
        except Exception as e:
            print(f"  ! TXT failed: {e}")
    if progress_mgr:
        progress_mgr.update_current(100, "Export complete")
    return True


# -----------------------------
# Raster export
# -----------------------------
def export_raster(corepoints, values, output_raster_path, resolution,
                  field_name="LOD", nodata=-9999.0, interp_method="IDW",
                  progress_mgr=None):
    try:
        import rasterio
        from rasterio.transform import from_origin
        from scipy.interpolate import griddata, NearestNDInterpolator
    except ImportError as e:
        _ui_error("Chyba", f"Chýba knižnica: {e}\npip install rasterio scipy")
        return False
    if progress_mgr:
        progress_mgr.update_current(0, f"Raster [{interp_method}]: {field_name}...")
    xs, ys = corepoints[:, 0], corepoints[:, 1]
    zv = np.asarray(values, dtype=np.float64)
    valid = ~np.isnan(zv)
    if valid.sum() < 4:
        _ui_error("Chyba", "Príliš málo platných bodov.")
        return False
    x_min, x_max = xs[valid].min(), xs[valid].max()
    y_min, y_max = ys[valid].min(), ys[valid].max()
    grid_x = np.arange(x_min, x_max + resolution, resolution)
    grid_y = np.arange(y_max, y_min - resolution, -resolution)
    gx, gy = np.meshgrid(grid_x, grid_y)
    if progress_mgr:
        progress_mgr.update_current(30, f"Raster: interpolácia ({interp_method})...")
    pts  = np.column_stack([xs[valid], ys[valid]])
    vals = zv[valid]
    m    = interp_method.upper()
    if m == "TIN":
        grid_z = griddata(pts, vals, (gx, gy), method="linear", fill_value=np.nan)
    elif m == "NEAREST":
        grid_z = NearestNDInterpolator(pts, vals)(gx, gy)
    elif m == "IDW":
        tree  = cKDTree(pts)
        k     = min(12, len(pts))
        dists, idx = tree.query(np.column_stack([gx.ravel(), gy.ravel()]), k=k, workers=-1)
        dists = np.where(dists == 0, 1e-10, dists)
        w     = 1.0 / dists**2
        grid_z = (np.sum(w * vals[idx], axis=1) / np.sum(w, axis=1)).reshape(gx.shape)
    elif m == "KRIGING":
        try:
            from pykrige.ok import OrdinaryKriging
        except ImportError:
            _ui_error("Chyba", "pip install pykrige")
            return False
        ok = OrdinaryKriging(xs[valid], ys[valid], vals,
                             variogram_model="linear", verbose=False, enable_plotting=False)
        z_krig, _ = ok.execute("grid", grid_x, grid_y[::-1])
        grid_z = np.flipud(z_krig.data)
    else:
        _ui_error("Chyba", f"Neznáma metóda: {interp_method}")
        return False
    grid_z = np.where(np.isnan(grid_z), nodata, grid_z).astype(np.float32)
    out_dir = os.path.dirname(output_raster_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    try:
        transform = from_origin(x_min, y_max, resolution, resolution)
        with rasterio.open(output_raster_path, 'w', driver='GTiff',
                           height=grid_z.shape[0], width=grid_z.shape[1],
                           count=1, dtype='float32', nodata=nodata,
                           transform=transform) as dst:
            dst.write(grid_z, 1)
        print(f"Raster: {output_raster_path}")
    except Exception as e:
        _ui_error("Chyba", f"Chyba rastra: {e}")
        return False
    if progress_mgr:
        progress_mgr.update_current(100, "Raster complete")
    return True


# -----------------------------
# Statistics
# -----------------------------
def print_comprehensive_stats(distances, significant, valid_mask, detection_limit,
                               max_depth, nz, nxy):
    valid_distances = distances[valid_mask & ~np.isnan(distances)]
    if len(valid_distances) == 0:
        print("Žiadne platné vzdialenosti!")
        return
    print(f"\n=== ŠTATISTIKA VÝSLEDKOV ===")
    print(f"Celkový počet corepoints : {len(distances)}")
    print(f"Platných bodov           : {len(valid_distances)}")
    print(f"Významných zmien         : {np.sum(significant[valid_mask])}")
    print(f"Detekčný limit           : {float(detection_limit):.4f} m")
    print(f"\nVZDIALENOSTI:")
    print(f"  Min={np.min(valid_distances):.4f}  Max={np.max(valid_distances):.4f}  "
          f"Mean={np.mean(valid_distances):.4f}  Median={np.median(valid_distances):.4f}  "
          f"SD={np.std(valid_distances):.4f}")
    print(f"\nNORMÁLOVÉ ZLOŽKY:")
    print(f"  Nz  min={nz.min():.3f}  max={nz.max():.3f}  mean={nz.mean():.3f}")
    print(f"  Nxy min={nxy.min():.3f}  max={nxy.max():.3f}  mean={nxy.mean():.3f}")
    bins   = [0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, np.inf]
    labels = ["<0.1m","0.1-0.5m","0.5-1.0m","1.0-2.0m","2.0-5.0m","5.0-10.0m",">10.0m"]
    hist, _ = np.histogram(np.abs(valid_distances), bins=bins)
    print(f"\nROZDELENIE ZMIEN:")
    for lbl, cnt in zip(labels, hist):
        if cnt > 0:
            print(f"  {lbl}: {cnt} ({cnt/len(valid_distances)*100:.1f}%)")
    if max_depth > 0:
        print(f"\nMax depth {max_depth} m → odfiltrovaných: {len(distances)-len(valid_distances)}")


# ─────────────────────────────────────────────────────────────────────
# Profile helpers
# ─────────────────────────────────────────────────────────────────────

def load_profile_line(txt_path):
    coords = []
    with open(txt_path, 'r') as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.replace(',', ' ').replace(';', ' ').split()
            if len(parts) >= 2:
                y_val, x_val = float(parts[0]), float(parts[1])
                coords.append([x_val, y_val])
    if len(coords) < 2:
        raise ValueError("Profile TXT must contain at least 2 vertices (Y X per line).")
    return np.array(coords, dtype=np.float64)


def _project_to_polyline(pts_xy, profile_xy):
    n          = len(pts_xy)
    best_along = np.zeros(n, dtype=np.float64)
    best_perp  = np.zeros(n, dtype=np.float64)
    best_abs   = np.full(n, np.inf, dtype=np.float64)
    cumlen     = 0.0

    for i in range(len(profile_xy) - 1):
        p0, p1  = profile_xy[i], profile_xy[i + 1]
        seg     = p1 - p0
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-10:
            continue
        tdir = seg / seg_len
        v    = pts_xy - p0
        t    = v @ tdir
        tc   = np.clip(t, 0.0, seg_len)
        foot = p0 + np.outer(tc, tdir)
        dp   = pts_xy - foot
        absd = np.linalg.norm(dp, axis=1)
        signed = tdir[0] * dp[:, 1] - tdir[1] * dp[:, 0]
        closer = absd < best_abs
        best_along[closer] = cumlen + tc[closer]
        best_perp[closer]  = signed[closer]
        best_abs[closer]   = absd[closer]
        cumlen += seg_len

    return best_along, best_perp, best_abs, cumlen


def export_profile_png(corepoints, distances, significant, profile_xy,
                       corridor_width, output_path,
                       detection_limit=None, profile_label="", progress_mgr=None):
    """
    Two-panel profile PNG.
    Upper  – elevation cross-section coloured by M3C2
    Lower  – M3C2 distance along the profile with LoD bands
    Reference points are shown in blue, comparison in the diverging colour scale.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        _ui_error("Chyba", "Chýba matplotlib.\npip install matplotlib")
        return False

    if progress_mgr:
        progress_mgr.update_current(10, f"Profile '{profile_label}': projecting onto line...")

    pts_xy = corepoints[:, :2].astype(np.float64)
    dist_along, _perp, abs_perp, total_len = _project_to_polyline(pts_xy, profile_xy)

    half_w = corridor_width / 2.0
    mask   = abs_perp <= half_w
    n_in   = int(mask.sum())

    if n_in == 0:
        _ui_error(
            "Chyba",
            f"No corepoints found within {corridor_width} m of profile '{profile_label}'.\n"
            "Check coordinates and corridor width.")
        return False

    d_al = dist_along[mask]
    z_v  = corepoints[mask, 2]
    m3c2 = distances[mask].astype(np.float64)
    sig  = (significant[mask].astype(bool)
            if significant is not None else np.ones(n_in, dtype=bool))

    order = np.argsort(d_al)
    d_al  = d_al[order]; z_v  = z_v[order]
    m3c2  = m3c2[order]; sig  = sig[order]

    if progress_mgr:
        progress_mgr.update_current(30, f"Profile '{profile_label}': building figure ({n_in} pts)...")

    valid_m = m3c2[~np.isnan(m3c2)]
    abs_max = float(np.nanpercentile(np.abs(valid_m), 98)) if len(valid_m) > 0 else 1.0
    if abs_max == 0:
        abs_max = 1.0
    norm_c = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    cmap   = plt.cm.RdBu_r

    fig = plt.figure(figsize=(16, 9), dpi=150, facecolor='white')
    gs  = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.42)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    for ax in (ax1, ax2):
        ax.set_facecolor('#f7f7f7')
        ax.grid(True, which='major', color='white',   linewidth=1.0, zorder=0)
        ax.grid(True, which='minor', color='#eeeeee', linewidth=0.5, zorder=0)
        ax.minorticks_on()
        for sp in ax.spines.values(): sp.set_color('#bbbbbb')
        ax.tick_params(axis='both', colors='#444444', labelsize=8, length=4)
        ax.set_xlabel("Distance along profile [m]", fontsize=9, color='#333333', labelpad=6)

    # Upper panel – elevation
    sc = ax1.scatter(d_al, z_v, c=m3c2, cmap=cmap, norm=norm_c,
                     s=7, linewidths=0, zorder=3, rasterized=True)
    ax1.set_ylabel("Elevation Z [m]", fontsize=9, color='#333333', labelpad=6)
    title_str = f"M3C2 Profile Cut"
    if profile_label:
        title_str += f"  ·  {profile_label}"
    title_str += (f"  ·  corridor {corridor_width} m  ·  {n_in} corepoints"
                  f"  ·  length {total_len:.1f} m")
    ax1.set_title(title_str, fontsize=10, color='#111111', pad=10,
                  fontweight='bold', loc='left')

    cb = fig.colorbar(sc, ax=ax1, fraction=0.022, pad=0.02, aspect=32)
    cb.set_label("M3C2 Distance [m]", fontsize=8, color='#333333', labelpad=6)
    cb.ax.tick_params(labelsize=7, colors='#444444')
    cb.outline.set_edgecolor('#bbbbbb')

    xlim = (-total_len * 0.01, total_len * 1.01)
    ax1.set_xlim(xlim)

    # Lower panel – M3C2 distance
    not_sig = ~sig & ~np.isnan(m3c2)
    is_sig  =  sig & ~np.isnan(m3c2)
    if not_sig.any():
        ax2.scatter(d_al[not_sig], m3c2[not_sig], color='#aaaaaa',
                    s=5, linewidths=0, zorder=3, label='Not significant', rasterized=True)
    if is_sig.any():
        ax2.scatter(d_al[is_sig], m3c2[is_sig], c=m3c2[is_sig], cmap=cmap, norm=norm_c,
                    s=6, linewidths=0, zorder=4, label='Significant', rasterized=True)

    ax2.axhline(0, color='#555555', linewidth=0.9, zorder=5)
    if detection_limit is not None and detection_limit > 0:
        ax2.axhline(+detection_limit, color='#c62828', linewidth=1.1,
                    linestyle='--', zorder=5, label=f'+LoD  {detection_limit:.3f} m')
        ax2.axhline(-detection_limit, color='#1565c0', linewidth=1.1,
                    linestyle='--', zorder=5, label=f'\u2212LoD  {detection_limit:.3f} m')

    ax2.set_ylabel("M3C2 Distance [m]", fontsize=9, color='#333333', labelpad=6)
    ax2.set_xlim(xlim)
    leg = ax2.legend(fontsize=7, framealpha=0.95, edgecolor='#cccccc',
                     facecolor='white', loc='upper right', ncol=2)
    for txt in leg.get_texts(): txt.set_color('#333333')

    fig.text(0.01, 0.005,
             f"Profile length: {total_len:.2f} m  ·  corridor ±{half_w:.1f} m  "
             f"·  {n_in} points  ·  M3C2 Tool – CHECKPOINT13",
             fontsize=6.5, color='#888888', va='bottom')
    fig.text(0.99, 0.005,
             f"Start: ({profile_xy[0,0]:.1f}, {profile_xy[0,1]:.1f})  "
             f"End: ({profile_xy[-1,0]:.1f}, {profile_xy[-1,1]:.1f})",
             fontsize=6.5, color='#888888', va='bottom', ha='right')

    if progress_mgr:
        progress_mgr.update_current(85, f"Profile '{profile_label}': saving PNG...")

    try:
        fig.savefig(output_path, bbox_inches='tight', facecolor='white', dpi=150)
        plt.close(fig)
        print(f"Profile PNG: {output_path}")
    except Exception as e:
        _ui_error("Chyba", f"Chyba pri ukladaní profilu: {e}")
        plt.close(fig)
        return False

    if progress_mgr:
        progress_mgr.update_current(100, "Profile export complete")
    return True


# -----------------------------
# Main M3C2 pipeline
# -----------------------------
def run_m3c2_gui(las_file1, las_file2, output_file, cyl_radius, normal_radii,
                 core_spacing, save_txt, export_components=True,
                 confidence_level=0.95, max_depth=0.0,
                 use_manual_sigma=False, sigma1_manual=0.1, sigma2_manual=0.1,
                 progress_mgr=None,
                 make_raster=False, raster_resolution=1.0,
                 raster_fields=None, raster_interp_method="IDW",
                 make_dem=False, dem_resolution=0.1, dem_interp_method="IDW",
                 quiver_normal=False, quiver_motion=False,
                 quiver_grid_size=1.0, quiver_png=True, quiver_html=True, quiver_gpkg=True,
                 las_fields=None,
                 # Profile – list of (label, np.ndarray [[x0,y0],[x1,y1]]) tuples
                 profile_lines=None,
                 profile_width=50.0):

    if not output_file:
        _ui_error("Chyba", "Zadajte výstupný LAS/LAZ súbor!")
        return

    # Step 1
    if progress_mgr: progress_mgr.update_current(0, "Loading point clouds...")
    epoch1, epoch2 = py4dgeo.read_from_las(las_file1, las_file2)
    ref_points = epoch1.cloud
    if progress_mgr:
        progress_mgr.update_current(100, "Point clouds loaded")
        progress_mgr.update_overall("Point clouds loaded")

    # Step 2
    if core_spacing and core_spacing > 0:
        corepoints = subsample_by_distance(ref_points, core_spacing, progress_mgr)
    else:
        corepoints = ref_points
        if progress_mgr: progress_mgr.update_current(100, "Using all points as corepoints")
    if progress_mgr: progress_mgr.update_overall("Subsampling complete")

    # Step 3
    if progress_mgr: progress_mgr.update_current(0, "Running M3C2 algorithm...")
    m3c2 = py4dgeo.M3C2(epochs=(epoch1, epoch2), corepoints=corepoints,
                         cyl_radius=cyl_radius, normal_radii=normal_radii)
    if progress_mgr: progress_mgr.update_current(50, "M3C2: Computing distances...")
    distances, uncertainties = m3c2.run()
    if progress_mgr:
        progress_mgr.update_current(100, "M3C2 complete")
        progress_mgr.update_overall("M3C2 distances computed")

    # Step 4
    normals = calculate_normals(corepoints, ref_points, k_neighbors=10, progress_mgr=progress_mgr)
    if progress_mgr: progress_mgr.update_overall("Normals computed")

    # Step 5
    if max_depth > 0:
        distances, valid_mask = apply_max_depth_filter(distances, max_depth, progress_mgr)
    else:
        valid_mask = np.ones_like(distances, dtype=bool)
        if progress_mgr: progress_mgr.update_current(100, "No max depth filter")
    if progress_mgr: progress_mgr.update_overall("Depth filtering complete")

    # Step 6
    significant, detection_limit, sigma1, sigma2 = calculate_significance_improved(
        distances, uncertainties, confidence_level, sigma1_manual, sigma2_manual,
        use_manual_sigma, progress_mgr)
    if progress_mgr: progress_mgr.update_overall("Significance computed")

    # Step 7
    nz, nxy = calculate_normal_angles(normals, progress_mgr)
    dx, dy, dz = calculate_normal_components(distances, normals, progress_mgr)
    if progress_mgr: progress_mgr.update_overall("Normal components computed")

    # Step 8
    success = save_results_simplified(corepoints, distances, nz, nxy, significant,
                                      output_file, save_txt, progress_mgr,
                                      uncertainties=uncertainties,
                                      dx=dx, dy=dy, dz=dz, las_fields=las_fields)
    if progress_mgr: progress_mgr.update_overall("Results saved")
    if not success:
        _ui_error("Chyba", "Nastala chyba pri ukladaní výsledkov.")
        return

    # Step 9
    if progress_mgr: progress_mgr.update_current(0, "Computing statistics...")
    print_comprehensive_stats(distances, significant, valid_mask, detection_limit,
                               max_depth, nz, nxy)
    if progress_mgr:
        progress_mgr.update_current(100, "Statistics computed")
        progress_mgr.update_overall("Complete!")

    field_map = {
        "M3C2_DISTANCE": distances.astype(np.float32),
        "SIGNIFICANT":   significant.astype(np.float32),
        "Nz": nz.astype(np.float32), "Nxy": nxy.astype(np.float32),
        "DX": dx.astype(np.float32), "DY": dy.astype(np.float32), "DZ": dz.astype(np.float32),
    }
    if uncertainties is not None:
        for py4d_key, col_name in [("lodetection","LOD"),("spread1","SPREAD1"),("spread2","SPREAD2")]:
            try:
                arr = np.asarray(uncertainties[py4d_key], dtype=np.float32).ravel()
                if len(arr) == len(corepoints): field_map[col_name] = arr
            except Exception: pass

    saved_extras = []

    if make_raster:
        for fld in (raster_fields or ["M3C2_DISTANCE"]):
            if fld not in field_map: continue
            rp = output_file.replace(".las", f"_{fld}.tif").replace(".laz", f"_{fld}.tif")
            if export_raster(corepoints, field_map[fld], rp, raster_resolution, fld,
                             interp_method=raster_interp_method, progress_mgr=progress_mgr):
                saved_extras.append(rp)

    if make_dem:
        dp = output_file.replace(".las","_DEM.tif").replace(".laz","_DEM.tif")
        if export_raster(corepoints, corepoints[:, 2], dp, dem_resolution, "Z (DEM)",
                         interp_method=dem_interp_method, progress_mgr=progress_mgr):
            saved_extras.append(dp)

    quiver_kwargs = dict(grid_size=quiver_grid_size, export_png=quiver_png,
                         export_html=quiver_html, export_gpkg=quiver_gpkg,
                         progress_mgr=progress_mgr)
    if quiver_normal:
        result = export_quiver(corepoints, distances, normals, significant, output_file,
                               mode='normal', **quiver_kwargs)
        if result: saved_extras.extend(result)
    if quiver_motion:
        result = export_quiver(corepoints, distances, normals, significant, output_file,
                               mode='motion', dx=dx, dy=dy, **quiver_kwargs)
        if result: saved_extras.extend(result)

    # ── Profile exports (one PNG per line) ─────────────────────────────
    if profile_lines:
        for i, (lbl, pxy) in enumerate(profile_lines):
            stem   = os.path.splitext(output_file)[0]
            pp     = f"{stem}_profile_{i+1:02d}.png"
            if export_profile_png(corepoints, distances, significant, pxy,
                                  profile_width, pp, detection_limit,
                                  profile_label=lbl, progress_mgr=progress_mgr):
                saved_extras.append(pp)

    extras_str = "\n".join(saved_extras) if saved_extras else "—"
    _ui_info("Hotovo",
        "M3C2 dokončené!\n"
        "Polia: X, Y, Z, M3C2_DISTANCE, SIGNIFICANT, Nz, Nxy, LOD, SPREAD1, SPREAD2\n\n"
        f"Dodatočné súbory:\n{extras_str}")


# ═══════════════════════════════════════════════════════════
# GUI
# ═══════════════════════════════════════════════════════════
def browse_file(entry, save=False):
    fp = (filedialog.asksaveasfilename(defaultextension=".las",
                                       filetypes=[("LAS/LAZ files", "*.las *.laz")])
          if save else
          filedialog.askopenfilename(filetypes=[("LAS/LAZ files", "*.las *.laz")]))
    if fp:
        entry.delete(0, tk.END); entry.insert(0, fp)

def toggle_sigma_entries():
    s = "normal" if var_use_manual_sigma.get() else "disabled"
    entry_sigma1.config(state=s); entry_sigma2.config(state=s)

def toggle_raster_entries():
    s = "normal" if var_make_raster.get() else "disabled"
    entry_raster_res.config(state=s); option_raster_interp.config(state=s)
    for chk in raster_field_chks.values(): chk.config(state=s)

def toggle_dem_entries():
    s = "normal" if var_make_dem.get() else "disabled"
    entry_dem_res.config(state=s); option_dem_interp.config(state=s)

def toggle_quiver_entries():
    either = var_quiver_normal.get() or var_quiver_motion.get()
    s = "normal" if either else "disabled"
    entry_quiver_grid.config(state=s)
    for chk in _qfmt_chks: chk.config(state=s)

def toggle_profile_entries():
    s = "normal" if var_make_profile.get() else "disabled"
    entry_profile_width.config(state=s)
    entry_profile_txt.config(state=s)
    btn_profile_browse.config(state=s)
    btn_pick_profiles.config(state=s)

def start_m3c2_threaded():
    """
    Called from the main thread (button click).
    ALL tkinter .get() calls happen here — never inside the worker thread.
    The worker receives only plain Python values.
    """
    # ── Read every GUI value here, in the main thread ─────────────────
    try:
        cyl          = float(entry_cyl.get())
        normals_r    = [float(x) for x in entry_normals.get().split(",")]
        core_spacing = (0 if var_use_all_points.get()
                        else (float(entry_spacing.get()) if entry_spacing.get() else 0))
        max_depth    = float(entry_max_depth.get()) if entry_max_depth.get() else 0.0
        conf_level   = float(var_confidence.get())
        use_man_s    = var_use_manual_sigma.get()
        s1           = float(entry_sigma1.get()) if use_man_s else 0.1
        s2           = float(entry_sigma2.get()) if use_man_s else 0.1
        raster_res   = float(entry_raster_res.get()) if entry_raster_res.get() else 1.0
        dem_res      = float(entry_dem_res.get())    if entry_dem_res.get()    else 0.1
        quiver_grid  = float(entry_quiver_grid.get()) if entry_quiver_grid.get() else 1.0
        profile_w    = float(entry_profile_width.get()) if entry_profile_width.get() else 50.0
    except ValueError:
        messagebox.showerror("Chyba", "Skontrolujte numerické polia.")
        return

    las1_path   = entry_las1.get()
    las2_path   = entry_las2.get()
    output_path = entry_output.get()
    save_txt    = var_txt.get()

    make_raster        = var_make_raster.get()
    raster_fields_sel  = [f for f, v in raster_field_vars.items() if v.get()]
    raster_interp      = var_raster_interp.get()
    make_dem           = var_make_dem.get()
    dem_interp         = var_dem_interp.get()
    quiver_normal      = var_quiver_normal.get()
    quiver_motion      = var_quiver_motion.get()
    quiver_png         = var_quiver_png.get()
    quiver_html        = var_quiver_html.get()
    quiver_gpkg        = var_quiver_gpkg.get()
    selected_las_fields = {f for f, v in las_field_vars.items() if v.get()}

    # Profile lines
    prof_lines = []
    if var_make_profile.get():
        if picked_profile_lines:
            prof_lines = list(picked_profile_lines)
        txt_path = entry_profile_txt.get().strip()
        if txt_path:
            try:
                pxy = load_profile_line(txt_path)
                lbl = os.path.splitext(os.path.basename(txt_path))[0]
                prof_lines.append((lbl, pxy))
            except Exception as e:
                messagebox.showerror("Chyba", f"Profile TXT chyba: {e}")
                return

    # ── Disable button, launch worker ────────────────────────────────
    btn_start.config(state="disabled")
    progress_mgr.reset()

    def _worker():
        try:
            run_m3c2_gui(
                las1_path, las2_path, output_path,
                cyl, normals_r, core_spacing, save_txt,
                True, conf_level, max_depth, use_man_s, s1, s2, progress_mgr,
                make_raster=make_raster, raster_resolution=raster_res,
                raster_fields=raster_fields_sel,
                raster_interp_method=raster_interp,
                make_dem=make_dem, dem_resolution=dem_res,
                dem_interp_method=dem_interp,
                quiver_normal=quiver_normal, quiver_motion=quiver_motion,
                quiver_grid_size=quiver_grid, quiver_png=quiver_png,
                quiver_html=quiver_html, quiver_gpkg=quiver_gpkg,
                las_fields=selected_las_fields if selected_las_fields else None,
                profile_lines=prof_lines if prof_lines else None,
                profile_width=profile_w,
            )
        finally:
            # Must re-enable the button on the main thread
            root.after(0, lambda: btn_start.config(state="normal"))

    threading.Thread(target=_worker, daemon=True).start()


def toggle_spacing_entry():
    s = "disabled" if var_use_all_points.get() else "normal"
    entry_spacing.config(state=s)


def guess_parameters():
    path = entry_las1.get()
    if not path:
        messagebox.showerror("Chyba", "Najprv vyberte referenčný LAS/LAZ súbor."); return
    try:
        las = laspy.read(path)
        pts = np.column_stack([las.x, las.y, las.z])
    except Exception as e:
        messagebox.showerror("Chyba", f"Chyba čítania súboru: {e}"); return
    n = len(pts)
    if n > 5000:
        idx = np.random.choice(n, 5000, replace=False); pts = pts[idx]
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2, workers=-1)
    d_avg = float(np.mean(dists[:, 1]))
    cyl_r  = round(2.0 * d_avg, 3); nr1 = round(3.0 * d_avg, 3)
    nr2    = round(5.0 * d_avg, 3); nr3 = round(8.0 * d_avg, 3)
    spacing = round(2.0 * d_avg, 3)
    entry_cyl.delete(0, tk.END);     entry_cyl.insert(0, str(cyl_r))
    entry_normals.delete(0, tk.END); entry_normals.insert(0, f"{nr1},{nr2},{nr3}")
    if not var_use_all_points.get():
        entry_spacing.delete(0, tk.END); entry_spacing.insert(0, str(spacing))
    messagebox.showinfo("Guess Parameters",
        "Estimated avg. point spacing: {:.4f} m\n\n"
        "Cylinder radius : {} m\n"
        "Normal radii    : {}, {}, {} m\n"
        "Core spacing    : {} m".format(d_avg, cyl_r, nr1, nr2, nr3, spacing))


# ── Profile picker callback ──────────────────────────────────────────
# Holds list of (label, np.ndarray) tuples for the current session
picked_profile_lines = []   # populated by _open_profile_picker()

def _open_profile_picker():
    f1 = entry_las1.get().strip()
    f2 = entry_las2.get().strip()
    if not f1 or not f2:
        messagebox.showerror("Chyba",
            "Please fill in both LAS/LAZ file paths before picking profiles.")
        return

    existing = [pxy for _lbl, pxy in picked_profile_lines]
    picker   = ProfilePicker(root, f1, f2, existing_lines=existing)
    result   = picker.wait()   # blocks until window is closed

    if result is None:
        return   # cancelled

    # Replace picked lines
    picked_profile_lines.clear()
    for i, pxy in enumerate(result):
        lbl = f"Line {i+1}"
        picked_profile_lines.append((lbl, pxy))

    _refresh_profile_listbox()


def _refresh_profile_listbox():
    lbox_profiles.delete(0, tk.END)
    for lbl, pxy in picked_profile_lines:
        x0, y0 = pxy[0]
        x1, y1 = pxy[1]
        length = float(np.linalg.norm(pxy[1] - pxy[0]))
        lbox_profiles.insert(tk.END,
            f"  {lbl}   ({x0:.1f}, {y0:.1f}) → ({x1:.1f}, {y1:.1f})   {length:.1f} m")
    n = len(picked_profile_lines)
    lbl_picked_count.config(
        text=f"{n} line{'s' if n != 1 else ''} picked" if n > 0 else "No lines picked yet")
    btn_clear_picked.config(state="normal" if n > 0 else "disabled")


def _clear_picked_lines():
    if not messagebox.askyesno("Clear", "Remove all interactively picked profile lines?"):
        return
    picked_profile_lines.clear()
    _refresh_profile_listbox()


# ─────────────────────────────────────────────────────────────────────
# Build main window
# ─────────────────────────────────────────────────────────────────────
root = tk.Tk()
_root_ref.append(root)   # enables _ui_error / _ui_info from worker threads
root.title("M3C2 Tool - CHECKPOINT13")
root.resizable(True, True)

style = ttk.Style()
style.theme_use("clam")
style.configure("TProgressbar", troughcolor="#d0d0d0", background="#1565c0", thickness=14)

PAD = dict(padx=6, pady=3)

_canvas    = tk.Canvas(root, highlightthickness=0)
_scrollbar = ttk.Scrollbar(root, orient="vertical", command=_canvas.yview)
_canvas.configure(yscrollcommand=_scrollbar.set)
_scrollbar.pack(side="right", fill="y")
_canvas.pack(side="left", fill="both", expand=True)

main_frame = tk.Frame(_canvas)
_canvas_win = _canvas.create_window((0, 0), window=main_frame, anchor="nw")

def _on_frame_resize(event): _canvas.configure(scrollregion=_canvas.bbox("all"))
def _on_canvas_resize(event): _canvas.itemconfig(_canvas_win, width=event.width)
def _on_mousewheel(event): _canvas.yview_scroll(int(-1*(event.delta/120)), "units")
main_frame.bind("<Configure>", _on_frame_resize)
_canvas.bind("<Configure>", _on_canvas_resize)
_canvas.bind_all("<MouseWheel>", _on_mousewheel)


def _lf(parent, text, row, col=0, colspan=3, sticky="ew", pady=(8, 2)):
    f = tk.LabelFrame(parent, text=text, font=("Arial", 9, "bold"),
                      padx=8, pady=6, bd=1, relief="groove")
    f.grid(row=row, column=col, columnspan=colspan, sticky=sticky, padx=8, pady=pady)
    f.columnconfigure(1, weight=1)
    return f

def _entry(parent, row, label, default, col=1, w=10):
    tk.Label(parent, text=label).grid(row=row, column=0, sticky="e", **PAD)
    e = tk.Entry(parent, width=w); e.insert(0, default)
    e.grid(row=row, column=col, sticky="w", **PAD)
    return e

# ── Files ──────────────────────────────────────────────────────────
ff = _lf(main_frame, "Input / Output Files", row=0)
for r, lbl in enumerate(["Reference LAS/LAZ", "Comparison LAS/LAZ", "Output LAS/LAZ (save)"]):
    tk.Label(ff, text=lbl).grid(row=r, column=0, sticky="e", **PAD)
entries = []
for r in range(3):
    e = tk.Entry(ff, width=55); e.grid(row=r, column=1, sticky="ew", **PAD); entries.append(e)
    s = (r == 2)
    tk.Button(ff, text="Browse\u2026", command=lambda e=e, s=s: browse_file(e, s)
              ).grid(row=r, column=2, **PAD)
ff.columnconfigure(1, weight=1)
entry_las1, entry_las2, entry_output = entries

# ── M3C2 Parameters ────────────────────────────────────────────────
pf = _lf(main_frame, "M3C2 Parameters", row=1)
entry_cyl = _entry(pf, 0, "Cylinder radius [m]", "2.0")
tk.Button(pf, text="Guess Parameters \u25ba", command=guess_parameters,
          bg="#1565c0", fg="white", font=("Arial", 9, "bold"),
          relief="flat", padx=6, cursor="hand2").grid(row=0, column=2, sticky="w", **PAD)
entry_normals = _entry(pf, 1, "Normal radii [m] (comma)", "1.5,2.5,3.5", w=22)
entry_spacing = _entry(pf, 2, "Corepoint spacing [m]", "1.0")
var_use_all_points = tk.BooleanVar()
tk.Checkbutton(pf, text="Use all points as corepoints",
               variable=var_use_all_points, command=toggle_spacing_entry
               ).grid(row=2, column=2, sticky="w", **PAD)
entry_max_depth = _entry(pf, 3, "Max depth detection [m]", "8.0")
tk.Label(pf, text="(0 = no limit)", fg="#666666").grid(row=3, column=2, sticky="w", **PAD)
tk.Label(pf, text="Confidence level").grid(row=4, column=0, sticky="e", **PAD)
var_confidence = tk.StringVar(value="0.95")
tk.OptionMenu(pf, var_confidence, "0.95", "0.99", "0.90").grid(row=4, column=1, sticky="w", **PAD)
var_use_manual_sigma = tk.BooleanVar()
tk.Checkbutton(pf, text="Use manual sigma values",
               variable=var_use_manual_sigma, command=toggle_sigma_entries
               ).grid(row=5, column=1, sticky="w", **PAD)
entry_sigma1 = _entry(pf, 6, "Sigma 1 [m]", "0.1"); entry_sigma1.config(state="disabled")
entry_sigma2 = _entry(pf, 7, "Sigma 2 [m]", "0.1"); entry_sigma2.config(state="disabled")

# ── Output Options ──────────────────────────────────────────────────
of = _lf(main_frame, "Output Options", row=2)
var_txt = tk.BooleanVar()
tk.Checkbutton(of, text="Also save results to TXT", variable=var_txt
               ).grid(row=0, column=0, columnspan=4, sticky="w", **PAD)
tk.Label(of, text="LAS extra fields:").grid(row=1, column=0, sticky="ne", **PAD)
las_fld_fr = tk.Frame(of); las_fld_fr.grid(row=1, column=1, columnspan=3, sticky="w", **PAD)
LAS_OPTIONAL_FIELDS = ["Nz","Nxy","DX","DY","DZ","LOD","SPREAD1","SPREAD2"]
las_field_vars = {}; las_field_chks = {}
for i, fname in enumerate(LAS_OPTIONAL_FIELDS):
    var = tk.BooleanVar(value=True); las_field_vars[fname] = var
    chk = tk.Checkbutton(las_fld_fr, text=fname, variable=var); las_field_chks[fname] = chk
    chk.grid(row=i // 4, column=i % 4, sticky="w", padx=6)

# ── Raster + DEM ────────────────────────────────────────────────────
RASTER_FIELDS  = ["M3C2_DISTANCE","LOD","SPREAD1","SPREAD2","SIGNIFICANT","Nz","Nxy","DX","DY","DZ"]
INTERP_METHODS = ["IDW","TIN","Nearest","Kriging"]
raster_field_vars = {}; raster_field_chks = {}

rd_row = tk.Frame(main_frame)
rd_row.grid(row=3, column=0, columnspan=3, sticky="ew", padx=8, pady=(8, 2))
rd_row.columnconfigure(0, weight=1); rd_row.columnconfigure(1, weight=1)

rf = tk.LabelFrame(rd_row, text="Raster Export  (Core Points \u2192 GeoTIFF)",
                   font=("Arial",9,"bold"), padx=8, pady=6, bd=1, relief="groove")
rf.grid(row=0, column=0, sticky="nsew", padx=(0,4)); rf.columnconfigure(1, weight=1)
var_make_raster = tk.BooleanVar()
tk.Checkbutton(rf, text="Enable raster export", variable=var_make_raster,
               command=toggle_raster_entries).grid(row=0, column=0, columnspan=3, sticky="w", padx=4, pady=2)
tk.Label(rf, text="Resolution [m]").grid(row=1, column=0, sticky="e", **PAD)
entry_raster_res = tk.Entry(rf, width=8, state="disabled"); entry_raster_res.insert(0,"0.5")
entry_raster_res.grid(row=1, column=1, sticky="w", **PAD)
tk.Label(rf, text="Interpolation").grid(row=2, column=0, sticky="e", **PAD)
var_raster_interp = tk.StringVar(value="IDW")
option_raster_interp = tk.OptionMenu(rf, var_raster_interp, *INTERP_METHODS)
option_raster_interp.config(state="disabled", width=10)
option_raster_interp.grid(row=2, column=1, sticky="w", **PAD)
tk.Label(rf, text="Fields:").grid(row=3, column=0, sticky="ne", **PAD)
rf_fld = tk.Frame(rf); rf_fld.grid(row=3, column=1, columnspan=2, sticky="w", **PAD)
for i, fname in enumerate(RASTER_FIELDS):
    var = tk.BooleanVar(value=(fname=="M3C2_DISTANCE")); raster_field_vars[fname] = var
    chk = tk.Checkbutton(rf_fld, text=fname, variable=var, state="disabled")
    raster_field_chks[fname] = chk; chk.grid(row=i//3, column=i%3, sticky="w", padx=4)

df = tk.LabelFrame(rd_row, text="DEM Export  (Z values \u2192 GeoTIFF)",
                   font=("Arial",9,"bold"), padx=8, pady=6, bd=1, relief="groove")
df.grid(row=0, column=1, sticky="nsew", padx=(4,0)); df.columnconfigure(1, weight=1)
var_make_dem = tk.BooleanVar()
tk.Checkbutton(df, text="Enable DEM export", variable=var_make_dem,
               command=toggle_dem_entries).grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=2)
tk.Label(df, text="Resolution [m]").grid(row=1, column=0, sticky="e", **PAD)
entry_dem_res = tk.Entry(df, width=8, state="disabled"); entry_dem_res.insert(0,"0.1")
entry_dem_res.grid(row=1, column=1, sticky="w", **PAD)
tk.Label(df, text="Interpolation").grid(row=2, column=0, sticky="e", **PAD)
var_dem_interp = tk.StringVar(value="IDW")
option_dem_interp = tk.OptionMenu(df, var_dem_interp, *INTERP_METHODS)
option_dem_interp.config(state="disabled", width=10)
option_dem_interp.grid(row=2, column=1, sticky="w", **PAD)

# ── Quiver Export ───────────────────────────────────────────────────
qf = _lf(main_frame, "Quiver Export", row=4); qf.columnconfigure(1, weight=1)
var_quiver_normal = tk.BooleanVar(); var_quiver_motion = tk.BooleanVar()
var_quiver_png    = tk.BooleanVar(value=True)
var_quiver_html   = tk.BooleanVar(value=True)
var_quiver_gpkg   = tk.BooleanVar(value=True)
_qfmt_chks = []

mode_fr = tk.Frame(qf); mode_fr.grid(row=0, column=0, columnspan=4, sticky="w", padx=4, pady=(2,4))
tk.Checkbutton(mode_fr, text="Export normal-direction quiver",
               variable=var_quiver_normal, command=toggle_quiver_entries,
               font=("Arial",9)).grid(row=0, column=0, sticky="w", padx=(0,20))
tk.Checkbutton(mode_fr, text="Export motion vector  (DX · DY)",
               variable=var_quiver_motion, command=toggle_quiver_entries,
               font=("Arial",9)).grid(row=0, column=1, sticky="w")

tk.Label(qf, text="Grid cell [m]").grid(row=1, column=0, sticky="e", **PAD)
entry_quiver_grid = tk.Entry(qf, width=8, state="disabled"); entry_quiver_grid.insert(0,"1.0")
entry_quiver_grid.grid(row=1, column=1, sticky="w", **PAD)
tk.Label(qf, text="(independent of corepoint spacing)", fg="#666666").grid(row=1, column=2, sticky="w", **PAD)

tk.Label(qf, text="Formats:").grid(row=2, column=0, sticky="e", **PAD)
qfmt_fr = tk.Frame(qf); qfmt_fr.grid(row=2, column=1, columnspan=3, sticky="w", **PAD)
for i, (var, lbl, tip) in enumerate([(var_quiver_png,"PNG","static image"),
                                       (var_quiver_html,"HTML","interactive Plotly"),
                                       (var_quiver_gpkg,"GPKG","QGIS vector arrows")]):
    chk = tk.Checkbutton(qfmt_fr, text=f"{lbl} ({tip})", variable=var, state="disabled")
    chk.grid(row=0, column=i, sticky="w", padx=10); _qfmt_chks.append(chk)

tk.Label(qf, text="Both modes can run together.  Normal → _quiver_normal.*  ·  Motion → _quiver_motion.*",
         fg="#666666", font=("Arial",8), wraplength=580, justify="left"
         ).grid(row=3, column=0, columnspan=4, sticky="w", padx=4, pady=(0,2))

# ══════════════════════════════════════════════════════════════════════
# ── Profile Cut Export ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════
prf = _lf(main_frame, "Profile Cut Export  (interactive picker  or  TXT \u2192 PNG)", row=5)
prf.columnconfigure(1, weight=1)

var_make_profile = tk.BooleanVar()
tk.Checkbutton(prf, text="Enable profile export",
               variable=var_make_profile, command=toggle_profile_entries
               ).grid(row=0, column=0, columnspan=3, sticky="w", padx=4, pady=(2, 6))

# ── Corridor width (shared) ─────────────────────────────────────────
tk.Label(prf, text="Corridor width [m]").grid(row=1, column=0, sticky="e", **PAD)
entry_profile_width = tk.Entry(prf, width=8, state="disabled")
entry_profile_width.insert(0, "50.0"); entry_profile_width.grid(row=1, column=1, sticky="w", **PAD)
tk.Label(prf, text="points within ± width/2 of each line",
         fg="#666666").grid(row=1, column=2, sticky="w", **PAD)

# ── Interactive picker ──────────────────────────────────────────────
sep1 = ttk.Separator(prf, orient="horizontal")
sep1.grid(row=2, column=0, columnspan=3, sticky="ew", padx=4, pady=(6, 4))

pick_fr = tk.Frame(prf)
pick_fr.grid(row=3, column=0, columnspan=3, sticky="ew", padx=4, pady=2)
pick_fr.columnconfigure(2, weight=1)

btn_pick_profiles = tk.Button(
    pick_fr,
    text="\u25b6   Pick profiles interactively…",
    command=_open_profile_picker,
    state="disabled",
    font=("Arial", 9, "bold"), relief="flat",
    bg="#1565c0", fg="white", padx=12, pady=5,
    activebackground="#0d47a1", activeforeground="white",
    cursor="hand2")
btn_pick_profiles.grid(row=0, column=0, sticky="w", padx=(0, 12))

tk.Label(pick_fr, text="reference = blue   ·   comparison = red",
         fg="#666666", font=("Arial", 8)).grid(row=0, column=1, sticky="w")

# Picked-lines listbox
lbox_fr = tk.Frame(prf)
lbox_fr.grid(row=4, column=0, columnspan=3, sticky="ew", padx=4, pady=(4, 2))
lbox_fr.columnconfigure(0, weight=1)

lbl_picked_count = tk.Label(lbox_fr, text="No lines picked yet",
                              font=("Arial", 8), fg="#888888")
lbl_picked_count.grid(row=0, column=0, sticky="w")

btn_clear_picked = tk.Button(lbox_fr, text="Clear picked lines",
                              command=_clear_picked_lines, state="disabled",
                              font=("Arial", 8), relief="flat", bg="#eeeeee", padx=6)
btn_clear_picked.grid(row=0, column=1, sticky="e")

lbox_profiles = tk.Listbox(lbox_fr, height=4, font=("Courier", 8),
                             selectmode="browse", relief="sunken", bd=1)
lbox_profiles.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))
_lbox_sb = ttk.Scrollbar(lbox_fr, orient="vertical", command=lbox_profiles.yview)
_lbox_sb.grid(row=1, column=2, sticky="ns")
lbox_profiles.config(yscrollcommand=_lbox_sb.set)

# ── TXT fallback ────────────────────────────────────────────────────
sep2 = ttk.Separator(prf, orient="horizontal")
sep2.grid(row=5, column=0, columnspan=3, sticky="ew", padx=4, pady=(6, 4))

tk.Label(prf, text="Or load from TXT  (Y X per line):",
         fg="#555555", font=("Arial", 8)).grid(row=6, column=0, columnspan=3,
                                                sticky="w", padx=4, pady=(0, 2))
tk.Label(prf, text="Profile TXT").grid(row=7, column=0, sticky="e", **PAD)
entry_profile_txt = tk.Entry(prf, width=46, state="disabled")
entry_profile_txt.grid(row=7, column=1, sticky="ew", **PAD)

def _browse_profile():
    p = filedialog.askopenfilename(
        filetypes=[("Text files", "*.txt *.csv *.dat"), ("All files", "*.*")])
    if p:
        entry_profile_txt.delete(0, tk.END); entry_profile_txt.insert(0, p)

btn_profile_browse = tk.Button(prf, text="Browse\u2026", state="disabled",
                                command=_browse_profile)
btn_profile_browse.grid(row=7, column=2, **PAD)

tk.Label(prf,
         text="TXT adds one extra profile on top of any interactively picked lines.  "
              "Output files: _profile_01.png, _profile_02.png …",
         fg="#666666", font=("Arial", 8), wraplength=560, justify="left"
         ).grid(row=8, column=0, columnspan=3, sticky="w", padx=4, pady=(0, 2))

# ── Progress bars ────────────────────────────────────────────────────
ttk.Separator(main_frame, orient="horizontal").grid(
    row=6, column=0, columnspan=3, sticky="ew", padx=8, pady=(10, 4))

lbl_current_task = tk.Label(main_frame, text="Current Task: Ready", anchor="w", fg="#444444")
lbl_current_task.grid(row=7, column=0, columnspan=3, sticky="w", padx=10)
progress_current = ttk.Progressbar(main_frame, length=600, mode="determinate")
progress_current.grid(row=8, column=0, columnspan=3, padx=10, pady=(0,4), sticky="ew")

lbl_overall = tk.Label(main_frame, text="Overall Progress: Ready", anchor="w", fg="#444444")
lbl_overall.grid(row=9, column=0, columnspan=3, sticky="w", padx=10)
progress_overall = ttk.Progressbar(main_frame, length=600, mode="determinate")
progress_overall.grid(row=10, column=0, columnspan=3, padx=10, pady=(0,6), sticky="ew")

progress_mgr = ProgressManager(
    current_bar=progress_current, current_label=lbl_current_task,
    overall_bar=progress_overall, overall_label=lbl_overall)

# ── Start button ─────────────────────────────────────────────────────
btn_start = tk.Button(main_frame, text="\u25ba   Run M3C2 Analysis",
                      command=start_m3c2_threaded,
                      bg="#2e7d32", fg="white", font=("Arial", 12, "bold"),
                      relief="flat", padx=30, pady=10, cursor="hand2",
                      activebackground="#1b5e20", activeforeground="white")
btn_start.grid(row=11, column=0, columnspan=3, pady=(4, 16))

main_frame.columnconfigure(1, weight=1)
root.mainloop()