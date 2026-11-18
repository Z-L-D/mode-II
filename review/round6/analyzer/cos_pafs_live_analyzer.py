#!/usr/bin/env python3
"""
CoS-PAFS Live Analyzer v3 — Enhanced Dashboard
================================================
v3 additions (2026-05-06):
  - Mean-profile β(g) plot (3-crossing method alongside cloud)
  - Early vs Late token comparison
  - Key Results summary panel
  - Interval decomposition (Coxeter)
  - Bootstrap outer ratio histogram
  - Attn vs MLP β(g) comparison
  - φ_micro classifier correction (fermionic inversion fix)
  - Export prefix from source CSV name
  - Improved dark styling

v2 base:
  1. Export all graphs to CSV source directory
  2. Interactive 3D Plotly graphs (orbit, zoom)
  3. Adjustable binning (temporal and generational)
  4. Data windowing (layer range, step/token range, generation range)
  5. GIF/MP4 export of generation process
  6. CMB power spectrum measurement at E→C boundary
  7. Phase boundary auto-detection with live overlay
  8. Prompt entropy indicator

Usage:
    python cos_pafs_live_analyzer_v3.py
    # Then open http://127.0.0.1:7860

Requirements:
    pip install gradio pandas numpy matplotlib scipy plotly imageio
"""

import os
import sys
import time
import threading
import tempfile
from pathlib import Path
from datetime import datetime

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    from scipy.interpolate import CubicSpline, UnivariateSpline
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import imageio
    _HAS_IMAGEIO = True
except ImportError:
    _HAS_IMAGEIO = False

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Matplotlib defaults
# ---------------------------------------------------------------------------
DARK_STYLE = {
    "figure.dpi": 120, "font.family": "sans-serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "#1a1a1a", "axes.facecolor": "#1a1a1a",
    "axes.edgecolor": "#888888", "axes.labelcolor": "#cccccc",
    "xtick.color": "#888888", "ytick.color": "#888888",
    "text.color": "#cccccc", "grid.color": "#444444", "grid.alpha": 0.3,
}
plt.rcParams.update(DARK_STYLE)


# =============================================================================
# Data Manager (enhanced with windowing)
# =============================================================================
class DataManager:
    def __init__(self):
        self.df = None
        self._raw_df = None  # unfiltered
        self._lock = threading.Lock()
        self._watch_paths = []
        self._watch_thread = None
        self._watch_running = False
        self._last_mtime = {}
        self._poll_interval = 5
        self._status = "Idle"
        self._source_dir = None
        self._source_filename = None

    def default_prefix(self):
        if self._source_filename:
            return Path(self._source_filename).stem
        return f"cospafs_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    @staticmethod
    def _deduplicate_gen_idx(dfs):
        """Reassign gen_idx across DataFrames so each file gets unique values.

        When multiple CSVs share the same gen_idx (e.g. all have gen_idx=1),
        groupby("gen_idx") merges them into one generation, collapsing
        inter-generation variation.  This remaps gen_idx per file so that
        each file occupies a unique range, preserving any internal structure
        (e.g. a single file with gen_idx 1..5 keeps 5 distinct gens).
        """
        if len(dfs) <= 1:
            return dfs
        # Check for collision: does any gen_idx appear in more than one file?
        seen = set()
        collision = False
        for df in dfs:
            if "gen_idx" not in df.columns:
                continue
            file_vals = set(df["gen_idx"].unique())
            if file_vals & seen:
                collision = True
                break
            seen |= file_vals
        if not collision:
            return dfs
        # Remap: each file's gen_idx is offset so ranges don't overlap
        offset = 0
        remapped = []
        for df in dfs:
            df = df.copy()
            if "gen_idx" in df.columns:
                local_min = df["gen_idx"].min()
                df["gen_idx"] = df["gen_idx"] - local_min + offset + 1
                offset = df["gen_idx"].max()
            remapped.append(df)
        return remapped

    def load_uploads(self, files):
        if not files:
            self._status = "No files uploaded"
            return None
        dfs, names = [], []
        for f in files:
            try:
                path = f.name if hasattr(f, "name") else str(f)
                dfs.append(pd.read_csv(path))
                names.append(Path(path).name)
                self._source_dir = str(Path(path).parent)
                if self._source_filename is None:
                    self._source_filename = Path(path).name
            except Exception as e:
                names.append(f"ERROR: {e}")
        with self._lock:
            dfs = self._deduplicate_gen_idx(dfs)
            self._raw_df = pd.concat(dfs, ignore_index=True) if dfs else None
            self.df = self._raw_df.copy() if self._raw_df is not None else None
            self._status = f"Loaded {len(dfs)} file(s): {', '.join(names)}"
        return self.summary()

    def load_paths(self, path_text):
        paths = [p.strip() for p in path_text.split(",") if p.strip()]
        valid = [p for p in paths if os.path.isfile(p)]
        if not valid:
            self._status = f"No valid paths (checked {len(paths)})"
            return None
        dfs = [pd.read_csv(p) for p in valid]
        with self._lock:
            dfs = self._deduplicate_gen_idx(dfs)
            self._raw_df = pd.concat(dfs, ignore_index=True)
            self.df = self._raw_df.copy()
            self._watch_paths = valid
            self._last_mtime = {p: os.path.getmtime(p) for p in valid}
            self._source_dir = str(Path(valid[0]).parent)
            self._status = f"Loaded {len(valid)} path(s) | Rows: {len(self.df)}"
        return self.summary()

    def apply_window(self, layer_min, layer_max, token_min, token_max, gen_min, gen_max, sigma_min=0.0, sigma_max=1.0):
        """Apply windowing filters to the raw data."""
        with self._lock:
            if self._raw_df is None:
                return "No data loaded"
            df = self._raw_df.copy()
            df = df[(df["layer"] >= layer_min) & (df["layer"] <= layer_max)]
            df = df[(df["token_idx"] >= token_min) & (df["token_idx"] <= token_max)]
            df = df[(df["gen_idx"] >= gen_min) & (df["gen_idx"] <= gen_max)]
            if "sigma" in df.columns and (sigma_min > 0.0 or sigma_max < 1.0):
                df = df[(df["sigma"] >= sigma_min) & (df["sigma"] <= sigma_max)]
            self.df = df
            self._status = f"Windowed: L[{layer_min}-{layer_max}] T[{token_min}-{token_max}] G[{gen_min}-{gen_max}] σ[{sigma_min:.2f}-{sigma_max:.2f}] | {len(df)} rows"
        return self._status

    def reset_window(self):
        with self._lock:
            if self._raw_df is not None:
                self.df = self._raw_df.copy()
                self._status = f"Window reset | {len(self.df)} rows"
        return self._status

    def start_watch(self, interval):
        if not self._watch_paths:
            self._status = "No watch paths — use 'Load Paths' first"
            return self._status
        self._poll_interval = max(1, int(interval))
        self._watch_running = True
        self._watch_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._watch_thread.start()
        self._status = f"Live watch ({self._poll_interval}s) — {len(self._watch_paths)} file(s)"
        return self._status

    def stop_watch(self):
        self._watch_running = False
        self._status = "Watch stopped"
        return self._status

    def get_df(self):
        with self._lock:
            return self.df.copy() if self.df is not None else None

    def get_ranges(self):
        """Get min/max for all dimensions."""
        df = self._raw_df if self._raw_df is not None else self.df
        if df is None:
            return (0, 31, 0, 255, 0, 0)
        return (
            int(df["layer"].min()), int(df["layer"].max()),
            int(df["token_idx"].min()), int(df["token_idx"].max()),
            int(df["gen_idx"].min()), int(df["gen_idx"].max()),
        )

    def summary(self):
        df = self.get_df()
        if df is None:
            return "No data loaded."
        lines = [
            f"Rows: {len(df):,}",
            f"Generations: {df['gen_idx'].nunique()}",
            f"Layers: {df['layer'].nunique()} ({df['layer'].min()}-{df['layer'].max()})",
            f"Token range: {df['token_idx'].min()} — {df['token_idx'].max()}",
            f"Hidden dim: {df['hidden_dim'].iloc[0] if 'hidden_dim' in df.columns else 'N/A'}",
        ]
        if 'submodule' in df.columns:
            subs = sorted(df['submodule'].dropna().unique().tolist())
            lines.append(f"Submodules: {', '.join(subs)} ({'dual-hook' if len(subs) > 1 else 'single-hook'})")
        if 'phi_micro_skip' in df.columns and df['phi_micro_skip'].abs().sum() > 1e-6:
            lines.append("phi_micro_skip: present")
        return "\n".join(lines)

    def status(self):
        return self._status

    def _poll_loop(self):
        while self._watch_running:
            time.sleep(self._poll_interval)
            changed = False
            for p in self._watch_paths:
                if not os.path.isfile(p):
                    continue
                mtime = os.path.getmtime(p)
                if mtime > self._last_mtime.get(p, 0):
                    self._last_mtime[p] = mtime
                    changed = True
            if changed:
                try:
                    dfs = [pd.read_csv(p) for p in self._watch_paths if os.path.isfile(p)]
                    with self._lock:
                        dfs = self._deduplicate_gen_idx(dfs)
                        self._raw_df = pd.concat(dfs, ignore_index=True)
                        self.df = self._raw_df.copy()
                        self._status = f"Reloaded {datetime.now().strftime('%H:%M:%S')} | {len(self.df)} rows"
                except Exception as e:
                    self._status = f"Reload error: {e}"


# =============================================================================
# Utilities
# =============================================================================
def smooth(y, w=5):
    y = np.asarray(y, dtype=float)
    if len(y) < w:
        return y
    pad = w // 2
    return np.convolve(np.pad(y, pad, mode="edge"), np.ones(w) / w, mode="valid")


def detect_phase_boundaries(df):
    """Detect E→C and C→C boundaries from token/step trajectory."""
    if df is None or df.empty:
        return []
    tm = df.groupby("token_idx")["g_micro"].mean()
    t, g = tm.index.values, tm.values
    if len(g) < 10:
        return []
    g_s = smooth(g, w=7)
    dg = np.diff(g_s)
    crossings = []
    for i in range(len(dg) - 1):
        if dg[i] * dg[i + 1] < 0:
            frac = -dg[i] / (dg[i + 1] - dg[i])
            x_cross = t[i] + frac * (t[i + 1] - t[i])
            direction = "peak" if dg[i] > 0 else "trough"
            crossings.append((x_cross, direction))
    return crossings


# =============================================================================
# 2D Plot Functions (Matplotlib) — with binning support
# =============================================================================
def plot_depth_profile(df, bin_size=1):
    if df is None or df.empty:
        return None
    if bin_size > 1:
        df = df.copy()
        df["layer"] = (df["layer"] // bin_size) * bin_size
    layer_means = df.groupby("layer")["g_micro"].agg(["mean", "std", "count"])
    layer_means["sem"] = layer_means["std"] / np.sqrt(layer_means["count"])
    layers = layer_means.index.values
    g_mean = layer_means["mean"].values
    g_sem = layer_means["sem"].values

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.errorbar(layers, g_mean, yerr=g_sem, color="#00d4aa", lw=1.8, capsize=3,
                marker="o", markersize=5, markerfacecolor="#00d4aa",
                markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xlabel("Layer L")
    ax.set_ylabel(r"$\langle g_{\mathrm{micro}}(L) \rangle$")
    ax.set_title("Depth Profile")
    ax.grid(True)

    ax = axes[1]
    if len(layers) > 2:
        odd_g = g_mean[::2]; even_g = g_mean[1::2]
        odd_l = layers[::2]; even_l = layers[1::2]
        ax.bar(odd_l - 0.2, odd_g, width=0.4, color="#00d4aa", alpha=0.8, label="Even-indexed")
        ax.bar(even_l + 0.2, even_g[:len(even_l)], width=0.4, color="#ff6b6b", alpha=0.8, label="Odd-indexed")
        ax.set_xlabel("Layer L"); ax.set_ylabel(r"$\langle g(L) \rangle$")
        ax.set_title("Sublattice Test"); ax.legend(); ax.grid(True)
    plt.tight_layout()
    return fig


def plot_token_trajectory(df, bin_size=1):
    if df is None or df.empty:
        return None
    if bin_size > 1:
        df = df.copy()
        df["token_idx"] = (df["token_idx"] // bin_size) * bin_size

    gen_groups = df.groupby("gen_idx")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    all_traj = []
    ax0 = axes[0]
    for gen_idx, gdf in gen_groups:
        tm = gdf.groupby("token_idx")["g_micro"].mean()
        t, g = tm.index.values, tm.values
        if len(t) > 10:
            g_s = smooth(g, w=7)
            ax0.plot(t, g_s, alpha=0.35, lw=0.8, color="#888888")
            all_traj.append(g[:min(len(g), 500)])
    ax0.set_xlabel("Step / Token index")
    ax0.set_ylabel(r"$\langle g_{\mathrm{micro}}(t) \rangle_L$")
    ax0.set_title(f"Step Trajectories ({len(all_traj)} gens)")

    # Mark phase boundaries
    boundaries = detect_phase_boundaries(df)
    for x, d in boundaries:
        color = "#ff6b6b" if d == "peak" else "#00d4aa"
        ax0.axvline(x, color=color, ls=":", lw=1.5, alpha=0.6)
    ax0.grid(True)

    ax1 = axes[1]
    if all_traj:
        min_len = min(len(x) for x in all_traj)
        aligned = np.array([x[:min_len] for x in all_traj])
        mean_t = smooth(np.nanmean(aligned, axis=0), w=7)
        std_t = smooth(np.nanstd(aligned, axis=0), w=7)
        t_c = np.arange(min_len)
        ax1.plot(t_c, mean_t, color="#00d4aa", lw=2)
        ax1.fill_between(t_c, mean_t - std_t, mean_t + std_t, color="#00d4aa", alpha=0.2)

        # Phase boundary overlay on mean
        for x, d in boundaries:
            if x < min_len:
                color = "#ff6b6b" if d == "peak" else "#00d4aa"
                ax1.axvline(x, color=color, ls=":", lw=1.5, alpha=0.6)

        ax1.set_xlabel("Step / Token index")
        ax1.set_ylabel(r"$\langle g(t) \rangle$")
        ax1.set_title("Mean Trajectory ±1σ")
        ax1.grid(True)
    plt.tight_layout()
    return fig


def plot_heatmap(df, temporal_bin=1):
    if df is None or df.empty:
        return None
    if temporal_bin > 1:
        df = df.copy()
        df["token_idx"] = (df["token_idx"] // temporal_bin) * temporal_bin

    mats = []
    for _, gdf in df.groupby("gen_idx"):
        p = gdf.pivot_table(index="layer", columns="token_idx", values="g_micro", aggfunc="mean")
        mats.append(p.values)
    if not mats:
        return None
    min_l = min(m.shape[0] for m in mats)
    min_t = min(m.shape[1] for m in mats)
    aligned = np.array([m[:min_l, :min_t] for m in mats])
    mean_m = np.nanmean(aligned, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    im0 = axes[0].imshow(mean_m, aspect="auto", origin="lower", cmap="viridis", interpolation="nearest")
    axes[0].set_xlabel("Step / Token index"); axes[0].set_ylabel("Layer L")
    axes[0].set_title(r"$\langle g(L, t) \rangle$")
    plt.colorbar(im0, ax=axes[0], label=r"$g_{\mathrm{micro}}$")

    layer_means = np.nanmean(mean_m, axis=1, keepdims=True)
    layer_means[layer_means < 1e-8] = 1e-8
    norm = mean_m / layer_means
    im1 = axes[1].imshow(norm, aspect="auto", origin="lower", cmap="RdBu_r", vmin=0.8, vmax=1.2, interpolation="nearest")
    axes[1].set_xlabel("Step / Token index"); axes[1].set_ylabel("Layer L")
    axes[1].set_title("Normalized by Layer Mean")
    plt.colorbar(im1, ax=axes[1], label="Normalized g")
    plt.tight_layout()
    return fig


def plot_beta_layer(df):
    if df is None or df.empty:
        return None
    all_g, all_beta = [], []
    for _, gdf in df.groupby("gen_idx"):
        p = gdf.pivot_table(index="layer", columns="token_idx", values="g_micro", aggfunc="mean")
        gmat = p.values
        beta = np.diff(gmat, axis=0)
        g_for = gmat[:-1, :]
        m = ~np.isnan(beta) & ~np.isnan(g_for)
        all_g.append(g_for[m])
        all_beta.append(beta[m])
    if not all_g:
        return None
    g = np.concatenate(all_g)
    beta = np.concatenate(all_beta)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(g, beta, s=2, alpha=0.12, color="#ff6b6b", edgecolors="none")

    edges = np.linspace(g.min(), g.max(), 30)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bmean = np.full_like(centers, np.nan)
    bstd = np.full_like(centers, np.nan)
    for i in range(len(centers)):
        mask = (g >= edges[i]) & (g < edges[i + 1])
        if mask.sum() >= 10:
            bmean[i] = beta[mask].mean()
            bstd[i] = beta[mask].std(ddof=1)
    valid = ~np.isnan(bmean)
    ax.plot(centers[valid], bmean[valid], color="white", lw=2)
    ax.fill_between(centers[valid], (bmean - bstd)[valid], (bmean + bstd)[valid], color="white", alpha=0.15)
    ax.axhline(0, color="#888888", lw=0.5)

    crossings = []
    for i in range(len(centers) - 1):
        if np.isnan(bmean[i]) or np.isnan(bmean[i + 1]):
            continue
        if bmean[i] * bmean[i + 1] < 0:
            x0 = centers[i] + (centers[i + 1] - centers[i]) * (-bmean[i]) / (bmean[i + 1] - bmean[i])
            direction = "neg→pos" if bmean[i] < 0 else "pos→neg"
            crossings.append((x0, direction))
            color = "#00d4aa" if direction == "neg→pos" else "#ff6b6b"
            ax.axvline(x0, color=color, lw=1, ls="--", alpha=0.8)

    title = r"$\beta_{\mathrm{layer}}(g)$ — Layer-Direction Flow"
    if crossings:
        title += "\n" + ", ".join(f"g*={x:.4f} ({d})" for x, d in crossings)
    ax.set_title(title)
    ax.set_xlabel(r"$g_{\mathrm{micro}}$")
    ax.set_ylabel(r"$\beta_{\mathrm{layer}}(g) = g(L+1) - g(L)$")
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_spiral_analysis(df):
    if df is None or df.empty:
        return None
    has_gap = "spectral_gap" in df.columns
    has_ent = "sv_entropy" in df.columns
    if not has_gap and not has_ent:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No spectral_gap / sv_entropy columns",
                ha="center", va="center", transform=ax.transAxes, color="#888888", fontsize=12)
        ax.set_title("Spiral Analysis — No Data")
        return fig

    sec_col = "spectral_gap" if has_gap else "sv_entropy"
    sec_label = r"Spectral Gap ($\sigma_1/\sigma_2$)" if has_gap else r"SV Entropy $H(\sigma)$"

    layer_g = df.groupby("layer")["g_micro"].agg(["mean", "std"]).rename(columns={"mean": "g_mean", "std": "g_std"})
    layer_sec = df.groupby("layer")[sec_col].agg(["mean", "std"]).rename(columns={"mean": "sec_mean", "std": "sec_std"})
    ld = layer_g.join(layer_sec)
    layers = ld.index.values
    g_mean = ld["g_mean"].values
    sec_mean = ld["sec_mean"].values
    g_std = ld["g_std"].values
    beta_layer = np.diff(g_mean)
    beta_layers = layers[:-1]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel A: secondary at beta crossings
    ax = axes[0, 0]
    ax.plot(layers, sec_mean, color="#00d4aa", lw=2, marker="o", markersize=4, label=sec_label)
    ax.set_xlabel("Layer L"); ax.set_ylabel(sec_label, color="#00d4aa")
    ax2 = ax.twinx()
    ax2.plot(beta_layers, beta_layer, color="#888888", lw=1.5, ls="--", alpha=0.7)
    ax2.axhline(0, color="#444444", lw=0.5); ax2.set_ylabel(r"$\beta_{\mathrm{layer}}$")
    crossing_layers = []
    for i in range(len(beta_layer) - 1):
        if beta_layer[i] * beta_layer[i + 1] < 0:
            frac = -beta_layer[i] / (beta_layer[i + 1] - beta_layer[i])
            cl = beta_layers[i] + frac
            crossing_layers.append(cl)
            ax.axvline(cl, color="#ff6b6b", ls=":", lw=1.5, alpha=0.7)
    ax.set_title("A: Secondary at Beta Crossings"); ax.grid(True)

    # Panel B: phase portrait
    ax = axes[0, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    for i in range(len(layers) - 1):
        ax.annotate("", xy=(g_mean[i + 1], sec_mean[i + 1]), xytext=(g_mean[i], sec_mean[i]),
                    arrowprops=dict(arrowstyle="->", color=colors[i], lw=1.5))
    sc = ax.scatter(g_mean, sec_mean, c=layers, cmap="viridis", s=30, zorder=5, edgecolors="white", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label="Layer L")
    for cl in crossing_layers:
        idx = int(cl)
        if idx < len(g_mean) - 1:
            frac = cl - idx
            cx = g_mean[idx] * (1 - frac) + g_mean[idx + 1] * frac
            cy = sec_mean[idx] * (1 - frac) + sec_mean[idx + 1] * frac
            ax.plot(cx, cy, "rx", markersize=10, markeredgewidth=2, zorder=10)
    ax.set_xlabel(r"$\langle g_{\mathrm{micro}}(L) \rangle$"); ax.set_ylabel(sec_label)
    ax.set_title("B: Phase Portrait"); ax.grid(True)

    # Panel C: variance
    ax = axes[1, 0]
    ax.bar(layers, g_std, color="#ff9f43", alpha=0.7)
    for cl in crossing_layers:
        ax.axvline(cl, color="#ff6b6b", ls=":", lw=1.5, alpha=0.7)
    ax.set_xlabel("Layer L"); ax.set_ylabel(r"$\mathrm{std}(g_{\mathrm{micro}})$")
    ax.set_title("C: g Variance by Layer"); ax.grid(True)

    # Panel D: crossing ratios
    ax = axes[1, 1]
    if len(crossing_layers) >= 2:
        g_stars = sorted(crossing_layers)
        ratios = [g_stars[i + 1] / g_stars[i] for i in range(len(g_stars) - 1)]
        x_pos = np.arange(len(ratios))
        ax.bar(x_pos, ratios, color="#9b59b6", alpha=0.8, width=0.4)
        ax.axhline(2.0, color="#ff6b6b", ls="--", lw=1, alpha=0.7, label="ratio=2")
        ax.axhline(1.618, color="#00d4aa", ls="--", lw=1, alpha=0.7, label="φ=1.618")
        ax.axhline(1.414, color="#6baed6", ls="--", lw=1, alpha=0.7, label="√2=1.414")
        ax.axhline(1.732, color="#fdae6b", ls="--", lw=1, alpha=0.7, label="√3=1.732")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{r:.3f}" for r in ratios])
        for i, r in enumerate(ratios):
            if r > 1:
                theta = np.pi / np.log(r)
                ax.annotate(f"θ≈{theta:.2f}", xy=(i, r), xytext=(i + 0.15, r + 0.05), fontsize=9, color="#bbbbbb")
        ax.set_title("D: Crossing Ratios → Coxeter Class"); ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "Need ≥2 crossings\nfor ratio analysis",
                ha="center", va="center", transform=ax.transAxes, color="#888888")
        ax.set_title("D: Crossing Ratios")
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_phase_portrait(df):
    if df is None or df.empty:
        return None
    has_gap = "spectral_gap" in df.columns
    has_ent = "sv_entropy" in df.columns
    if not has_gap and not has_ent:
        return None
    sec_col = "spectral_gap" if has_gap else "sv_entropy"
    sec_label = r"Spectral Gap ($\sigma_1/\sigma_2$)" if has_gap else r"SV Entropy $H(\sigma)$"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    gen_groups = df.groupby("gen_idx")
    ax0 = axes[0]
    n_plot = 0
    for _, gdf in gen_groups:
        if n_plot >= 50:
            break
        lm = gdf.groupby("layer").agg(g=("g_micro", "mean"), sec=(sec_col, "mean"))
        if len(lm) < 5:
            continue
        ax0.plot(lm["g"].values, lm["sec"].values, alpha=0.3, lw=0.8, color="#ff6b6b")
        n_plot += 1
    ax0.set_xlabel(r"$g_{\mathrm{micro}}(L)$"); ax0.set_ylabel(sec_label)
    ax0.set_title(f"Per-Gen Phase Portraits (n={n_plot})"); ax0.grid(True)

    ax1 = axes[1]
    all_lm = df.groupby("layer").agg(g=("g_micro", "mean"), sec=(sec_col, "mean"))
    g_vals, sec_vals, layers = all_lm["g"].values, all_lm["sec"].values, all_lm.index.values
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    for i in range(len(layers) - 1):
        ax1.annotate("", xy=(g_vals[i + 1], sec_vals[i + 1]), xytext=(g_vals[i], sec_vals[i]),
                     arrowprops=dict(arrowstyle="->", color=colors[i], lw=2, mutation_scale=12))
    sc = ax1.scatter(g_vals, sec_vals, c=layers, cmap="viridis", s=40, zorder=5, edgecolors="white", linewidths=0.5)
    plt.colorbar(sc, ax=ax1, label="Layer L")
    ax1.set_xlabel(r"$\langle g_{\mathrm{micro}}(L) \rangle$"); ax1.set_ylabel(sec_label)
    ax1.set_title("Mean Phase Portrait"); ax1.grid(True)
    plt.tight_layout()
    return fig


# =============================================================================
# Power Spectrum (CMB analog)
# =============================================================================
def plot_power_spectrum(df, boundary_step=None):
    """Compute 2D power spectrum of g_micro at a given step (token_idx)."""
    if df is None or df.empty:
        return None

    # Build the layer x token matrix averaged across generations
    mats = []
    for _, gdf in df.groupby("gen_idx"):
        p = gdf.pivot_table(index="layer", columns="token_idx", values="g_micro", aggfunc="mean")
        mats.append(p.values)
    if not mats:
        return None
    min_l = min(m.shape[0] for m in mats)
    min_t = min(m.shape[1] for m in mats)
    aligned = np.array([m[:min_l, :min_t] for m in mats])
    mean_m = np.nanmean(aligned, axis=0)  # layers x tokens

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: power spectrum of layer profile at each step
    ax = axes[0]
    n_steps = mean_m.shape[1]
    step_indices = np.linspace(0, n_steps - 1, min(5, n_steps), dtype=int)
    colors = plt.cm.plasma(np.linspace(0, 1, len(step_indices)))
    for idx, si in enumerate(step_indices):
        profile = mean_m[:, si]
        profile = profile - np.mean(profile)
        ps = np.abs(np.fft.rfft(profile))**2
        freqs = np.fft.rfftfreq(len(profile))
        ax.semilogy(freqs[1:], ps[1:], color=colors[idx], lw=1.5, alpha=0.8, label=f"step {si}")
    ax.set_xlabel("Spatial frequency (1/layer)")
    ax.set_ylabel("Power")
    ax.set_title("Power Spectrum of Layer Profile")
    ax.legend(fontsize=7); ax.grid(True)

    # Right: power spectrum along temporal axis at each layer
    ax = axes[1]
    n_layers = mean_m.shape[0]
    layer_indices = np.linspace(0, n_layers - 1, min(5, n_layers), dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, len(layer_indices)))
    for idx, li in enumerate(layer_indices):
        profile = mean_m[li, :]
        profile = profile - np.mean(profile)
        ps = np.abs(np.fft.rfft(profile))**2
        freqs = np.fft.rfftfreq(len(profile))
        ax.semilogy(freqs[1:], ps[1:], color=colors[idx], lw=1.5, alpha=0.8, label=f"L{li}")
    ax.set_xlabel("Temporal frequency (1/step)")
    ax.set_ylabel("Power")
    ax.set_title("Power Spectrum Along Trajectory")
    ax.legend(fontsize=7); ax.grid(True)

    plt.tight_layout()
    return fig


# =============================================================================
# φ_micro (Phase Field) Plots
# =============================================================================
def _has_dual_hook(df):
    """Detect dual-hook capture: 'submodule' column with both 'attn' and 'mlp' values."""
    if df is None or 'submodule' not in df.columns:
        return False
    subs = set(df['submodule'].dropna().unique())
    return ('attn' in subs) and ('mlp' in subs)


def _has_phi_skip(df):
    """Detect non-trivial phi_micro_skip data (column present and not all zeros)."""
    if df is None or 'phi_micro_skip' not in df.columns:
        return False
    return df['phi_micro_skip'].abs().sum() > 1e-6


def plot_phi_micro_depth(df, bin_size=1):
    """φ_micro depth profile.
    
    Dual-hook data (submodule column with attn+mlp): 2x2 panel grid showing
    phi_micro and phi_micro_skip depth profiles per submodule, the cross-type
    vs same-type comparison, and the Madelung scatter colored by submodule.
    
    Single-hook (legacy) data: 1x3 grid as before — phi_micro depth, sublattice
    classifier, Madelung scatter.
    """
    if df is None or df.empty or 'phi_micro' not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No phi_micro column in data",
                ha="center", va="center", transform=ax.transAxes, color="#888888", fontsize=12)
        ax.set_title("φ_micro — No Data")
        return fig

    if bin_size > 1:
        df = df.copy()
        df["layer"] = (df["layer"] // bin_size) * bin_size

    dual = _has_dual_hook(df)
    has_skip = _has_phi_skip(df)

    if dual:
        return _plot_phi_micro_depth_dual(df, has_skip)
    else:
        return _plot_phi_micro_depth_legacy(df, has_skip)


def _plot_phi_micro_depth_legacy(df, has_skip):
    """Legacy 1x3 layout (single-hook), with optional phi_skip overlay."""
    layer_means = df.groupby("layer")["phi_micro"].agg(["mean", "std", "count"])
    layer_means["sem"] = layer_means["std"] / np.sqrt(layer_means["count"])
    layers = layer_means.index.values
    phi_mean = layer_means["mean"].values
    phi_sem = layer_means["sem"].values

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: Depth profile (with optional phi_skip overlay)
    ax = axes[0]
    colors = ["#ff6b6b" if v < 0 else "#00d4aa" for v in phi_mean]
    ax.bar(layers, phi_mean, color=colors, alpha=0.8, edgecolor="white", linewidth=0.3,
           label=r"$\phi_{\mathrm{micro}}$")
    ax.errorbar(layers, phi_mean, yerr=phi_sem, fmt="none", ecolor="#888888", capsize=2)
    if has_skip:
        skip_mean = df.groupby("layer")["phi_micro_skip"].mean().reindex(layers).values
        ax.plot(layers, skip_mean, 'o-', color="#ffd166", markersize=4, lw=1.2,
                label=r"$\phi_{\mathrm{micro,skip}}$ (stride-2)")
        ax.legend(loc='best', fontsize=8)
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xlabel("Layer L")
    ax.set_ylabel(r"$\langle \phi(L) \rangle$")
    ax.set_title("φ_micro Depth Profile")
    ax.grid(True, alpha=0.3)

    # Panel 2: Phase classification
    ax = axes[1]
    if len(layers) > 4:
        even_phi = phi_mean[::2]
        odd_phi = phi_mean[1::2]
        even_l = layers[::2]
        odd_l = layers[1::2]
        ax.bar(even_l - 0.2, even_phi, width=0.4, color="#00d4aa", alpha=0.8, label="Even-indexed")
        ax.bar(odd_l + 0.2, odd_phi[:len(odd_l)], width=0.4, color="#ff6b6b", alpha=0.8, label="Odd-indexed")
        ax.axhline(0, color="#888888", lw=0.8)
        ax.axhspan(0.3, 1.0, color="#00d4aa", alpha=0.06, label="_bosonic zone")
        ax.axhspan(-1.0, -0.3, color="#ff6b6b", alpha=0.06, label="_fermionic zone")
        ax.set_xlabel("Layer L")
        ax.set_ylabel(r"$\langle \phi(L) \rangle$")

        overall_mean = np.mean(phi_mean[2:])
        overall_std = np.std(phi_mean[2:])
        has_positive = np.any(phi_mean[2:] > 0.1)
        has_negative = np.any(phi_mean[2:] < -0.1)

        if has_positive and has_negative and overall_std > 0.4:
            sign_changes = []
            for ii in range(2, len(phi_mean) - 1):
                if phi_mean[ii] * phi_mean[ii + 1] < 0:
                    sign_changes.append(layers[ii])
            trans_str = f" at L≈{sign_changes[0]}" if sign_changes else ""
            verdict = f"MIXED (phase transition{trans_str})"
            verdict_color = "#ff9f43"
        elif overall_mean < -0.3 and overall_std < 0.3:
            verdict = "FERMIONIC (period-2 reversal)"
            verdict_color = "#ff6b6b"
        elif overall_mean > 0.3 and overall_std < 0.3:
            verdict = "BOSONIC (constructive)"
            verdict_color = "#00d4aa"
        elif overall_std > 0.4:
            verdict = "MIXED (phase transition present)"
            verdict_color = "#ff9f43"
        else:
            verdict = "DECOHERENT (no phase structure)"
            verdict_color = "#888888"
        ax.set_title(f"Phase Classification: {verdict}", color=verdict_color)

        # Reference bands
        ax.axhspan(-1.0, -0.3, alpha=0.05, color="#ff6b6b")  # fermionic zone
        ax.axhspan(0.3, 1.0, alpha=0.05, color="#00d4aa")     # bosonic zone
        ax.axhspan(-0.3, 0.3, alpha=0.05, color="#888888")    # decoherent zone
        ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Madelung scatter
    ax = axes[2]
    if "g_micro" in df.columns:
        layer_g = df.groupby("layer")["g_micro"].mean()
        layer_phi = df.groupby("layer")["phi_micro"].mean()
        common = layer_g.index.intersection(layer_phi.index)
        if len(common) > 3:
            g_vals = layer_g.loc[common].values
            phi_vals = layer_phi.loc[common].values
            sc = ax.scatter(g_vals, phi_vals, c=common.values, cmap="viridis",
                           s=50, edgecolors="white", linewidths=0.5, zorder=5)
            plt.colorbar(sc, ax=ax, label="Layer L")
            ax.set_xlabel(r"$\langle g_{\mathrm{micro}}(L) \rangle$ (amplitude)")
            ax.set_ylabel(r"$\langle \phi_{\mathrm{micro}}(L) \rangle$ (phase)")
            ax.set_title("Madelung: Amplitude vs Phase")
            ax.axhline(0, color="#888888", lw=0.5)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def _plot_phi_micro_depth_dual(df, has_skip):
    """Dual-hook 2x2 layout: phi_micro by submodule, phi_skip by submodule,
    cross-type vs same-type comparison, Madelung scatter colored by submodule."""
    df_attn = df[df['submodule'] == 'attn']
    df_mlp = df[df['submodule'] == 'mlp']

    attn_phi = df_attn.groupby("layer")["phi_micro"].mean()
    mlp_phi = df_mlp.groupby("layer")["phi_micro"].mean()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Panel 1: phi_micro depth — both submodules overlaid
    ax = axes[0, 0]
    common_layers = sorted(set(attn_phi.index) | set(mlp_phi.index))
    a_vals = [attn_phi.get(L, np.nan) for L in common_layers]
    m_vals = [mlp_phi.get(L, np.nan) for L in common_layers]
    width = 0.4
    x = np.array(common_layers, dtype=float)
    ax.bar(x - width/2, a_vals, width=width, color="#ff6b6b", alpha=0.85, label="attn", edgecolor="white", linewidth=0.3)
    ax.bar(x + width/2, m_vals, width=width, color="#4dabf7", alpha=0.85, label="mlp", edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="#888888", lw=0.8)
    ax.axhspan(0.3, 1.0, color="#00d4aa", alpha=0.06)
    ax.axhspan(-1.0, -0.3, color="#ff6b6b", alpha=0.06)
    ax.set_xlabel("Layer L"); ax.set_ylabel(r"$\langle \phi_{\mathrm{micro}} \rangle$ (stride-1)")
    ax.set_title("φ_micro (cross-type / inter-sublattice)")
    ax.legend(loc='best'); ax.grid(True, alpha=0.3)

    # Panel 2: phi_micro_skip depth — both submodules overlaid
    ax = axes[0, 1]
    if has_skip:
        attn_skip = df_attn.groupby("layer")["phi_micro_skip"].mean()
        mlp_skip = df_mlp.groupby("layer")["phi_micro_skip"].mean()
        as_vals = [attn_skip.get(L, np.nan) for L in common_layers]
        ms_vals = [mlp_skip.get(L, np.nan) for L in common_layers]
        ax.bar(x - width/2, as_vals, width=width, color="#ff6b6b", alpha=0.85, label="attn")
        ax.bar(x + width/2, ms_vals, width=width, color="#4dabf7", alpha=0.85, label="mlp")
        ax.axhline(0, color="#888888", lw=0.8)
        ax.axhspan(0.3, 1.0, color="#00d4aa", alpha=0.06)
        ax.axhspan(-1.0, -0.3, color="#ff6b6b", alpha=0.06)
        ax.set_xlabel("Layer L"); ax.set_ylabel(r"$\langle \phi_{\mathrm{micro,skip}} \rangle$ (stride-2)")
        ax.set_title("φ_micro_skip (same-type / intra-sublattice)")
        ax.legend(loc='best'); ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No phi_micro_skip data", ha="center", va="center",
                transform=ax.transAxes, color="#888888", fontsize=12)
        ax.set_title("φ_micro_skip — Not available")

    # Panel 3: Two-sector classifier (compare phi_micro vs phi_micro_skip means)
    ax = axes[1, 0]
    if has_skip:
        a_phi_mean = np.nanmean(a_vals[2:]) if len(a_vals) > 2 else np.nan
        m_phi_mean = np.nanmean(m_vals[2:]) if len(m_vals) > 2 else np.nan
        a_skip_mean = np.nanmean(as_vals[3:]) if len(as_vals) > 3 else np.nan
        m_skip_mean = np.nanmean(ms_vals[3:]) if len(ms_vals) > 3 else np.nan
        labels_4 = ['attn φ', 'attn φ_skip', 'mlp φ', 'mlp φ_skip']
        vals_4 = [a_phi_mean, a_skip_mean, m_phi_mean, m_skip_mean]
        cols_4 = ['#ff6b6b', '#ffa3a3', '#4dabf7', '#a3ccff']
        ax.bar(labels_4, vals_4, color=cols_4, edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="#888888", lw=0.8)
        ax.axhspan(0.3, 1.0, color="#00d4aa", alpha=0.06)
        ax.axhspan(-1.0, -0.3, color="#ff6b6b", alpha=0.06)
        ax.set_ylabel(r"$\langle \phi \rangle$ (mean over layers ≥ 3)")
        # Two-sector verdict: cross-type negative AND same-type positive = TWO-SECTOR confirmed
        cross_mean = np.nanmean([a_phi_mean, m_phi_mean])
        same_mean = np.nanmean([a_skip_mean, m_skip_mean])
        if not (np.isnan(cross_mean) or np.isnan(same_mean)):
            if cross_mean < -0.2 and same_mean > 0.2:
                verdict = "TWO-SECTOR CONFIRMED (fermionic ⟂ bosonic)"
                vc = "#00d4aa"
            elif cross_mean < -0.2 and same_mean < -0.2:
                verdict = "SINGLE-SECTOR — fermionic both"
                vc = "#ff6b6b"
            elif cross_mean > 0.2 and same_mean > 0.2:
                verdict = "SINGLE-SECTOR — bosonic both"
                vc = "#4dabf7"
            else:
                verdict = "AMBIGUOUS / DECOHERENT"
                vc = "#888888"
            ax.set_title(f"Two-Sector Test: {verdict}", color=vc, fontsize=10)
        else:
            ax.set_title("Two-Sector Test: insufficient data")
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, "Two-sector test requires phi_micro_skip", ha="center", va="center",
                transform=ax.transAxes, color="#888888", fontsize=12)

    # Panel 4: Madelung scatter — colored by submodule
    ax = axes[1, 1]
    if "g_micro" in df.columns:
        for sub_label, sub_color in [('attn', '#ff6b6b'), ('mlp', '#4dabf7')]:
            sub_df = df[df['submodule'] == sub_label]
            if sub_df.empty:
                continue
            g_by_l = sub_df.groupby("layer")["g_micro"].mean()
            phi_by_l = sub_df.groupby("layer")["phi_micro"].mean()
            common = g_by_l.index.intersection(phi_by_l.index)
            if len(common) > 1:
                ax.scatter(g_by_l.loc[common].values, phi_by_l.loc[common].values,
                          c=sub_color, s=40, alpha=0.75, edgecolors="white", linewidths=0.5,
                          label=sub_label, zorder=5)
        ax.axhline(0, color="#888888", lw=0.5)
        ax.set_xlabel(r"$\langle g_{\mathrm{micro}} \rangle$ (amplitude)")
        ax.set_ylabel(r"$\langle \phi_{\mathrm{micro}} \rangle$ (phase)")
        ax.set_title("Madelung: g vs φ (by submodule)")
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_phi_micro_heatmap(df, temporal_bin=1):
    """φ_micro heatmap: phase field across layers and steps.
    
    Dual-hook data: 4-panel grid showing phi_micro (attn, mlp) and phi_micro_skip (attn, mlp).
    Single-hook (legacy) data: 2-3 panel grid as before — phi heatmap, sign map, optional g comparison.
    """
    if df is None or df.empty or 'phi_micro' not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No phi_micro column in data",
                ha="center", va="center", transform=ax.transAxes, color="#888888", fontsize=12)
        ax.set_title("φ_micro Heatmap — No Data")
        return fig

    if temporal_bin > 1:
        df = df.copy()
        df["token_idx"] = (df["token_idx"] // temporal_bin) * temporal_bin

    dual = _has_dual_hook(df)
    has_skip = _has_phi_skip(df)

    if dual:
        return _plot_phi_micro_heatmap_dual(df, has_skip)
    else:
        return _plot_phi_micro_heatmap_legacy(df)


def _build_phi_heatmap_matrix(df, value_col):
    """Pivot a dataframe to a (layer, token_idx) matrix for `value_col`, averaged across gens."""
    mats = []
    for _, gdf in df.groupby("gen_idx"):
        p = gdf.pivot_table(index="layer", columns="token_idx", values=value_col, aggfunc="mean")
        mats.append(p.values)
    if not mats:
        return None
    min_l = min(m.shape[0] for m in mats)
    min_t = min(m.shape[1] for m in mats)
    aligned = np.array([m[:min_l, :min_t] for m in mats])
    return np.nanmean(aligned, axis=0)


def _plot_phi_micro_heatmap_legacy(df):
    """Legacy 2-3 panel layout (single-hook)."""
    mean_m = _build_phi_heatmap_matrix(df, "phi_micro")
    if mean_m is None:
        return None

    g_mats = []
    if "g_micro" in df.columns:
        g_mean = _build_phi_heatmap_matrix(df, "g_micro")
    else:
        g_mean = None

    n_cols = 3 if g_mean is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.5))
    if n_cols == 2:
        axes = list(axes)

    # Panel 1: phi_micro heatmap
    ax = axes[0]
    vmax = max(abs(np.nanmin(mean_m)), abs(np.nanmax(mean_m)))
    im = ax.imshow(mean_m, aspect="auto", origin="lower", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xlabel("Step / Token index"); ax.set_ylabel("Layer L")
    ax.set_title(r"$\langle \phi_{\mathrm{micro}}(L, t) \rangle$")
    plt.colorbar(im, ax=ax, label=r"$\phi_{\mathrm{micro}}$")

    # Panel 2: phi_micro sign map
    ax = axes[1]
    sign_map = np.sign(mean_m)
    im = ax.imshow(sign_map, aspect="auto", origin="lower", cmap="RdBu_r",
                   vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xlabel("Step / Token index"); ax.set_ylabel("Layer L")
    ax.set_title("φ_micro Sign Map (+1=constructive, -1=destructive)")
    plt.colorbar(im, ax=ax, label="sign(φ)")

    # Panel 3: g_micro heatmap for 90° offset check
    if g_mean is not None and n_cols == 3:
        ax = axes[2]
        im = ax.imshow(g_mean, aspect="auto", origin="lower", cmap="viridis",
                       interpolation="nearest")
        ax.set_xlabel("Step / Token index"); ax.set_ylabel("Layer L")
        ax.set_title(r"$\langle g_{\mathrm{micro}}(L, t) \rangle$ (for comparison)")
        plt.colorbar(im, ax=ax, label=r"$g_{\mathrm{micro}}$")

    plt.tight_layout()
    return fig


def _plot_phi_micro_heatmap_dual(df, has_skip):
    """Dual-hook 2x2 layout: phi_micro × {attn, mlp} and phi_micro_skip × {attn, mlp}."""
    df_attn = df[df['submodule'] == 'attn']
    df_mlp = df[df['submodule'] == 'mlp']

    phi_attn = _build_phi_heatmap_matrix(df_attn, "phi_micro")
    phi_mlp = _build_phi_heatmap_matrix(df_mlp, "phi_micro")

    if has_skip:
        skip_attn = _build_phi_heatmap_matrix(df_attn, "phi_micro_skip")
        skip_mlp = _build_phi_heatmap_matrix(df_mlp, "phi_micro_skip")
    else:
        skip_attn = None
        skip_mlp = None

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Build a shared color scale for the phi_micro pair
    phi_panels = [m for m in (phi_attn, phi_mlp) if m is not None]
    if phi_panels:
        vmax_phi = max(max(abs(np.nanmin(m)), abs(np.nanmax(m))) for m in phi_panels)
    else:
        vmax_phi = 1.0

    # Top-left: phi_micro attn
    ax = axes[0, 0]
    if phi_attn is not None:
        im = ax.imshow(phi_attn, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-vmax_phi, vmax=vmax_phi, interpolation="nearest")
        plt.colorbar(im, ax=ax, label=r"$\phi$")
        ax.set_title(r"$\phi_{\mathrm{micro}}(L, t)$ — attn (cross-type, stride-1)")
    else:
        ax.text(0.5, 0.5, "No attn data", ha="center", va="center", transform=ax.transAxes, color="#888")
        ax.set_title("φ_micro attn — empty")
    ax.set_xlabel("Step / Token index"); ax.set_ylabel("Layer L")

    # Top-right: phi_micro mlp
    ax = axes[0, 1]
    if phi_mlp is not None:
        im = ax.imshow(phi_mlp, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-vmax_phi, vmax=vmax_phi, interpolation="nearest")
        plt.colorbar(im, ax=ax, label=r"$\phi$")
        ax.set_title(r"$\phi_{\mathrm{micro}}(L, t)$ — mlp (cross-type, stride-1)")
    else:
        ax.text(0.5, 0.5, "No mlp data", ha="center", va="center", transform=ax.transAxes, color="#888")
        ax.set_title("φ_micro mlp — empty")
    ax.set_xlabel("Step / Token index"); ax.set_ylabel("Layer L")

    # Build color scale for phi_skip pair
    skip_panels = [m for m in (skip_attn, skip_mlp) if m is not None]
    if skip_panels:
        vmax_skip = max(max(abs(np.nanmin(m)), abs(np.nanmax(m))) for m in skip_panels)
    else:
        vmax_skip = 1.0

    # Bottom-left: phi_micro_skip attn
    ax = axes[1, 0]
    if skip_attn is not None:
        im = ax.imshow(skip_attn, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-vmax_skip, vmax=vmax_skip, interpolation="nearest")
        plt.colorbar(im, ax=ax, label=r"$\phi_{\mathrm{skip}}$")
        ax.set_title(r"$\phi_{\mathrm{micro,skip}}(L, t)$ — attn (same-type, stride-2)")
    else:
        ax.text(0.5, 0.5, "phi_micro_skip not available", ha="center", va="center",
                transform=ax.transAxes, color="#888")
        ax.set_title("φ_micro_skip attn — empty")
    ax.set_xlabel("Step / Token index"); ax.set_ylabel("Layer L")

    # Bottom-right: phi_micro_skip mlp
    ax = axes[1, 1]
    if skip_mlp is not None:
        im = ax.imshow(skip_mlp, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-vmax_skip, vmax=vmax_skip, interpolation="nearest")
        plt.colorbar(im, ax=ax, label=r"$\phi_{\mathrm{skip}}$")
        ax.set_title(r"$\phi_{\mathrm{micro,skip}}(L, t)$ — mlp (same-type, stride-2)")
    else:
        ax.text(0.5, 0.5, "phi_micro_skip not available", ha="center", va="center",
                transform=ax.transAxes, color="#888")
        ax.set_title("φ_micro_skip mlp — empty")
    ax.set_xlabel("Step / Token index"); ax.set_ylabel("Layer L")

    plt.tight_layout()
    return fig


# =============================================================================
# NEW v3 Plot Functions (2026-05-06)
# =============================================================================

def _get_mean_profile_crossings(df, submodule="attn", spline_s=5e-5):
    """Compute β(g) from mean g(L) profile. Returns (g_fine, b_fine, fixed_pts, mean_g)."""
    gen_only = df[df.token_type == "generated"] if "token_type" in df.columns else df
    if "submodule" in gen_only.columns:
        gen_only = gen_only[gen_only.submodule == submodule]
    agg = gen_only.groupby(["gen_idx", "layer"]).g_micro.mean().reset_index()
    piv = agg.pivot(index="gen_idx", columns="layer", values="g_micro")
    if piv.shape[0] < 3 or piv.shape[1] < 5:
        return None, None, [], None
    mean_g = piv.mean(axis=0).values
    beta = np.diff(mean_g)
    g_at = mean_g[:-1]
    order = np.argsort(g_at)
    try:
        spl = UnivariateSpline(g_at[order], beta[order], s=spline_s)
    except Exception:
        return None, None, [], mean_g
    g_fine = np.linspace(g_at.min(), g_at.max(), 5000)
    b_fine = spl(g_fine)
    fps = []
    for i in range(1, len(b_fine)):
        if b_fine[i] * b_fine[i - 1] < 0:
            gs = g_fine[i - 1] + (g_fine[i] - g_fine[i - 1]) * (-b_fine[i - 1]) / (b_fine[i] - b_fine[i - 1])
            d = "pos→neg" if b_fine[i - 1] > 0 else "neg→pos"
            fps.append((gs, d))
    return g_fine, b_fine, fps, mean_g


def plot_beta_g_mean_profile(df):
    """Mean-profile β(g): averages over tokens first, then computes β from converged depth profile."""
    if df is None or df.empty:
        return None
    gen_only = df[df.token_type == "generated"] if "token_type" in df.columns else df
    has_sub = "submodule" in gen_only.columns
    submodules = gen_only.submodule.unique().tolist() if has_sub else ["all"]
    n_sub = len(submodules)
    fig, axes = plt.subplots(1, n_sub, figsize=(8 * n_sub, 5), squeeze=False)
    for col, sm in enumerate(submodules):
        ax = axes[0, col]
        sub = gen_only[gen_only.submodule == sm] if has_sub else gen_only
        agg = sub.groupby(["gen_idx", "layer"]).g_micro.mean().reset_index()
        piv = agg.pivot(index="gen_idx", columns="layer", values="g_micro")
        if piv.shape[0] < 3:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
            continue
        # Per-gen trajectories
        for _, row in piv.iterrows():
            v = row.values
            ax.plot(v[:-1], np.diff(v), color="#ff6b6b", alpha=0.08, lw=0.5)
        # Mean profile spline
        g_fine, b_fine, fps, _ = _get_mean_profile_crossings(df, sm)
        if g_fine is not None:
            ax.plot(g_fine, b_fine, color="white", lw=2.5, label=f"spline mean (s=5e-5)")
            for gs, d in fps:
                c = "#ff4444" if "pos→neg" in d else "#00d4aa"
                ax.axvline(gs, color=c, ls="--", lw=1.5, alpha=0.8)
                ax.text(gs, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 0.015,
                        f"g*={gs:.4f}", color=c, fontsize=8, ha="center", fontweight="bold")
        ax.axhline(0, color="#888888", lw=0.5)
        title = f"{sm.upper()} — Mean-Profile β(g)"
        if fps:
            title += f" ({len(fps)} crossings)"
        ax.set_title(title)
        ax.set_xlabel(r"$g_{\rm micro}$")
        ax.set_ylabel(r"$\beta_{\rm layer}(g)$")
        ax.grid(True, alpha=0.15)
        ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_early_vs_late_tokens(df):
    """Compare β(g) clouds for early vs late token positions."""
    if df is None or df.empty:
        return None
    gen_only = df[df.token_type == "generated"] if "token_type" in df.columns else df
    if "submodule" in gen_only.columns:
        gen_only = gen_only[gen_only.submodule == "attn"]
    max_tok = gen_only.token_idx.max()
    if max_tok < 30:
        return None
    early_cut = min(30, max_tok // 4)
    late_cut = max(max_tok - max_tok // 4, max_tok // 2)
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, tok_lo, tok_hi, color in [("early", 0, early_cut, "#00d4aa"), ("late", late_cut, max_tok + 1, "#ff9f43")]:
        gs, bs = [], []
        sub = gen_only[(gen_only.token_idx >= tok_lo) & (gen_only.token_idx < tok_hi)]
        for _, gdf in sub.groupby("gen_idx"):
            p = gdf.pivot_table(index="layer", columns="token_idx", values="g_micro", aggfunc="mean")
            gmat = p.values
            beta = np.diff(gmat, axis=0)
            gf = gmat[:-1, :]
            m = ~np.isnan(beta) & ~np.isnan(gf)
            gs.append(gf[m])
            bs.append(beta[m])
        if not gs:
            continue
        g = np.concatenate(gs)
        b = np.concatenate(bs)
        ax.scatter(g[::5], b[::5], s=1, alpha=0.06, color=color)
        # Binned mean
        edges = np.linspace(np.percentile(g, 2), np.percentile(g, 98), 25)
        centers = 0.5 * (edges[:-1] + edges[1:])
        bm = np.full_like(centers, np.nan)
        for i in range(len(centers)):
            mask = (g >= edges[i]) & (g < edges[i + 1])
            if mask.sum() >= 5:
                bm[i] = b[mask].mean()
        v = ~np.isnan(bm)
        ax.plot(centers[v], bm[v], color=color, lw=2.5, label=f"{label} (tok {tok_lo}-{tok_hi - 1})")
    ax.axhline(0, color="#888888", lw=0.5)
    ax.set_title("Early vs Late Tokens — β(g) cloud comparison\nLate tokens have higher g → smear cloud method")
    ax.set_xlabel(r"$g_{\rm micro}$")
    ax.set_ylabel(r"$\beta_{\rm layer}(g)$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    return fig


def plot_interval_decomposition(df):
    """Fixed-point interval decomposition with Coxeter class comparison."""
    if df is None or df.empty:
        return None
    g_fine, b_fine, fps, mean_g = _get_mean_profile_crossings(df, "attn")
    if len(fps) < 3:
        # Try without submodule filter
        g_fine, b_fine, fps, mean_g = _get_mean_profile_crossings(df, "all")
    if len(fps) < 3 or mean_g is None:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.text(0.5, 0.5, f"Need ≥3 crossings for interval decomposition\n(found {len(fps)})",
                ha="center", va="center", transform=ax.transAxes, fontsize=14, color="#888")
        return fig
    phi = (1 + np.sqrt(5)) / 2
    g1, g2, g3 = fps[0][0], fps[1][0], fps[2][0]
    d1 = g2 - g1
    d2 = g3 - g2
    total = g3 - g1
    r_outer = g3 / g1
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [2, 1]})
    # Top: depth profile with fixed-point lines
    ax = axes[0]
    L = np.arange(len(mean_g))
    ax.plot(L, mean_g, "o-", color="#00d4aa", lw=2, markersize=4)
    for gs, d in fps:
        c = "#ff4444" if "pos→neg" in d else "#00d4aa"
        lab = "UV" if "pos→neg" in d else "IR"
        ax.axhline(gs, color=c, ls=":", lw=1, alpha=0.6)
        ax.text(len(mean_g) - 1, gs, f" g*={gs:.4f} ({lab})", color=c, fontsize=8, va="center")
    ax.set_xlabel("Layer L")
    ax.set_ylabel(r"$\langle g_{\rm micro}(L) \rangle$")
    ax.set_title("Depth Profile with Fixed-Point Lines")
    ax.grid(True, alpha=0.15)
    # Bottom: interval bar
    ax = axes[1]
    ax.barh(0.5, d1, left=g1, height=0.15, color="#ff6b6b", edgecolor="white", lw=1)
    ax.barh(0.5, d2, left=g2, height=0.15, color="#00d4aa", edgecolor="white", lw=1)
    ax.text(g1 + d1 / 2, 0.7, f"Δ₁ = {d1 / total * 100:.1f}%", ha="center", fontsize=11,
            color="#ff6b6b", fontweight="bold")
    ax.text(g2 + d2 / 2, 0.7, f"Δ₂ = {d2 / total * 100:.1f}%", ha="center", fontsize=11,
            color="#00d4aa", fontweight="bold")
    # Prediction bar
    ax.barh(0.2, 0.46 * total, left=g1, height=0.06, color="#ff6b6b", alpha=0.3)
    ax.barh(0.2, 0.54 * total, left=g1 + 0.46 * total, height=0.06, color="#00d4aa", alpha=0.3)
    ax.text(g1 + total / 2, 0.08, "pred: 46% / 54%", ha="center", fontsize=9, color="#888")
    # Reference ratios
    txt = (f"g₃/g₁ = {r_outer:.4f}\n"
           f"  vs φ = {phi:.4f} ({(r_outer / phi - 1) * 100:+.1f}%)\n"
           f"  vs √3 = {np.sqrt(3):.4f} ({(r_outer / np.sqrt(3) - 1) * 100:+.1f}%)\n"
           f"  vs √2 = {np.sqrt(2):.4f} ({(r_outer / np.sqrt(2) - 1) * 100:+.1f}%)\n"
           f"Δ₁/span = {d1 / total * 100:.1f}% (pred: 46%)")
    ax.text(0.02, 0.95, txt, transform=ax.transAxes, fontsize=10, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e", alpha=0.9))
    ax.set_xlim(g1 * 0.85, g3 * 1.15)
    ax.set_ylim(-0.1, 0.9)
    ax.set_yticks([])
    ax.set_xlabel(r"$g_{\rm micro}$")
    ax.set_title("Interval Decomposition — Coxeter Comparison")
    plt.tight_layout()
    return fig


def plot_bootstrap_ratio(df, n_boot=300):
    """Bootstrap distribution of outer ratio g₃/g₁."""
    if df is None or df.empty or not _HAS_SCIPY:
        return None
    gen_only = df[df.token_type == "generated"] if "token_type" in df.columns else df
    if "submodule" in gen_only.columns:
        gen_only = gen_only[gen_only.submodule == "attn"]
    agg = gen_only.groupby(["gen_idx", "layer"]).g_micro.mean().reset_index()
    piv = agg.pivot(index="gen_idx", columns="layer", values="g_micro")
    if piv.shape[0] < 10:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, f"Need ≥10 gens for bootstrap (have {piv.shape[0]})",
                ha="center", va="center", transform=ax.transAxes, fontsize=14, color="#888")
        return fig
    np.random.seed(42)
    r_boot, frac1_boot = [], []
    gen_ids = piv.index.tolist()
    for _ in range(n_boot):
        s = np.random.choice(gen_ids, size=len(gen_ids), replace=True)
        mg = piv.loc[s].mean(axis=0).values
        bl = np.diff(mg)
        ga = mg[:-1]
        o = np.argsort(ga)
        try:
            sp = UnivariateSpline(ga[o], bl[o], s=5e-5)
            gf = np.linspace(ga.min(), ga.max(), 3000)
            bf = sp(gf)
            fp = []
            for i in range(1, len(bf)):
                if bf[i] * bf[i - 1] < 0:
                    fp.append(gf[i - 1] + (gf[i] - gf[i - 1]) * (-bf[i - 1]) / (bf[i] - bf[i - 1]))
            if len(fp) >= 3:
                r_boot.append(fp[2] / fp[0])
                frac1_boot.append((fp[1] - fp[0]) / (fp[2] - fp[0]))
        except Exception:
            pass
    if len(r_boot) < 10:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, f"Bootstrap failed: only {len(r_boot)}/{n_boot} had ≥3 crossings",
                ha="center", va="center", transform=ax.transAxes, fontsize=14, color="#888")
        return fig
    r_boot = np.array(r_boot)
    frac1_boot = np.array(frac1_boot)
    phi = (1 + np.sqrt(5)) / 2
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Left: outer ratio
    ax = axes[0]
    ax.hist(r_boot, bins=30, color="#666", edgecolor="white", lw=0.5, density=True, alpha=0.7)
    for val, name, c in [(phi, f"φ={phi:.3f}", "#ffd700"), (np.sqrt(3), f"√3={np.sqrt(3):.3f}", "#ff8800"),
                          (np.sqrt(2), f"√2={np.sqrt(2):.3f}", "#00ff88"), (2.0, "2.0", "#ff4444")]:
        ax.axvline(val, color=c, lw=2, ls="--", label=name)
    med = np.median(r_boot)
    ax.axvline(med, color="white", lw=2, label=f"median={med:.3f}")
    p5, p95 = np.percentile(r_boot, [5, 95])
    ax.set_title(f"Outer Ratio g₃/g₁ — Bootstrap (n={n_boot})\nmedian={med:.3f}, 90% CI [{p5:.3f}, {p95:.3f}]")
    ax.set_xlabel(r"$g^*_3 / g^*_1$")
    ax.set_ylabel("density")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.15)
    # Right: Δ₁/span
    ax = axes[1]
    ax.hist(frac1_boot, bins=30, color="#666", edgecolor="white", lw=0.5, density=True, alpha=0.7)
    ax.axvline(0.46, color="#ff6b6b", lw=2, ls="--", label="pred=0.46")
    med_f = np.median(frac1_boot)
    ax.axvline(med_f, color="white", lw=2, label=f"median={med_f:.3f}")
    p5f, p95f = np.percentile(frac1_boot, [5, 95])
    ax.set_title(f"Interval Split Δ₁/span — Bootstrap\nmedian={med_f:.3f}, 90% CI [{p5f:.3f}, {p95f:.3f}]")
    ax.set_xlabel(r"$\Delta_1 / \mathrm{span}$")
    ax.set_ylabel("density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    return fig


def plot_key_results(df):
    """Summary text panel with key numerical results."""
    if df is None or df.empty:
        return None
    gen_only = df[df.token_type == "generated"] if "token_type" in df.columns else df
    has_sub = "submodule" in gen_only.columns
    n_gens = gen_only.gen_idx.nunique()
    n_layers = gen_only.layer.nunique()
    lines = []
    lines.append("═" * 72)
    lines.append("  KEY RESULTS SUMMARY")
    lines.append("═" * 72)
    lines.append(f"  Generations: {n_gens}   Layers: {n_layers}")
    if has_sub:
        lines.append(f"  Submodules: {', '.join(gen_only.submodule.unique())}")
    # β(g) fixed points
    for sm in (["attn", "mlp"] if has_sub else ["all"]):
        _, _, fps, mg = _get_mean_profile_crossings(df, sm)
        if fps:
            lines.append(f"\n  β(g) Fixed Points ({sm}):")
            for i, (gs, d) in enumerate(fps):
                stability = "UV" if "pos→neg" in d else "IR"
                lines.append(f"    g*_{i + 1} = {gs:.6f} ({stability})")
            if len(fps) >= 3:
                r = fps[2][0] / fps[0][0]
                d1 = fps[1][0] - fps[0][0]
                total = fps[2][0] - fps[0][0]
                phi = (1 + np.sqrt(5)) / 2
                lines.append(f"    Outer ratio g₃/g₁ = {r:.4f} (φ={phi:.4f}, Δ={((r / phi) - 1) * 100:+.1f}%)")
                lines.append(f"    Δ₁/span = {d1 / total * 100:.1f}% (pred: 46%)")
    # Two-sector
    if has_sub and "phi_micro" in gen_only.columns:
        for sm in ["attn", "mlp"]:
            sub = gen_only[gen_only.submodule == sm]
            if "phi_micro_skip" in sub.columns:
                bulk = sub[(sub.layer >= 3) & (sub.layer <= n_layers - 4)]
                if len(bulk) > 0:
                    phi_m = bulk.groupby(["gen_idx", "layer"]).phi_micro.mean().values
                    phi_s = bulk.groupby(["gen_idx", "layer"]).phi_micro_skip.mean().values
                    pm, ps = np.nanmean(phi_m), np.nanmean(phi_s)
                    if pm < -0.3 and ps > 0.1:
                        lines.append(f"\n  Two-Sector ({sm}): ✓ CONFIRMED")
                    else:
                        lines.append(f"\n  Two-Sector ({sm}): not detected")
                    lines.append(f"    φ_micro = {pm:.3f}, φ_skip = {ps:.3f}")
    # Spectral gap
    if "spectral_gap" in gen_only.columns and has_sub:
        for sm in ["attn", "mlp"]:
            sub = gen_only[gen_only.submodule == sm]
            sg_by_layer = sub.groupby("layer").spectral_gap.mean()
            peak_L = sg_by_layer.idxmax()
            peak_v = sg_by_layer.max()
            if peak_v > 50:
                lines.append(f"\n  Spectral Gap Spike ({sm}): σ₁/σ₂ = {peak_v:.1f} at L={peak_L}")
    lines.append("\n" + "═" * 72)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, fontsize=10,
            va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#1a1a2e", alpha=0.95))
    plt.tight_layout()
    return fig


def plot_beta_g_comparison(df):
    """Side-by-side attn vs MLP β(g) with per-gen trajectories."""
    if df is None or df.empty:
        return None
    gen_only = df[df.token_type == "generated"] if "token_type" in df.columns else df
    has_sub = "submodule" in gen_only.columns
    if not has_sub:
        return plot_beta_g_mean_profile(df)
    submodules = [sm for sm in ["attn", "mlp"] if sm in gen_only.submodule.unique()]
    if not submodules:
        return None
    phi = (1 + np.sqrt(5)) / 2
    fig, axes = plt.subplots(1, len(submodules), figsize=(8 * len(submodules), 5.5))
    if len(submodules) == 1:
        axes = [axes]
    for col, sm in enumerate(submodules):
        ax = axes[col]
        sub = gen_only[gen_only.submodule == sm]
        agg = sub.groupby(["gen_idx", "layer"]).g_micro.mean().reset_index()
        piv = agg.pivot(index="gen_idx", columns="layer", values="g_micro")
        if piv.shape[0] < 3:
            continue
        # Per-gen traces
        for _, row in piv.iterrows():
            v = row.values
            ax.plot(v[:-1], np.diff(v), color="#ff6b6b", alpha=0.06, lw=0.5)
        # Cloud binned mean (dashboard method)
        all_g, all_b = [], []
        for _, gdf in sub.groupby("gen_idx"):
            p = gdf.pivot_table(index="layer", columns="token_idx", values="g_micro", aggfunc="mean")
            gmat = p.values
            beta = np.diff(gmat, axis=0)
            gf = gmat[:-1, :]
            m = ~np.isnan(beta) & ~np.isnan(gf)
            all_g.append(gf[m])
            all_b.append(beta[m])
        g_cloud = np.concatenate(all_g) if all_g else np.array([])
        b_cloud = np.concatenate(all_b) if all_b else np.array([])
        if len(g_cloud) > 0:
            edges = np.linspace(g_cloud.min(), g_cloud.max(), 30)
            centers = 0.5 * (edges[:-1] + edges[1:])
            bm_c = np.full_like(centers, np.nan)
            for i in range(len(centers)):
                mask = (g_cloud >= edges[i]) & (g_cloud < edges[i + 1])
                if mask.sum() >= 10:
                    bm_c[i] = b_cloud[mask].mean()
            vc = ~np.isnan(bm_c)
            ax.fill_between(centers[vc], np.zeros_like(bm_c[vc]), bm_c[vc], color="gray", alpha=0.15)
            ax.plot(centers[vc], bm_c[vc], color="gray", lw=1.5, ls=":", label="cloud (dashboard)", alpha=0.7)
        # Mean-profile spline
        g_fine, b_fine, fps, _ = _get_mean_profile_crossings(df, sm)
        if g_fine is not None:
            ax.plot(g_fine, b_fine, color="white", lw=2.5, label="mean-profile (spline)")
            for gs, d in fps:
                c = "#ff4444" if "pos→neg" in d else "#00d4aa"
                ax.axvline(gs, color=c, ls="--", lw=1.5, alpha=0.8)
        ax.axhline(0, color="#888888", lw=0.5)
        n_c = len(fps) if g_fine is not None else 0
        ax.set_title(f"{sm.upper()} — β(g) ({n_c} crossings)")
        ax.set_xlabel(r"$g_{\rm micro}$")
        ax.set_ylabel(r"$\beta_{\rm layer}(g)$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)
    plt.tight_layout()
    return fig



def _plotly_to_html(fig, height=650):
    """Convert a Plotly figure to an HTML string that renders reliably in Gradio.
    Uses base64 data URI in an iframe to avoid escaping issues."""
    if fig is None:
        return "<p style='color:#888; text-align:center; padding:2em;'>No data for this plot</p>"
    import base64
    inner_html = fig.to_html(full_html=True, include_plotlyjs="cdn", config={
        "displayModeBar": True,
        "scrollZoom": True,
    })
    b64 = base64.b64encode(inner_html.encode("utf-8")).decode("ascii")
    return f'<iframe src="data:text/html;base64,{b64}" style="width:100%; height:{height}px; border:none;"></iframe>'


def _auto_range(vals, pad_frac=0.1):
    """Compute axis range that uses the full visual space, no forced zero."""
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    span = vmax - vmin
    if span < 1e-12:
        span = abs(vmin) * 0.1 if abs(vmin) > 0 else 1.0
    return [vmin - span * pad_frac, vmax + span * pad_frac]


def _dark_scene(xlab, ylab, zlab, x_range=None, y_range=None, z_range=None):
    """Standard dark scene config with auto-ranged axes."""
    scene = dict(
        xaxis=dict(title=xlab, backgroundcolor="#1a1a1a", gridcolor="#333",
                   showbackground=True, zerolinecolor="#555"),
        yaxis=dict(title=ylab, backgroundcolor="#1e1e1e", gridcolor="#333",
                   showbackground=True, zerolinecolor="#555"),
        zaxis=dict(title=zlab, backgroundcolor="#222222", gridcolor="#333",
                   showbackground=True, zerolinecolor="#555"),
        bgcolor="#1a1a1a",
    )
    if x_range is not None:
        scene["xaxis"]["range"] = x_range
    if y_range is not None:
        scene["yaxis"]["range"] = y_range
    if z_range is not None:
        scene["zaxis"]["range"] = z_range
    return scene


def _add_spline_or_linear(fig, x, y, z, use_spline, color="crimson", width=4, name="Trajectory", row=None, col=None):
    """Add trajectory as either straight lines or cubic spline."""
    if use_spline and _HAS_SCIPY and len(x) >= 4:
        chords = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)
        t_param = np.zeros(len(x))
        t_param[1:] = np.cumsum(chords)
        if t_param[-1] > 0:
            t_param /= t_param[-1]
        cs_x = CubicSpline(t_param, x, bc_type="natural")
        cs_y = CubicSpline(t_param, y, bc_type="natural")
        cs_z = CubicSpline(t_param, z, bc_type="natural")
        t_s = np.linspace(0, 1, 300)
        xs, ys, zs = cs_x(t_s), cs_y(t_s), cs_z(t_s)
        line_name = f"{name} (spline)"
    else:
        xs, ys, zs = x, y, z
        line_name = f"{name} (linear)"

    trace = go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=color, width=width), name=line_name,
    )
    if row is not None and col is not None:
        fig.add_trace(trace, row=row, col=col)
    else:
        fig.add_trace(trace)


def plot_per_gen_diagnostic(df):
    """Per-generation diagnostic: min(β), nether depth, orbit fraction.

    Ranks each generation by how deeply it plunges past the consolidation
    boundary (min β across all layer pairs).  Gens that stay near zero
    (orbit the limit cycle) are healthy; gens with deeply negative β
    overshot into pathological over-consolidation ("nether region").

    Also shows sigma-band breakdown: what fraction of each gen's nether
    plunge comes from low-sigma (late) steps vs high-sigma (early) steps.
    """
    if df is None or df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="#888888", fontsize=12)
        ax.set_title("Per-Gen Diagnostic — No Data")
        return fig

    has_sigma = "sigma" in df.columns

    gen_stats = []
    for gen_id, gdf in df.groupby("gen_idx"):
        p = gdf.pivot_table(index="layer", columns="token_idx",
                            values="g_micro", aggfunc="mean")
        gmat = p.values
        beta = np.diff(gmat, axis=0)
        g_for = gmat[:-1, :]
        m = ~np.isnan(beta) & ~np.isnan(g_for)
        if m.sum() == 0:
            continue
        bvals = beta[m]
        gvals = g_for[m]

        min_beta = bvals.min()
        mean_beta = bvals.mean()
        max_g = gvals.max()
        nether_frac = (bvals < -0.02).mean()
        orbit_frac = (np.abs(bvals) < 0.005).mean()

        # Sigma-band breakdown if available
        sigma_lo_nether = np.nan
        sigma_hi_nether = np.nan
        if has_sigma:
            sig_p = gdf.pivot_table(index="layer", columns="token_idx",
                                    values="sigma", aggfunc="mean")
            sig_mat = sig_p.reindex(index=p.index, columns=p.columns).values
            sig_for = sig_mat[:-1, :]
            sig_valid = sig_for[m]
            lo_mask = sig_valid < 0.3  # late steps (low sigma)
            hi_mask = sig_valid >= 0.3  # early steps (high sigma)
            if lo_mask.sum() > 0:
                sigma_lo_nether = (bvals[lo_mask] < -0.02).mean()
            if hi_mask.sum() > 0:
                sigma_hi_nether = (bvals[hi_mask] < -0.02).mean()

        # Get a label — use prompt_text if available
        label = str(gen_id)
        if "prompt_text" in gdf.columns:
            texts = gdf["prompt_text"].dropna().unique()
            if len(texts) > 0:
                label = str(texts[0])[:40]

        gen_stats.append(dict(
            gen_id=gen_id, label=label,
            min_beta=min_beta, mean_beta=mean_beta,
            max_g=max_g, nether_frac=nether_frac,
            orbit_frac=orbit_frac,
            sigma_lo_nether=sigma_lo_nether,
            sigma_hi_nether=sigma_hi_nether,
        ))

    if not gen_stats:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No per-gen data", ha="center", va="center",
                transform=ax.transAxes, color="#888888", fontsize=12)
        return fig

    # Sort by min_beta (most negative first)
    gen_stats.sort(key=lambda x: x["min_beta"])

    n = len(gen_stats)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Panel A: min(β) ranked bar chart
    ax = axes[0, 0]
    min_betas = [s["min_beta"] for s in gen_stats]
    colors_a = ["#ff6b6b" if b < -0.05 else "#ffaa44" if b < -0.02 else "#00d4aa"
                for b in min_betas]
    ax.barh(range(n), min_betas, color=colors_a, alpha=0.8)
    ax.axvline(-0.02, color="#ffaa44", ls="--", lw=1, alpha=0.7, label="nether threshold")
    ax.axvline(-0.05, color="#ff6b6b", ls="--", lw=1, alpha=0.7, label="deep nether")
    ax.set_xlabel(r"min $\beta$ (most negative)")
    ax.set_ylabel("Gen rank")
    ax.set_title("A: Nether Depth Ranking")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    if n <= 30:
        ax.set_yticks(range(n))
        ax.set_yticklabels([s["label"][:20] for s in gen_stats], fontsize=6)
    ax.invert_yaxis()

    # Panel B: orbit fraction vs nether fraction scatter
    ax = axes[0, 1]
    orbit_fracs = [s["orbit_frac"] for s in gen_stats]
    nether_fracs = [s["nether_frac"] for s in gen_stats]
    sc = ax.scatter(nether_fracs, orbit_fracs,
                    c=min_betas, cmap="RdYlGn_r", s=50,
                    edgecolors="white", linewidths=0.5, vmin=min(min_betas), vmax=0)
    plt.colorbar(sc, ax=ax, label=r"min $\beta$")
    ax.set_xlabel("Nether fraction (β < -0.02)")
    ax.set_ylabel("Orbit fraction (|β| < 0.005)")
    ax.set_title("B: Orbit vs Nether")
    ax.grid(True, alpha=0.3)

    # Panel C: sigma-band breakdown (if available)
    ax = axes[1, 0]
    if has_sigma and not all(np.isnan(s["sigma_lo_nether"]) for s in gen_stats):
        lo_vals = [s["sigma_lo_nether"] for s in gen_stats]
        hi_vals = [s["sigma_hi_nether"] for s in gen_stats]
        valid = [(l, h) for l, h in zip(lo_vals, hi_vals)
                 if not np.isnan(l) and not np.isnan(h)]
        if valid:
            lo_v, hi_v = zip(*valid)
            ax.scatter(hi_v, lo_v, c=[s["min_beta"] for s, l, h in
                       zip(gen_stats, lo_vals, hi_vals) if not np.isnan(l) and not np.isnan(h)],
                       cmap="RdYlGn_r", s=50, edgecolors="white", linewidths=0.5)
            ax.plot([0, max(max(lo_v), max(hi_v))], [0, max(max(lo_v), max(hi_v))],
                    "k--", alpha=0.3)
            ax.set_xlabel("Nether % at σ ≥ 0.3 (early/exploration)")
            ax.set_ylabel("Nether % at σ < 0.3 (late/consolidation)")
            ax.set_title("C: Sigma-Band Nether Source")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "Insufficient sigma data", ha="center",
                    va="center", transform=ax.transAxes, color="#888888")
            ax.set_title("C: Sigma-Band (no data)")
    else:
        ax.text(0.5, 0.5, "No sigma column", ha="center",
                va="center", transform=ax.transAxes, color="#888888")
        ax.set_title("C: Sigma-Band (no sigma)")

    # Panel D: distribution of min(β) with statistics
    ax = axes[1, 1]
    ax.hist(min_betas, bins=min(20, max(5, n // 3)), color="#9b59b6",
            alpha=0.7, edgecolor="white")
    med = np.median(min_betas)
    ax.axvline(med, color="#00d4aa", ls="-", lw=2, label=f"median={med:.4f}")
    ax.axvline(-0.02, color="#ffaa44", ls="--", lw=1, alpha=0.7)
    ax.axvline(-0.05, color="#ff6b6b", ls="--", lw=1, alpha=0.7)
    deep = sum(1 for b in min_betas if b < -0.05)
    moderate = sum(1 for b in min_betas if -0.05 <= b < -0.02)
    healthy = sum(1 for b in min_betas if b >= -0.02)
    ax.set_title(f"D: min(β) Distribution — {healthy} healthy / {moderate} moderate / {deep} deep")
    ax.set_xlabel(r"min $\beta$")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Per-Generation Diagnostic — {n} gens", fontsize=13, y=1.01)
    plt.tight_layout()
    return fig


def plot_3d_phase_trajectory(df, use_spline=False):
    """3D phase portrait: g_micro × secondary × layer with proper scaling."""
    if df is None or df.empty:
        return None
    has_gap = "spectral_gap" in df.columns
    has_ent = "sv_entropy" in df.columns
    if not has_gap and not has_ent:
        return None

    sec_col = "spectral_gap" if has_gap else "sv_entropy"
    sec_label = "Spectral Gap" if has_gap else "SV Entropy"

    layer_data = df.groupby("layer").agg(
        g_mean=("g_micro", "mean"), sec_mean=(sec_col, "mean"))
    layers = layer_data.index.values.astype(float)
    g_mean = layer_data["g_mean"].values
    sec_mean = layer_data["sec_mean"].values

    fig = go.Figure()

    # Per-generation ghost traces
    n_shown = 0
    for _, gdf in df.groupby("gen_idx"):
        if n_shown >= 25:
            break
        gl = gdf.groupby("layer").agg(g=("g_micro", "mean"), s=(sec_col, "mean"))
        if len(gl) < 5:
            continue
        fig.add_trace(go.Scatter3d(
            x=gl["g"].values, y=gl["s"].values, z=gl.index.values.astype(float),
            mode="lines", line=dict(color="rgba(255,107,107,0.15)", width=1),
            showlegend=False, name="",
        ))
        n_shown += 1

    # Mean trajectory
    _add_spline_or_linear(fig, g_mean, sec_mean, layers, use_spline,
                          color="cyan", width=5, name="Mean trajectory")

    # Data points colored by layer
    fig.add_trace(go.Scatter3d(
        x=g_mean, y=sec_mean, z=layers, mode="markers",
        marker=dict(size=6, color=layers, colorscale="Viridis", showscale=True,
                    colorbar=dict(title="Layer", x=1.02)),
        name="Layer points",
    ))

    fig.update_layout(
        scene=_dark_scene(
            "⟨g_micro⟩", sec_label, "Layer",
            x_range=_auto_range(g_mean),
            y_range=_auto_range(sec_mean),
        ),
        paper_bgcolor="#1a1a1a", font=dict(color="#ccc"),
        title="3D Phase Trajectory (orbit: drag, zoom: scroll)",
        showlegend=False, margin=dict(l=0, r=0, t=40, b=0), height=650,
    )
    return fig


def plot_3d_heatmap_surface(df, temporal_bin=1):
    """3D surface plot of g(layer, step) — the heatmap lifted into 3D."""
    if df is None or df.empty:
        return None
    if temporal_bin > 1:
        df = df.copy()
        df["token_idx"] = (df["token_idx"] // temporal_bin) * temporal_bin

    mats = []
    for _, gdf in df.groupby("gen_idx"):
        p = gdf.pivot_table(index="layer", columns="token_idx", values="g_micro", aggfunc="mean")
        mats.append(p.values)
    if not mats:
        return None
    min_l = min(m.shape[0] for m in mats)
    min_t = min(m.shape[1] for m in mats)
    aligned = np.array([m[:min_l, :min_t] for m in mats])
    mean_m = np.nanmean(aligned, axis=0)

    layers = np.arange(min_l)
    steps = np.arange(min_t)
    step_grid, layer_grid = np.meshgrid(steps, layers)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=("g(Layer, Step) — Raw", "g(Layer, Step) — Normalized"),
    )

    # Raw surface
    fig.add_trace(go.Surface(
        x=step_grid, y=layer_grid, z=mean_m,
        colorscale="Viridis", showscale=True,
        colorbar=dict(title="g_micro", x=0.45, len=0.8),
        name="Raw",
    ), row=1, col=1)

    # Normalized surface
    layer_means = np.nanmean(mean_m, axis=1, keepdims=True)
    layer_means[layer_means < 1e-8] = 1e-8
    norm_m = mean_m / layer_means
    fig.add_trace(go.Surface(
        x=step_grid, y=layer_grid, z=norm_m,
        colorscale="RdBu_r", showscale=True, cmin=0.8, cmax=1.2,
        colorbar=dict(title="Normalized", x=1.0, len=0.8),
        name="Normalized",
    ), row=1, col=2)

    for col_idx in [1, 2]:
        fig.update_scenes(
            xaxis_title="Step/Token", yaxis_title="Layer", zaxis_title="g_micro",
            bgcolor="#1a1a1a",
            xaxis=dict(backgroundcolor="#1a1a1a", gridcolor="#333"),
            yaxis=dict(backgroundcolor="#1e1e1e", gridcolor="#333"),
            zaxis=dict(backgroundcolor="#222", gridcolor="#333"),
            row=1, col=col_idx,
        )

    fig.update_layout(
        paper_bgcolor="#1a1a1a", font=dict(color="#ccc"),
        title="3D Surface: g_micro(Layer, Step)",
        margin=dict(l=0, r=0, t=40, b=0), height=600,
    )
    return fig


def plot_3d_beta_flow(df, use_spline=False):
    """3D beta layer flow: g_micro × β_layer × layer — shows the RG flow in full 3D."""
    if df is None or df.empty:
        return None

    layer_data = df.groupby("layer")["g_micro"].mean()
    layers = layer_data.index.values.astype(float)
    g_mean = layer_data.values

    beta = np.diff(g_mean)
    beta_layers = layers[:-1] + 0.5  # midpoints
    g_at_beta = (g_mean[:-1] + g_mean[1:]) / 2.0

    has_sec = "spectral_gap" in df.columns or "sv_entropy" in df.columns
    sec_col = "spectral_gap" if "spectral_gap" in df.columns else ("sv_entropy" if "sv_entropy" in df.columns else None)

    if sec_col:
        sec_data = df.groupby("layer")[sec_col].mean()
        sec_mean = sec_data.values
        sec_at_beta = (sec_mean[:-1] + sec_mean[1:]) / 2.0
        sec_label = "Spectral Gap" if sec_col == "spectral_gap" else "SV Entropy"
    else:
        sec_at_beta = beta_layers
        sec_label = "Layer"

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=("β Flow: g × β × Layer", "β Flow: g × β × Secondary"),
    )

    # Panel 1: g × β × layer
    _add_spline_or_linear(fig, g_at_beta, beta, beta_layers, use_spline,
                          color="cyan", width=4, name="β trajectory", row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=g_at_beta, y=beta, z=beta_layers, mode="markers",
        marker=dict(size=5, color=beta, colorscale="RdBu_r", showscale=True,
                    cmid=0, colorbar=dict(title="β", x=0.45, len=0.7)),
        name="β values",
    ), row=1, col=1)
    # Zero-plane for β
    g_range = [g_at_beta.min(), g_at_beta.max()]
    l_range = [beta_layers.min(), beta_layers.max()]
    fig.add_trace(go.Surface(
        x=[[g_range[0], g_range[1]], [g_range[0], g_range[1]]],
        y=[[0, 0], [0, 0]],
        z=[[l_range[0], l_range[0]], [l_range[1], l_range[1]]],
        showscale=False, opacity=0.15, colorscale=[(0, "gray"), (1, "gray")],
        name="β=0 plane",
    ), row=1, col=1)

    fig.update_scenes(
        _dark_scene("⟨g_micro⟩", "β_layer", "Layer",
                    x_range=_auto_range(g_at_beta),
                    y_range=_auto_range(beta)),
        row=1, col=1,
    )

    # Panel 2: g × β × secondary
    if sec_col:
        _add_spline_or_linear(fig, g_at_beta, beta, sec_at_beta, use_spline,
                              color="orange", width=4, name="β vs secondary", row=1, col=2)
        fig.add_trace(go.Scatter3d(
            x=g_at_beta, y=beta, z=sec_at_beta, mode="markers",
            marker=dict(size=5, color=beta, colorscale="RdBu_r", cmid=0, showscale=False),
            name="",
        ), row=1, col=2)
        fig.update_scenes(
            _dark_scene("⟨g_micro⟩", "β_layer", sec_label,
                        x_range=_auto_range(g_at_beta),
                        y_range=_auto_range(beta)),
            row=1, col=2,
        )

    fig.update_layout(
        paper_bgcolor="#1a1a1a", font=dict(color="#ccc"),
        title="3D Beta Flow — RG Trajectory",
        showlegend=False, margin=dict(l=0, r=0, t=40, b=0), height=600,
    )
    return fig


def plot_3d_secondary_at_crossings(df, use_spline=False):
    """3D: layer × g_micro × secondary — the spiral analysis panel A lifted to 3D."""
    if df is None or df.empty:
        return None
    has_gap = "spectral_gap" in df.columns
    has_ent = "sv_entropy" in df.columns
    if not has_gap and not has_ent:
        return None

    sec_col = "spectral_gap" if has_gap else "sv_entropy"
    sec_label = "Spectral Gap" if has_gap else "SV Entropy"

    ld = df.groupby("layer").agg(g=("g_micro", "mean"), s=(sec_col, "mean"))
    layers = ld.index.values.astype(float)
    g_mean = ld["g"].values
    sec_mean = ld["s"].values

    beta = np.diff(g_mean)

    fig = go.Figure()

    # Per-gen ghost traces
    n_shown = 0
    for _, gdf in df.groupby("gen_idx"):
        if n_shown >= 30:
            break
        gl = gdf.groupby("layer").agg(g=("g_micro", "mean"), s=(sec_col, "mean"))
        if len(gl) < 5:
            continue
        fig.add_trace(go.Scatter3d(
            x=gl.index.values.astype(float), y=gl["g"].values, z=gl["s"].values,
            mode="lines", line=dict(color="rgba(255,107,107,0.12)", width=1),
            showlegend=False, name="",
        ))
        n_shown += 1

    # Mean trajectory
    _add_spline_or_linear(fig, layers, g_mean, sec_mean, use_spline,
                          color="cyan", width=5, name="Mean")

    # Points colored by layer
    fig.add_trace(go.Scatter3d(
        x=layers, y=g_mean, z=sec_mean, mode="markers",
        marker=dict(size=6, color=layers, colorscale="Viridis", showscale=True,
                    colorbar=dict(title="Layer")),
        name="",
    ))

    # Mark beta crossings
    for i in range(len(beta) - 1):
        if beta[i] * beta[i + 1] < 0:
            frac = -beta[i] / (beta[i + 1] - beta[i])
            cl = layers[i] + frac
            cg = g_mean[i] + frac * (g_mean[i + 1] - g_mean[i])
            cs = sec_mean[i] + frac * (sec_mean[i + 1] - sec_mean[i])
            fig.add_trace(go.Scatter3d(
                x=[cl], y=[cg], z=[cs], mode="markers",
                marker=dict(size=12, color="red", symbol="x"),
                name=f"β crossing L≈{cl:.1f}",
            ))

    fig.update_layout(
        scene=_dark_scene(
            "Layer", "⟨g_micro⟩", sec_label,
            y_range=_auto_range(g_mean),
            z_range=_auto_range(sec_mean),
        ),
        paper_bgcolor="#1a1a1a", font=dict(color="#ccc"),
        title="3D: Layer × g_micro × Secondary — Spiral Visualization",
        showlegend=False, margin=dict(l=0, r=0, t=40, b=0), height=650,
    )
    return fig


# =============================================================================
# Export functions
# =============================================================================
def export_all_plots(df, source_dir, custom_prefix=None):
    """Save all plots as PNGs to the source directory."""
    if df is None or source_dir is None:
        return "No data or source directory"

    out_dir = Path(source_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = custom_prefix.strip() if custom_prefix and custom_prefix.strip() else f"cospafs_export_{timestamp}"
    saved = []

    plot_funcs = {
        "depth_profile": lambda: plot_depth_profile(df),
        "token_trajectory": lambda: plot_token_trajectory(df),
        "heatmap": lambda: plot_heatmap(df),
        "beta_layer": lambda: plot_beta_layer(df),
        "spiral_analysis": lambda: plot_spiral_analysis(df),
        "phase_portrait": lambda: plot_phase_portrait(df),
        "power_spectrum": lambda: plot_power_spectrum(df),
        "phi_micro_depth": lambda: plot_phi_micro_depth(df),
        "phi_micro_heatmap": lambda: plot_phi_micro_heatmap(df),
        # v3 plots
        "beta_g_mean_profile": lambda: plot_beta_g_mean_profile(df),
        "early_vs_late_tokens": lambda: plot_early_vs_late_tokens(df),
        "key_results": lambda: plot_key_results(df),
        "interval_decomposition": lambda: plot_interval_decomposition(df),
        "bootstrap_ratio": lambda: plot_bootstrap_ratio(df),
        "beta_g_comparison": lambda: plot_beta_g_comparison(df),
    }

    for name, func in plot_funcs.items():
        try:
            fig = func()
            if fig is not None:
                path = out_dir / f"{prefix}_{name}.png"
                fig.savefig(str(path), dpi=150, bbox_inches="tight")
                plt.close(fig)
                saved.append(name)
        except Exception as e:
            saved.append(f"{name}: ERROR {e}")

    # Save 3D as interactive HTML
    try:
        fig3d = plot_3d_phase_trajectory(df)
        if fig3d is not None:
            path = out_dir / f"{prefix}_3d_phase_trajectory.html"
            fig3d.write_html(str(path))
            saved.append("3d_phase_trajectory (HTML)")
    except Exception as e:
        saved.append(f"3d_phase: ERROR {e}")

    try:
        fig3d = plot_3d_heatmap_surface(df)
        if fig3d is not None:
            path = out_dir / f"{prefix}_3d_heatmap_surface.html"
            fig3d.write_html(str(path))
            saved.append("3d_heatmap_surface (HTML)")
    except Exception as e:
        saved.append(f"3d_surface: ERROR {e}")

    try:
        fig3d = plot_3d_beta_flow(df)
        if fig3d is not None:
            path = out_dir / f"{prefix}_3d_beta_flow.html"
            fig3d.write_html(str(path))
            saved.append("3d_beta_flow (HTML)")
    except Exception as e:
        saved.append(f"3d_beta: ERROR {e}")

    try:
        fig3d = plot_3d_secondary_at_crossings(df)
        if fig3d is not None:
            path = out_dir / f"{prefix}_3d_secondary_crossings.html"
            fig3d.write_html(str(path))
            saved.append("3d_secondary_crossings (HTML)")
    except Exception as e:
        saved.append(f"3d_secondary: ERROR {e}")

    return f"Exported to {out_dir}:\n" + "\n".join(f"  ✓ {s}" for s in saved)


def export_gif(df, source_dir, plot_type="token_trajectory", fps=4, max_frames=50):
    """Export an animated GIF showing the plot building up across generations."""
    if df is None or not _HAS_IMAGEIO:
        return "Need data + imageio installed"

    out_dir = Path(source_dir) if source_dir else Path(".")
    gen_indices = sorted(df["gen_idx"].unique())
    total = len(gen_indices)
    if total < 2:
        return "Need ≥2 generations"

    step = max(1, total // max_frames)
    checkpoints = list(range(1, total + 1, step))
    if checkpoints[-1] != total:
        checkpoints.append(total)

    frames = []
    for n in checkpoints:
        subset = df[df["gen_idx"].isin(gen_indices[:n])]
        if plot_type == "token_trajectory":
            fig = plot_token_trajectory(subset)
        elif plot_type == "depth_profile":
            fig = plot_depth_profile(subset)
        elif plot_type == "beta_layer":
            fig = plot_beta_layer(subset)
        elif plot_type == "heatmap":
            fig = plot_heatmap(subset)
        else:
            fig = plot_token_trajectory(subset)

        if fig is not None:
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
            frames.append(buf)
            plt.close(fig)

    if not frames:
        return "No frames generated"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = out_dir / f"cospafs_{plot_type}_{timestamp}.gif"
    imageio.mimsave(str(gif_path), frames, fps=fps, loop=0)
    return f"Saved {gif_path} ({len(frames)} frames, {fps} fps)"


# =============================================================================
# Gradio App
# =============================================================================
def make_app():
    dm = DataManager()

    def _refresh_all(layer_bin, temporal_bin, use_spline):
        df = dm.get_df()
        lb = max(1, int(layer_bin))
        tb = max(1, int(temporal_bin))
        spline = bool(use_spline)
        if df is None:
            no_3d = "<p style='color:#888; text-align:center; padding:2em;'>No data</p>"
            return (None,) * 16 + (no_3d,) * 4 + ("No data.", "Idle")

        errors = []
        def safe(func, *args):
            try:
                r = func(*args)
                return r
            except Exception as e:
                errors.append(f"{func.__name__}: {e}")
                return None

        r = []
        r.append(safe(plot_depth_profile, df, lb))            # 0
        r.append(safe(plot_token_trajectory, df, tb))          # 1
        r.append(safe(plot_heatmap, df, tb))                   # 2
        r.append(safe(plot_beta_layer, df))                    # 3
        r.append(safe(plot_spiral_analysis, df))               # 4
        r.append(safe(plot_phase_portrait, df))                # 5
        r.append(safe(plot_power_spectrum, df))                # 6
        r.append(safe(plot_phi_micro_depth, df, lb))             # 7
        r.append(safe(plot_phi_micro_heatmap, df, tb))           # 8
        # NEW v3 plots
        r.append(safe(plot_beta_g_mean_profile, df))             # 9
        r.append(safe(plot_early_vs_late_tokens, df))            # 10
        r.append(safe(plot_key_results, df))                     # 11
        r.append(safe(plot_interval_decomposition, df))          # 12
        r.append(safe(plot_bootstrap_ratio, df))                 # 13
        r.append(safe(plot_beta_g_comparison, df))               # 14
        r.append(safe(plot_per_gen_diagnostic, df))               # 15
        # 3D plots → convert to HTML for gr.HTML rendering
        def safe_3d(func, *args):
            try:
                fig = func(*args)
                return _plotly_to_html(fig)
            except Exception as e:
                errors.append(f"3D {func.__name__}: {e}")
                return f"<p style='color:#f66; padding:1em;'>Error: {e}</p>"

        r.append(safe_3d(plot_3d_phase_trajectory, df, spline))   # 16
        r.append(safe_3d(plot_3d_heatmap_surface, df, tb))        # 17
        r.append(safe_3d(plot_3d_beta_flow, df, spline))          # 18
        r.append(safe_3d(plot_3d_secondary_at_crossings, df, spline))  # 19

        summary = dm.summary()
        status = dm.status()
        if errors:
            status += "\n" + "\n".join(errors)
        # 20 outputs for plots + summary + status = 22
        return tuple(r) + (summary, status)

    with gr.Blocks(title="CoS-PAFS Live Analyzer v3.1") as app:
        gr.HTML("""
        <div style="text-align:center; font-size:1.6rem; font-weight:700; color:#e0e0e0;">
            🌀 CoS-PAFS Live Analyzer v3.1
        </div>
        <p style="text-align:center; color:#888; margin-top:0;">
            Enhanced telemetry — per-gen diagnostic, sigma-band windowing, gen_idx deduplication
        </p>
        """)

        timer = gr.Timer(value=5, active=False)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Data Source")
                file_upload = gr.File(file_count="multiple", label="Upload CSVs", file_types=[".csv"])
                btn_upload = gr.Button("Load Uploads", variant="primary")
                gr.Markdown("### 👁 Live Watch")
                watch_paths = gr.Textbox(label="Watch Paths (comma-sep)", placeholder="/path/to/run.csv")
                with gr.Row():
                    btn_load = gr.Button("Load Paths")
                    btn_start = gr.Button("▶ Watch", variant="secondary")
                    btn_stop = gr.Button("⏹ Stop")
                poll_interval = gr.Slider(1, 60, 5, step=1, label="Poll (s)")

                gr.Markdown("### 🔧 Controls")
                with gr.Row():
                    layer_bin = gr.Slider(1, 8, 1, step=1, label="Layer bin")
                    temporal_bin = gr.Slider(1, 20, 1, step=1, label="Temporal bin")
                use_spline = gr.Checkbox(label="Spline curves (3D plots)", value=False)
                btn_refresh = gr.Button("🔄 Refresh All", variant="primary")

                gr.Markdown("### 🪟 Windowing")
                with gr.Row():
                    w_layer_min = gr.Number(0, label="Layer min", precision=0)
                    w_layer_max = gr.Number(31, label="Layer max", precision=0)
                with gr.Row():
                    w_token_min = gr.Number(0, label="Step/Token min", precision=0)
                    w_token_max = gr.Number(9999, label="Step/Token max", precision=0)
                with gr.Row():
                    w_gen_min = gr.Number(0, label="Gen min", precision=0)
                    w_gen_max = gr.Number(9999, label="Gen max", precision=0)
                with gr.Row():
                    w_sigma_min = gr.Number(0.0, label="Sigma min", precision=2)
                    w_sigma_max = gr.Number(1.0, label="Sigma max", precision=2)
                with gr.Row():
                    btn_window = gr.Button("Apply Window")
                    btn_reset_window = gr.Button("Reset Window")

                gr.Markdown("### 💾 Export")
                export_prefix = gr.Textbox(label="Export prefix", placeholder="auto from CSV name")
                btn_export = gr.Button("Export All Plots", variant="secondary")
                with gr.Row():
                    gif_type = gr.Dropdown(
                        ["token_trajectory", "depth_profile", "beta_layer", "heatmap"],
                        value="token_trajectory", label="GIF plot type"
                    )
                    btn_gif = gr.Button("Export GIF")
                export_status = gr.Textbox(label="Export Status", lines=4, interactive=False)

                gr.Markdown("### 📊 Summary")
                summary_box = gr.Textbox(label="Dataset", lines=6, interactive=False)
                status_box = gr.Textbox(label="Status", lines=3, interactive=False)

            with gr.Column(scale=3):
                gr.Markdown("### 2D Plots — Base")
                with gr.Tab("Depth & Sublattice"):
                    plot_depth = gr.Plot()
                with gr.Tab("Step Trajectory"):
                    plot_token = gr.Plot()
                with gr.Tab("Heatmap"):
                    plot_heat = gr.Plot()
                with gr.Tab("Beta Layer (Cloud)"):
                    plot_beta = gr.Plot()
                with gr.Tab("Spiral Analysis"):
                    plot_spiral = gr.Plot()
                with gr.Tab("Phase Portrait"):
                    plot_phase = gr.Plot()
                with gr.Tab("Power Spectrum"):
                    plot_power = gr.Plot()
                with gr.Tab("φ_micro Depth"):
                    plot_phi_depth = gr.Plot()
                with gr.Tab("φ_micro Heatmap"):
                    plot_phi_heat = gr.Plot()

                gr.Markdown("### 2D Plots — v3 Analysis")
                with gr.Tab("β(g) Mean-Profile"):
                    plot_beta_mp = gr.Plot()
                with gr.Tab("Early vs Late Tokens"):
                    plot_early_late = gr.Plot()
                with gr.Tab("Key Results"):
                    plot_key = gr.Plot()
                with gr.Tab("Interval Decomposition"):
                    plot_interval = gr.Plot()
                with gr.Tab("Bootstrap Ratio"):
                    plot_bootstrap = gr.Plot()
                with gr.Tab("Attn vs MLP β(g)"):
                    plot_attn_mlp = gr.Plot()
                with gr.Tab("Per-Gen Diagnostic"):
                    plot_gen_diag = gr.Plot()

                gr.Markdown("### 3D Interactive (drag to orbit, scroll to zoom)")
                with gr.Tab("3D Phase Trajectory"):
                    plot_3d_phase = gr.HTML()
                with gr.Tab("3D Surface (Heatmap)"):
                    plot_3d_surface = gr.HTML()
                with gr.Tab("3D Beta Flow"):
                    plot_3d_beta = gr.HTML()
                with gr.Tab("3D Spiral / Secondary"):
                    plot_3d_spiral = gr.HTML()

        # All plot outputs in order
        all_plots = [
            plot_depth, plot_token, plot_heat, plot_beta, plot_spiral,
            plot_phase, plot_power, plot_phi_depth, plot_phi_heat,
            plot_beta_mp, plot_early_late, plot_key, plot_interval,
            plot_bootstrap, plot_attn_mlp, plot_gen_diag,
            plot_3d_phase, plot_3d_surface, plot_3d_beta, plot_3d_spiral,
            summary_box, status_box,
        ]
        control_inputs = [layer_bin, temporal_bin, use_spline]

        # Wiring
        def do_upload(files):
            dm.load_uploads(files)
            return dm.summary(), dm.status(), dm.default_prefix()

        def do_load_paths(txt):
            dm.load_paths(txt)
            return dm.summary(), dm.status(), dm.default_prefix()

        def do_start(interval):
            dm.start_watch(interval)
            return gr.Timer(active=True), dm.status()

        def do_stop():
            dm.stop_watch()
            return gr.Timer(active=False), dm.status()

        def do_window(lmin, lmax, tmin, tmax, gmin, gmax, smin, smax):
            return dm.apply_window(int(lmin), int(lmax), int(tmin), int(tmax), int(gmin), int(gmax), float(smin), float(smax))

        def do_reset_window():
            return dm.reset_window()

        def do_export(pfx):
            df = dm.get_df()
            return export_all_plots(df, dm._source_dir, pfx)

        def do_gif(plot_type):
            df = dm.get_df()
            return export_gif(df, dm._source_dir, plot_type)

        btn_upload.click(do_upload, file_upload, [summary_box, status_box, export_prefix])\
                  .then(_refresh_all, control_inputs, all_plots)
        btn_load.click(do_load_paths, watch_paths, [summary_box, status_box, export_prefix])\
                .then(_refresh_all, control_inputs, all_plots)
        btn_start.click(do_start, poll_interval, [timer, status_box])
        btn_stop.click(do_stop, outputs=[timer, status_box])
        btn_refresh.click(_refresh_all, control_inputs, all_plots)
        btn_window.click(do_window, [w_layer_min, w_layer_max, w_token_min, w_token_max, w_gen_min, w_gen_max, w_sigma_min, w_sigma_max], status_box)\
                  .then(_refresh_all, control_inputs, all_plots)
        btn_reset_window.click(do_reset_window, outputs=status_box)\
                        .then(_refresh_all, control_inputs, all_plots)
        btn_export.click(do_export, export_prefix, export_status)
        btn_gif.click(do_gif, gif_type, export_status)
        timer.tick(_refresh_all, control_inputs, all_plots)

    return app


if __name__ == "__main__":
    demo = make_app()
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
        share=False, show_error=True,
        theme=gr.themes.Soft(primary_hue="teal", secondary_hue="orange", neutral_hue="zinc"),
    )
