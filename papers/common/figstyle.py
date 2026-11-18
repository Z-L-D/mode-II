"""Shared figure style for the Mode-I and Mode-II papers (arXiv, single-column 6.5in text width).
Palette: validated categorical slots (dataviz reference palette), fixed order, never cycled.
Every figure script: `from figstyle import *`, build the figure, then `save(fig, 'I-F3_observable_hierarchy')`,
which writes <paper>/figures/<name>.pdf (vector, used by LaTeX) and .png (preview)."""
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']   # categorical slots 1-5, in this order
GRAY = '#9a9893'; LIGHT = '#d9d7d2'; INK = '#1a1a19'; INK2 = '#52514e'
# fixed model colours (identity follows the entity, never its rank)
MODEL_COLOR = {'ZIT': C[0], 'ZIB': C[1], 'MOCHI': C[2], 'SD35M': C[3], 'SD35L': C[4]}
MODEL_LABEL = {'ZIT': 'Z-Image Turbo', 'ZIB': 'Z-Image Turbo 25-step (ZIT-25)', 'MOCHI': 'Mochi 1', 'SD35M': 'SD3.5 Medium', 'SD35L': 'SD3.5 Large'}
TEXTWIDTH = 6.5
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif',
    'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'legend.fontsize': 7.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'axes.edgecolor': '#8a8984', 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.7,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#ecebe7', 'grid.linewidth': 0.6,
    'lines.linewidth': 1.6, 'lines.markersize': 4, 'legend.frameon': False, 'figure.dpi': 150, 'savefig.bbox': 'tight',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
def save(fig, name, paper=None):
    """paper: 'mode-I' or 'mode-II'; inferred from the name prefix (I- / II-) when omitted."""
    paper = paper or ('mode-II' if name.startswith('II-') else 'mode-I')
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), paper, 'figures')
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + '.pdf')); fig.savefig(os.path.join(out, name + '.png'), dpi=200)
    plt.close(fig)
def panel_label(ax, s):
    ax.text(-0.12, 1.04, s, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='left', color=INK)
