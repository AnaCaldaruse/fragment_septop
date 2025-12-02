import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
from scipy.stats import kendalltau
rng = np.random.default_rng(1234)

# Data (load data in from abfe_cypd_absolute_values.csv)
df = pd.read_csv("abfe_cypd_absolute_values.csv")
ligands = df["ligand"].tolist()
dG_exp  = df["dG_exp"].values
dG_calc = df["dG_calc"].values
dG_err  = df["dG_err"].values

# Metrics (with optional bootstrap CIs)
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true)**2))

def mue(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def bootstrap_ci(metric_fn, y_true, y_pred, n=10000, alpha=0.05):
    npts = len(y_true)
    samples = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.integers(0, npts, size=npts)
        samples[i] = metric_fn(y_true[idx], y_pred[idx])
    lo = np.quantile(samples, alpha/2)
    hi = np.quantile(samples, 1 - alpha/2)
    return lo, hi

do_bootstrap = True  # set False if you want to skip resampling

RMSE = rmse(dG_exp, dG_calc)
MUE  = mue(dG_exp, dG_calc)
KTAU, _ = kendalltau(dG_exp, dG_calc)

if do_bootstrap:
    rmse_lo, rmse_hi = bootstrap_ci(rmse, dG_exp, dG_calc)
    mue_lo,  mue_hi  = bootstrap_ci(mue,  dG_exp, dG_calc)

    # bootstrap Kendall tau by resampling pairs
    def tau_boot(y_true, y_pred):
        t, _ = kendalltau(y_true, y_pred)
        return t
    ktau_lo, ktau_hi = bootstrap_ci(tau_boot, dG_exp, dG_calc)
else:
    rmse_lo = rmse_hi = mue_lo = mue_hi = ktau_lo = ktau_hi = np.nan

# Styling parameters
figsize = (5, 5)
font_sizes = {"title": 14, "labels": 14, "other": 14}
scatter_kwargs = {"s": 50, "marker": "o", "edgecolors": "k"}  

# Apply base font sizes
rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.labelsize": font_sizes["labels"],
    "xtick.labelsize": font_sizes["other"],
    "ytick.labelsize": font_sizes["other"],
})

# Color logic (Cinnabar-like)
cmap = plt.colormaps["coolwarm"] 

dist = np.abs(dG_calc - dG_exp)
max_error_for_scaling = 2.372
norm_errors = np.clip(dist / max_error_for_scaling, 0, 1.0)
point_colors = [cmap(val) for val in norm_errors]

# Plot
fig, ax = plt.subplots(figsize=figsize, facecolor="white")

# leave headroom for the right-column text
plt.subplots_adjust(top=0.70)

pad = 1.0
mn = min(dG_exp.min(), dG_calc.min()) - pad
mx = max(dG_exp.max(), dG_calc.max()) + pad
x  = np.linspace(mn, mx, 500)

# Cinnabar band styling
ax.fill_between(
    x, x-1.0, x+1.0,
    color="grey", alpha=0.2,
    edgecolor="grey", linewidth=1.0, zorder=0,
)
ax.fill_between(
    x, x-0.5, x+0.5,
    color="grey", alpha=0.2,
    edgecolor="grey", linewidth=1.0, zorder=1,
)
ax.plot(x, x, color="k", linestyle=":", linewidth=1.7)


# Draw error bars only 
for xi, yi, ei in zip(dG_exp, dG_calc, dG_err):
    ax.errorbar(
        xi, yi, yerr=ei,
        fmt="none", 
        ecolor="gray",
        elinewidth=1.8,
        capsize=0,
        zorder=1
    )

# Draw colored points with black outlines on top
ax.scatter(
    dG_exp, dG_calc,
    c=point_colors,
    edgecolors="k",
    s=scatter_kwargs.get("s", 50),
    marker=scatter_kwargs.get("marker", "o"),
    zorder=2
)

# Axes
ax.set_xlabel("Experimental ΔG (kcal/mol)")
ax.set_ylabel("Calculated ΔG (kcal/mol)")
ax.set_xlim(mn, mx)
ax.set_ylim(mn, mx)
ax.set_aspect("equal", adjustable="box")
ax.grid(False)


right_x = 0.78
title_fs = font_sizes["title"]
other_fs = font_sizes["other"]

# Format stats string
if do_bootstrap:
    stats_text = (
        f"RMSE:  {RMSE:.2f} [95%: {rmse_lo:.2f}, {rmse_hi:.2f}]\n"
        f" MUE:  {MUE:.2f} [95%: {mue_lo:.2f}, {mue_hi:.2f}]\n"
        f"KTAU:  {KTAU:.2f} [95%: {ktau_lo:.2f}, {ktau_hi:.2f}]"
    )
else:
    stats_text = (
        f"RMSE:  {RMSE:.2f}\n"
        f" MUE:  {MUE:.2f}\n"
        f"KTAU:  {KTAU:.2f}"
    )

# Combine title, N, and stats
long_title = (
    "CypD ABFE ΔG Results\n"
    f"(N = {len(dG_exp)})\n"
    + stats_text
)

# Plot all together in one right-aligned monospace block
fig.text(
    right_x, 0.965, long_title,
    ha="right", va="top",
    fontsize=font_sizes["title"],
    family="DejaVu Sans Mono",
    linespacing=1.1
)

plt.savefig("cypd_abfe_dg", dpi=300, bbox_inches="tight")

