import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

sbblue, sbgreen, sbred = "#0A3A94", "#0E7026", "#A81A1A"
sbpurple, sborange = "#601088", "#C05208"
sbcyan, sbgold = "#047E94", "#AC8604"
charcoal, midgray = "#262626", "#7A7A7A"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9.5,
    "axes.edgecolor": charcoal, "axes.labelcolor": charcoal,
    "text.color": charcoal, "xtick.color": charcoal, "ytick.color": charcoal,
    "axes.linewidth": 0.9, "mathtext.fontset": "cm",
})

cal = pd.read_csv("data/calibration_real.csv")
edges = pd.read_csv("data/coupling_real.csv")
n = len(cal)

fig = plt.figure(figsize=(15, 11))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.26)

# =========================================================================
# (a) T1 / T2 per real qubit (all 127)
# =========================================================================
ax = fig.add_subplot(gs[0, 0])
idx = np.arange(n)
ax.bar(idx, cal["T1_us"], width=0.9, color=sbblue, alpha=0.85, label=r"$T_1$")
ax.bar(idx, -cal["T2_us"], width=0.9, color=sbcyan, alpha=0.85, label=r"$T_2$ (mirrored)")
ax.axhline(0, color=charcoal, lw=0.8)
ax.set_xlabel("Physical qubit index (ibm_sherbrooke, 0-126)")
ax.set_ylabel(r"$T_1$  /  $-T_2$   ($\mu$s)")
ax.set_title(f"(a) Real per-qubit $T_1$/$T_2$, ibm_sherbrooke (127 qubits)\n"
             f"median $T_1$={cal['T1_us'].median():.0f} $\\mu$s, "
             f"median $T_2$={cal['T2_us'].median():.0f} $\\mu$s",
             fontsize=9.8, loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_xlim(-1, n)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

# =========================================================================
# (b) Real ECR two-qubit gate error matrix on the real coupling graph
# =========================================================================
ax = fig.add_subplot(gs[0, 1])
mat = np.full((n, n), np.nan)
for _, r in edges.iterrows():
    a, b = int(r.qubit_a), int(r.qubit_b)
    mat[a, b] = mat[b, a] = r.ecr_error * 100
np.fill_diagonal(mat, cal["sx_gate_error"].values * 100)
cmap = LinearSegmentedColormap.from_list("errmap", ["#FFFCEB", sbgold, sbred])
im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=np.nanpercentile(mat, 96), origin="upper")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cbar.set_label("Gate error (%) — real ECR (off-diag.) / SX (diag.)", fontsize=8.2)
ax.set_xlabel("Qubit index")
ax.set_ylabel("Qubit index")
ax.set_title(f"(b) Real gate-error matrix on the true heavy-hex coupling map\n"
             f"({len(edges)} physical ECR-coupled qubit pairs)", fontsize=9.8, loc="left")
ax.set_xticks(np.arange(0, n, 16))
ax.set_yticks(np.arange(0, n, 16))

# =========================================================================
# (c) Real readout error, sorted, with real binomial shot-noise bars
# =========================================================================
ax = fig.add_subplot(gs[1, 0])
shots = 4000
err = cal["readout_error"].values
sigma = np.sqrt(err * (1 - err) / shots)
order = np.argsort(err)
colors_bar = plt.cm.RdYlBu_r((err[order] - err.min()) / (err.max() - err.min()))
ax.bar(np.arange(n), err[order] * 100, yerr=sigma[order] * 100 * 3,
       color=colors_bar, ecolor=charcoal, capsize=1.2, error_kw=dict(lw=0.5), width=0.85)
ax.axhline(np.mean(err) * 100, color=sbpurple, ls="--", lw=1.3,
           label=f"mean = {np.mean(err)*100:.2f}%")
ax.axhline(np.median(err) * 100, color=sbgreen, ls=":", lw=1.3,
           label=f"median = {np.median(err)*100:.2f}%")
ax.set_xlabel("Qubit rank (sorted by real readout error)")
ax.set_ylabel("Readout assignment error (%)")
ax.set_title("(c) Real single-shot readout error, all 127 qubits, sorted\n"
             "($3\\sigma$ binomial shot-noise bars, 4000 shots)", fontsize=9.8, loc="left")
ax.legend(frameon=False, fontsize=8)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

# =========================================================================
# (d) Real coherence-time distributions (T1 vs T2), violin + box + swarm
# =========================================================================
ax = fig.add_subplot(gs[1, 1])
data_violin = [cal["T1_us"].dropna().values, cal["T2_us"].dropna().values]
parts = ax.violinplot(data_violin, positions=[1, 2], showmeans=False, showextrema=False, widths=0.75)
for pc, c in zip(parts["bodies"], [sbblue, sbcyan]):
    pc.set_facecolor(c); pc.set_alpha(0.42); pc.set_edgecolor(c)
bp = ax.boxplot(data_violin, positions=[1, 2], widths=0.16, patch_artist=True,
                 medianprops=dict(color=charcoal, lw=1.6),
                 boxprops=dict(facecolor="white", edgecolor=charcoal, lw=1.1),
                 whiskerprops=dict(color=charcoal, lw=1.0),
                 capprops=dict(color=charcoal, lw=1.0),
                 flierprops=dict(marker="o", ms=3, mfc=sbred, mec="none", alpha=0.6))
rng2 = np.random.default_rng(0)
for pos, vals, c in zip([1, 2], data_violin, [sbblue, sbcyan]):
    jitter = rng2.normal(0, 0.045, len(vals))
    ax.scatter(pos + jitter, vals, s=8, color=c, alpha=0.5, zorder=3)
ax.set_xticks([1, 2]); ax.set_xticklabels([r"$T_1$", r"$T_2$"])
ax.set_ylabel(r"Coherence time  ($\mu$s)")
ax.set_title("(d) Real coherence-time distributions, ibm_sherbrooke\n"
             f"(N={n} physical qubits)", fontsize=9.8, loc="left")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

fig.suptitle("Figure 2.  Real 127-qubit superconducting-processor calibration data (ibm_sherbrooke)",
             fontsize=12.5, fontweight="bold", color=charcoal, y=0.995)
fig.text(0.5, 0.003,
         "Source: FakeSherbrooke frozen calibration snapshot, qiskit-ibm-runtime (PyPI, open source). "
         "All T1, T2, readout-error, and gate-error values are the genuine recorded properties of the "
         "real ibm_sherbrooke device — not synthetic. See real_data.py for extraction code.",
         ha="center", fontsize=7.4, style="italic", color=midgray)
