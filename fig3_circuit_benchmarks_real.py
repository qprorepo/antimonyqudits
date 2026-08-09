import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

bench = pd.read_csv("data/circuit_benchmarks_real.csv")
tp = pd.read_csv("data/transpile_overhead_real.csv")
smq = pd.read_csv("data/supermarq_real.csv")

fig = plt.figure(figsize=(14.5, 11))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.30)

family_colors = {
    "arithmetic": sbblue, "algebraic": sbred, "search": sbgold,
    "simulation": sbcyan, "variational": sbpurple, "entanglement": sbgreen,
    "QEC": sborange, "linear-algebra": "#8B4513", "ML": "#444444",
    "sampling": "#C71585", "other": midgray,
}

# =========================================================================
# (a) Real circuit depth vs qubit count, log-log, coloured by family
# =========================================================================
ax = fig.add_subplot(gs[0, 0])
# clip extreme outlier circuits (a few QASMBench entries have >1e5 depth)
plot_df = bench[bench["depth"] <= bench["depth"].quantile(0.97)]
for fam, sub in plot_df.groupby("family"):
    c = family_colors.get(fam, midgray)
    ax.scatter(sub["n_qubits"], sub["depth"], color=c, s=34, label=fam,
               edgecolor="white", linewidth=0.4, alpha=0.85, zorder=3)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Number of qubits (log scale)")
ax.set_ylabel("Circuit depth (log scale)")
ax.set_title(f"(a) Real circuit depth vs. width, QASMBench "
             f"(N={len(plot_df)} real circuits,\nsmall+medium+large sets, 97th-pct depth clip)",
             fontsize=9.6, loc="left")
ax.legend(frameon=False, fontsize=6.6, ncol=3, loc="upper left")

# =========================================================================
# (b) Real gate-count composition, largest tractable instance per family
# =========================================================================
ax = fig.add_subplot(gs[0, 1])
reasonable = bench[bench["gate_count"] <= 5000]
largest = (reasonable.loc[reasonable.groupby("family")["n_qubits"].idxmax()]
           .sort_values("gate_count"))
y = np.arange(len(largest))
ax.barh(y, largest["one_qubit_gates"], color=sbcyan, label="1-qubit gates")
ax.barh(y, largest["two_qubit_gates"], left=largest["one_qubit_gates"],
        color=sbred, label="2-qubit gates")
ax.set_yticks(y)
ax.set_yticklabels([f"{c} (n={q})" for c, q in zip(largest["circuit"], largest["n_qubits"])],
                    fontsize=7.6)
ax.set_xlabel("Gate count (exact, from real .qasm source)")
ax.set_title("(b) Real gate-count composition, one representative\n"
             "circuit per family (gate_count $\\leq$ 5000)", fontsize=9.6, loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower right")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

# =========================================================================
# (c) Real transpilation overhead onto the real ibm_sherbrooke coupling map
# =========================================================================
ax = fig.add_subplot(gs[1, 0])
tp_s = tp.sort_values("logical_depth")
x = np.arange(len(tp_s))
w = 0.38
ax.bar(x - w/2, tp_s["logical_depth"], width=w, color=sbblue,
       label="Logical depth (unrouted)")
ax.bar(x + w/2, tp_s["physical_depth"], width=w, color=sborange,
       label="Physical depth (ibm_sherbrooke, real transpile)")
for xi, (ld, pd_) in enumerate(zip(tp_s["logical_depth"], tp_s["physical_depth"])):
    ratio = pd_ / ld if ld > 0 else np.nan
    ax.text(xi, max(ld, pd_) * 1.05, f"{ratio:.1f}$\\times$", ha="center",
            fontsize=7.3, color=charcoal)
ax.set_xticks(x)
ax.set_xticklabels(tp_s["circuit"], rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Circuit depth")
ax.set_yscale("log")
ax.set_title("(c) Real transpilation overhead: optimization_level=1,\n"
             "qiskit.transpile() onto the real 127-qubit heavy-hex map", fontsize=9.6, loc="left")
ax.legend(frameon=False, fontsize=7.6)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

# =========================================================================
# (d) Real SupermarQ capability radar, 6 real benchmark circuits
# =========================================================================
ax = fig.add_subplot(gs[1, 1], projection="polar")
feat_cols = ["communication", "liveness", "parallelism", "measurement",
             "entanglement", "critical_depth"]
feat_labels = ["comm.", "liveness", "parallelism", "measurement", "entanglement", "crit.-depth"]
n_axes = len(feat_cols)
angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
angles += angles[:1]
palette = [sbblue, sbred, sbgold, sbcyan, sbpurple, sbgreen]
for (_, row), c in zip(smq.iterrows(), palette):
    vals = row[feat_cols].values.astype(float).tolist()
    vals += vals[:1]
    ax.plot(angles, vals, color=c, lw=1.8, label=f"{row['circuit']} (n={row['n_qubits']})")
    ax.fill(angles, vals, color=c, alpha=0.08)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(feat_labels, fontsize=8.3)
ax.set_ylim(0, 1.05)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=6.3)
ax.set_title("(d) Real SupermarQ capability-feature radar\n"
             "(Tomesh et al. formulas, computed on real QASMBench circuits)",
             fontsize=9.4, loc="center", pad=16)
ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.18), frameon=False, fontsize=7.2)

fig.suptitle("Figure 3.  Real circuit-benchmark suite (QASMBench) and real transpilation / SupermarQ analysis",
             fontsize=12.3, fontweight="bold", color=charcoal, y=0.997)
fig.text(0.5, 0.003,
         "Sources: QASMBench (github.com/pnnl/QASMBench, real .qasm circuits); qiskit.transpile() onto the "
         "real ibm_sherbrooke coupling map (FakeSherbrooke); SupermarQ feature formulas (PyPI: supermarq, "
         "arXiv:2202.11045). No synthetic values in this figure. See real_data.py for extraction code.",
         ha="center", fontsize=7.3, style="italic", color=midgray)

# ----------------------------------------------------------------------
# Save output (run from the repository root)
# ----------------------------------------------------------------------
import os
os.makedirs("output", exist_ok=True)
fig.savefig("output/fig3_circuit_benchmarks.pdf", dpi=300, bbox_inches="tight")
fig.savefig("output/fig3_circuit_benchmarks.png", dpi=300, bbox_inches="tight")
