import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

cal = pd.read_csv("data/calibration_real.csv").dropna().reset_index(drop=True)
# note: sx_gate_length_ns is a real but device-wide *fixed* pulse duration on this
# backend (zero variance across all 127 qubits) -- so the real per-qubit-averaged
# ECR (2-qubit) error/length is used here instead, which does vary meaningfully.
feat_cols = ["frequency_GHz", "T1_us", "T2_us", "readout_error",
             "sx_gate_error", "mean_ecr_error", "mean_ecr_length_ns"]
feat_labels = [r"$f_{01}$", r"$T_1$", r"$T_2$", r"$\epsilon_{RO}$",
               r"$\epsilon_{SX}$", r"$\bar\epsilon_{ECR}$", r"$\bar\tau_{ECR}$"]
X = cal[feat_cols].values
Xs = StandardScaler().fit_transform(X)
n = len(cal)

fig = plt.figure(figsize=(14.5, 11))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

# =========================================================================
# (a) Correlation matrix, real features
# =========================================================================
ax = fig.add_subplot(gs[0, 0])
corr = pd.DataFrame(X, columns=feat_labels).corr()
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
for i in range(len(feat_labels)):
    for j in range(len(feat_labels)):
        v = corr.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=8.2, color="white" if abs(v) > 0.55 else charcoal)
ax.set_xticks(range(len(feat_labels))); ax.set_yticks(range(len(feat_labels)))
ax.set_xticklabels(feat_labels, fontsize=9.5)
ax.set_yticklabels(feat_labels, fontsize=9.5)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Pearson $r$", fontsize=8.5)
ax.set_title(f"(a) Real correlation structure, ibm_sherbrooke\n(N={n} physical qubits)",
             fontsize=9.8, loc="left")

# =========================================================================
# (b) PCA of the real 127-qubit feature space
# =========================================================================
ax = fig.add_subplot(gs[0, 1])
pca = PCA(n_components=2, random_state=0)
Xp = pca.fit_transform(Xs)
sca = ax.scatter(Xp[:, 0], Xp[:, 1], c=cal["T1_us"], cmap="viridis",
                  s=42, edgecolor=charcoal, linewidth=0.35)
cbar = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r"real $T_1$ ($\mu$s)", fontsize=8.5)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)")
ax.set_title("(b) PCA of the real 6-dimensional calibration\nfeature space, all 127 qubits",
             fontsize=9.8, loc="left")

# =========================================================================
# (c) t-SNE + KMeans on real data
# =========================================================================
ax = fig.add_subplot(gs[1, 0])
tsne = TSNE(n_components=2, perplexity=30, random_state=0, init="pca", learning_rate="auto")
Xt = tsne.fit_transform(Xs)
km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(Xs)
cluster_colors = [sbblue, sbred, sbgold, sbgreen]
for k in range(4):
    m = km.labels_ == k
    ax.scatter(Xt[m, 0], Xt[m, 1], s=42, color=cluster_colors[k],
               edgecolor="white", linewidth=0.4, label=f"cluster {k} (n={m.sum()})")
ax.set_xlabel("t-SNE dim. 1"); ax.set_ylabel("t-SNE dim. 2")
ax.set_title("(c) t-SNE embedding of real qubits, K-means-labelled\n"
             "(4 clusters, standardised real calibration features)", fontsize=9.8, loc="left")
ax.legend(frameon=False, fontsize=7.6)

# =========================================================================
# (d) Real Pareto frontier: real mean-ECR error vs real mean-ECR length
# =========================================================================
ax = fig.add_subplot(gs[1, 1])
e = cal["mean_ecr_error"].values * 100
t = cal["mean_ecr_length_ns"].values
order = np.argsort(t)
e_o, t_o = e[order], t[order]
pareto_mask = np.zeros(len(e_o), dtype=bool)
best = np.inf
for i in range(len(e_o)):
    if e_o[i] < best:
        pareto_mask[i] = True
        best = e_o[i]
ax.scatter(t, e, s=32, color=midgray, alpha=0.55, label="all 127 real qubits", zorder=2)
ax.scatter(t_o[pareto_mask], e_o[pareto_mask], s=85, color=sbred,
           edgecolor=charcoal, linewidth=0.8, zorder=4, label="Pareto-optimal")
ax.plot(t_o[pareto_mask], e_o[pareto_mask], color=sbred, lw=1.3, ls="--", zorder=3)
ax.set_xlabel(r"real mean ECR gate duration $\bar\tau_{ECR}$ (ns)")
ax.set_ylabel(r"real mean ECR gate error (%)")
ax.set_yscale("log")
ax.set_title("(d) Real speed-fidelity Pareto frontier,\ntwo-qubit ECR gate (per-qubit neighbour average)",
             fontsize=9.8, loc="left")
ax.legend(frameon=False, fontsize=8)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

fig.suptitle("Figure 4.  Statistical / unsupervised-learning analysis of real ibm_sherbrooke calibration data",
             fontsize=12.3, fontweight="bold", color=charcoal, y=0.996)
fig.text(0.5, 0.003,
         "All panels computed directly from data/calibration_real.csv — the genuine frozen calibration "
         f"snapshot of the real 127-qubit ibm_sherbrooke processor (FakeSherbrooke, qiskit-ibm-runtime). "
         "No synthetic values. See real_data.py for extraction code.",
         ha="center", fontsize=7.3, style="italic", color=midgray)
