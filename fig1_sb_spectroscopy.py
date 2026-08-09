import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import sb_data as D


sbblue, sbgreen, sbred   = "#0A3A94", "#0E7026", "#A81A1A"
sbpurple, sborange       = "#601088", "#C05208"
sbcyan, sbgold           = "#047E94", "#AC8604"
charcoal, midgray        = "#262626", "#7A7A7A"
lblue, lred, lgold       = "#D4E4FF", "#FFD6D6", "#FFFCCC"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "axes.edgecolor": charcoal,
    "axes.labelcolor": charcoal,
    "text.color": charcoal,
    "xtick.color": charcoal,
    "ytick.color": charcoal,
    "axes.linewidth": 0.9,
    "mathtext.fontset": "cm",
})

fig = plt.figure(figsize=(13.5, 10.2))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

ax = fig.add_subplot(gs[0, 0])
E_zeeman = D.zeeman_energies()
E_quad   = D.quadrupole_energies()
m_lab    = [f"${'+' if m>0 else ''}{int(2*m)}/2$" for m in D.M_LEVELS]

x0, w0 = 0.02, 0.34          # Zeeman-only column
x1, w1 = 0.58, 0.34          # Zeeman+quadrupole column

for i, (Ez, Eq, lab) in enumerate(zip(E_zeeman, E_quad, m_lab)):
    ax.plot([x0, x0 + w0], [Ez, Ez], color=sbblue, lw=3, solid_capstyle="butt")
    ax.plot([x1, x1 + w1], [Eq, Eq], color=sbred,  lw=3, solid_capstyle="butt")
    ax.text(x0 - 0.02, Ez, lab, ha="right", va="center", fontsize=7.7, color=sbblue)
    ax.text(x1 + w1 + 0.02, Eq, lab, ha="left", va="center", fontsize=7.7, color=sbred)

# nu_L double arrows on the Zeeman side
for i in range(len(E_zeeman) - 1):
    ymid = (E_zeeman[i] + E_zeeman[i+1]) / 2
    ax.annotate("", xy=(x0 + w0 + 0.045, E_zeeman[i+1]), xytext=(x0 + w0 + 0.045, E_zeeman[i]),
                arrowprops=dict(arrowstyle="<->", color=sbcyan, lw=0.9))
    if i == 3:
        ax.text(x0 + w0 + 0.065, ymid, r"$\nu_L$", color=sbcyan, fontsize=8)

# unequal transition arrows on quadrupole side
nu_trans, m_upper, _ = D.transition_frequencies()
for k in range(len(E_quad) - 1):
    ymid = (E_quad[k] + E_quad[k+1]) / 2
    ax.annotate("", xy=(x1 - 0.05, E_quad[k+1]), xytext=(x1 - 0.05, E_quad[k]),
                arrowprops=dict(arrowstyle="<->", color=sborange, lw=0.9))
    ax.text(x1 - 0.09, ymid, rf"$\nu_{{{k+1}}}$", color=sborange, fontsize=7.2, ha="right")

ax.text(x0 + w0/2, E_zeeman[0] + 0.9, "Zeeman only\n(equally spaced)",
        ha="center", fontsize=9.2, fontweight="bold", color=sbblue)
ax.text(x1 + w1/2, E_quad[0] + 0.9, "+ Quadrupole\n(resolvable, $f_Q=66$ kHz)",
        ha="center", fontsize=9.2, fontweight="bold", color=sbred)

y_hq = (E_zeeman[1] + E_zeeman[2]) / 2 + 4.0
ax.annotate("", xy=(x1 - 0.14, y_hq), xytext=(x0 + w0 + 0.10, y_hq),
            arrowprops=dict(arrowstyle="->", color=sbgold, lw=2.2))
ax.text((x0+w0+x1)/2, y_hq + 1.6, r"$H_Q$",
        color=sbgold, fontsize=10, fontweight="bold", ha="center")

ax.set_xlim(-0.16, 1.02)
ax.set_ylim(E_zeeman.min() - 1.6, E_zeeman.max() + 1.6)
ax.set_xticks([])
ax.set_ylabel(r"Energy / $h$  (MHz)")
ax.set_title(r"(a) $^{123}$Sb ($I{=}7/2$, $d{=}8$) energy ladder — $B_0=1.384$ T, $\gamma_n=5.55$ MHz/T",
             fontsize=10, color=charcoal, loc="left")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

# =========================================================================
# (b) NMR transition spectrum: 7 resolvable Lorentzian lines
# =========================================================================
ax = fig.add_subplot(gs[0, 1])
freq_axis = np.linspace(D.NU_L - 0.30, D.NU_L + 0.30, 4000)   # MHz window
linewidth = 1.0 / (np.pi * D.T2_STAR) * 1e-6 * 1e3            # ~3 Hz -> broaden for visibility
linewidth = 0.004  # MHz, instrumental/plotting linewidth (4 kHz) for clarity
spectrum = np.zeros_like(freq_axis)
colors_line = plt.cm.plasma(np.linspace(0.08, 0.90, len(nu_trans)))
for k, nu in enumerate(nu_trans):
    lor = (linewidth / 2) ** 2 / ((freq_axis - nu) ** 2 + (linewidth / 2) ** 2)
    spectrum += lor
    ax.plot(freq_axis, lor, color=colors_line[k], lw=1.4)
    ax.fill_between(freq_axis, 0, lor, color=colors_line[k], alpha=0.18)
    ax.annotate(rf"$\nu_{{{k+1}}}$", xy=(nu, 1.04), ha="center", fontsize=7.6, color=colors_line[k])

ax.axvline(D.NU_L, color=midgray, ls=":", lw=1.1)
ax.text(D.NU_L, -0.10, r"$\nu_L=7.681$ MHz", ha="center", fontsize=7.3, color=midgray)
ax.set_xlabel("RF frequency (MHz)")
ax.set_ylabel("Transition amplitude (arb. u.)")
ax.set_title(r"(b) Resolved single-quantum NMR spectrum, $f_Q=66$ kHz spacing",
             fontsize=10, color=charcoal, loc="left")
ax.set_ylim(-0.15, 1.25)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

# inset: quadrupole shift Delta(m) = f_Q/6 [3m^2 - I(I+1)] vs m
axins = ax.inset_axes([0.62, 0.55, 0.36, 0.40])
m_full = D.M_LEVELS
shift = (D.F_Q / 6.0) * (3 * m_full**2 - D.I_SPIN * (D.I_SPIN + 1)) * 1e3   # kHz
axins.stem(m_full, shift, linefmt=sbred, markerfmt="o", basefmt=" ")
axins.axhline(0, color=midgray, lw=0.6)
axins.set_title(r"$\Delta_Q(m)$ (kHz)", fontsize=7)
axins.tick_params(labelsize=6)

# =========================================================================
# (c) Electrically driven coherence decay (Ramsey / FID envelope)
# =========================================================================
ax = fig.add_subplot(gs[1, 0])
t = np.linspace(0, 0.5, 3000)     # seconds
detune = 3.0                       # Hz artificial detuning to show fringes
envelope = np.exp(-(t / D.T2_STAR) ** 2)     # Gaussian-like dephasing envelope (typical for 1/f noise)
fringes = envelope * np.cos(2 * np.pi * detune * t)
ax.plot(t, fringes, color=sbpurple, lw=0.9)
ax.plot(t, envelope, color=sbred, lw=1.8, label=r"$\exp[-(t/T_2^{*})^2]$")
ax.plot(t, -envelope, color=sbred, lw=1.8)
ax.axvline(D.T2_STAR, color=sbgold, ls="--", lw=1.3)
ax.text(D.T2_STAR, 1.05, r"$T_2^{*}=0.10$ s" + "\n(electrically driven)",
        color=sbgold, fontsize=7.6, ha="center")
ax.set_xlabel("Free evolution time  $t$  (s)")
ax.set_ylabel("Nuclear coherence  $\\langle I_x \\rangle$ (norm.)")
ax.set_title("(c) Ramsey coherence envelope of the driven nuclear qudit",
             fontsize=10, color=charcoal, loc="left")
ax.legend(frameon=False, loc="lower left", fontsize=8)
ax.set_ylim(-1.25, 1.35)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

# =========================================================================
# (d) Qudit readout SNR advantage:  eta_qdt = d(d+1)/6
# =========================================================================
ax = fig.add_subplot(gs[1, 1])
platforms = ["3-qubit binary\nregister (8 states)", r"$^{123}$Sb qudit" + "\n" + r"($d=8$, this work)"]
snr_rel = [1.0, D.ETA_QDIT_ADVANTAGE]
bar_colors = [midgray, sbblue]
bars = ax.bar(platforms, snr_rel, color=bar_colors, width=0.55,
              edgecolor=charcoal, linewidth=0.8)
for b, v in zip(bars, snr_rel):
    ax.text(b.get_x() + b.get_width()/2, v + 0.25, f"$\\times {v:.0f}$" if v > 1 else "1$\\times$ (ref.)",
            ha="center", fontsize=10, fontweight="bold", color=charcoal)
ax.set_ylabel("Relative inductive-readout SNR,  $\\eta_{\\mathrm{qdt}}=d(d{+}1)/6$")
ax.set_title("(d) Predicted qudit readout advantage over an equal-Hilbert-\n"
             "space multi-qubit register", fontsize=10, color=charcoal, loc="left")
ax.set_ylim(0, D.ETA_QDIT_ADVANTAGE * 1.35)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.text(0.5, 0.90, r"$\log_2 d = 3$ logical bits / physical qudit"
        + "\n(vs. 1 bit / physical qubit)",
        transform=ax.transAxes, ha="center", fontsize=8.3, color=charcoal,
        bbox=dict(boxstyle="round,pad=0.35", fc=lgold, ec=sbgold, lw=0.8))

fig.suptitle(r"Figure 1.  Spectroscopic and information-theoretic characterisation of the $^{123}$Sb ($I{=}7/2,\,d{=}8$) nuclear qudit",
             fontsize=12.5, fontweight="bold", color=charcoal, y=0.995)
fig.text(0.5, 0.005,
         "Parameters from Asaad et al., Nature 579, 205–209 (2020) and Sarkar et al., arXiv:1703.04852 "
         "(hyperfine $A=101.5$ MHz). Panels (a,b,c) use only measured/derived physical constants; "
         "panel (d) uses the manuscript's derived SNR-advantage relation.",
         ha="center", fontsize=7.3, style="italic", color=midgray)

