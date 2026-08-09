# Antimony Qudits for High-Dimensional Error-Corrected Quantum Storage

Code and data supporting a manuscript on using the nuclear spin of
$^{123}$Sb ($I = 7/2$, natural dimension $d = 8$) as a physical qudit for
quantum storage. The paper works out the full spin Hamiltonian, a
$\mathbb{Z}_8$ Heisenberg-Weyl stabiliser formalism, a $[[3,1,2]]_8$
error-correcting code, an $SU(8)$ gate decomposition, and a classical
inductive-readout theory with a predicted $\sim\!12\times$ SNR advantage
over an equal-Hilbert-space multi-qubit register. This repo holds the
numerical pipeline behind the paper's four main figures, plus the
manuscript source itself.

The manuscript is currently anonymised for double-blind review, so this
repo is too -- no author names in the code, `LICENSE`, or `CITATION.cff`
until that changes.

## What's real here, and what isn't

Worth being upfront about, since the two get mixed together across the
figures:

- **Figures 2, 3, and 4** run on genuine hardware data: the frozen
  127-qubit calibration snapshot of IBM's `ibm_sherbrooke` processor
  (via `FakeSherbrooke` in `qiskit-ibm-runtime`) and real circuits from
  [QASMBench](https://github.com/pnnl/QASMBench), transpiled onto that
  device's actual coupling map. None of the numbers in those three
  figures are made up -- they're either read straight off the snapshot
  or computed deterministically from it.
- **Figure 1** is a *model*, not a measurement. It uses real physical
  constants from the literature (the field strength from Asaad et al.,
  *Nature* 579, 205-209 (2020); the hyperfine coupling from Sarkar et
  al., arXiv:1703.04852) plugged into a Zeeman + first-order quadrupole
  Hamiltonian, to show what the spectrum *should* look like. It isn't a
  measured spectrum, and the manuscript doesn't claim it is.
- The $^{123}$Sb qudit platform itself -- the Hamiltonian, the stabiliser
  code, the gate sequences -- is a **theoretical proposal**. The
  ibm_sherbrooke data in Figures 2-4 is there as a real-world benchmarking
  and comparison reference (transpilation overhead, circuit structure,
  device-level noise), not because the proposed qudit hardware has been
  built. Don't let "real" in half these filenames read as "we fabricated
  the device" -- it means "these particular numbers came from a real
  chip or a real benchmark suite," which is a narrower and more honest
  claim.

One more disclosure: `figures/sb_data.py` -- the module that
`fig1_sb_spectroscopy.py` imports for the energy-level and transition-
frequency helpers -- wasn't part of the original file set. It's
reconstructed here from the physics described in the manuscript
(Zeeman term, first-order quadrupole correction, the $\eta_{\mathrm{qdt}}
= d(d+1)/6$ readout-advantage relation) and from the constants cited in
`fig1_sb_spectroscopy.py`'s own footer text. It's been checked against
the manuscript's formulas and runs end-to-end, but since it wasn't in
the uploaded set, it's worth a second look before you treat it as
authoritative.

## Repository layout

```
antimonyqudits/
├── data/                       real calibration + benchmark data (CSV)
│   ├── calibration_real.csv        127-qubit T1/T2/readout/gate-error snapshot
│   ├── coupling_real.csv           real ECR-coupled qubit pairs + errors
│   ├── circuit_benchmarks_real.csv 123 real QASMBench circuits, parsed
│   ├── transpile_overhead_real.csv logical vs. physical depth after routing
│   ├── supermarq_real.csv          SupermarQ capability features, 6 circuits
│   └── README.md                   data dictionary
├── figures/                    figure-generation scripts (run from repo root)
│   ├── sb_data.py                   Sb-123 physics helpers (see note above)
│   ├── fig1_sb_spectroscopy.py      energy ladder, NMR spectrum, SNR advantage
│   ├── fig2_hardware_calibration_real.py   T1/T2, gate-error matrix, readout
│   ├── fig3_circuit_benchmarks_real.py     depth/width, transpilation, SupermarQ
│   └── fig4_statistical_analysis_real.py   correlation, PCA, t-SNE, Pareto front
├── scripts/
│   └── real_data.py             extracts everything in data/ from source
├── manuscript/
│   └── main.tex                 manuscript source
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

## Setup

```bash
git clone https://github.com/qprorepo/antimonyqudits.git
cd antimonyqudits
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Regenerating the figures

Each figure script reads from `data/*.csv` with a relative path, so run
them from the repository root, not from inside `figures/`:

```bash
python figures/fig1_sb_spectroscopy.py
python figures/fig2_hardware_calibration_real.py
python figures/fig3_circuit_benchmarks_real.py
python figures/fig4_statistical_analysis_real.py
```

Each one writes a PDF and PNG to `output/` (created automatically,
gitignored). `output/` isn't tracked, so `git status` staying clean
after a run is expected, not a bug.

## Regenerating the data

`scripts/real_data.py` is what produced every CSV in `data/`. The
calibration and coupling files only need `qiskit-ibm-runtime` and take
a few seconds:

```bash
python scripts/real_data.py
```

The circuit-benchmark, transpilation, and SupermarQ extraction steps
additionally need a local clone of
[QASMBench](https://github.com/pnnl/QASMBench), which isn't vendored in
this repo (it's a large third-party suite with its own license). Point
the script at your clone with an environment variable:

```bash
git clone https://github.com/pnnl/QASMBench.git
export QASMBENCH_DIR=$(pwd)/QASMBench   # Windows: set QASMBENCH_DIR=...
python scripts/real_data.py
```

## Building the manuscript

`manuscript/main.tex` compiles with a standard `pdflatex` + `biber`
toolchain (it uses `biblatex`, `glossaries`, `tikz`/`pgfplots`, and
`quantikz`). This snapshot includes the `.tex` source only -- the
bibliography (`bibliography.bib`), glossary definitions (`glossary.tex`),
and the referenced figure graphics (`fig_CSUM_encoder`,
`fig_quantum_memory`, `fig_s23_qudit_array`, and the main energy-level
diagram) aren't part of this code/data upload and will need to be added
to `manuscript/` before it compiles cleanly.

## License

Code and data in this repository are MIT-licensed (see `LICENSE`) --
that covers `figures/`, `scripts/`, and `data/`. The manuscript text in
`manuscript/` is not covered by that license; treat it as all-rights-
reserved while it's under review, the usual arrangement for a paper
that hasn't been published yet.

## Citing this work

See `CITATION.cff`. Author metadata is a placeholder until the
manuscript is de-anonymised.
