import warnings
warnings.filterwarnings("ignore")

import glob
import os
import numpy as np
import pandas as pd
from qiskit import qasm2, transpile
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

# Path to a local clone of QASMBench (https://github.com/pnnl/QASMBench).
# Override with an environment variable so this runs on any machine, e.g.:
#   export QASMBENCH_DIR=/path/to/QASMBench   (Linux/macOS)
#   set QASMBENCH_DIR=C:\path\to\QASMBench    (Windows)
QASMBENCH_DIR = os.environ.get("QASMBENCH_DIR", "data/QASMBench")

# ----------------------------------------------------------------------
# 1) REAL 127-qubit calibration snapshot (ibm_sherbrooke, frozen in-package)
# ----------------------------------------------------------------------
def extract_real_calibration(path="data/calibration_real.csv",
                              edge_path="data/coupling_real.csv"):
    backend = FakeSherbrooke()
    props = backend.properties()
    n = backend.num_qubits

    rows = []
    for q in range(n):
        t1 = props.t1(q) * 1e6 if props.t1(q) else np.nan          # -> us
        t2 = props.t2(q) * 1e6 if props.t2(q) else np.nan          # -> us
        freq = props.frequency(q) / 1e9 if props.frequency(q) else np.nan  # -> GHz
        ro = props.readout_error(q)
        try:
            sx_err = props.gate_error("sx", q)
            sx_len = props.gate_length("sx", q) * 1e9   # ns
        except Exception:
            sx_err, sx_len = np.nan, np.nan
        rows.append(dict(qubit=q, T1_us=t1, T2_us=t2, frequency_GHz=freq,
                          readout_error=ro, sx_gate_error=sx_err,
                          sx_gate_length_ns=sx_len))
    df = pd.DataFrame(rows)

    # real ECR (2-qubit) gate errors on the real physical coupling map
    edges = list(backend.coupling_map.get_edges())
    seen = set()
    erows = []
    for (a, b) in edges:
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        try:
            err = props.gate_error("ecr", [a, b])
            length = props.gate_length("ecr", [a, b]) * 1e9  # ns
        except Exception:
            try:
                err = props.gate_error("ecr", [b, a])
                length = props.gate_length("ecr", [b, a]) * 1e9
            except Exception:
                continue
        erows.append(dict(qubit_a=key[0], qubit_b=key[1],
                           ecr_error=err, ecr_length_ns=length))
    edf = pd.DataFrame(erows)

    # per-qubit average real ECR (2-qubit) error/length over its real neighbours
    # (note: sx_gate_length_ns is a real but *device-wide fixed* pulse duration on
    # this backend -- zero variance -- so the ECR aggregate is used instead wherever
    # a varying two-qubit-linked feature is needed downstream)
    ecr_by_qubit = {}
    for _, r in edf.iterrows():
        for q in (int(r.qubit_a), int(r.qubit_b)):
            ecr_by_qubit.setdefault(q, []).append((r.ecr_error, r.ecr_length_ns))
    df["mean_ecr_error"] = df["qubit"].map(
        lambda q: np.mean([e for e, _ in ecr_by_qubit.get(q, [(np.nan, np.nan)])]))
    df["mean_ecr_length_ns"] = df["qubit"].map(
        lambda q: np.mean([l for _, l in ecr_by_qubit.get(q, [(np.nan, np.nan)])]))

    df.to_csv(path, index=False)
    edf.to_csv(edge_path, index=False)
    return df, edf, backend


# ----------------------------------------------------------------------
# 2) REAL QASMBench circuit metrics (parsed from the real .qasm files)
# ----------------------------------------------------------------------
FAMILY_MAP = {
    "adder": "arithmetic", "qft": "algebraic", "grover": "search",
    "bv": "algebraic", "ising": "simulation", "qpe": "algebraic",
    "vqe": "variational", "ghz": "entanglement", "qaoa": "variational",
    "dnn": "ML", "hhl": "linear-algebra", "qec": "QEC",
    "error_correctiond3": "QEC", "teleportation": "entanglement",
    "toffoli": "arithmetic", "basis_change": "algebraic",
    "iswap": "entanglement", "cat_state": "entanglement",
    "deutsch": "algebraic", "fredkin": "arithmetic", "inverseqft": "algebraic",
    "linearsolver": "linear-algebra", "pea": "algebraic", "qrng": "sampling",
    "shor": "algebraic", "simon": "algebraic", "swap_test": "entanglement",
    "variational": "variational", "wstate": "entanglement",
    "bwt": "arithmetic", "multiplier": "arithmetic", "multiply": "arithmetic",
    "sat": "search", "qsvm": "ML", "vqc": "ML",
}

def classify(name):
    base = name.split("_n")[0]
    for key, fam in FAMILY_MAP.items():
        if base.startswith(key):
            return fam
    return "other"

def try_load(path):
    for kwargs in ({}, {"custom_instructions": qasm2.LEGACY_CUSTOM_INSTRUCTIONS}):
        try:
            return qasm2.load(path, **kwargs)
        except Exception:
            continue
    return None

def extract_real_benchmarks(path="data/circuit_benchmarks_real.csv"):
    rows = []
    for cat in ("small", "medium", "large"):
        for f in glob.glob(f"{QASMBENCH_DIR}/{cat}/**/*.qasm", recursive=True):
            if "transpiled" in f:
                continue
            name = os.path.splitext(os.path.basename(f))[0]
            qc = try_load(f)
            if qc is None:
                continue
            ops = qc.count_ops()
            twoq = sum(v for k, v in ops.items()
                       if k in ("cx", "cz", "cu1", "cp", "swap", "iswap",
                                 "rxx", "ryy", "rzz", "ecr", "crx", "cry", "crz"))
            oneq = sum(v for k, v in ops.items()
                       if k not in ("barrier", "measure", "reset") ) - twoq
            rows.append(dict(
                circuit=name, family=classify(name), category=cat,
                n_qubits=qc.num_qubits, depth=qc.depth(),
                gate_count=sum(v for k, v in ops.items() if k not in ("barrier", "measure")),
                one_qubit_gates=max(oneq, 0), two_qubit_gates=twoq,
                source_file=os.path.relpath(f, QASMBENCH_DIR),
            ))
    df = pd.DataFrame(rows).drop_duplicates(subset="circuit")
    df.to_csv(path, index=False)
    return df


# ----------------------------------------------------------------------
# 3) REAL transpilation overhead onto the REAL FakeSherbrooke device
# ----------------------------------------------------------------------
def extract_real_transpilation(bench_df, backend, path="data/transpile_overhead_real.csv",
                                max_qubits=27, max_gates=4000, per_family=1):
    rows = []
    tractable = bench_df[(bench_df.n_qubits <= max_qubits) & (bench_df.gate_count <= max_gates)]
    picked = (tractable.sort_values("n_qubits", ascending=False)
              .groupby("family").head(per_family))
    for _, r in picked.iterrows():
        f = os.path.join(QASMBENCH_DIR, r["source_file"])
        qc = try_load(f)
        if qc is None:
            continue
        try:
            qct = transpile(qc, backend=backend, optimization_level=1, seed_transpiler=42)
        except Exception as e:
            continue
        rows.append(dict(circuit=r["circuit"], family=r["family"], n_qubits=r["n_qubits"],
                          logical_depth=qc.depth(), physical_depth=qct.depth(),
                          logical_2q=r["two_qubit_gates"],
                          physical_2q=sum(v for k, v in qct.count_ops().items()
                                            if k in ("ecr", "cx"))))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


# ----------------------------------------------------------------------
# 4) REAL SupermarQ capability features, computed on real QASMBench circuits
# ----------------------------------------------------------------------
def extract_real_supermarq(bench_df, path="data/supermarq_real.csv",
                            max_qubits=12, n_pick=6):
    from supermarq import features as smf
    small = bench_df[(bench_df.n_qubits <= max_qubits) & (bench_df.n_qubits >= 4)]
    # pick the LARGEST tractable circuit per family for a richer feature profile
    idx = small.groupby("family")["n_qubits"].idxmax()
    per_family_largest = small.loc[idx].sort_values("n_qubits", ascending=False)
    picked = per_family_largest.head(n_pick)
    rows = []
    for _, r in picked.iterrows():
        f = os.path.join(QASMBENCH_DIR, r["source_file"])
        qc = try_load(f)
        if qc is None:
            continue
        qc_nomeas = qc.remove_final_measurements(inplace=False) or qc
        try:
            comm = smf.compute_communication_with_qiskit(qc_nomeas)
            live = smf.compute_liveness_with_qiskit(qc_nomeas)
            para = smf.compute_parallelism_with_qiskit(qc_nomeas)
            meas = smf.compute_measurement_with_qiskit(qc)
            ent = smf.compute_entanglement_with_qiskit(qc_nomeas)
            cdep = smf.compute_depth_with_qiskit(qc_nomeas)
        except Exception as e:
            continue
        rows.append(dict(circuit=r["circuit"], family=r["family"], n_qubits=r["n_qubits"],
                          communication=comm, liveness=live, parallelism=para,
                          measurement=meas, entanglement=ent, critical_depth=cdep))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    cal, edges, backend = extract_real_calibration()
    print(f"[1] Real calibration: {len(cal)} qubits, {len(edges)} coupled pairs "
          f"(source: FakeSherbrooke / ibm_sherbrooke snapshot)")

    bench = extract_real_benchmarks()
    print(f"[2] Real QASMBench circuits parsed: {len(bench)}")

    tp = extract_real_transpilation(bench, backend)
    print(f"[3] Real transpilation runs (onto real coupling map): {len(tp)}")

    smq = extract_real_supermarq(bench)
    print(f"[4] Real SupermarQ feature vectors computed: {len(smq)}")
