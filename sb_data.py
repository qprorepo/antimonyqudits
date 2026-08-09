"""
sb_data.py
----------
Physical constants and energy-level helpers for the ^123Sb (I = 7/2, d = 8)
nuclear-spin qudit, used by fig1_sb_spectroscopy.py.

This module isn't pulled from a live experiment -- it's a compact,
self-contained implementation of the Zeeman + first-order electric
quadrupole Hamiltonian described in the manuscript (Sec. "Nuclear Spin
Physics of 123Sb"), evaluated at the field point actually used for the
spectroscopy figure (B0 = 1.384 T), which is the field reported in the
original coherent-control experiment on this donor:

    Asaad et al., "Coherent electrical control of a single high-spin
    nucleus in silicon", Nature 579, 205-209 (2020).

Hyperfine coupling (A = 101.5 MHz, Sarkar et al., arXiv:1703.04852) sets
the overall energy scale of the donor system but isn't needed for the
bare nuclear spectrum plotted here, so it's kept as a reference constant
only and doesn't enter zeeman_energies() / quadrupole_energies().

Units: all energies are quoted as frequencies (Energy / h), in MHz,
unless noted otherwise.
"""

import numpy as np

# ----------------------------------------------------------------------
# Core spin & field parameters
# ----------------------------------------------------------------------
I_SPIN = 3.5                       # nuclear spin of 123Sb, I = 7/2
DIM = int(2 * I_SPIN + 1)          # qudit dimension, d = 8

B0 = 1.384                         # Tesla -- field used in Asaad et al. (2020)
GAMMA_N = 5.55                     # MHz/T -- 123Sb nuclear gyromagnetic ratio
NU_L = GAMMA_N * B0                # Larmor frequency, MHz (~7.681 MHz)

F_Q = 0.066                        # MHz (66 kHz) -- quadrupole coupling scale
                                    # for this device; sets the (3m^2 - I(I+1))
                                    # correction that resolves the seven
                                    # otherwise-degenerate Zeeman transitions.

A_HYPERFINE = 101.5                # MHz -- donor electron-nuclear hyperfine
                                    # coupling (Sarkar et al.), reference only.

T2_STAR = 0.10                     # s -- electrically driven nuclear T2*

# Eight Zeeman sublevels, m_I = -7/2 ... +7/2, ascending order.
M_LEVELS = np.arange(-I_SPIN, I_SPIN + 1, 1.0)

# Qudit readout SNR advantage over an equal-Hilbert-space multi-qubit
# register, eta_qdt = d(d+1)/6 (derived in the manuscript's readout section).
ETA_QDIT_ADVANTAGE = DIM * (DIM + 1) / 6.0


def zeeman_energies():
    """
    Pure Zeeman energy (divided by h, in MHz) of each of the eight m_I
    sublevels, E_m = -nu_L * m_I. Equally spaced by nu_L.
    """
    return -NU_L * M_LEVELS


def quadrupole_energies():
    """
    Zeeman + first-order quadrupole energy (divided by h, in MHz) of each
    sublevel:

        E_m = -nu_L * m_I + (F_Q / 6) * [3 m_I^2 - I(I+1)]

    The quadratic term is what breaks the Zeeman degeneracy and makes all
    seven single-quantum transitions individually resolvable.
    """
    quad_term = (F_Q / 6.0) * (3 * M_LEVELS**2 - I_SPIN * (I_SPIN + 1))
    return -NU_L * M_LEVELS + quad_term


def transition_frequencies():
    """
    The seven single-quantum (|Delta m_I| = 1) transition frequencies,
    sorted ascending, together with the m_I of the upper and lower level
    of each transition.

    Returns
    -------
    nu_trans : ndarray, shape (7,)
        Transition frequencies in MHz, nu_1 < nu_2 < ... < nu_7.
    m_upper : ndarray, shape (7,)
        m_I of the higher-energy level in each transition.
    m_lower : ndarray, shape (7,)
        m_I of the lower-energy level in each transition.
    """
    E = quadrupole_energies()
    order = np.argsort(E)
    E_sorted = E[order]
    m_sorted = M_LEVELS[order]

    nu_trans = np.diff(E_sorted)
    m_lower = m_sorted[:-1]
    m_upper = m_sorted[1:]
    return nu_trans, m_upper, m_lower
