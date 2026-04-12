"""
find_best_table_1d.py

Python port of Tucker McClure's find_best_table_1d (MATLAB, 2013).

Given a dense 1-D signal, finds the optimal placement of n_x breakpoints
so that a piecewise-linear reconstruction minimises mean-squared error.

The optimisation state is the *spacing* between interior breakpoints
(length n_x - 2).  The first and last breakpoints are always pinned to
x_0[0] and x_0[-1].

Dependencies: numpy, scipy, matplotlib
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def spacing_to_x(spacing, x0_start, x0_end):
    """Convert interior spacings → full breakpoint array (length n_x)."""
    interior = np.cumsum(spacing) + x0_start
    return np.concatenate([[x0_start], interior, [x0_end]])


def evaluate_spacing(spacing, x_0, z_0, x0_start, x0_end):
    """
    Reconstruct the table from spacing, interpolate back to x_0, return MSE.

    Returns (mse, max_error, z_i)  — scipy minimize uses only the scalar mse.
    """
    x_i = spacing_to_x(spacing, x0_start, x0_end)
    z_i = np.interp(x_i, x_0, z_0)                      # table values
    z_reconstructed = np.interp(x_0, x_i, z_i)          # back to original grid
    se = (z_0 - z_reconstructed) ** 2
    mse = np.mean(se)
    max_error = np.sqrt(np.max(se))
    return mse, max_error, z_i


def find_best_table_1d(x_0, z_0, n_x):
    """
    Find the best n_x-point table for the signal (x_0, z_0).

    Parameters
    ----------
    x_0 : 1-D array  – independent variable (must be sorted)
    z_0 : 1-D array  – dependent variable sampled at x_0
    n_x : int        – desired number of breakpoints in the output table

    Returns
    -------
    x_f       : optimised breakpoints (length n_x)
    z_f       : table values at x_f
    mse_f     : mean-squared error of final table
    max_error : maximum point-wise error
    """
    x0_start = x_0[0]
    x0_end   = x_0[-1]
    span     = x0_end - x0_start
    n_free   = n_x - 2          # number of free (interior) spacings

    if n_free <= 0:
        # Degenerate: only 2 breakpoints → pin endpoints
        x_f = np.array([x0_start, x0_end])
        z_f = np.interp(x_f, x_0, z_0)
        mse_f, max_error, _ = evaluate_spacing(np.array([]), x_0, z_0, x0_start, x0_end)
        return x_f, z_f, mse_f, max_error

    # Uniform initial spacing
    dx_uniform = span / (n_x - 1)
    spacing_0  = dx_uniform * np.ones(n_free)

    # Lower bound: 0.1 % of uniform spacing (prevents degenerate collapse)
    lb = 1e-3 * dx_uniform * np.ones(n_free)

    # Inequality constraint: sum(spacing) <= span - lb[0]
    # (mirrors the MATLAB  A*x <= b  with A = ones(...), b = span - lb)
    # scipy form: fun(x) >= 0  →  (span - lb[0]) - sum(spacing) >= 0
    constraints = {
        'type': 'ineq',
        'fun': lambda s: (span - lb[0]) - np.sum(s)
    }

    bounds = [(lb[i], None) for i in range(n_free)]

    result = minimize(
        fun     = lambda s: evaluate_spacing(s, x_0, z_0, x0_start, x0_end)[0],
        x0      = spacing_0,
        method  = 'SLSQP',          # closest scipy equivalent to MATLAB active-set
        bounds  = bounds,
        constraints = constraints,
        options = {'maxiter': 10000, 'ftol': 1e-9, 'disp': False}
    )

    spacing_f = result.x
    x_f = spacing_to_x(spacing_f, x0_start, x0_end)
    mse_f, max_error, z_f = evaluate_spacing(spacing_f, x_0, z_0, x0_start, x0_end)

    return x_f, z_f, mse_f, max_error


# ---------------------------------------------------------------------------
# Demo: two summed cosines as the signal
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(42)

    # Dense original signal: sum of two cosines with random freq / phase / amp
    x_0 = np.linspace(0, 10, 1000)

    freqs  = rng.uniform(0.3, 2.0, size=2)
    phases = rng.uniform(0,   2*np.pi, size=2)
    amps   = rng.uniform(0.5, 2.0,    size=2)

    z_0 = (amps[0] * np.cos(2 * np.pi * freqs[0] * x_0 + phases[0]) +
           amps[1] * np.cos(2 * np.pi * freqs[1] * x_0 + phases[1]))

    print(f"Signal: {amps[0]:.2f}·cos(2π·{freqs[0]:.2f}·x + {phases[0]:.2f})"
          f"  +  {amps[1]:.2f}·cos(2π·{freqs[1]:.2f}·x + {phases[1]:.2f})")
    print(f"Original table size: {len(x_0)} points\n")

    # ---- Fit at several table sizes ----------------------------------------
    sizes = [6, 10, 20]
    results = {}

    for n in sizes:
        x_f, z_f, mse_f, me_f = find_best_table_1d(x_0, z_0, n)
        results[n] = (x_f, z_f, mse_f, me_f)
        print(f"n={n:>3}  MSE={mse_f:.6f}  MaxErr={me_f:.6f}"
              f"  breakpoints={np.round(x_f, 3)}")

    # ---- Plot ----------------------------------------------------------------
    fig, axes = plt.subplots(len(sizes), 2, figsize=(13, 4 * len(sizes)))
    fig.suptitle("Guided breakpoint optimisation  –  two-cosine signal", fontsize=13)

    for row, n in enumerate(sizes):
        x_f, z_f, mse_f, me_f = results[n]

        # Reconstruct on the original grid for the residual plot
        z_reconstructed = np.interp(x_0, x_f, z_f)
        residual = z_0 - z_reconstructed

        ax_fit = axes[row, 0]
        ax_fit.plot(x_0, z_0, 'k', lw=1.2, label='Original (1000 pts)')
        ax_fit.plot(x_f, z_f, 'o-', color='tab:blue', lw=1.8,
                    ms=6, label=f'Optimised ({n} pts)')
        ax_fit.vlines(x_f, *ax_fit.get_ylim(), colors='tab:blue',
                      alpha=0.15, lw=0.8)
        ax_fit.set_title(f'n={n}  |  MSE={mse_f:.4f}  MaxErr={me_f:.4f}')
        ax_fit.set_xlabel('x')
        ax_fit.set_ylabel('z')
        ax_fit.legend(fontsize=8)

        ax_res = axes[row, 1]
        ax_res.plot(x_0, residual, color='tab:red', lw=0.9)
        ax_res.axhline(0, color='k', lw=0.7, ls='--')
        ax_res.set_title(f'Residual  (n={n})')
        ax_res.set_xlabel('x')
        ax_res.set_ylabel('error')

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/find_best_table_demo.png', dpi=130)
    print("\nPlot saved to find_best_table_demo.png")
    plt.show()


if __name__ == '__main__':
    main()
