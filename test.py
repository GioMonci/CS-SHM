"""
Compressive Sensing (1D Time Series)
Goal:
  Reconstruct a length-N time series x[t] from only M randomly sampled points.
Model:
  - True signal:        x ∈ R^N
  - Measurements:       y = x[measured_indices] ∈ R^M
  - Dictionary (basis): D ∈ R^(M×P) where each column is a sine/cosine atom evaluated at the measured time indices.
  - Sparse recovery:    find sparse coefficients a such that y ≈ D a (we use Orthogonal Matching Pursuit, OMP)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit

# 1) Important Values (change these to test different settings)

SIGNAL_LENGTH_N = 512          # Total number of time samples in the full signal x[t]
NUM_MEASUREMENTS_M = 180       # How many samples we "observe" (compression amount)

# True signal content (two sinusoids)
TONE1_FREQ_CYCLES = 7          # frequency as "cycles across N samples"
TONE2_FREQ_CYCLES = 23
TONE2_AMPLITUDE = 0.6

# Recovery configuration
MAX_FREQUENCY = SIGNAL_LENGTH_N // 2   # candidate frequency grid (0..N/2)
SPARSITY_K = 8                         # expected number of nonzero coefficients

RANDOM_SEED = 0


# 2) Build the true signal (ground truth)

rng = np.random.default_rng(RANDOM_SEED)

time_index = np.arange(SIGNAL_LENGTH_N)

x_true = (np.sin(2 * np.pi * TONE1_FREQ_CYCLES * time_index / SIGNAL_LENGTH_N)
    + TONE2_AMPLITUDE * np.sin(2 * np.pi * TONE2_FREQ_CYCLES * time_index / SIGNAL_LENGTH_N)
          )

# 3) Compressive measurements: random time sampling

measured_indices = np.sort(rng.choice(SIGNAL_LENGTH_N, size=NUM_MEASUREMENTS_M, replace=False))

y_measured = x_true[measured_indices]

# 4) Build a Fourier-style dictionary at the measured indices
#    We use real atoms: cos(2π f t/N) and sin(2π f t/N)

def build_fourier_dictionary(sample_times: np.ndarray, N: int, f_max: int) -> np.ndarray:
    """
    Create a real-valued Fourier dictionary evaluated at specific time indices.

    Columns:
      - cos(2π f t/N) for f = 0..f_max
      - sin(2π f t/N) for f = 1..f_max   (skip f=0 sine because it's always 0)

    Returns:
      D: shape (len(sample_times), (f_max+1) + f_max)
    """
    # Cosine atoms (includes DC component f=0)
    cosine_atoms = [np.cos(2 * np.pi * f * sample_times / N) for f in range(f_max + 1)]
    # Sine atoms (start at f=1)
    sine_atoms = [np.sin(2 * np.pi * f * sample_times / N) for f in range(1, f_max + 1)]

    D = np.column_stack(cosine_atoms + sine_atoms)

    # Normalize columns so OMP isn't biased toward higher-energy atoms
    column_norms = np.linalg.norm(D, axis=0, keepdims=True) + 1e-12
    D = D / column_norms

    return D


D_measured = build_fourier_dictionary(
    sample_times=measured_indices,
    N=SIGNAL_LENGTH_N,
    f_max=MAX_FREQUENCY
)


# 5) Sparse recovery with OMP: y ≈ D a (with a sparse)

omp = OrthogonalMatchingPursuit(
    n_nonzero_coefs=SPARSITY_K,
    fit_intercept=False
)

omp.fit(D_measured, y_measured)
a_hat = omp.coef_


# 6) Reconstruct the full signal on all time indices

D_full = build_fourier_dictionary(
    sample_times=time_index,
    N=SIGNAL_LENGTH_N,
    f_max=MAX_FREQUENCY
)

x_recon = D_full @ a_hat

# 7) Quick evaluation + visualization

relative_l2_error = np.linalg.norm(x_true - x_recon) / (np.linalg.norm(x_true) + 1e-12)
print(f"Relative L2 reconstruction error: {relative_l2_error:.6f}")

plt.figure(figsize=(12, 4))
plt.plot(time_index, x_true, label="Original signal x(t)")
plt.plot(time_index, x_recon, "--", label="Reconstructed signal x̂(t)")
plt.scatter(measured_indices, y_measured, s=12, label="Measurements (random samples)", zorder=3)
plt.title(
    "Compressive Sensing Demo: Random Time Sampling + Fourier Dictionary + OMP\n"
    f"N={SIGNAL_LENGTH_N}, M={NUM_MEASUREMENTS_M}, k={SPARSITY_K}"
)
plt.xlabel("Time sample index t")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.show()


# Optional: “what did we recover?” — peek at the sparse coeffs

plt.figure(figsize=(12, 3))
plt.stem(np.abs(a_hat), markerfmt=" ", basefmt=" ")
plt.title("Recovered coefficient magnitudes |â| (sparse solution)")
plt.xlabel("Dictionary atom index (cos first, then sin)")
plt.ylabel("|coefficient|")
plt.tight_layout()
plt.show()
