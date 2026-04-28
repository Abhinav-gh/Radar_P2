import numpy as np
import matplotlib.pyplot as plt

# Keep array printing compact but readable for educational console output.
np.set_printoptions(precision=4, suppress=True)

# Make the demo reproducible so the noisy reconstruction is stable in a presentation.
np.random.seed(7)


def print_step_header(step_id: int, title: str) -> None:
    print("\n" + "=" * 72)
    print(f"Step {step_id}: {title}")
    print("=" * 72)


# -----------------------------------------------------------------------------
# 1) SETUP
# -----------------------------------------------------------------------------
c = 3e8
f = np.linspace(1e9, 5e9, 20)  # real systems often have limited frequency samples
k = 2 * np.pi * f / c

N = 50
x = np.linspace(0, 0.45, N)  # 1D voxel positions (m)

antenna_positions = [ 0, 0.45]
num_antennas = len(antenna_positions)


# Create a clustered, continuous object profile instead of isolated spikes.
chi_true = np.zeros(N)

for i in range(20, 30):
    chi_true[i] = np.exp(-0.5 * ((i-25)/10)**2)

# for i in range(12, 16):
#     chi_true[i] += 0.6 * np.exp(-0.5 * ((i - 14) / 1.5) ** 2)

object_indices = np.where(chi_true > 0)[0]

print_step_header(1, "Setup")
print("We discretize the scene into voxels and assign object reflectivity chi.")
print("This represents a continuous object instead of isolated scatterers.")
print("Multiple antennas provide more measurement diversity.")
print(f"Number of voxels: {N}")
print("Voxel positions x (m):")
print(np.array2string(x, separator=", "))
print(f"Object voxel indices: {object_indices.tolist()}")
print(f"Antenna position(s): {antenna_positions}")
print(f"Number of antennas: {num_antennas}")

plt.figure(figsize=(9, 3.5))
plt.plot(x, np.zeros_like(x), "bo", label="Voxel grid")
plt.plot(x[object_indices], np.zeros_like(object_indices, dtype=float), "r^", markersize=8, label="Object-support voxels")
for antenna_x in antenna_positions:
    plt.plot(antenna_x, 0.0, "ks", markersize=8)
    plt.text(antenna_x, 0.035, "antenna", ha="center", fontsize=8)
plt.yticks([])
plt.xlabel("Position x (m)")
plt.title("1D Imaging Setup: Voxel Grid and Object")
plt.grid(alpha=0.35)
plt.legend()
plt.tight_layout()
plt.show()

# Plot 1: continuous object profile for presentation clarity.
plt.figure(figsize=(9, 3.5))
plt.plot(x, chi_true, "b-o", linewidth=2, markersize=4)
plt.xlabel("Position (m)")
plt.ylabel("True Reflectivity")
plt.title("Continuous Object Shape (Ground Truth)")
plt.grid(alpha=0.35)
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# 2) FORWARD PHYSICS
# -----------------------------------------------------------------------------
print_step_header(2, "Forward Physics")
print("Each voxel contributes a phase-shifted signal depending on distance.")
print("Signal decays with distance (approximate spherical spreading).")
print("For each tx/rx pair and each frequency, we compute:")
print("S = sum over voxels of chi_n * (1/(r_tx+1e-3)) * (1/(r_rx+1e-3)) * exp(-j k (r_tx+r_rx))")
print("Added Gaussian noise to simulate measurement uncertainty.")
print("In real systems, measurements are limited.")

num_freq = len(f)
num_measurements = num_antennas * num_antennas * num_freq
S_clean = np.zeros(num_measurements, dtype=complex)

for tx_idx, tx_pos in enumerate(antenna_positions):
    for rx_idx, rx_pos in enumerate(antenna_positions):
        for i in range(num_freq):
            row_idx = tx_idx * (num_antennas * num_freq) + rx_idx * num_freq + i
            for n in range(N):
                r_tx = np.abs(x[n] - tx_pos)
                r_rx = np.abs(x[n] - rx_pos)
                contribution = (
                    chi_true[n]
                    * (1 / (r_tx + 1e-3))
                    * (1 / (r_rx + 1e-3))
                    * np.exp(-1j * k[i] * (r_tx + r_rx))
                )
                S_clean[row_idx] += contribution

# A bit of noise helps expose the ill-posed nature of the inversion.
noise_level = 0.08
noise = noise_level * (np.random.randn(*S_clean.shape) + 1j * np.random.randn(*S_clean.shape))
S_noisy = S_clean + noise

# Detailed walkthrough for only the first 2 frequencies.
tx_pos_demo = antenna_positions[0]
rx_pos_demo = antenna_positions[0]
for i in range(2):
    print("\n" + "-" * 72)
    print(
        f"Frequency index {i}: f = {f[i] / 1e9:.4f} GHz, k = {k[i]:.6f} rad/m "
        f"(Tx={tx_pos_demo:.2f} m, Rx={rx_pos_demo:.2f} m)"
    )
    running_sum = 0.0 + 0.0j

    for n in range(N):
        r_tx = np.abs(x[n] - tx_pos_demo)
        r_rx = np.abs(x[n] - rx_pos_demo)
        attenuation_tx = 1 / (r_tx + 1e-3)
        attenuation_rx = 1 / (r_rx + 1e-3)
        phase_term = np.exp(-1j * k[i] * (r_tx + r_rx))
        contribution = chi_true[n] * attenuation_tx * attenuation_rx * phase_term
        running_sum += contribution

        print(
            f"  voxel n={n:02d} | r_tx={r_tx:.4f} m | r_rx={r_rx:.4f} m | "
            f"1/(r_tx+1e-3)={attenuation_tx:.4f} | 1/(r_rx+1e-3)={attenuation_rx:.4f} | "
            f"exp(-jk(r_tx+r_rx))={phase_term.real:+.4f}{phase_term.imag:+.4f}j | "
            f"chi_n={chi_true[n]:.2f} | contrib={contribution.real:+.4f}{contribution.imag:+.4f}j"
        )

    print(f"  Final S(f[{i}]) = {running_sum.real:+.6f}{running_sum.imag:+.6f}j")


# -----------------------------------------------------------------------------
# 3) FULL MEASUREMENT VIEW
# -----------------------------------------------------------------------------
print_step_header(3, "Measurement S(f)")
print("First 5 complex values of S_clean:")
for i in range(5):
    print(f"  S_clean[{i}] = {S_clean[i].real:+.6f}{S_clean[i].imag:+.6f}j")

print(f"Total number of measurements: {num_measurements} (= Ntx * Nrx * Nf)")
print("Using the noisy measurement vector S_noisy for inversion.")
print("With multiple antennas, reconstruction should be smoother and more accurate than single-antenna data.")

plt.figure(figsize=(8.5, 4))
plt.plot(np.abs(S_clean), linewidth=2)
plt.plot(np.abs(S_noisy), linewidth=1.2, linestyle="--")
plt.xlabel("Measurement Index")
plt.ylabel("Magnitude")
plt.title("Measurement Comparison: |S_clean| vs |S_noisy|")
plt.legend(["|S_clean|", "|S_noisy|"])
plt.grid(alpha=0.35)
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# 4) MATRIX MODEL
# -----------------------------------------------------------------------------
print_step_header(4, "Matrix Model A")
print("Matrix A captures how each voxel affects each frequency.")
print("A matrix maps object -> measurements")
print("Each row corresponds to one (tx antenna, rx antenna, frequency) tuple.")
print("A[row,n] = (1/(r_tx+1e-3))*(1/(r_rx+1e-3))*exp(-j k_i (r_tx+r_rx))")

A = np.zeros((num_measurements, N), dtype=complex)
for tx_idx, tx_pos in enumerate(antenna_positions):
    for rx_idx, rx_pos in enumerate(antenna_positions):
        for i in range(num_freq):
            row_idx = tx_idx * (num_antennas * num_freq) + rx_idx * num_freq + i
            for n in range(N):
                r_tx = np.abs(x[n] - tx_pos)
                r_rx = np.abs(x[n] - rx_pos)
                A[row_idx, n] = (
                    (1 / (r_tx + 1e-3))
                    * (1 / (r_rx + 1e-3))
                    * np.exp(-1j * k[i] * (r_tx + r_rx))
                )

print(f"Shape of A: {A.shape}")
print("First 3 rows of A:")
print(A[:3, :])

plt.figure(figsize=(8, 4))
plt.imshow(np.angle(A), cmap="twilight", aspect="auto", origin="lower")
plt.colorbar(label="Phase (rad)")
plt.xlabel("Voxel index n")
plt.ylabel("Frequency index i")
plt.title("Phase of System Matrix A")
print("Information is encoded in phase, not magnitude.")
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# 5) INVERSE PROBLEM STATEMENT
# -----------------------------------------------------------------------------
print_step_header(5, "Inverse Problem")
print("We solve S = A * chi using least squares with Tikhonov regularization.")
print("We stabilize the inversion because the problem is ill-posed.")
print("There are more equations than unknowns, so we compute a best-fit solution.")


# -----------------------------------------------------------------------------
# 6) SOLUTION
# -----------------------------------------------------------------------------
print_step_header(6, "Reconstruction")
lambda_reg = 0.2
identity = np.eye(N)
cond_A = np.linalg.cond(A)
chi_est = np.linalg.inv(A.conj().T @ A + lambda_reg * identity) @ A.conj().T @ S_noisy

# Baseline for interpretation: single-antenna (first Tx/Rx pair only).
A_single = A[:num_freq, :]
S_single_noisy = S_noisy[:num_freq]
chi_est_single = (
    np.linalg.inv(A_single.conj().T @ A_single + lambda_reg * identity)
    @ A_single.conj().T
    @ S_single_noisy
)

print(f"Regularization strength lambda_reg: {lambda_reg}")
print(f"Shape of normal-equation inverse term: {(A.conj().T @ A + lambda_reg * identity).shape}")
print(f"Condition number of A: {cond_A:.4e}")
print("First few values of chi_est:")
for i in range(min(8, len(chi_est))):
    print(f"  chi_est[{i}] = {chi_est[i].real:+.6f}{chi_est[i].imag:+.6f}j")

print("Reconstruction is imperfect due to:")
print("  - noise")
print("  - limited measurements")
print("  - ill-conditioning")
print("With multiple antennas, diversity improves reconstruction quality compared to single-antenna scans.")

error_multi = np.abs(chi_true - chi_est)
error_single = np.abs(chi_true - chi_est_single)
mean_err_multi = np.mean(error_multi)
mean_err_single = np.mean(error_single)
max_err_multi = np.max(error_multi)
max_err_single = np.max(error_single)

print("Single-antenna vs multi-antenna comparison:")
print(f"  Mean error (single antenna): {mean_err_single:.6e}")
print(f"  Mean error (multi antenna):  {mean_err_multi:.6e}")
print(f"  Max error  (single antenna): {max_err_single:.6e}")
print(f"  Max error  (multi antenna):  {max_err_multi:.6e}")

if mean_err_multi < mean_err_single:
    improvement = 100.0 * (mean_err_single - mean_err_multi) / (mean_err_single + 1e-12)
    print(f"  Improvement from multiple antennas: {improvement:.2f}% lower mean error")
else:
    print("  In this noise realization, multi-antenna did not outperform single-antenna by mean error.")


# -----------------------------------------------------------------------------
# 7) VISUAL COMPARISON
# -----------------------------------------------------------------------------
plt.figure(figsize=(9, 4))
plt.plot(x, chi_true, "b-", linewidth=2, label="True chi")
plt.plot(x, np.abs(chi_est), "r--", linewidth=2, label="Estimated |chi|")
plt.xlabel("Position (m)")
plt.ylabel("Scattering Strength")
plt.title("Ground Truth vs Reconstruction (Multi-Antenna)")
plt.legend()
plt.grid(alpha=0.35)
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# 8) ERROR ANALYSIS
# -----------------------------------------------------------------------------
print_step_header(7, "Error Analysis")
error = error_multi
print("Reconstruction error per voxel is |chi_true - chi_est|.")
print(f"Max error: {np.max(error):.6e}")
print(f"Mean error: {np.mean(error):.6e}")

plt.figure(figsize=(9, 3.8))
plt.plot(x, error, "k-o", linewidth=1.8, markersize=4)
plt.xlabel("Position (m)")
plt.ylabel("Absolute Error")
plt.title("Reconstruction Error vs Position")
plt.grid(alpha=0.35)
plt.tight_layout()
plt.show()