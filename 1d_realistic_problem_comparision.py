import numpy as np
import matplotlib.pyplot as plt

# Keep array printing compact but readable for educational console output.
np.set_printoptions(precision=4, suppress=True)

# Make the demo reproducible so the noisy reconstruction is stable in a presentation.
np.random.seed(42)


def print_experiment_header(exp_num: int, title: str) -> None:
    print("\n" + "=" * 72)
    print(f"Experiment {exp_num}: {title}")
    print("=" * 72)


# =============================================================================
# GLOBAL PARAMETERS
# =============================================================================
c = 3e8
f = np.linspace(1e9, 5e9, 20)
k = 2 * np.pi * f / c

N = 50
x = np.linspace(0, 0.45, N)

noise_level = 0.08
lambda_reg = 0.2

# Define true object (same for all experiments)
chi_true = np.zeros(N)
for i in range(20, 30):
    chi_true[i] = np.exp(-0.5 * ((i - 25) / 10) ** 2)

object_indices = np.where(chi_true > 0)[0]


# =============================================================================
# STEP 1: SETUP VISUALIZATION
# =============================================================================
print("=" * 72)
print("SETUP: Global Voxel Grid and Object")
print("=" * 72)
print(f"Number of voxels: {N}")
print(f"Voxel range: {x[0]:.2f} m to {x[-1]:.2f} m")
print(f"Object support indices: {object_indices.tolist()}")
print(f"Frequency range: {f[0]/1e9:.1f} GHz to {f[-1]/1e9:.1f} GHz")

plt.figure(figsize=(10, 3.5))
plt.plot(x, np.zeros_like(x), "co", alpha=0.6, markersize=4, label="Voxel grid")
plt.plot(x[object_indices], np.zeros_like(object_indices, dtype=float), "r^", markersize=8, label="Object region")
plt.yticks([])
plt.xlabel("Position x (m)", fontsize=11)
plt.ylabel("")
plt.title("Voxel Grid and Object Placement", fontsize=12, fontweight="bold")
plt.grid(alpha=0.3, linestyle=":")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 3.5))
plt.fill_between(x, 0, chi_true, alpha=0.3, color="blue", label="Object profile")
plt.plot(x, chi_true, "b-", linewidth=2, label="True reflectivity χ(x)")
plt.xlabel("Position x (m)", fontsize=11)
plt.ylabel("Reflectivity χ", fontsize=11)
plt.title("Continuous Object Shape (Ground Truth)", fontsize=12, fontweight="bold")
plt.grid(alpha=0.3, linestyle=":")
plt.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# STEP 2: DEFINE COMMON FUNCTIONS
# =============================================================================
def simulate_S(antenna_positions, x, chi_true, f, k, noise_level):
    """Simulate noisy measurement vector S."""
    num_antennas = len(antenna_positions)
    num_freq = len(f)
    num_measurements = num_antennas * num_antennas * num_freq
    S_clean = np.zeros(num_measurements, dtype=complex)

    for tx_idx, tx_pos in enumerate(antenna_positions):
        for rx_idx, rx_pos in enumerate(antenna_positions):
            for i in range(num_freq):
                row_idx = tx_idx * (num_antennas * num_freq) + rx_idx * num_freq + i
                for n in range(len(x)):
                    r_tx = np.abs(x[n] - tx_pos)
                    r_rx = np.abs(x[n] - rx_pos)
                    contribution = (
                        chi_true[n]
                        * (1 / (r_tx + 1e-3))
                        * (1 / (r_rx + 1e-3))
                        * np.exp(-1j * k[i] * (r_tx + r_rx))
                    )
                    S_clean[row_idx] += contribution

    noise = noise_level * (
        np.random.randn(*S_clean.shape) + 1j * np.random.randn(*S_clean.shape)
    )
    S_noisy = S_clean + noise
    return S_clean, S_noisy


def build_A(antenna_positions, x, f, k):
    """Build system matrix A."""
    num_antennas = len(antenna_positions)
    num_freq = len(f)
    num_measurements = num_antennas * num_antennas * num_freq
    N = len(x)
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
    return A


def reconstruct(A, S_noisy, lambda_reg, N):
    """Reconstruct chi using regularized least squares."""
    identity = np.eye(N)
    chi_est = (
        np.linalg.inv(A.conj().T @ A + lambda_reg * identity)
        @ A.conj().T
        @ S_noisy
    )
    return chi_est


def compute_error(chi_true, chi_est):
    """Compute reconstruction error metrics."""
    error = np.abs(chi_true - chi_est)
    mean_error = np.mean(error)
    max_error = np.max(error)
    return error, mean_error, max_error


# =============================================================================
# STORAGE FOR RESULTS
# =============================================================================
results = {}


# =============================================================================
# EXPERIMENT 1: SINGLE ANTENNA
# =============================================================================
print_experiment_header(1, "Single Antenna Configuration")
antenna_config_1 = [0.0]
print(f"Antenna positions: {antenna_config_1}")

S_clean_1, S_noisy_1 = simulate_S(antenna_config_1, x, chi_true, f, k, noise_level)
A_1 = build_A(antenna_config_1, x, f, k)
chi_est_1 = reconstruct(A_1, S_noisy_1, lambda_reg, N)
error_1, mean_err_1, max_err_1 = compute_error(chi_true, chi_est_1)

print(f"Matrix A shape: {A_1.shape}")
print(f"Condition number: {np.linalg.cond(A_1):.4e}")
print(f"Mean error: {mean_err_1:.6e}")
print(f"Max error:  {max_err_1:.6e}")

# Plot 1: Ground truth vs reconstruction
plt.figure(figsize=(10, 3.5))
plt.plot(x, chi_true, "b-", linewidth=2.5, label="True χ(x)")
plt.plot(x, np.abs(chi_est_1), "r--", linewidth=2, label="Estimated |χ(x)|")
plt.xlabel("Position x (m)", fontsize=11)
plt.ylabel("Reflectivity χ", fontsize=11)
plt.title("Experiment 1: Single Antenna - Reconstruction", fontsize=12, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(alpha=0.3, linestyle=":")
plt.tight_layout()
plt.show()

# Plot 2: Error
plt.figure(figsize=(10, 3.5))
plt.plot(x, error_1, "k-o", linewidth=1.5, markersize=3, alpha=0.7)
plt.xlabel("Position x (m)", fontsize=11)
plt.ylabel("Absolute Error", fontsize=11)
plt.title("Experiment 1: Single Antenna - Reconstruction Error", fontsize=12, fontweight="bold")
plt.grid(alpha=0.3, linestyle=":")
plt.tight_layout()
plt.show()

# Plot 3: Measurement spectrum
plt.figure(figsize=(10, 3.5))
plt.plot(np.abs(S_clean_1), "b-", linewidth=2, label="|S_clean|")
plt.plot(np.abs(S_noisy_1), "r--", linewidth=1.5, alpha=0.7, label="|S_noisy|")
plt.xlabel("Measurement Index", fontsize=11)
plt.ylabel("Magnitude", fontsize=11)
plt.title("Experiment 1: Single Antenna - Measurement Spectrum", fontsize=12, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(alpha=0.3, linestyle=":")
plt.tight_layout()
plt.show()

# Plot 4: Antenna positions relative to object
plt.figure(figsize=(10, 3.5))
plt.fill_between(x, 0, chi_true, alpha=0.3, color="blue", label="Object")
plt.plot(x, chi_true, "b-", linewidth=2)
for ant_pos in antenna_config_1:
    plt.axvline(ant_pos, color="red", linestyle="--", linewidth=2.5, alpha=0.8)
plt.plot([], [], "r--", linewidth=2.5, label="Antenna position")
plt.xlabel("Position x (m)", fontsize=11)
plt.ylabel("Reflectivity χ", fontsize=11)
plt.title("Experiment 1: Single Antenna - Geometry", fontsize=12, fontweight="bold")
plt.legend(fontsize=10, loc="upper right")
plt.grid(alpha=0.3, linestyle=":")
plt.tight_layout()
plt.show()

# Plot 5: Phase pattern of system matrix
plt.figure(figsize=(10, 3.5))
plt.imshow(np.angle(A_1), cmap="twilight", aspect="auto", origin="lower")
plt.colorbar(label="Phase (rad)")
plt.xlabel("Voxel index n", fontsize=11)
plt.ylabel("Frequency index i", fontsize=11)
plt.title("Experiment 1: Single Antenna - Phase Pattern of A", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

print(f"\nPhase Statistics (Experiment 1):")
print(f"  Phase range: [{np.min(np.angle(A_1)):.4f}, {np.max(np.angle(A_1)):.4f}] rad")
print(f"  Phase mean: {np.mean(np.angle(A_1)):.4f} rad")
print(f"  Phase std: {np.std(np.angle(A_1)):.4f} rad")

results["Single antenna"] = {
    "mean_error": mean_err_1,
    "max_error": max_err_1,
    "cond_A": np.linalg.cond(A_1),
}


# =============================================================================
# EXPERIMENT 2: TWO ANTENNAS (SYMMETRIC)
# =============================================================================
print_experiment_header(2, "Two Antennas (Symmetric Placement)")
antenna_config_2 = [0.0, 0.45]
print(f"Antenna positions: {antenna_config_2}")
print(f"Object center: ~{x[object_indices[len(object_indices)//2]]:.3f} m")

S_clean_2, S_noisy_2 = simulate_S(antenna_config_2, x, chi_true, f, k, noise_level)
A_2 = build_A(antenna_config_2, x, f, k)
chi_est_2 = reconstruct(A_2, S_noisy_2, lambda_reg, N)
error_2, mean_err_2, max_err_2 = compute_error(chi_true, chi_est_2)

print(f"Matrix A shape: {A_2.shape}")
print(f"Condition number: {np.linalg.cond(A_2):.4e}")
print(f"Mean error: {mean_err_2:.6e}")
print(f"Max error:  {max_err_2:.6e}")

# Plot 1: Ground truth vs reconstruction
plt.figure(figsize=(10, 3.5))
plt.plot(x, chi_true, "b-", linewidth=2.5, label="True χ(x)")
plt.plot(x, np.abs(chi_est_2), "r--", linewidth=2, label="Estimated |χ(x)|")
plt.xlabel("Position x (m)", fontsize=11)
plt.ylabel("Reflectivity χ", fontsize=11)
plt.title("Experiment 2: Two Antennas (Symmetric) - Reconstruction", fontsize=12, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(alpha=0.3, linestyle=":")
plt.tight_layout()
plt.show()

# Plot 2: Error
plt.figure(figsize=(10, 3.5))
plt.plot(x, error_2, "k-o", linewidth=1.5, markersize=3, alpha=0.7)
plt.xlabel("Position x (m)", fontsize=11)
plt.ylabel("Absolute Error", fontsize=11)
plt.title("Experiment 2: Two Antennas (Symmetric) - Reconstruction Error", fontsize=12, fontweight="bold")
plt.grid(alpha=0.3, linestyle=":")
plt.tight_layout()
plt.show()

# Plot 3: Measurement spectrum
plt.figure(figsize=(10, 3.5))
plt.plot(np.abs(S_clean_2), "b-", linewidth=2, label="|S_clean|")
plt.plot(np.abs(S_noisy_2), "r--", linewidth=1.5, alpha=0.7, label="|S_noisy|")
plt.xlabel("Measurement Index", fontsize=11)
plt.ylabel("Magnitude", fontsize=11)
plt.title("Experiment 2: Two Antennas (Symmetric) - Measurement Spectrum", fontsize=12, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(alpha=0.3, linestyle=":")
plt.tight_layout()
plt.show()

# Plot 4: Antenna positions relative to object
plt.figure(figsize=(10, 3.5))
plt.fill_between(x, 0, chi_true, alpha=0.3, color="blue", label="Object")
plt.plot(x, chi_true, "b-", linewidth=2)
for ant_pos in antenna_config_2:
    plt.axvline(ant_pos, color="red", linestyle="--", linewidth=2.5, alpha=0.8)
plt.plot([], [], "r--", linewidth=2.5, label="Antenna position")
plt.xlabel("Position x (m)", fontsize=11)
plt.ylabel("Reflectivity χ", fontsize=11)
plt.title("Experiment 2: Two Antennas (Symmetric) - Geometry", fontsize=12, fontweight="bold")
plt.legend(fontsize=10, loc="upper right")
plt.grid(alpha=0.3, linestyle=":")
plt.tight_layout()
plt.show()

# Plot 5: Phase pattern of system matrix
plt.figure(figsize=(10, 3.5))
plt.imshow(np.angle(A_2), cmap="twilight", aspect="auto", origin="lower")
plt.colorbar(label="Phase (rad)")
plt.xlabel("Voxel index n", fontsize=11)
plt.ylabel("Measurement index", fontsize=11)
plt.title("Experiment 2: Two Antennas (Symmetric) - Phase Pattern of A", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

print(f"\nPhase Statistics (Experiment 2):")
print(f"  Phase range: [{np.min(np.angle(A_2)):.4f}, {np.max(np.angle(A_2)):.4f}] rad")
print(f"  Phase mean: {np.mean(np.angle(A_2)):.4f} rad")
print(f"  Phase std: {np.std(np.angle(A_2)):.4f} rad")

results["Two antennas symmetric"] = {
    "mean_error": mean_err_2,
    "max_error": max_err_2,
    "cond_A": np.linalg.cond(A_2),
}


# =============================================================================
# EXPERIMENT 3: TWO ANTENNAS (ASYMMETRIC)
# =============================================================================
print_experiment_header(3, "Two Antennas (Asymmetric Placement)")
antenna_config_3 = [0.0, 0.3]
print(f"Antenna positions: {antenna_config_3}")
print(f"Object center: ~{x[object_indices[len(object_indices)//2]]:.3f} m")
print("Note: Object is NOT centered between antennas")

S_clean_3, S_noisy_3 = simulate_S(antenna_config_3, x, chi_true, f, k, noise_level)
A_3 = build_A(antenna_config_3, x, f, k)
chi_est_3 = reconstruct(A_3, S_noisy_3, lambda_reg, N)
error_3, mean_err_3, max_err_3 = compute_error(chi_true, chi_est_3)

print(f"Matrix A shape: {A_3.shape}")
print(f"Condition number: {np.linalg.cond(A_3):.4e}")
print(f"Mean error: {mean_err_3:.6e}")
print(f"Max error:  {max_err_3:.6e}")

# Plot 1: Ground truth vs reconstruction
plt.figure(figsize=(10, 3.5))
plt.plot(x, chi_true, "b-", linewidth=2.5, label="True χ(x)")
plt.plot(x, np.abs(chi_est_3), "r--", linewidth=2, label="Estimated |χ(x)|")
plt.xlabel("Position x (m)", fontsize=11)
plt.ylabel("Reflectivity χ", fontsize=11)
plt.title("Experiment 3: Two Antennas (Asymmetric) - Reconstruction", fontsize=12, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(alpha=0.3, linestyle=":")
plt.tight_layout()
plt.show()

# Plot 2: Error
plt.figure(figsize=(10, 3.5))
plt.plot(x, error_3, "k-o", linewidth=1.5, markersize=3, alpha=0.7)
plt.xlabel("Position x (m)", fontsize=11)
plt.ylabel("Absolute Error", fontsize=11)
plt.title("Experiment 3: Two Antennas (Asymmetric) - Reconstruction Error", fontsize=12, fontweight="bold")
plt.grid(alpha=0.3, linestyle=":")
plt.tight_layout()
plt.show()

# Plot 3: Measurement spectrum
plt.figure(figsize=(10, 3.5))
plt.plot(np.abs(S_clean_3), "b-", linewidth=2, label="|S_clean|")
plt.plot(np.abs(S_noisy_3), "r--", linewidth=1.5, alpha=0.7, label="|S_noisy|")
plt.xlabel("Measurement Index", fontsize=11)
plt.ylabel("Magnitude", fontsize=11)
plt.title("Experiment 3: Two Antennas (Asymmetric) - Measurement Spectrum", fontsize=12, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(alpha=0.3, linestyle=":")
plt.tight_layout()
plt.show()

# Plot 4: Antenna positions relative to object
plt.figure(figsize=(10, 3.5))
plt.fill_between(x, 0, chi_true, alpha=0.3, color="blue", label="Object")
plt.plot(x, chi_true, "b-", linewidth=2)
for ant_pos in antenna_config_3:
    plt.axvline(ant_pos, color="red", linestyle="--", linewidth=2.5, alpha=0.8)
plt.plot([], [], "r--", linewidth=2.5, label="Antenna position")
plt.xlabel("Position x (m)", fontsize=11)
plt.ylabel("Reflectivity χ", fontsize=11)
plt.title("Experiment 3: Two Antennas (Asymmetric) - Geometry", fontsize=12, fontweight="bold")
plt.legend(fontsize=10, loc="upper right")
plt.grid(alpha=0.3, linestyle=":")
plt.tight_layout()
plt.show()

# Plot 5: Phase pattern of system matrix
plt.figure(figsize=(10, 3.5))
plt.imshow(np.angle(A_3), cmap="twilight", aspect="auto", origin="lower")
plt.colorbar(label="Phase (rad)")
plt.xlabel("Voxel index n", fontsize=11)
plt.ylabel("Measurement index", fontsize=11)
plt.title("Experiment 3: Two Antennas (Asymmetric) - Phase Pattern of A", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

print(f"\nPhase Statistics (Experiment 3):")
print(f"  Phase range: [{np.min(np.angle(A_3)):.4f}, {np.max(np.angle(A_3)):.4f}] rad")
print(f"  Phase mean: {np.mean(np.angle(A_3)):.4f} rad")
print(f"  Phase std: {np.std(np.angle(A_3)):.4f} rad")

results["Two antennas asymmetric"] = {
    "mean_error": mean_err_3,
    "max_error": max_err_3,
    "cond_A": np.linalg.cond(A_3),
}


# =============================================================================
# COMPARISON SUMMARY
# =============================================================================
print("\n" + "=" * 72)
print("COMPARISON SUMMARY")
print("=" * 72)
print("\nReconstruction Error Metrics:")
print("-" * 72)
print(f"{'Configuration':<30} {'Mean Error':<20} {'Max Error':<20}")
print("-" * 72)
for config_name, metrics in results.items():
    print(f"{config_name:<30} {metrics['mean_error']:<20.6e} {metrics['max_error']:<20.6e}")

print("\nCondition Numbers:")
print("-" * 72)
print(f"{'Configuration':<30} {'Condition Number':<20}")
print("-" * 72)
for config_name, metrics in results.items():
    print(f"{config_name:<30} {metrics['cond_A']:<20.4e}")

# Calculate improvements
improvement_2v1 = 100.0 * (mean_err_1 - mean_err_2) / (mean_err_1 + 1e-12)
improvement_3v1 = 100.0 * (mean_err_1 - mean_err_3) / (mean_err_1 + 1e-12)

print(f"\nImprovement (symmetric vs single): {improvement_2v1:.2f}% lower mean error")
print(f"Improvement (asymmetric vs single): {improvement_3v1:.2f}% lower mean error")


# =============================================================================
# FINAL COMPARISON PLOT
# =============================================================================
config_names = list(results.keys())
mean_errors = [results[name]["mean_error"] for name in config_names]

plt.figure(figsize=(8, 5))
bars = plt.bar(config_names, mean_errors, color=["#FF6B6B", "#4ECDC4", "#45B7D1"], alpha=0.8, edgecolor="black", linewidth=1.5)
plt.ylabel("Mean Reconstruction Error", fontsize=12, fontweight="bold")
plt.title("Reconstruction Error Comparison Across Configurations", fontsize=13, fontweight="bold")
plt.yscale("log")
plt.grid(alpha=0.3, linestyle=":", axis="y")

# Add value labels on bars
for bar, err in zip(bars, mean_errors):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{err:.2e}',
             ha='center', va='bottom', fontsize=10, fontweight="bold")

plt.tight_layout()
plt.show()


print("\n" + "=" * 72)
print("KEY INSIGHTS:")
print("=" * 72)
print("1. Single antenna: Limited reconstruction due to few measurements.")
print("2. Symmetric antennas: Improved reconstruction with better measurement diversity.")
print("3. Asymmetric antennas: Performance depends on antenna-object geometry.")
print("\n>> Antenna placement and diversity DIRECTLY affect reconstruction quality.")
print("=" * 72)
