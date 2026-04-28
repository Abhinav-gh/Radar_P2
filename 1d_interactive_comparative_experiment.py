import numpy as np
import matplotlib.pyplot as plt

# Keep array printing compact but readable for educational console output.
np.set_printoptions(precision=4, suppress=True)

# Make the demo reproducible so the noisy reconstruction is stable in a presentation.
np.random.seed(42)


def print_section_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def print_experiment_header(exp_num: int, title: str) -> None:
    print("\n" + "=" * 72)
    print(f"Experiment {exp_num}: {title}")
    print("=" * 72)


# =============================================================================
# INTERACTIVE SETUP: USER INPUT SECTION
# =============================================================================
print_section_header("INTERACTIVE MICROWAVE IMAGING COMPARATIVE EXPERIMENT")
print("Configure the simulation parameters interactively.\n")

# --- 1) Number of voxels and spatial domain ---
print_section_header("Step 1: Spatial Domain Configuration")
while True:
    try:
        N = int(input("Enter number of voxels (default=50): ") or "50")
        if N < 5:
            print("  ERROR: Must have at least 5 voxels. Try again.")
            continue
        break
    except ValueError:
        print("  ERROR: Invalid input. Enter an integer.")

while True:
    try:
        x_max = float(input("Enter spatial domain upper limit in meters (default=0.45): ") or "0.45")
        if x_max <= 0:
            print("  ERROR: Upper limit must be positive. Try again.")
            continue
        break
    except ValueError:
        print("  ERROR: Invalid input. Enter a number.")

x = np.linspace(0, x_max, N)
print(f"✓ Voxel grid: {N} voxels from 0 to {x_max:.3f} m")


# --- 2) Object definition ---
print_section_header("Step 2: Object Definition")
print("Define one or more reflective objects in the spatial domain.")
print(f"Voxel range: {x[0]:.3f} to {x[-1]:.3f} m (indices 0 to {N-1})")

chi_true = np.zeros(N)
object_regions = []

while True:
    print("\nDefine object region:")
    while True:
        try:
            start_idx = int(input(f"  Start voxel index (0 to {N-1}): "))
            if 0 <= start_idx < N:
                break
            print(f"    ERROR: Must be between 0 and {N-1}.")
        except ValueError:
            print("    ERROR: Invalid input. Enter an integer.")
    
    while True:
        try:
            end_idx = int(input(f"  End voxel index (>={start_idx}): "))
            if end_idx >= start_idx and end_idx < N:
                break
            print(f"    ERROR: Must be >= {start_idx} and < {N}.")
        except ValueError:
            print("    ERROR: Invalid input. Enter an integer.")
    
    while True:
        try:
            chi_val = float(input("  Reflectivity χ value for this region (0.0 to 1.0): "))
            if 0 <= chi_val <= 1:
                break
            print("    ERROR: χ must be between 0 and 1.")
        except ValueError:
            print("    ERROR: Invalid input. Enter a number.")
    
    # Define shape: constant, Gaussian, or triangular
    print("  Shape options: (1) Constant, (2) Gaussian, (3) Triangular")
    while True:
        try:
            shape_choice = int(input("  Select shape (1/2/3, default=1): ") or "1")
            if shape_choice in [1, 2, 3]:
                break
            print("    ERROR: Enter 1, 2, or 3.")
        except ValueError:
            print("    ERROR: Invalid input.")
    
    region_indices = np.arange(start_idx, end_idx + 1)
    center_idx = (start_idx + end_idx) / 2.0
    
    if shape_choice == 1:
        # Constant
        chi_true[region_indices] = chi_val
        print(f"  ✓ Added constant region: χ={chi_val:.3f} at indices [{start_idx}, {end_idx}]")
    elif shape_choice == 2:
        # Gaussian
        sigma = (end_idx - start_idx) / 4.0 + 0.5
        for i in region_indices:
            chi_true[i] = chi_val * np.exp(-0.5 * ((i - center_idx) / sigma) ** 2)
        print(f"  ✓ Added Gaussian region: χ_peak={chi_val:.3f} at indices [{start_idx}, {end_idx}]")
    elif shape_choice == 3:
        # Triangular
        for i in region_indices:
            dist_from_center = np.abs(i - center_idx)
            max_dist = (end_idx - start_idx) / 2.0
            chi_true[i] = max(0, chi_val * (1 - dist_from_center / max_dist))
        print(f"  ✓ Added triangular region: χ_peak={chi_val:.3f} at indices [{start_idx}, {end_idx}]")
    
    object_regions.append((start_idx, end_idx, chi_val, shape_choice))
    
    add_more = input("\nAdd another object region? (y/n, default=n): ").strip().lower()
    if add_more != 'y':
        break

object_indices = np.where(chi_true > 0)[0]
print(f"\n✓ Object configuration complete. Object spans indices {object_indices[0]} to {object_indices[-1]}")


# --- 3) Frequency configuration ---
print_section_header("Step 3: Frequency Configuration")
while True:
    try:
        f_min = float(input("Enter minimum frequency in GHz (default=1.0): ") or "1.0")
        f_max = float(input("Enter maximum frequency in GHz (default=5.0): ") or "5.0")
        num_freq = int(input("Enter number of frequency points (default=20): ") or "20")
        if f_min < f_max and num_freq >= 3:
            break
        print("  ERROR: f_min < f_max and num_freq >= 3 required.")
    except ValueError:
        print("  ERROR: Invalid input.")

f = np.linspace(f_min * 1e9, f_max * 1e9, num_freq)
c = 3e8
k = 2 * np.pi * f / c
print(f"✓ Frequency range: {f_min:.1f} to {f_max:.1f} GHz ({num_freq} points)")


# --- 4) Noise and regularization ---
print_section_header("Step 4: Noise and Regularization")
while True:
    try:
        noise_level = float(input("Enter noise level as percentage (0-100%, default=8): ") or "8")
        lambda_reg = float(input("Enter regularization parameter λ (default=0.2): ") or "0.2")
        if 0 <= noise_level <= 100 and lambda_reg > 0:
            noise_level = noise_level / 100.0
            break
        print("  ERROR: noise in [0,100]% and λ > 0 required.")
    except ValueError:
        print("  ERROR: Invalid input.")

print(f"✓ Noise: {noise_level*100:.1f}%, Regularization: λ={lambda_reg:.3f}")


# --- 5) Antenna configurations ---
print_section_header("Step 5: Antenna Configurations")
print("Define antenna configurations to compare.\n")

antenna_configs = []
config_count = 0

while True:
    config_count += 1
    print(f"Configuration {config_count}:")
    config_name = input("  Name for this configuration (e.g., 'Single antenna'): ").strip()
    if not config_name:
        config_name = f"Configuration {config_count}"
    
    print(f"  Enter antenna positions (as space-separated values in meters)")
    print(f"  Example: 0.0 (single) or 0.0 0.45 (two antennas)")
    
    while True:
        try:
            antenna_str = input("  Antenna positions (e.g., 0.0 0.45): ").strip()
            antenna_pos = [float(pos) for pos in antenna_str.split()]
            
            # Validate positions
            if all(0 <= pos <= x_max for pos in antenna_pos):
                break
            print(f"    ERROR: Positions must be in [0, {x_max}].")
        except ValueError:
            print("    ERROR: Invalid input. Enter space-separated numbers.")
    
    antenna_configs.append({
        "name": config_name,
        "positions": antenna_pos,
    })
    print(f"  ✓ Added: {config_name} with antennas at {antenna_pos}")
    
    add_more = input("\nAdd another configuration? (y/n, default=n): ").strip().lower()
    if add_more != 'y':
        break

print(f"\n✓ Defined {len(antenna_configs)} antenna configurations")


# =============================================================================
# DEFINE COMMON FUNCTIONS
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
# SETUP VISUALIZATION
# =============================================================================
print_section_header("SETUP: Global Voxel Grid and Object")
print(f"Number of voxels: {N}")
print(f"Voxel range: {x[0]:.4f} m to {x[-1]:.4f} m")
print(f"Object support indices: {object_indices.tolist() if len(object_indices) > 0 else 'None'}")
print(f"Frequency range: {f[0]/1e9:.1f} GHz to {f[-1]/1e9:.1f} GHz")

plt.figure(figsize=(10, 3.5))
plt.plot(x, np.zeros_like(x), "co", alpha=0.6, markersize=4, label="Voxel grid")
if len(object_indices) > 0:
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
# STORAGE FOR RESULTS
# =============================================================================
results = {}


# =============================================================================
# RUN ALL EXPERIMENTS
# =============================================================================
print_section_header("RUNNING COMPARATIVE EXPERIMENTS")

for exp_idx, config in enumerate(antenna_configs, start=1):
    config_name = config["name"]
    antenna_positions = config["positions"]
    
    print_experiment_header(exp_idx, config_name)
    print(f"Antenna positions: {antenna_positions}")
    
    # Simulate
    S_clean, S_noisy = simulate_S(antenna_positions, x, chi_true, f, k, noise_level)
    A = build_A(antenna_positions, x, f, k)
    chi_est = reconstruct(A, S_noisy, lambda_reg, N)
    error, mean_err, max_err = compute_error(chi_true, chi_est)
    
    print(f"Matrix A shape: {A.shape}")
    print(f"Condition number: {np.linalg.cond(A):.4e}")
    print(f"Mean error: {mean_err:.6e}")
    print(f"Max error:  {max_err:.6e}")
    
    # Plot 1: Ground truth vs reconstruction
    plt.figure(figsize=(10, 3.5))
    plt.plot(x, chi_true, "b-", linewidth=2.5, label="True χ(x)")
    plt.plot(x, np.abs(chi_est), "r--", linewidth=2, label="Estimated |χ(x)|")
    plt.xlabel("Position x (m)", fontsize=11)
    plt.ylabel("Reflectivity χ", fontsize=11)
    plt.title(f"Experiment {exp_idx}: {config_name} - Reconstruction", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Error
    plt.figure(figsize=(10, 3.5))
    plt.plot(x, error, "k-o", linewidth=1.5, markersize=3, alpha=0.7)
    plt.xlabel("Position x (m)", fontsize=11)
    plt.ylabel("Absolute Error", fontsize=11)
    plt.title(f"Experiment {exp_idx}: {config_name} - Reconstruction Error", fontsize=12, fontweight="bold")
    plt.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.show()
    
    # Plot 3: Measurement spectrum
    plt.figure(figsize=(10, 3.5))
    plt.plot(np.abs(S_clean), "b-", linewidth=2, label="|S_clean|")
    plt.plot(np.abs(S_noisy), "r--", linewidth=1.5, alpha=0.7, label="|S_noisy|")
    plt.xlabel("Measurement Index", fontsize=11)
    plt.ylabel("Magnitude", fontsize=11)
    plt.title(f"Experiment {exp_idx}: {config_name} - Measurement Spectrum", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.show()
    
    # Plot 4: Antenna positions relative to object
    plt.figure(figsize=(10, 3.5))
    plt.fill_between(x, 0, chi_true, alpha=0.3, color="blue", label="Object")
    plt.plot(x, chi_true, "b-", linewidth=2)
    for ant_pos in antenna_positions:
        plt.axvline(ant_pos, color="red", linestyle="--", linewidth=2.5, alpha=0.8)
    plt.plot([], [], "r--", linewidth=2.5, label="Antenna position")
    plt.xlabel("Position x (m)", fontsize=11)
    plt.ylabel("Reflectivity χ", fontsize=11)
    plt.title(f"Experiment {exp_idx}: {config_name} - Geometry", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10, loc="upper right")
    plt.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.show()
    
    # Plot 5: Phase pattern of system matrix
    plt.figure(figsize=(10, 3.5))
    plt.imshow(np.angle(A), cmap="twilight", aspect="auto", origin="lower")
    plt.colorbar(label="Phase (rad)")
    plt.xlabel("Voxel index n", fontsize=11)
    plt.ylabel("Measurement index", fontsize=11)
    plt.title(f"Experiment {exp_idx}: {config_name} - Phase Pattern of A", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
    
    print(f"\nPhase Statistics ({config_name}):")
    print(f"  Phase range: [{np.min(np.angle(A)):.4f}, {np.max(np.angle(A)):.4f}] rad")
    print(f"  Phase mean: {np.mean(np.angle(A)):.4f} rad")
    print(f"  Phase std: {np.std(np.angle(A)):.4f} rad")
    
    results[config_name] = {
        "mean_error": mean_err,
        "max_error": max_err,
        "cond_A": np.linalg.cond(A),
    }


# =============================================================================
# COMPARISON SUMMARY
# =============================================================================
print_section_header("COMPARISON SUMMARY")
print("\nReconstruction Error Metrics:")
print("-" * 72)
print(f"{'Configuration':<35} {'Mean Error':<18} {'Max Error':<18}")
print("-" * 72)
for config_name, metrics in results.items():
    print(f"{config_name:<35} {metrics['mean_error']:<18.6e} {metrics['max_error']:<18.6e}")

print("\nCondition Numbers:")
print("-" * 72)
print(f"{'Configuration':<35} {'Condition Number':<18}")
print("-" * 72)
for config_name, metrics in results.items():
    print(f"{config_name:<35} {metrics['cond_A']:<18.4e}")

# Calculate improvements
if len(results) > 1:
    errors = list(results.values())
    baseline_error = errors[0]["mean_error"]
    print("\nImprovement Analysis:")
    print("-" * 72)
    for i, (config_name, metrics) in enumerate(results.items()):
        if i == 0:
            print(f"{config_name:<35} (baseline)")
        else:
            improvement = 100.0 * (baseline_error - metrics["mean_error"]) / (baseline_error + 1e-12)
            print(f"{config_name:<35} {improvement:+.2f}% vs baseline")


# =============================================================================
# FINAL COMPARISON PLOT
# =============================================================================
if len(results) > 1:
    config_names = list(results.keys())
    mean_errors = [results[name]["mean_error"] for name in config_names]
    
    plt.figure(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(config_names)))
    bars = plt.bar(range(len(config_names)), mean_errors, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    plt.xticks(range(len(config_names)), config_names, rotation=15, ha='right')
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


print_section_header("SIMULATION COMPLETE")
print("All experiments have been run and visualized.")
print("=" * 72)
