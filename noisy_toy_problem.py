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
f = np.linspace(1e9, 5e9, 30)  # real systems often have limited frequency samples
k = 2 * np.pi * f / c

N = 20
x = np.linspace(0.1, 1.0, N)  # 1D voxel positions (m)

antenna_pos = 0.0
# Optional extension: antenna_positions = [0.0, 0.2]
# Keeping the default to one antenna makes the inverse problem visibly imperfect.
antenna_positions = [antenna_pos]

chi_true = np.zeros(N, dtype=float)
chi_true[5] = 1.0
chi_true[12] = 0.8
chi_true[16] = 0.6
object_indices = np.where(chi_true > 0)[0]

print_step_header(1, "Setup")
print("We discretize the scene into voxels and assign object reflectivity chi.")
print("This is a monostatic setup: transmitter and receiver at same position.")
print(f"Number of voxels: {N}")
print("Voxel positions x (m):")
print(np.array2string(x, separator=", "))
print(f"Object voxel indices: {object_indices.tolist()}")
print(f"Antenna position(s): {antenna_positions}")

plt.figure(figsize=(9, 3.5))
plt.plot(x, np.zeros_like(x), "bo", label="All voxels")
plt.plot(x[object_indices], np.zeros_like(object_indices, dtype=float), "r^", markersize=10, label="Active scatterers")
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


# -----------------------------------------------------------------------------
# 2) FORWARD PHYSICS
# -----------------------------------------------------------------------------
print_step_header(2, "Forward Physics")
print("Each voxel contributes a phase-shifted signal depending on distance.")
print("Signal decays with distance (approximate spherical spreading).")
print("For each frequency, we compute S(f) = sum over voxels of chi_n * (1/(r_n+1e-3)) * exp(-j 2k r_n)")
print("Added Gaussian noise to simulate measurement uncertainty.")
print("In real systems, measurements are limited.")

S = np.zeros(len(f), dtype=complex)
S_stacked = np.zeros(len(f) * len(antenna_positions), dtype=complex)

for i in range(len(f)):
    for n in range(N):
        r_n = np.abs(x[n] - antenna_pos)
        S[i] += chi_true[n] * (1 / (r_n + 1e-3)) * np.exp(-1j * 2 * k[i] * r_n)

# A bit of noise helps expose the ill-posed nature of the inversion.
noise_level = 0.08
noise = noise_level * (np.random.randn(*S.shape) + 1j * np.random.randn(*S.shape))
S_noisy = S + noise

for antenna_idx, antenna_x in enumerate(antenna_positions):
    for i in range(len(f)):
        row_idx = antenna_idx * len(f) + i
        for n in range(N):
            r_n = np.abs(x[n] - antenna_x)
            S_stacked[row_idx] += chi_true[n] * (1 / (r_n + 1e-3)) * np.exp(-1j * 2 * k[i] * r_n)

noise_stacked = noise_level * (
    np.random.randn(*S_stacked.shape) + 1j * np.random.randn(*S_stacked.shape)
)
S_stacked_noisy = S_stacked + noise_stacked

# Detailed walkthrough for only the first 2 frequencies.
for i in range(2):
    print("\n" + "-" * 72)
    print(f"Frequency index {i}: f = {f[i] / 1e9:.4f} GHz, k = {k[i]:.6f} rad/m")
    running_sum = 0.0 + 0.0j

    for n in range(N):
        r_n = np.abs(x[n] - antenna_pos)
        phase_term = np.exp(-1j * 2 * k[i] * r_n)
        attenuation = 1 / (r_n + 1e-3)
        contribution = chi_true[n] * attenuation * phase_term
        running_sum += contribution

        print(
            f"  voxel n={n:02d} | r_n={r_n:.4f} m | 1/(r_n+1e-3)={attenuation:.4f} | "
            f"exp(-j2kr_n)={phase_term.real:+.4f}{phase_term.imag:+.4f}j | "
            f"chi_n={chi_true[n]:.2f} | contrib={contribution.real:+.4f}{contribution.imag:+.4f}j"
        )

    print(f"  Final S(f[{i}]) = {running_sum.real:+.6f}{running_sum.imag:+.6f}j")


# -----------------------------------------------------------------------------
# 3) FULL MEASUREMENT VIEW
# -----------------------------------------------------------------------------
print_step_header(3, "Measurement S(f)")
print("First 5 complex values of S:")
for i in range(5):
    print(f"  S[{i}] = {S[i].real:+.6f}{S[i].imag:+.6f}j")

print("Using the noisy measurement vector S_stacked_noisy for inversion.")

plt.figure(figsize=(8.5, 4))
plt.plot(f / 1e9, np.abs(S), linewidth=2)
plt.plot(f / 1e9, np.abs(S_noisy), linewidth=1.5, linestyle="--")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Magnitude")
plt.title("Measured Signal Magnitude vs Frequency")
plt.legend(["|S|", "|S_noisy|"])
plt.grid(alpha=0.35)
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# 4) MATRIX MODEL
# -----------------------------------------------------------------------------
print_step_header(4, "Matrix Model A")
print("Matrix A captures how each voxel affects each frequency.")
print("A matrix maps object -> measurements")
print("A[i,n] = (1 / (r_n + 1e-3)) * exp(-j 2 k_i r_n)")

if len(antenna_positions) > 1:
    print("Multiple antennas improve reconstruction.")

A = np.zeros((len(f) * len(antenna_positions), N), dtype=complex)
for antenna_idx, antenna_x in enumerate(antenna_positions):
    for i in range(len(f)):
        row_idx = antenna_idx * len(f) + i
        for n in range(N):
            r_n = np.abs(x[n] - antenna_x)
            A[row_idx, n] = (1 / (r_n + 1e-3)) * np.exp(-1j * 2 * k[i] * r_n)

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
chi_est = np.linalg.inv(A.conj().T @ A + lambda_reg * identity) @ A.conj().T @ S_stacked_noisy

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


# -----------------------------------------------------------------------------
# 7) VISUAL COMPARISON
# -----------------------------------------------------------------------------
plt.figure(figsize=(9, 4))
plt.stem(x, chi_true, linefmt="b-", markerfmt="bo", basefmt=" ", label="True chi")
plt.stem(x, np.abs(chi_est), linefmt="r--", markerfmt="rx", basefmt=" ", label="Estimated |chi|")
plt.xlabel("Position (m)")
plt.ylabel("Scattering Strength")
plt.title("Ground Truth vs Reconstruction")
plt.legend()
plt.grid(alpha=0.35)
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# 8) ERROR ANALYSIS
# -----------------------------------------------------------------------------
print_step_header(7, "Error Analysis")
error = np.abs(chi_true - chi_est)
print("Reconstruction error per voxel is |chi_true - chi_est|.")
print(f"Max error: {np.max(error):.6e}")
print(f"Mean error: {np.mean(error):.6e}")

plt.figure(figsize=(9, 3.8))
plt.stem(x, error, linefmt="k-", markerfmt="ko", basefmt=" ")
plt.xlabel("Position (m)")
plt.ylabel("Absolute Error")
plt.title("Reconstruction Error vs Position")
plt.grid(alpha=0.35)
plt.tight_layout()
plt.show()