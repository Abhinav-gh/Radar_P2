import numpy as np
import matplotlib.pyplot as plt

# Keep array printing compact but readable for educational console output.
np.set_printoptions(precision=4, suppress=True)


def print_step_header(step_id: int, title: str) -> None:
    print("\n" + "=" * 72)
    print(f"Step {step_id}: {title}")
    print("=" * 72)


# -----------------------------------------------------------------------------
# 1) SETUP
# -----------------------------------------------------------------------------
c = 3e8
f = np.linspace(1e9, 5e9, 200)  # frequency sweep
k = 2 * np.pi * f / c

N = 20
x = np.linspace(0.1, 1.0, N)  # 1D voxel positions (m)

chi_true = np.zeros(N, dtype=float)
chi_true[5] = 1.0
chi_true[12] = 0.8
chi_true[16] = 0.6
object_indices = np.where(chi_true > 0)[0]

print_step_header(1, "Setup")
print("We discretize the scene into voxels and assign object reflectivity chi.")
print(f"Number of voxels: {N}")
print("Voxel positions x (m):")
print(np.array2string(x, separator=", "))
print(f"Object voxel indices: {object_indices.tolist()}")

plt.figure(figsize=(9, 3.5))
plt.plot(x, np.zeros_like(x), "bo", label="All voxels")
plt.plot(x[object_indices], np.zeros_like(object_indices, dtype=float), "r^", markersize=10, label="Active scatterers")
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
print("For each frequency, we compute S(f) = sum over voxels of chi_n * exp(-j 2k r_n)")

S = np.zeros(len(f), dtype=complex)

for i in range(len(f)):
    for n in range(N):
        S[i] += chi_true[n] * np.exp(-1j * 2 * k[i] * x[n])

# Detailed walkthrough for only the first 2 frequencies.
for i in range(2):
    print("\n" + "-" * 72)
    print(f"Frequency index {i}: f = {f[i] / 1e9:.4f} GHz, k = {k[i]:.6f} rad/m")
    running_sum = 0.0 + 0.0j

    for n in range(N):
        r_n = x[n]
        phase_term = np.exp(-1j * 2 * k[i] * r_n)
        contribution = chi_true[n] * phase_term
        running_sum += contribution

        print(
            f"  voxel n={n:02d} | r_n={r_n:.4f} m | "
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

plt.figure(figsize=(8.5, 4))
plt.plot(f / 1e9, np.abs(S), linewidth=2)
plt.xlabel("Frequency (GHz)")
plt.ylabel("|S(f)|")
plt.title("Measured Signal Magnitude vs Frequency")
plt.grid(alpha=0.35)
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# 4) MATRIX MODEL
# -----------------------------------------------------------------------------
print_step_header(4, "Matrix Model A")
print("Matrix A captures how each voxel affects each frequency.")
print("A matrix maps object -> measurements")
print("A[i,n] = exp(-j 2 k_i x_n)")

A = np.zeros((len(f), N), dtype=complex)
for i in range(len(f)):
    for n in range(N):
        A[i, n] = np.exp(-1j * 2 * k[i] * x[n])

print(f"Shape of A: {A.shape}")
print("First 3 rows of A:")
print(A[:3, :])

plt.figure(figsize=(8, 4))
plt.imshow(np.abs(A), aspect="auto", origin="lower", cmap="viridis")
plt.colorbar(label="|A[i,n]|")
plt.xlabel("Voxel index n")
plt.ylabel("Frequency index i")
plt.title("System Matrix A (Magnitude Heatmap)")
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# 5) INVERSE PROBLEM STATEMENT
# -----------------------------------------------------------------------------
print_step_header(5, "Inverse Problem")
print("We solve S = A * chi using least squares (pseudo-inverse)")
print("There are more equations than unknowns, so we compute a best-fit solution.")


# -----------------------------------------------------------------------------
# 6) SOLUTION
# -----------------------------------------------------------------------------
print_step_header(6, "Reconstruction")
A_pinv = np.linalg.pinv(A)
cond_A = np.linalg.cond(A)
chi_est = A_pinv @ S

print(f"Shape of pseudo-inverse pinv(A): {A_pinv.shape}")
print(f"Condition number of A: {cond_A:.4e}")
print("First few values of chi_est:")
for i in range(min(8, len(chi_est))):
    print(f"  chi_est[{i}] = {chi_est[i].real:+.6f}{chi_est[i].imag:+.6f}j")


# -----------------------------------------------------------------------------
# 7) VISUAL COMPARISON
# -----------------------------------------------------------------------------
plt.figure(figsize=(9, 4))
plt.stem(x, chi_true, linefmt="b-", markerfmt="bo", basefmt=" ", label="True chi")
plt.stem(x, np.real(chi_est), linefmt="r--", markerfmt="rx", basefmt=" ", label="Estimated chi (real part)")
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