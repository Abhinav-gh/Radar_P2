# 📡 Microwave Imaging: Detailed Mathematical Comparison of Antenna Configurations

---

# 🧠 Core Model (Applies to All Cases)

We solve the inverse problem:

$$
S = A \chi
$$

Where:

* $S$: measured signal (complex)
* $A$: system matrix
* $\chi$: voxel dielectric contrast (unknown)

---

## 🔹 Forward Model (General Form)

$$
A[i,n] = \frac{1}{r_{in} , r_{jn}} \cdot e^{-j k (r_{in} + r_{jn})}
$$

* $r_{in}$: distance from Tx antenna $i$ to voxel $n$
* $r_{jn}$: distance from voxel $n$ to Rx antenna $j$
* $k = \frac{2\pi f}{c}$

---

## 🔹 Reconstruction

$$
\hat{\chi} = (A^H A + \lambda I)^{-1} A^H S
$$

---

# 🧠 Key Idea

Reconstruction quality depends on:

$$
\text{Independence of columns of } A
$$

Each column = response of one voxel across all measurements.

---

# 🔵 CASE 1: Single Antenna (Monostatic)

## Geometry:

* One antenna at position $x_a$
* Only reflection ($S_{11}$)

---

## Phase Model:

$$
\text{Path length} = r_n + r_n = 2r_n
$$

$$
A[i,n] = \frac{1}{r_n^2} e^{-j 2k_i r_n}
$$

---

## Column Structure:

$$
a_n =
\begin{bmatrix}
e^{-j2k_1 r_n} \
e^{-j2k_2 r_n} \
\vdots
\end{bmatrix}
$$

---

## Column Relationship:

$$
a_m[i] = a_n[i] \cdot e^{-j2k_i (r_m - r_n)}
$$

---

## Interpretation:

* Columns differ by **frequency-dependent phase shift**
* If $\Delta r = r_m - r_n$ is small:

$$
e^{-j2k_i \Delta r} \approx 1
$$

So:

$$
a_m \approx a_n
$$

---

## Result:

* Moderate conditioning (~$10^8$)
* Blurred reconstruction

---

# 🔴 CASE 2: Double Antenna (BAD GEOMETRY)

## Geometry:

* Antennas on same side of object
* $x_1 \approx x_2$

---

## Distance Relation:

$$
r_{2n} \approx r_{1n} + C
$$

---

## Phase:

$$
r_{1n} + r_{2n} = 2r_{1n} + C
$$

$$
A[i,n] = e^{-j k (2r_{1n} + C)} = e^{-j2k r_{1n}} \cdot e^{-jkC}
$$

---

## 🔥 Critical Result:

$$
a_n \approx \text{constant} \cdot e^{-j2k r_{1n}}
$$

---

## Consequence:

* Columns become **linearly dependent**
* $A^H A$ becomes nearly singular

---

## Result:

* Condition number $\sim 10^{19}$
* Worse than single antenna
* Noise amplification

---

# 🟢 CASE 3: Double Antenna (ASYMMETRIC — BEST)

## Geometry:

* Antennas at different positions (e.g., $x_1=0$, $x_2=0.3$)

---

## Phase:

$$
A[i,n] = e^{-j k (r_{1n} + r_{2n})}
$$

---

## Column Relationship:

$$
a_m[i] = e^{-jk[(r_{1m}+r_{2m}) - (r_{1n}+r_{2n})]} \cdot a_n[i]
$$

---

## 🔥 Key Insight:

$$
(r_{1m}+r_{2m}) - (r_{1n}+r_{2n}) \not\propto (r_m - r_n)
$$

---

## Meaning:

* Columns are **not simple phase shifts**
* Each voxel has **unique phase signature**

---

## Result:

* Strong column independence
* Best reconstruction (~60% improvement)

---

# 🟡 CASE 4: Triple Antenna

## Geometry:

* Three antennas (e.g., left, center, right)

---

## Phase:

$$
A[i,n] = e^{-j k (r_{in} + r_{jn})}
$$

(for multiple antenna pairs)

---

## Effect:

* Multiple independent path combinations:

  * reflection
  * transmission
  * cross paths

---

## Result:

* Improved conditioning (~$10^{10}$)
* Good reconstruction
* Slight redundancy

---

# 🧠 PHASE COMPARISON SUMMARY

| Case              | Phase Expression         | Behavior                    |
| ----------------- | ------------------------ | --------------------------- |
| Single antenna    | $2r_n$                   | Round-trip reflection       |
| Double bad        | $2r_{1n} + C$            | Collapses to single antenna |
| Double asymmetric | $r_{1n} + r_{2n}$        | Unique nonlinear variation  |
| Triple antenna    | multiple $r_{in}+r_{jn}$ | Richest diversity           |

---

# 🧠 COLUMN DEPENDENCE SUMMARY

| Case              | Column Independence      |
| ----------------- | ------------------------ |
| Single antenna    | moderate                 |
| Double bad        | very poor                |
| Double asymmetric | strong                   |
| Triple antenna    | strong (some redundancy) |

---

# 🧠 CONDITIONING VS PERFORMANCE

| Case              | Condition Number | Performance |
| ----------------- | ---------------- | ----------- |
| Single            | $\sim 10^8$      | baseline    |
| Double bad        | $\sim 10^{19}$   | worst       |
| Double asymmetric | $\sim 10^{14}$   | best        |
| Triple            | $\sim 10^{10}$   | good        |

---

## 🔥 Important Note:

Condition number alone does NOT determine performance.
Geometry and column independence matter more.

---

# 🧠 FINAL TAKEAWAYS

---

## 🔹 1. Phase Encodes Geometry

$$
\text{Phase} = k \cdot (\text{path length})
$$

---

## 🔹 2. Column Independence is Critical

$$
\text{Good reconstruction} \iff \text{columns of } A \text{ are independent}
$$

---

## 🔹 3. Bad Geometry Causes Rank Collapse

$$
r_{2n} \approx r_{1n} + C
$$

→ redundant measurements

---

## 🔹 4. More Antennas ≠ Better

Only useful if they introduce:

* new paths
* new phase diversity

---

## 🔹 5. Imaging = Designing Phase Diversity

Each voxel must produce a **unique phase response**.

---

# 🏆 ONE-LINE SUMMARY

> Microwave imaging works by encoding spatial information in phase differences across propagation paths; poor geometry collapses this encoding, while good geometry maximizes it.

---
