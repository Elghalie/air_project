import numpy as np

# Matrice mal conditionnée (type Hilbert)
A = np.array([
    [1.0,     1.0/2.0, 1.0/3.0],
    [1.0/2.0, 1.0/3.0, 1.0/4.0],
    [1.0/3.0, 1.0/4.0, 1.0/5.0]
])

b = np.array([1.0, 0.5, 0.3333])

# -----------------------------
# 1. Calcul du conditionnement
# -----------------------------
kappa_A = np.linalg.cond(A)

print(f"Conditionnement de la matrice kappa(A) = {kappa_A}")

# -----------------------------
# 2. Résolution du système exact
# -----------------------------
x_exact = np.linalg.solve(A, b)

print("\nSolution exacte :")
print(x_exact)

# -----------------------------
# 3. Perturbation de b
# -----------------------------
b_perturbe = b.copy()
b_perturbe[2] += 1e-14

# -----------------------------
# 4. Résolution perturbée
# -----------------------------
x_perturbe = np.linalg.solve(A, b_perturbe)

print("\nSolution perturbée :")
print(x_perturbe)

# -----------------------------
# 5. Erreur relative
# -----------------------------
erreur_relative = (
    np.linalg.norm(x_exact - x_perturbe)
    / np.linalg.norm(x_exact)
)

print(
    f"\nErreur relative induite sur x : "
    f"{erreur_relative:.6e}"
)