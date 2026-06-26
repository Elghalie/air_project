import numpy as np
import time

# Génération d'un système linéaire dense
n = 4000
rng = np.random.RandomState(42)

A = rng.rand(n, n)
b = rng.rand(n)

# ------------------------------------
# Approche A : Inversion explicite
# ------------------------------------
t0 = time.time()

A_inv = np.linalg.inv(A)
x_inv = A_inv @ b

t_inv = time.time() - t0
residu_inv = np.linalg.norm(A @ x_inv - b)

# ------------------------------------
# Approche B : Solveur direct
# ------------------------------------
t0 = time.time()

x_solve = np.linalg.solve(A, b)

t_solve = time.time() - t0
residu_solve = np.linalg.norm(A @ x_solve - b)

# ------------------------------------
# Résultats
# ------------------------------------
print(
    f"Inversion Explicite -> Temps: "
    f"{t_inv:.4f}s | Norme du residu: {residu_inv:e}"
)

print(
    f"Solveur Direct -> Temps: "
    f"{t_solve:.4f}s | Norme du residu: {residu_solve:e}"
)