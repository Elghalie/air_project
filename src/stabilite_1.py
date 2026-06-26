import numpy as np


def sommation_naive(tableau):
    somme = 0.0
    for x in tableau:
        somme += x
    return somme


def sommation_kahan(tableau):
    somme = 0.0
    c = 0.0  # compensation des erreurs d'arrondi

    for x in tableau:
        # Valeur corrigée
        y = x - c

        # Nouvelle somme
        t = somme + y

        # Calcul de l'erreur d'arrondi
        c = (t - somme) - y

        # Mise à jour de la somme
        somme = t

    return somme


# -----------------------------
# Vérification expérimentale
# -----------------------------

epsilon_machine = np.finfo(float).eps
print(f"Epsilon Machine : {epsilon_machine}")

# Un grand nombre suivi de nombreuses petites valeurs
valeurs = [1.0] + [epsilon_machine / 2.0] * 10000

print(f"Somme Naive : {sommation_naive(valeurs)}")
print(f"Somme Kahan : {sommation_kahan(valeurs)}")