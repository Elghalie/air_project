"""
Module de prétraitement des données
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any


def clean_data(df: pd.DataFrame,
               config: Dict[str, Any]) -> pd.DataFrame:
    """
    Nettoyage des données :
    - Imputation des valeurs manquantes par la médiane
    - Traitement des outliers par la méthode IQR
    """

    df_clean = df.copy()

    handle_missing = config["preprocessing"]["handle_missing"]

    # ===============================
    # 1. Gestion des valeurs manquantes
    # ===============================
    for col in df_clean.columns:

        if df_clean[col].isnull().any():

            if handle_missing == "median":

                # Calcul de la médiane
                median = df_clean[col].median()

                # Remplacement des NaN par la médiane
                df_clean[col] = df_clean[col].fillna(median)

    # ===============================
    # 2. Gestion des valeurs aberrantes
    # ===============================
    threshold = config["preprocessing"]["outlier_threshold"]

    numeric_cols = df_clean.select_dtypes(
        include=[np.number]
    ).columns

    for col in numeric_cols:

        # On ne traite pas la variable cible
        if col != config["data"]["target_column"]:

            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)

            IQR = Q3 - Q1

            # Bornes IQR
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR

            # Clipping des valeurs aberrantes
            df_clean[col] = df_clean[col].clip(
                lower=lower,
                upper=upper
            )

    return df_clean


def split_and_scale(
        df: pd.DataFrame,
        config: Dict[str, Any]
) -> Tuple[np.ndarray,
           np.ndarray,
           pd.Series,
           pd.Series,
           StandardScaler]:
    """
    Séparation Train/Test et normalisation.
    """

    target = config["data"]["target_column"]

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["project"]["random_seed"]
    )

    scaler = StandardScaler()

    if config["preprocessing"]["scale_features"]:

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    )