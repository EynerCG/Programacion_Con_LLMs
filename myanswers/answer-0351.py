import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest


def preparar_datos(
    df,
    target_col,
    cols_numericas,
    cols_categoricas,
    contaminacion=0.05
):

    X = df[
        cols_numericas + cols_categoricas
    ].copy()

    y = df[target_col].to_numpy()

    rama_numerica = Pipeline(steps=[
        (
            'imputer',
            SimpleImputer(strategy='median')
        ),
        (
            'scaler',
            StandardScaler()
        )
    ])

    rama_categorica = Pipeline(steps=[
        (
            'imputer',
            SimpleImputer(
                strategy='constant',
                fill_value='desconocido'
            )
        ),
        (
            'onehot',
            OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False
            )
        )
    ])

    preprocesador = ColumnTransformer(
        transformers=[
            (
                'numerica',
                rama_numerica,
                cols_numericas
            ),
            (
                'categorica',
                rama_categorica,
                cols_categoricas
            )
        ]
    )

    X_prep = preprocesador.fit_transform(X)

    iso = IsolationForest(
        contamination=contaminacion,
        random_state=42
    )

    mascara = iso.fit_predict(X_prep)

    filas_normales = (mascara == 1)

    X_limpio = X_prep[filas_normales]

    y_limpio = y[filas_normales]

    return X_limpio, y_limpio