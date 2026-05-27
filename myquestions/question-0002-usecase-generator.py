import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def _ground_truth(df, modelo_tipo="linear", test_size=0.25):
    """Función interna para calcular el resultado esperado."""

    X = df.drop(columns=["produccion_oro_kg"])
    y = df["produccion_oro_kg"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    if modelo_tipo == "linear":
        modelo = LinearRegression()
    elif modelo_tipo == "ridge":
        modelo = Ridge(alpha=1.0)
    else:
        raise ValueError("Modelo no válido. Use 'linear' o 'ridge'")

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    metricas = {
        "MAE":  float(mae),
        "RMSE": float(rmse),
        "R2":   float(r2)
    }

    return metricas, modelo.coef_, modelo


def generar_caso_de_uso_produccion():

    n = random.randint(120, 250)

    toneladas    = np.random.uniform(800, 3000, n)
    horas        = np.random.uniform(16, 24, n)
    eficiencia   = np.random.uniform(70, 98, n)
    temperatura  = np.random.uniform(40, 75, n)
    mantenimiento = np.random.randint(1, 6, n)

    produccion = (
        0.0025 * toneladas
        + 0.8  * horas
        + 0.15 * eficiencia
        - 0.03 * temperatura
        + 1.5  * mantenimiento
        + np.random.normal(0, 2, n)
    )

    df = pd.DataFrame({
        "toneladas_procesadas": toneladas,
        "horas_operacion":      horas,
        "eficiencia_planta":    eficiencia,
        "temperatura_operativa": temperatura,
        "nivel_mantenimiento":  mantenimiento,
        "produccion_oro_kg":    produccion
    })

    modelo_tipo = random.choice(["linear", "ridge"])
    test_size   = round(random.uniform(0.2, 0.3), 2)

    # Usa _ground_truth en lugar de predecir_produccion
    # para evitar conflicto con la función del estudiante
    metricas, coeficientes, modelo = _ground_truth(
        df, modelo_tipo=modelo_tipo, test_size=test_size
    )

    input_data = {
        "df":          df.copy(),
        "modelo_tipo": modelo_tipo,
        "test_size":   test_size
    }

    output_data = {
        "metricas":        metricas,
        "coeficientes":    coeficientes,
        "modelo_entrenado": modelo
    }

    return input_data, output_data


if __name__ == "__main__":
    args, output = generar_caso_de_uso_produccion()
    print(output)