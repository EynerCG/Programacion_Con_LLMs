import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# =====================================================
# FUNCIÓN SOLUCIÓN
# =====================================================

def entrenar_modelo_viviendas(df, target):

    # ==========================================
    # 1. Separar variables predictoras y target
    # ==========================================

    X = df.drop(columns=[target])
    y = df[target]

    # ==========================================
    # 2. División train/test
    # ==========================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ==========================================
    # 3. Escalado de variables
    # ==========================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================
    # 4. Entrenamiento Ridge
    # ==========================================

    modelo = Ridge()

    modelo.fit(X_train_scaled, y_train)

    # ==========================================
    # 5. Predicciones
    # ==========================================

    y_pred = modelo.predict(X_test_scaled)

    # ==========================================
    # 6. Calcular MSE
    # ==========================================

    mse = mean_squared_error(y_test, y_pred)

    # ==========================================
    # 7. Retornar resultados
    # ==========================================

    return (float(mse), modelo)


# =====================================================
# GENERADOR DE CASOS DE USO
# =====================================================

def generar_caso_de_uso_entrenar_modelo_viviendas():

    # ==========================================
    # 1. Parámetros aleatorios
    # ==========================================

    n_rows = np.random.randint(50, 201)
    n_features = np.random.randint(3, 9)

    target_name = "precio"

    # ==========================================
    # 2. Crear features
    # ==========================================

    cols = [f"feature_{i}" for i in range(n_features)]

    X_values = np.random.rand(
        n_rows,
        n_features
    ) * 100

    df = pd.DataFrame(
        X_values,
        columns=cols
    )

    # ==========================================
    # 3. Crear target
    # ==========================================

    weights = np.random.uniform(
        100,
        1000,
        size=n_features
    )

    bias = 50000

    ruido = np.random.normal(
        0,
        5000,
        size=n_rows
    )

    df[target_name] = (
        X_values.dot(weights)
        + bias
        + ruido
    )

    # ==========================================
    # 4. Ground Truth
    # ==========================================

    X = df.drop(columns=[target_name])
    y = df[target_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    modelo = Ridge()

    modelo.fit(X_train_scaled, y_train)

    y_pred = modelo.predict(X_test_scaled)

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    # ==========================================
    # 5. Retorno
    # ==========================================

    input_data = {
        "df": df,
        "target": target_name
    }

    output_data = (
        mse,
        modelo
    )

    return input_data, output_data


# =====================================================
# TEST AUTOMÁTICO
# =====================================================

if __name__ == "__main__":

    print("=" * 60)
    print("🏠 TEST PREDICCIÓN DE VIVIENDAS")
    print("=" * 60)

    # ==========================================
    # Generar caso aleatorio
    # ==========================================

    input_data, output_data = generar_caso_de_uso_entrenar_modelo_viviendas()

    # ==========================================
    # Ejecutar solución
    # ==========================================

    mse_res, modelo_res = entrenar_modelo_viviendas(
        **input_data
    )

    # ==========================================
    # Obtener Ground Truth
    # ==========================================

    mse_gt, modelo_gt = output_data

    # ==========================================
    # Validaciones
    # ==========================================

    mse_correcto = np.isclose(
        mse_res,
        mse_gt
    )

    coef_correctos = np.allclose(
        modelo_res.coef_,
        modelo_gt.coef_
    )

    # ==========================================
    # Resultados
    # ==========================================

    print(f"\nFilas del DataFrame: {len(input_data['df'])}")

    print(f"Columnas: {list(input_data['df'].columns)}")

    print(f"\nMSE obtenido: {mse_res:.4f}")

    print(f"\nTipo de modelo:")
    print(type(modelo_res))

    print(f"\nCoeficientes del modelo:")
    print(modelo_res.coef_)

    print("\nValidaciones:")

    print("MSE correcto:", mse_correcto)

    print("Coeficientes correctos:", coef_correctos)
