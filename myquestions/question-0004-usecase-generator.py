import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def generar_caso_de_uso_preparar_datos_comercializacion_oro():
    """
    Genera un caso de uso aleatorio para la función preparar_datos(df, target_col)

    Retorna:
        input (dict): {"df": DataFrame, "target_col": str}
        output (tuple): (X_procesado, y)
    """

    # -----------------------------
    # 1️⃣ Generar tamaño aleatorio
    # -----------------------------
    n = np.random.randint(8, 20)

    # -----------------------------
    # 2️⃣ Generar datos aleatorios
    # -----------------------------
    data = {
        "precio_internacional_usd": np.random.uniform(1800, 2500, n),
        "tasa_cambio_cop": np.random.uniform(3500, 5000, n),
        "pureza_porcentaje": np.random.uniform(70, 99.9, n),
        "cantidad_gramos": np.random.uniform(5, 500, n),
        "costo_logistico_cop": np.random.uniform(50000, 500000, n),
        "volatilidad_mercado": np.random.uniform(0.1, 5.0, n),
        "margen_ganancia": np.random.uniform(100000, 2000000, n)
    }

    df = pd.DataFrame(data)

    # -----------------------------
    # 3️⃣ Introducir NaNs aleatorios
    # -----------------------------
    for col in df.columns[:-1]:  # evitar modificar target
        if np.random.rand() > 0.5:
            idx = np.random.randint(0, n)
            df.loc[idx, col] = np.nan

    target_col = "margen_ganancia"

    # -----------------------------
    # 4️⃣ Construir input
    # -----------------------------
    input = {
        "df": df.copy(),
        "target_col": target_col
    }

    # -----------------------------
    # 5️⃣ Construir output esperado
    # -----------------------------
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    # Imputación
    imputer = SimpleImputer(strategy="mean")
    X_imputado = imputer.fit_transform(X)

    # Escalado
    scaler = StandardScaler()
    X_escalado = scaler.fit_transform(X_imputado)

    output = (X_escalado, y)

    return input, output

# Vamos a ejecutar el generador y comprobar que funciona correctamente

input_data, output_data = generar_caso_de_uso_preparar_datos_comercializacion_oro()

df_generado = input_data["df"]
target_col = input_data["target_col"]
X_procesado, y = output_data

print("✔ Generador ejecutado correctamente\n")

print("Tamaño del DataFrame original:", df_generado.shape)
print("Columna objetivo:", target_col)

print("\n¿Existen NaNs en el DataFrame original?")
print(df_generado.isna().sum())

print("\nForma de X procesado:", X_procesado.shape)
print("Forma de y:", y.shape)

print("\n¿Existen NaNs en X procesado?")
print(np.isnan(X_procesado).sum())

print("\nPrimeras 3 filas de X procesado:")
print(X_procesado[:3])
