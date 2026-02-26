import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def generar_caso_de_uso_clasificar_ley_oro():
    """
    Genera un caso de uso aleatorio para la función
    clasificar_ley_oro(df, test_size).

    Returns
    -------
    tuple
        (input_data, output_data)
    """

    # ---------------------------------------------------------
    # 1. Generar dataset aleatorio
    # ---------------------------------------------------------
    n = random.randint(80, 200)

    df = pd.DataFrame({
        "profundidad": np.random.uniform(100, 800, n),
        "concentracion_arsenico": np.random.uniform(0, 500, n),
        "concentracion_cobre": np.random.uniform(0, 300, n),
        "densidad_roca": np.random.uniform(2.0, 3.5, n),
        "humedad": np.random.uniform(0, 15, n),
        "ley_oro": np.random.uniform(0.5, 5.0, n)
    })

    test_size = round(random.uniform(0.2, 0.35), 2)

    # ---------------------------------------------------------
    # 2. Lógica esperada de clasificar_ley_oro (Ground Truth)
    # ---------------------------------------------------------
    df_proc = df.copy()
    df_proc["alta_ley"] = (df_proc["ley_oro"] >= 2.5).astype(int)

    X = df_proc.drop(columns=["ley_oro", "alta_ley"])
    y = df_proc["alta_ley"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train_scaled, y_train)

    y_pred = modelo.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    matriz_conf = confusion_matrix(y_test, y_pred)

    # ---------------------------------------------------------
    # 3. Construcción de input y output
    # ---------------------------------------------------------
    input_data = {
        "df": df.copy(),
        "test_size": test_size
    }

    output_data = {
        "accuracy": accuracy,
        "f1_score": f1,
        "matriz_confusion": matriz_conf,
        "modelo_entrenado": modelo
    }

    return input_data, output_data