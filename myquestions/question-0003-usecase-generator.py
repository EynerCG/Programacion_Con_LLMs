
import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, roc_auc_score



def detectar_fallas(df, test_size=0.25):
    
    X = df.drop(columns=["falla"])
    y = df["falla"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    modelo = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    )
    
    modelo.fit(X_train, y_train)
    
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]
    
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    metricas = {
        "Recall": recall,
        "Precision": precision,
        "ROC_AUC": roc_auc
    }
    
    importancia = dict(zip(X.columns, modelo.feature_importances_))
    
    return metricas, importancia, modelo



def generar_caso_de_uso_fallas():
    
    n = random.randint(1000, 2000)
    
    vibracion = np.random.normal(5, 1.5, n)
    temperatura = np.random.normal(75, 8, n)
    presion = np.random.normal(250, 20, n)
    horas = np.random.uniform(0, 5000, n)
    combustible = np.random.uniform(10, 100, n)
    

    prob_falla = (
        0.03 * vibracion +
        0.02 * (temperatura - 70) +
        0.0003 * horas -
        0.01 * combustible
    )
    
    prob_falla = 1 / (1 + np.exp(-prob_falla))  # función sigmoide
    
    # Forzar 5% de fallas aproximadamente
    threshold = np.percentile(prob_falla, 95)
    falla = (prob_falla >= threshold).astype(int)
    
    df = pd.DataFrame({
        "vibracion": vibracion,
        "temperatura_motor": temperatura,
        "presion_hidraulica": presion,
        "horas_uso": horas,
        "nivel_combustible": combustible,
        "falla": falla
    })
    
    test_size = round(random.uniform(0.2, 0.3), 2)
    
    metricas, importancia, modelo = detectar_fallas(df, test_size=test_size)
    
    input_data = {
        "df": df.copy(),
        "test_size": test_size
    }
    
    output_data = {
        "metricas": metricas,
        "importancia_variables": importancia,
        "modelo_entrenado": modelo
    }
    
    return input_data, output_data



input_data, output_data = generar_caso_de_uso_fallas()

print("✔ Generador ejecutado correctamente\n")

print("Tamaño dataset:", input_data["df"].shape)
print("Proporción de fallas:", input_data["df"]["falla"].mean())
print("Test size:", input_data["test_size"])

print("\nMétricas obtenidas:")
print(output_data["metricas"])

print("\nImportancia de variables:")
print(output_data["importancia_variables"])
