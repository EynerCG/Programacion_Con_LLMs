import numpy as np
from sklearn.metrics import roc_auc_score

def evaluar_probabilidades(y_true, y_pred_proba):

    auc = roc_auc_score(y_true, y_pred_proba)

    mascara_alta_confianza = y_pred_proba > 0.85

    conteo_alta_confianza = int(np.sum(mascara_alta_confianza))

    return (float(auc), conteo_alta_confianza)


y_true = np.array([0, 1, 0, 1, 1])
y_pred_proba = np.array([0.1, 0.9, 0.2, 0.95, 0.87])

resultado = evaluar_probabilidades(y_true, y_pred_proba)

print(resultado)