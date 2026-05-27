import pandas as pd
import numpy as np
import random

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest


# =====================================================
# FUNCIÓN SOLUCIÓN
# =====================================================

def preparar_datos(df,
                   target_col,
                   cols_numericas,
                   cols_categoricas,
                   contaminacion=0.05):

    # ==========================================
    # 1. Separar X e y
    # ==========================================

    X = df[cols_numericas + cols_categoricas].copy()
    y = df[target_col].to_numpy()

    # ==========================================
    # 2. Pipeline numérico
    # ==========================================

    pipeline_numerico = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # ==========================================
    # 3. Pipeline categórico
    # ==========================================

    pipeline_categorico = Pipeline(steps=[
        ('imputer', SimpleImputer(
            strategy='constant',
            fill_value='desconocido'
        )),
        ('onehot', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False
        ))
    ])

    # ==========================================
    # 4. ColumnTransformer
    # ==========================================

    preprocesador = ColumnTransformer(transformers=[
        ('num', pipeline_numerico, cols_numericas),
        ('cat', pipeline_categorico, cols_categoricas)
    ])

    # ==========================================
    # 5. Transformar datos
    # ==========================================

    X_transformado = preprocesador.fit_transform(X)

    # ==========================================
    # 6. IsolationForest
    # ==========================================

    detector_anomalias = IsolationForest(
        contamination=contaminacion,
        random_state=42
    )

    mascara = detector_anomalias.fit_predict(X_transformado)

    # ==========================================
    # 7. Conservar filas normales (+1)
    # ==========================================

    filas_normales = (mascara == 1)

    X_limpio = X_transformado[filas_normales]
    y_limpio = y[filas_normales]

    # ==========================================
    # 8. Retornar resultado
    # ==========================================

    return X_limpio, y_limpio


# =====================================================
# GENERADOR DE CASOS DE USO
# =====================================================

_TIPOS_EMPLEO = [
    'formal',
    'independiente',
    'desempleado'
]

_NIVELES_ESTUDIO = [
    'primaria',
    'secundaria',
    'universitario',
    'posgrado'
]


def generar_caso_de_uso_0003():

    n_rows = random.randint(100, 200)

    contaminacion = round(
        random.uniform(0.05, 0.1),
        3
    )

    todas_num = [
        'ingresos',
        'deuda',
        'score_crediticio',
        'edad'
    ]

    cols_numericas = random.sample(
        todas_num,
        k=3
    )

    cols_categoricas = [
        'tipo_empleo',
        'nivel_estudio'
    ]

    data = {}

    # ==========================================
    # Generar columnas numéricas
    # ==========================================

    for col in cols_numericas:

        valores = np.random.uniform(
            1000,
            10000,
            n_rows
        )

        mascara_nan = np.random.choice(
            [True, False],
            size=n_rows,
            p=[0.1, 0.9]
        )

        valores[mascara_nan] = np.nan

        data[col] = valores

    # ==========================================
    # Generar columnas categóricas
    # ==========================================

    data['tipo_empleo'] = np.random.choice(
        _TIPOS_EMPLEO,
        size=n_rows
    )

    data['nivel_estudio'] = np.random.choice(
        _NIVELES_ESTUDIO,
        size=n_rows
    )

    # ==========================================
    # Target
    # ==========================================

    target_col = 'riesgo_impago'

    data[target_col] = np.random.randint(
        0,
        2,
        size=n_rows
    )

    df = pd.DataFrame(data)

    # ==========================================
    # Ground Truth
    # ==========================================

    X_raw = df[
        cols_numericas + cols_categoricas
    ].copy()

    y_raw = df[target_col].to_numpy()

    p_num = Pipeline([
        ('i', SimpleImputer(strategy='median')),
        ('s', StandardScaler())
    ])

    p_cat = Pipeline([
        ('i', SimpleImputer(
            strategy='constant',
            fill_value='desconocido'
        )),
        ('o', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False
        ))
    ])

    ct = ColumnTransformer([
        ('n', p_num, cols_numericas),
        ('c', p_cat, cols_categoricas)
    ])

    X_p = ct.fit_transform(X_raw)

    mask = IsolationForest(
        contamination=contaminacion,
        random_state=42
    ).fit_predict(X_p)

    X_exp = X_p[mask == 1]
    y_exp = y_raw[mask == 1]

    input_data = {
        'df': df.copy(),
        'target_col': target_col,
        'cols_numericas': cols_numericas,
        'cols_categoricas': cols_categoricas,
        'contaminacion': contaminacion
    }

    expected_output = (
        X_exp,
        y_exp
    )

    return input_data, expected_output


# =====================================================
# TEST AUTOMÁTICO
# =====================================================

if __name__ == "__main__":

    print("=" * 60)
    print("🏦 TEST SISTEMA BANCARIO")
    print("=" * 60)

    # ==========================================
    # Generar caso aleatorio
    # ==========================================

    entrada, salida_esperada = generar_caso_de_uso_0003()

    # ==========================================
    # Ejecutar solución
    # ==========================================

    X_res, y_res = preparar_datos(**entrada)

    # ==========================================
    # Obtener Ground Truth
    # ==========================================

    X_gt, y_gt = salida_esperada

    # ==========================================
    # Validaciones
    # ==========================================

    x_correcto = np.allclose(
        X_res,
        X_gt,
        atol=1e-8
    )

    y_correcto = np.array_equal(
        y_res,
        y_gt
    )

    # ==========================================
    # Resultados
    # ==========================================

    print(f"\nRegistros iniciales: {entrada['df'].shape[0]}")
    print(f"Registros finales: {X_res.shape[0]}")
    print(f"Features finales: {X_res.shape[1]}")

    print("\nComparación:")

    print("X coincide:", x_correcto)
    print("y coincide:", y_correcto)

