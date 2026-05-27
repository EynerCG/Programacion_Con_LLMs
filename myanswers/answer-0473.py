import pandas as pd
import numpy as np
import random

def limpiar_ventas(df):
    
    df_limpio = df.drop_duplicates()
    
    porcentaje_nulos = df_limpio.isnull().mean()
    
    columnas_validas = porcentaje_nulos[porcentaje_nulos <= 0.5].index
    
    df_limpio = df_limpio[columnas_validas]
    
    return df_limpio

def generar_caso_de_uso_limpiar_ventas():
    
    n_rows = random.randint(8, 20)
    n_cols = random.randint(3, 6)

    data = np.random.randn(n_rows, n_cols)
    cols = [f'col_{i}' for i in range(n_cols)]

    df = pd.DataFrame(data, columns=cols)

    for col in df.columns:
        if random.random() < 0.5:
            df.loc[df.sample(frac=0.6).index, col] = np.nan

    df = pd.concat([df, df.iloc[:2]], ignore_index=True)

    input_data = {'df': df.copy()}

    df_clean = limpiar_ventas(df)

    return input_data, df_clean

entrada, resultado = generar_caso_de_uso_limpiar_ventas()

print("DATAFRAME ORIGINAL:")
print(entrada['df'])

print("\nDATAFRAME LIMPIO:")
print(resultado)