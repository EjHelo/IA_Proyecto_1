import pandas as pd

def appen():
    df = pd.read_csv("prueba.csv")
    df['predicciones'] = 'predicciones'
    

    for i in df.index:
        df.at[i, 'predicciones'] = "siul"

    df.to_csv("prueba.csv")
