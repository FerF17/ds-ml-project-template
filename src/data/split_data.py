"""
Script para dividir los datos en conjunto de entrenamiento y conjunto de prueba.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

def split_and_save_data(raw_data_path: str, interim_data_path: str):
    """
    INSTRUCCIONES:
    1. Lee el archivo CSV descargado previamente en `raw_data_path` usando pandas.
    2. Separa los datos con `train_test_split()`. Te recomendamos un test_size=0.2 y random_state=42.
    3. (Opcional pero recomendado) Puedes usar `StratifiedShuffleSplit` basado en la variable
       del ingreso medio (median_income) para que la muestra sea representativa.
    4. Guarda los archivos resultantes (ej. train_set.csv y test_set.csv) en la carpeta `interim_data_path`.
    """
    raw_path = Path(raw_data_path)
    interim_path = Path(interim_data_path)
    interim_path.mkdir(parents=True, exist_ok=True)
    
    if not raw_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {raw_path}")
    
    # 1. Cargar los datos
    df = pd.read_csv(raw_path)
    
    # 2. Dividir los datos, estratos para 'median_income'
    # usamos ingreso medio categorizado para el estratificado en 5 categorías
    # creamos una nueva columna 'income_cat' para el estratificado
    df['income_cat'] = pd.cut(df['median_income'],
                              bins=[0, 1.5, 3, 4.5, 6, float('inf')],
                              labels=[1, 2, 3, 4, 5])

    # 3. Estratificado usando StratifiedShuffleSplit
    train_set, test_set = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['income_cat'])
    
    # 4. Limpieza: Eliminar columna temoral 'income_cat'
    for set_ in (train_set, test_set):
        set_.drop('income_cat', axis=1, inplace=True)

    # 5. Guardar los conjuntos de datos
    train_set.to_csv(interim_path / "train_set.csv", index=False)
    test_set.to_csv(interim_path / "test_set.csv", index=False)

    print(f"Datos divididos y guardados en: {interim_path.absolute()}")

if __name__ == "__main__":
    RAW_PATH = "data/raw/housing/housing.csv"
    INTERIM_PATH = "data/interim/"
    split_and_save_data(RAW_PATH, INTERIM_PATH)
    