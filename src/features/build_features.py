"""
Módulo para limpieza, enriquecimiento (Feature Engineering) y escalado.
Diseñado para evitar Data Leakage y ser utilizado en entornos de entrenamiento y producción.
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================
def _calcular_distancia_haversine(lat1, lon1, lat2, lon2):
    """Calcula la distancia en km entre dos coordenadas geográficas."""
    R = 6371.0 # Radio de la Tierra en km
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# ==========================================
# PIPELINE PRINCIPAL
# ==========================================

def clean_data(df: pd.DataFrame, is_train: bool = True, imputer: KNNImputer = None) -> tuple:
    """
    1. Filtra anomalías lógicas (solo en entrenamiento).
    2. Imputa valores faltantes usando KNN.
    Retorna el DataFrame limpio y el objeto imputer.
    """
    df_clean = df.copy()
    
    # 1. HARD CLEANING (Solo aplicable en entrenamiento)
    # En producción (is_train=False), no eliminamos filas porque el cliente necesita
    # una predicción para esa fila aunque sea un caso extremo.
    if is_train and 'median_house_value' in df_clean.columns:
        filtro_censura = df_clean['median_house_value'] < 500001.0
        filtro_fantasmas = df_clean['population'] >= df_clean['households']
        filtro_hacinamiento = (df_clean['population'] / df_clean['households']) <= 20.0
        df_clean = df_clean[filtro_censura & filtro_fantasmas & filtro_hacinamiento].reset_index(drop=True)
    
    # 2. IMPUTACIÓN DE FALTANTES (KNN)
    # Separamos columnas que el KNN no debe usar
    cols_to_exclude = ['ocean_proximity']
    if 'median_house_value' in df_clean.columns:
        cols_to_exclude.append('median_house_value')
        
    features_to_impute = df_clean.drop(columns=cols_to_exclude, errors='ignore')
    col_names = features_to_impute.columns
    
    if is_train:
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        imputed_data = imputer.fit_transform(features_to_impute)
    else:
        if imputer is None:
            raise ValueError("Debes proveer un 'imputer' ajustado cuando is_train=False")
        imputed_data = imputer.transform(features_to_impute)
        
    df_features = pd.DataFrame(imputed_data, columns=col_names, index=df_clean.index)
    
    # Reconstruimos el DataFrame uniendo las columnas excluidas
    df_clean = pd.concat([df_features, df_clean[cols_to_exclude]], axis=1)
    
    return df_clean, imputer

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega nuevas variables estructurales y espaciales.
    """
    df_feat = df.copy()
    
    # Ratios (añadimos epsilon 1e-6 para evitar DivisionByZero en datos nuevos anómalos)
    df_feat['rooms_per_household'] = df_feat['total_rooms'] / (df_feat['households'] + 1e-6)
    df_feat['bedrooms_per_room'] = df_feat['total_bedrooms'] / (df_feat['total_rooms'] + 1e-6)
    df_feat['population_per_household'] = df_feat['population'] / (df_feat['households'] + 1e-6)
    df_feat['rooms_per_population'] = df_feat['total_rooms'] / (df_feat['population'] + 1e-6)
    
    # Distancias Geográficas (Hubs económicos)
    LA_COORDS = (34.0522, -118.2437)
    SF_COORDS = (37.7749, -122.4194)
    
    df_feat['dist_to_LA'] = _calcular_distancia_haversine(
        df_feat['latitude'], df_feat['longitude'], LA_COORDS[0], LA_COORDS[1]
    )
    df_feat['dist_to_SF'] = _calcular_distancia_haversine(
        df_feat['latitude'], df_feat['longitude'], SF_COORDS[0], SF_COORDS[1]
    )
    
    return df_feat

def encode_and_scale(df: pd.DataFrame, is_train: bool = True, scaler: StandardScaler = None, train_columns: list = None) -> tuple:
    """
    Aplica One-Hot Encoding (evitando colinealidad) y estandariza (Z-score).
    """
    df_proc = df.copy()
    
    # 1. ONE-HOT ENCODING
    df_proc = pd.get_dummies(df_proc, columns=['ocean_proximity'], prefix='ocean', drop_first=True, dtype=int)
    
    # Alineación de columnas (CRÍTICO en producción): 
    # Garantiza que el test set tenga las mismas columnas que el train set, en el mismo orden.
    if is_train:
        train_columns = df_proc.columns.tolist()
    else:
        # Rellenar con 0 las columnas (categorías) que faltan en los nuevos datos
        for col in train_columns:
            if col not in df_proc.columns:
                df_proc[col] = 0
        df_proc = df_proc[train_columns] # Reordenar

    # 2. ESCALADO (StandardScaler)
    # Ignorar target y dummies
    cols_to_scale = [col for col in df_proc.columns if 'ocean_' not in col and col != 'median_house_value']
    
    if is_train:
        scaler = StandardScaler()
        df_proc[cols_to_scale] = scaler.fit_transform(df_proc[cols_to_scale])
    else:
        if scaler is None:
            raise ValueError("Debes proveer un 'scaler' ajustado cuando is_train=False")
        df_proc[cols_to_scale] = scaler.transform(df_proc[cols_to_scale])
        
    return df_proc, scaler, train_columns

def preprocess_pipeline(df: pd.DataFrame, is_train: bool = True, artifacts: dict = None) -> tuple:
    """
    Función orquestadora.
    artifacts: Diccionario conteniendo 'imputer', 'scaler', y 'train_columns'.
    Retorna (df_procesado, nuevos_artifacts)
    """
    if artifacts is None:
        artifacts = {'imputer': None, 'scaler': None, 'train_columns': None}
        
    # 1. Limpieza e Imputación
    df_clean, imputer = clean_data(df, is_train, artifacts['imputer'])
    
    # 2. Enriquecimiento
    df_featured = create_features(df_clean)
    
    # 3. Codificación y Escalado
    df_final, scaler, train_columns = encode_and_scale(
        df_featured, is_train, artifacts['scaler'], artifacts['train_columns']
    )
    
    # Guardamos los objetos entrenados
    nuevos_artifacts = {
        'imputer': imputer,
        'scaler': scaler,
        'train_columns': train_columns
    }
    
    return df_final, nuevos_artifacts

if __name__ == "__main__":
    print("Iniciando prueba del módulo de feature engineering...")
    
    # Simulamos datos
    df_dummy_train = pd.DataFrame({
        'longitude': [-122.23, -118.25], 'latitude': [37.88, 34.05],
        'housing_median_age': [41.0, 21.0], 'total_rooms': [880.0, 1000.0],
        'total_bedrooms': [129.0, np.nan], 'population': [322.0, 500.0],
        'households': [126.0, 200.0], 'median_income': [8.32, 4.0],
        'median_house_value': [452600.0, 200000.0], 'ocean_proximity': ['NEAR BAY', 'INLAND']
    })
    
    # 1. Ajuste con datos de entrenamiento
    df_train_procesado, artefactos_guardados = preprocess_pipeline(df_dummy_train, is_train=True)
    print("\nDataset de entrenamiento procesado:")
    print(df_train_procesado.head())
    
    # 2. Inferencia con nuevos datos (ej. df_test o producción)
    df_dummy_test = pd.DataFrame({ # Simulación de un dato nuevo sin target y con nulo
        'longitude': [-120.00], 'latitude': [36.00], 'housing_median_age': [30.0], 
        'total_rooms': [500.0], 'total_bedrooms': [np.nan], 'population': [200.0], 
        'households': [100.0], 'median_income': [3.0], 'ocean_proximity': ['<1H OCEAN']
    })
    
    df_test_procesado, _ = preprocess_pipeline(df_dummy_test, is_train=False, artifacts=artefactos_guardados)
    print("\nNuevo dato (producción/test) procesado usando los artefactos entrenados:")
    print(df_test_procesado.head())