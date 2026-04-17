"""
Script para descargar y extraer los datos originales del proyecto.
"""

import os
import urllib.request
import tarfile
from pathlib import Path

def fetch_housing_data(housing_url: str, housing_path: str):
    """
    INSTRUCCIONES:
    1. Asegúrate de que el directorio `housing_path` exista (usa os.makedirs o Path.mkdir).
    2. Usa urllib.request.urlretrieve para descargar el archivo .tgz desde `housing_url`.
    3. Usa tarfile.open para extraer el contenido en `housing_path`.
    
    URL de los datos: "https://github.com/ageron/data/raw/main/housing.tgz"
    Ruta de destino recomendada: "data/raw/"
    """
    path = Path(housing_path)
    path.mkdir(parents=True, exist_ok=True)
    tgz_path = path / "housing.tgz"
    
    try:
        urllib.request.urlretrieve(housing_url, tgz_path)
        with tarfile.open(tgz_path) as housing_tgz:
            housing_tgz.extractall(path=path)
        print(f"Datos descargados y extraídos en: {path.absolute()}")
        
    except Exception as e:
        print(f"Error al descargar o extraer los datos: {e}")
    finally:
        if tgz_path.exists():
            os.remove(tgz_path)
        
if __name__ == "__main__":
    housing_url = "https://github.com/ageron/data/raw/main/housing.tgz"
    housing_path = "data/raw/"
    fetch_housing_data(housing_url, housing_path)
