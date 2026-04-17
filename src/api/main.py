"""
API Básica usando FastAPI para servir el modelo entrenado.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
import sys

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS ABSOLUTAS
# ==========================================
# Garantizamos que la API encuentre los archivos sin importar desde dónde se ejecute
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_random_forest_model.pkl"
ARTIFACTS_PATH = MODELS_DIR / "preprocessing_artifacts.pkl"

# Conectar con src/features para importar tu pipeline de preprocesamiento
FEATURES_DIR = BASE_DIR / "src" / "features"
sys.path.append(str(FEATURES_DIR))
from build_features import preprocess_pipeline

# Inicializamos la app
app = FastAPI(
    title="API de Predicción de Precios de Vivienda (California)", 
    version="1.0",
    description="Predice el valor medio de una casa basado en características geográficas y demográficas."
)

# ==========================================
# 2. ESQUEMA DE ENTRADA (Datos Crudos)
# ==========================================
# Pedimos exactamente lo que un usuario llenaría en un formulario web (sin procesar)
class HousingFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str  # Variable categórica vital para tu pipeline

# Variables globales para cargar el modelo y los artefactos de limpieza
model = None
artifacts = None

# ==========================================
# 3. CARGA EN MEMORIA AL INICIAR EL SERVIDOR
# ==========================================
@app.on_event("startup")
def load_model():
    """
    Carga el modelo globalmente y los artefactos al iniciar el servidor.
    """
    global model, artifacts
    try:
        model = joblib.load(MODEL_PATH)
        artifacts = joblib.load(ARTIFACTS_PATH)
        print("[OK] Modelo Random Forest y Artefactos cargados exitosamente.")
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo o artefactos: {e}")
        print("Asegúrate de haber corrido train_model.py primero.")

# ==========================================
# 4. ENDPOINTS (Rutas de la API)
# ==========================================
@app.get("/")
def home():
    return {"mensaje": "Bienvenido a la API del Proyecto Final de Ciencia de Datos. Ve a /docs para probarla."}

@app.post("/predict")
def predict_price(features: HousingFeatures):
    """
    Recibe las características crudas, las preprocesa y retorna el precio predicho.
    """
    if model is None or artifacts is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible en este momento.")
    
    try:
        # 1. Convertir el objeto Pydantic a un DataFrame de Pandas (1 sola fila)
        # model_dump() es la forma moderna en Pydantic v2 (si falla usa .dict())
        df_raw = pd.DataFrame([features.model_dump() if hasattr(features, 'model_dump') else features.dict()])
        
        # 2. Aplicar el pipeline de preprocesamiento (usamos is_train=False)
        X_processed, _ = preprocess_pipeline(
            df=df_raw, 
            is_train=False, 
            artifacts=artifacts
        )
        
        # Por seguridad, si el pipeline dejara la columna objetivo vacía, la quitamos
        if "median_house_value" in X_processed.columns:
            X_processed = X_processed.drop("median_house_value", axis=1)
            
        # 3. Hacer la predicción (extraemos el primer elemento del array [0])
        prediction = model.predict(X_processed)[0]
        
        # 4. Retornar el resultado
        return {
            "predicted_price": float(prediction),
            "currency": "USD"
        }
        
    except Exception as e:
        # Capturamos cualquier error en el procesamiento y lo mostramos
        raise HTTPException(status_code=400, detail=f"Error procesando la solicitud: {str(e)}")

# Instrucciones para correr la API localmente:
# Abre tu terminal en la raíz de tu proyecto y ejecuta:
# uvicorn src.api.main:app --reload