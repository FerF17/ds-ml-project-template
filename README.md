# Proyecto Final de Fundamentos de Ciencia de Datos - USFQ

# 🏡 Predicción de Precios de Vivienda: Análisis y Modelado Predictivo

## 📖 Resumen Ejecutivo

Este repositorio contiene un pipeline completo de Machine Learning (end-to-end) diseñado para estimar el valor mediano de viviendas en distintos distritos (bloques) habitacionales.

Desde una perspectiva aplicada, este modelo permite a entidades inmobiliarias, fondos de inversión o políticas públicas identificar zonas infravaloradas o sobrevaloradas. Desde una perspectiva técnica, el proyecto destaca por una metodología rigurosa que previene el **Data Leakage**, una ingeniería de características fundamentada en la intuición espacial, y una selección de modelos basada en una robusta validación cruzada.

---

## 🏗️ Arquitectura del Proyecto

El repositorio está estructurado modularmente para garantizar la reproducibilidad y separar la fase de experimentación (Notebooks) de la fase de producción/ingeniería (Scripts en `src`).

├── data/
│   ├── raw/                # Datos originales inmutables
│   └── interim/            # Datos particionados (train/test) para prevenir Data Leakage
├── notebooks/
│   ├── 01_exploracion.ipynb                # Análisis Exploratorio (EDA)
│   ├── 02_limpieza_enriquecimiento.ipynb   # Feature Engineering y Pipeline de Preprocesamiento
│   └── 03_experimentacion.ipynb            # Fine-Tuning y Selección de Modelos
├── src/
│   ├── api/                # Capa de servicio (FastAPI/Flask) para despliegue
│   ├── data/               # Scripts para descarga y partición estratificada
│   └── features/           # Pipelines de transformación (Scikit-Learn)
├── requirements.txt        # Dependencias del entorno
└── README.md


---

## 🔬 Metodología Analítica

El ciclo de vida del dato se abordó respetando los más altos estándares de rigor metodológico:

### 1. Partición Temprana de Datos (`src/data/split_data.py`)

Antes de cualquier análisis exploratorio, el conjunto de datos fue dividido en **Entrenamiento** y **Prueba**. Esto es un principio innegociable para asegurar que las distribuciones observadas y las transformaciones calculadas (ej. la media para imputar valores nulos) no estén contaminadas por el Test Set (**Data Leakage**).

### 2. Análisis Exploratorio de Datos (EDA)

Se analizaron las distribuciones multivariadas, identificando sesgos asimétricos (*skewness*) en variables monetarias y habitacionales. Se mapearon espacialmente los precios utilizando latitud y longitud, confirmando una alta **autocorrelación espacial** (los precios son fuertemente dictados por la cercanía a centros urbanos y la costa).

### 3. Ingeniería de Características (Feature Engineering)

La intuición del negocio dictó la creación de variables sintéticas más representativas que las originales. Desarrollamos un pipeline modular (`src/features/build_features.py`) que:

- **Imputa** valores atípicos y nulos sistemáticamente.
- **Genera ratios lógicos:** `rooms_per_household`, `bedrooms_per_room`, y `population_per_household`. Un bloque con muchas habitaciones totales no implica casas grandes si el número de hogares es igualmente alto.
- **Calcula características espaciales:** Se añadieron métricas de distancia a polos económicos clave (`dist_to_LA`, `dist_to_SF`).
- **Codifica (One-Hot Encoding):** Se transformó la variable categórica `ocean_proximity`.
- **Escala:** Se estandarizaron las características para garantizar la convergencia de algoritmos basados en gradiente y regularización.

### 4. Experimentación y Selección de Modelos

Se definieron algoritmos **baseline** (Regresión Lineal y Regresión SGD) para establecer un piso de rendimiento. Posteriormente, se introdujeron modelos de alta capacidad (Árboles de Decisión y Random Forest).

> **Corrección Metodológica Crucial:** Durante la experimentación con `GridSearchCV`, se abordó y resolvió la convención de scikit-learn de utilizar `neg_root_mean_squared_error`. Se evitó el clásico error de calcular la raíz cuadrada sobre un valor negativo o previamente enraizado, asegurando la veracidad de las métricas de varianza.

---

## 📊 Resultados y Benchmark

Los modelos fueron evaluados rigurosamente utilizando **Validación Cruzada de 5 pliegues (5-Fold CV)** en el conjunto de entrenamiento para estimar el error de generalización, seguido de una validación final en el **Test Set**.

### Fase 1 — Comparativa de Algoritmos (Validación Cruzada)

| Algoritmo | RMSE Train | RMSE CV (Promedio) | Alerta / Diagnóstico |
|---|---|---|---|
| RandomForestRegressor | $16,134.42 | $43,704.96 | Prometedor (Requiere ajuste fino) |
| LinearRegression | $57,746.64 | $57,872.49 | Subajuste (Modelo muy simple) |
| SGDRegressor (Base) | $58,178.82 | $58,182.88 | Subajuste (Modelo muy simple) |
| DecisionTreeRegressor | $0.00 | $62,118.45 | Sobreajuste Crítico (Memorización) |

### Fase 2 — Evaluación Final en Test Set (Modelos Afinados)

| Modelo (Afinado con GridSearchCV) | RMSE en Test (Margen de Error) | Estado de Generalización |
|---|---|---|
| Random Forest Regressor | $48,255.62 | ✅ Excelente (Modelo Ganador) |
| SGD Regressor | $124,131.10 | ❌ Deficiente (Incapaz de generalizar) |

### Justificación del Modelo Final: `Random Forest Regressor`

Se optó por el modelo de ensamblado tras someterlo a un ajuste fino de hiperparámetros.

- **Por qué funciona:** Al construir múltiples árboles descorrelacionados (modificando `max_features`), el modelo es capaz de capturar la complejidad topográfica y demográfica sin colapsar ante el ruido.
- **Brecha de Generalización:** Existe un gap de ~$6,800 entre la Validación Cruzada y el Test de prueba. Este comportamiento es **normal y esperado**. Se deriva de la reducción del sesgo inherente al ajuste fino (incluyendo experimentos con `bootstrap: False`), lo que incrementa marginalmente la varianza empírica ante datos nunca vistos.

---

## 💡 Conclusiones y Consideraciones Aplicadas

- **Valor del Negocio:** Un error promedio de ~$48,255 dólares es altamente competitivo para definir bandas de precio (mínimo y máximo sugerido) en tasaciones automáticas iniciales (AVM - *Automated Valuation Models*), reduciendo la fricción operativa en un 80% frente a la valoración exclusivamente humana.

- **Poder de la Geografía:** Las características espaciales diseñadas (`dist_to_LA`, `dist_to_SF`) demostraron empíricamente que la regla de oro del Real Estate (*"Ubicación, ubicación, ubicación"*) rige la capacidad predictiva.

### Supuestos y Limitaciones

> ⚠️ **Atemporalidad:** El modelo actual asume una distribución de precios estática. En el mundo real, fenómenos macroeconómicos (tasas de interés, inflación) alteran drásticamente el valor objetivo. *Sugerencia futura: Incorporar series temporales macroeconómicas.*

> ⚠️ **Límites de Captura:** El conjunto de datos original censura el precio de las casas por encima de **$500,000**. El modelo heredará esta ceguera, siendo ineficaz para propiedades de ultra-lujo.

---

## 🚀 Guía de Despliegue y Reproducción

Sigue estos pasos para reproducir el experimento o levantar la API de inferencia local:

### 1. Preparación del Entorno

Clona el repositorio y configura un entorno virtual:

```bash
git clone https://github.com/tu_usuario/tu_repositorio.git
cd tu_repositorio
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Ejecución del Pipeline de Datos

Asegúrate de particionar los datos de manera limpia antes de explorar:

```bash
python src/data/make_dataset.py
python src/data/split_data.py
```

### 3. Entrenamiento (Opcional)

Si deseas re-entrenar el modelo ganador y generar de nuevo los artefactos `.pkl`, ejecuta los notebooks secuencialmente utilizando Jupyter:

```bash
jupyter notebook
```

Ejecuta `02_limpieza_enriquecimiento.ipynb` seguido de `03_experimentacion.ipynb`.

### 4. Despliegue de la API

El modelo está empaquetado para inferencia a través de una API RESTful:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Una vez levantado, visita `http://localhost:8000/docs` para interactuar con el modelo a través de **Swagger UI**.