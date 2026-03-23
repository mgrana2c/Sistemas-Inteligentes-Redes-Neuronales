# Agente Racional para Logística E-Commerce

Implementación de un agente racional basado en redes neuronales para predecir el tiempo de entrega (ETA) y clasificar el riesgo de retraso en pedidos del dataset Olist (e-commerce brasileño).

---

## Descripción del problema

Los sistemas logísticos tradicionales basados en reglas estáticas no capturan la naturaleza no lineal de las entregas urbanas. Este proyecto entrena dos modelos de redes neuronales profundas para:

- **Regresión:** predecir cuántos días tardará un pedido en ser entregado.
- **Clasificación:** estimar si un pedido llegará tarde respecto a la fecha prometida.

---

## Estructura del proyecto

```
SIST RN/
├── dataset/                        # Datos originales de Olist (CSV)
├── notebooks/
│   ├── 01_eda.ipynb                # Análisis exploratorio
│   ├── 02_ingenieria_features.ipynb# Construcción del dataset de entrenamiento
│   ├── 03_modelo_regresion.ipynb   # Red neuronal: predicción de días
│   ├── 04_modelo_clasificacion.ipynb# Red neuronal: riesgo de retraso
│   └── 05_evaluacion.ipynb         # Métricas finales y demo del agente
├── data/
│   └── processed/                  # Generado por el notebook 02
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── preprocesadores.pkl
├── models/                         # Generado por los notebooks 03 y 04
│   ├── modelo_regresion.keras
│   └── modelo_clasificacion.keras
├── outputs/
│   ├── graficas/                   # Gráficas generadas automáticamente
│   └── metricas.json               # Métricas finales en formato JSON
├── requirements.txt
└── dataset_download.py             # Script para descargar el dataset
```

---

## Instalación

```bash
pip install -r requirements.txt
```

Dependencias principales: `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `matplotlib`, `seaborn`, `jupyter`.

---

## Orden de ejecución de los notebooks

Los notebooks deben ejecutarse en orden. Cada uno depende de los resultados del anterior.

### Paso 1 — `01_eda.ipynb`

Análisis exploratorio del dataset. No genera archivos, solo visualizaciones.
Permite entender las distribuciones, nulos, desbalance de clases y rango temporal de los datos.

### Paso 2 — `02_ingenieria_features.ipynb`

**Obligatorio antes de entrenar cualquier modelo.**

Realiza los joins entre tablas, calcula la distancia Haversine entre vendedor y cliente, extrae features temporales y físicas, codifica variables categóricas y divide los datos cronológicamente (80% train, 10% val, 10% test).

Genera en `data/processed/`:
- `train.csv`, `val.csv`, `test.csv`
- `preprocesadores.pkl` (scaler, encoders, cardinalidades)

### Paso 3 — `03_modelo_regresion.ipynb`

Entrena la red neuronal de regresión. Requiere los archivos generados en el paso 2.

Arquitectura: embeddings para categorías + 3 capas densas (256-128-64) con BatchNorm y Dropout.
Loss: MAE. Guarda el modelo en `models/modelo_regresion.keras`.

### Paso 4 — `04_modelo_clasificacion.ipynb`

Entrena la red neuronal de clasificación. Puede ejecutarse en paralelo con el paso 3.

Misma arquitectura base, salida con activación sigmoid. Usa `class_weight` para compensar el desbalance de clases (~8% de retrasos). Guarda el modelo en `models/modelo_clasificacion.keras`.

### Paso 5 — `05_evaluacion.ipynb`

Requiere que los pasos 3 y 4 hayan finalizado.

Carga ambos modelos, evalúa sobre el conjunto de test, genera las visualizaciones finales (scatter, curva ROC, matriz de confusión) y guarda las métricas en `outputs/metricas.json`. Incluye una demostración de inferencia sobre un pedido individual.

---

## Métricas objetivo

| Modelo | Métrica | Umbral razonable |
|---|---|---|
| Regresión | R² | > 0.50 |
| Regresión | MAE | < 5 días |
| Clasificación | AUC-ROC | > 0.70 |
| Clasificación | F1-Score | > 0.40 |

---

## Dataset

Brazilian E-Commerce Public Dataset por Olist. Aproximadamente 100.000 órdenes entre 2016 y 2018.
Licencia: CC BY-NC-SA 4.0.

Para descargar el dataset:
```bash
python dataset_download.py
```
