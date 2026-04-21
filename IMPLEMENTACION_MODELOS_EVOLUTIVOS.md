# Implementación de Métodos Evolutivos para Selección de Features

## 1. Objetivo

Identificar el subconjunto óptimo de features para cada modelo de red neuronal utilizando un **Algoritmo Genético (GA)** como método de búsqueda. Se ejecutan dos GAs independientes:

- **GA-REG** → maximiza R² en validación (modelo de regresión)
- **GA-CLF** → maximiza F1-score en validación (modelo de clasificación)

---

## 2. Contexto

Los modelos base (notebook `08_experimentos_final.ipynb`) usan las 15 features disponibles. No todas contribuyen igualmente. El GA busca el subconjunto que mejore la métrica objetivo al entrenar con esas features y ninguna más.

---

## 3. Features disponibles (15 totales)

| # | Feature | Tipo |
|---|---------|------|
| 0 | `distancia_km` | Numérica |
| 1 | `precio_total` | Numérica |
| 2 | `flete_total` | Numérica |
| 3 | `product_weight_g` | Numérica |
| 4 | `volumen_cm3` | Numérica |
| 5 | `mes_compra` | Numérica |
| 6 | `dia_semana_compra` | Numérica |
| 7 | `hora_compra` | Numérica |
| 8 | `dias_estimados` | Numérica |
| 9 | `dias_limite_envio` | Numérica |
| 10 | `n_items` | Numérica |
| 11 | `mismo_estado` | Numérica |
| 12 | `categoria_producto` | Categórica |
| 13 | `customer_state` | Categórica |
| 14 | `seller_state` | Categórica |

---

## 4. Diseño del Algoritmo Genético

### Representación

Cromosoma binario de 15 bits. Cada bit indica si la feature correspondiente está activa (1) o inactiva (0).

```
[1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1,  1, 0, 1]
 ─────────────────────────────────────  ────────
          12 features numéricas          3 categ.
```

### Parámetros

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| Tamaño de población | 20 | Balance entre diversidad y costo computacional |
| Generaciones máx. | 15 | Con parada anticipada, suele converger antes |
| Selección | Torneo k=3 | Presión selectiva moderada |
| Crossover | Un punto, prob=0.80 | Estándar para cromosomas de longitud media |
| Mutación | 2% por bit | Exploración sin destruir buenas soluciones |
| Elitismo | Top-1 | El mejor individuo pasa directo a la siguiente generación |
| Mínimo de features | 3 | Evita cromosomas degenerados |
| Parada anticipada | 5 generaciones sin mejora | Ahorra tiempo cuando ya convergió |

### Función de fitness (proxy)

Entrenar la red completa en cada evaluación sería prohibitivo. Se usa un **proxy de 20% del conjunto de entrenamiento**:

- Sub-muestra fija de `~15.000 filas` (reproducible, `SEED=42`)
- `epochs=20`, `EarlyStopping patience=2`
- `batch_size=256`
- **GA-REG:** fitness = R² en validación completa
- **GA-CLF:** fitness = F1 con umbral óptimo en validación completa

### Caché

Se mantiene un `dict` con las evaluaciones ya realizadas, indexado por el cromosoma. Si dos individuos en distintas generaciones son idénticos, se reutiliza el resultado sin re-entrenar.

### Estrategia de masking

Las features desactivadas no se eliminan — se enmascaran antes de entrar a la red:
- **Numéricas desactivadas:** se reemplazan por `0.0` (espacio normalizado)
- **Categóricas desactivadas:** se reemplazan por el índice de `'desconocido'` en el LabelEncoder

Esto permite usar siempre la misma arquitectura de red.

---

## 5. Arquitectura de Red Neuronal (fija)

La arquitectura no cambia durante el GA. Es la configuración ganadora del notebook `08`:

| Componente | Configuración |
|-----------|---------------|
| Embeddings | Por cada variable categórica, `dim = min(50, card//2 + 1)` |
| Capas densas | [256, 128, 64] con ReLU |
| Regularización | BatchNormalization + Dropout (0.3, 0.2) |
| Salida regresión | 1 neurona, activación lineal, loss MAE |
| Salida clasificación | 1 neurona, activación sigmoid, Focal Loss γ=2.0 α=0.75 |
| Optimizador | Adam lr=1e-3 |

---

## 6. Split de datos

Split cronológico 80/10/10 sobre órdenes entregadas entre 2016-2018.

| Conjunto | Uso |
|---------|-----|
| Train (80%) | Entrenamiento del proxy y del modelo final |
| Validación (10%) | Evaluación de fitness en el GA; selección de umbral |
| Test (10%) | Evaluación final únicamente en notebook 10 |

---

## 7. Notebooks

| Notebook | Función |
|----------|---------|
| `09_metodos_evolutivos.ipynb` | Ejecuta los dos GAs y guarda `outputs/ga_resultados.json` |
| `10_experimento_ME.ipynb` | Carga los cromosomas, entrena los modelos finales y compara contra el baseline |

### Orden de ejecución

```
09_metodos_evolutivos.ipynb  →  outputs/ga_resultados.json
                                        ↓
                            10_experimento_ME.ipynb  →  outputs/metricas_ME.json
```

### Artefactos generados

| Archivo | Contenido |
|---------|-----------|
| `outputs/ga_resultados.json` | Cromosomas óptimos, historia de evolución, parámetros del GA |
| `outputs/metricas_ME.json` | Métricas del modelo Evolutivo vs Baseline |
| `models/modelo_regresion_ME.keras` | Modelo de regresión entrenado con features GA-REG |
| `models/modelo_clasificacion_ME.keras` | Modelo de clasificación entrenado con features GA-CLF |
| `outputs/graficas/ga_evolucion_fitness.png` | Curvas de fitness por generación |
| `outputs/graficas/ga_features_seleccionadas.png` | Mapa de features seleccionadas por cada GA |

---

## 8. Resultados

> *Sección pendiente de completar tras la ejecución de los notebooks 09 y 10.*

### 8.1 Features seleccionadas

| | GA-REG | GA-CLF |
|---|--------|--------|
| Número de features | — / 15 | — / 15 |
| Features activas | — | — |

### 8.2 Comparación de métricas vs Baseline

| Modelo | Features | MAE | RMSE | R² | AUC | F1 |
|--------|----------|-----|------|----|-----|----|
| Baseline (15 feat.) | 15 | 2.748 | 4.028 | 0.288 | 0.667 | 0.123 |
| GA-REG | — | — | — | — | — | — |
| GA-CLF | — | — | — | — | — | — |

### 8.3 Evolución del GA

*(Insertar imagen `outputs/graficas/ga_evolucion_fitness.png` tras la ejecución)*

### 8.4 Mapa de features

*(Insertar imagen `outputs/graficas/ga_features_seleccionadas.png` tras la ejecución)*

---

## 9. Comando de ejecución

```bash
# Con el entorno activado (Python 3.x, TensorFlow instalado):
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  "notebooks/09_metodos_evolutivos.ipynb"

# Una vez generado ga_resultados.json:
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  "notebooks/10_experimento_ME.ipynb"
```

Tiempo estimado de ejecución en CPU: **1.5 – 2.5 horas** para el notebook 09 (ambos GAs). El notebook 10 tarda ~15-20 minutos adicionales.
