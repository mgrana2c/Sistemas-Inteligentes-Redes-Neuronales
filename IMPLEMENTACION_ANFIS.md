# Implementación de Razonamiento con Incertidumbre: ANFIS para Predicción de ETA

## 1. Objetivo

Modelar el tiempo de entrega (`dias_entrega`) no como un valor determinista, sino como el resultado de **variables lingüísticas con grados de pertenencia**, permitiendo razonar bajo la incertidumbre inherente al transporte en Brasil.

Se implementa un **Sistema de Inferencia Neuro-Difuso Adaptativo (ANFIS)** tipo Sugeno, que:

- Fuzzifica las entradas en conjuntos lingüísticos interpretables (*Cercano / Lejano*, *Urgente / Normal / Holgado*, etc.)
- Aprende automáticamente los parámetros de las funciones de membresía mediante backpropagation
- Genera reglas difusas en lenguaje natural auditables por un humano

---

## 2. Contexto del Proyecto

| Entrega | Método | Notebook | Enfoque |
|---------|--------|----------|---------|
| 1 — Redes Neuronales | MLP + Embeddings | `08_experimentos_final.ipynb` | Predicción precisa con todas las features |
| 2 — Métodos Evolutivos | GA + MLP | `09/10_metodos_evolutivos.ipynb` | Selección óptima de subconjunto de features |
| **3 — Razonamiento con Inc.** | **ANFIS** | **`11_anfis_razonamiento_incierto.ipynb`** | **Reglas interpretables bajo incertidumbre** |



---

## 3. Variables lingüísticas de entrada (3 features)

La limitación arquitectónica de ANFIS —el número de reglas crece como $\text{MFs}^{n_\text{inputs}}$— impone un máximo práctico de 3–4 entradas para mantener la interpretabilidad. Se eligieron las 3 features con mayor semántica difusa del dominio logístico:

| # | Feature | Significado lingüístico | Conjuntos (3 MFs) |
|---|---------|------------------------|-------------------|
| 0 | `distancia_km` | ¿Qué tan lejos está el cliente? | Cercano · Medio · Lejano |
| 1 | `dias_limite_envio` | ¿Con qué urgencia debe enviarlo el vendedor? | Urgente · Normal · Holgado |
| 2 | `flete_total` | ¿Qué nivel de servicio logístico se contrató? | Económico · Estándar · Premium |

**Normalización:** MinMaxScaler `[0, 1]` ajustado exclusivamente sobre el conjunto de training (sin data leakage), requisito del espacio de parámetros del sistema difuso.

---

## 4. Arquitectura ANFIS (Sugeno tipo-1, Jang)

### Las 5 capas funcionales

```
Entradas x₁, x₂, x₃  →  [L1] → [L2] → [L3] → [L4] → [L5]  →  ŷ (días_entrega)
```

| Capa | Función | Parámetros aprendibles |
|------|---------|------------------------|
| **L1 — Fuzzificación** | $\mu_{ij}(x_i)$ — grado de pertenencia de $x_i$ al conjunto $j$ | Premisa: $(a, m, b)$ para trimf · $(\mu, \sigma)$ para gaussmf |
| **L2 — Firing strength** | $w_r = \prod_i \mu_{ij_r}$ — intensidad de activación de la regla $r$ | — (AND por producto, diferenciable) |
| **L3 — Normalización** | $\bar{w}_r = w_r \,/\, \sum_r w_r$ — pesos normalizados | — |
| **L4 — Consecuente** | $f_r = p_r^1 x_1 + p_r^2 x_2 + p_r^3 x_3 + p_r^0$ (Sugeno lineal) | Consecuente: $[p_r^0, p_r^1, p_r^2, p_r^3]$ por regla |
| **L5 — Defuzzificación** | $\hat{y} = \sum_r \bar{w}_r \cdot f_r$ — salida ponderada | — |

### Recuento de reglas y parámetros

| MFs por input | Reglas ($n^3$) | Params. premisa (gauss) | Params. consecuente | Total |
|:---:|:---:|:---:|:---:|:---:|
| 2 | 8 | 12 | 32 | 44 |
| 3 | 27 | 18 | 108 | 126 |
| 4 | 64 | 24 | 256 | 280 |

En comparación, el MLP baseline tiene **~100k parámetros**; el ANFIS con 4 MFs tiene 280 — órdenes de magnitud más interpretable.

### Implementación

La capa ANFIS se implementó **desde cero** como `tf.keras.layers.Layer` compatible con TensorFlow 2.10, sin librerías externas de lógica difusa. El gradiente fluye a través de todos los parámetros con `tf.GradientTape` estándar.

---

## 5. Protocolo Experimental

### 5.1 Variables independientes (factores de control)

| Factor | Valores | Descripción |
|--------|---------|-------------|
| **Número de MFs** | `{2, 3, 4}` | Granularidad del conocimiento difuso |
| **Tipo de MF** | `{trimf, gaussmf}` | Forma de los conjuntos difusos |
| **Método de optimización** | `{hybrid, gradient}` | Inicialización y ajuste de parámetros |

**Total de configuraciones:** $3 \times 2 \times 2 = \mathbf{12}$

### 5.2 Descripción de los métodos de optimización

**Método Hybrid** (aproximación al algoritmo clásico de Jang):
1. **Inicialización premisa:** centros de MFs con percentiles equiespaciados en `[0, 1]`; anchuras proporcionales al espaciado
2. **Inicialización consecuente:** Mínimos Cuadrados (LSQ) sobre los firing strengths $\bar{w}_r$ del train — resuelve $\Phi \cdot \theta = y$ analíticamente
3. **Fine-tuning global:** Adam `lr=1e-3` sobre todos los parámetros

**Método Gradient** (baseline puro):
1. Inicialización aleatoria (Glorot uniform) para todos los parámetros
2. Adam `lr=1e-3` desde cero, sin warmstart

### 5.3 Variables dependientes (métricas de salida)

| Métrica | Descripción |
|---------|-------------|
| **MAE** | Error absoluto medio en días (métrica principal, comparabilidad con E1 y E2) |
| **RMSE** | Raíz del error cuadrático medio en días |
| **R²** | Coeficiente de determinación |
| **Nº reglas** | Complejidad / interpretabilidad del sistema |
| **Época de convergencia** | Eficiencia del entrenamiento |
| **Tiempo (s)** | Costo computacional |

### 5.4 Configuración de entrenamiento

| Parámetro | Valor |
|-----------|-------|
| Epochs máx. | 200 |
| Early stopping | `patience=15`, monitor `val_mae`, `restore_best_weights=True` |
| ReduceLROnPlateau | `factor=0.5, patience=7, min_lr=1e-6` |
| Batch size | 512 |
| Optimizer | Adam `lr=1e-3` |
| Loss | MAE |
| Entorno | CPU (TF 2.10, Python 3.9, CUDA deshabilitado explícitamente) |

### 5.5 Split de datos

Split cronológico **80 / 10 / 10** — idéntico al de las entregas 1 y 2 para garantizar comparabilidad directa sobre el mismo test set.

| Conjunto | Uso en ANFIS |
|---------|-------------|
| Train (80%) | Entrenamiento + inicialización LSQ del hybrid |
| Validación (10%) | Early stopping, selección del mejor epoch |
| Test (10%) | Evaluación final — reporte de métricas comparativas |

### 5.6 Tabla de configuraciones

| ID | MFs | Tipo MF | Optimización | Reglas |
|----|:---:|---------|-------------|:------:|
| 1  | 2 | trimf   | hybrid   | 8  |
| 2  | 2 | trimf   | gradient | 8  |
| 3  | 2 | gaussmf | hybrid   | 8  |
| 4  | 2 | gaussmf | gradient | 8  |
| 5  | 3 | trimf   | hybrid   | 27 |
| 6  | 3 | trimf   | gradient | 27 |
| 7  | 3 | gaussmf | hybrid   | 27 |
| 8  | 3 | gaussmf | gradient | 27 |
| 9  | 4 | trimf   | hybrid   | 64 |
| 10 | 4 | trimf   | gradient | 64 |
| 11 | 4 | gaussmf | hybrid   | 64 |
| 12 | 4 | gaussmf | gradient | 64 |

---

## 6. Resultados

> **Pendiente de ejecución.** Completar tras ejecutar `11_anfis_razonamiento_incierto.ipynb`.

### 6.1 Tabla completa del grid (12 configuraciones)

| ID | MFs | Tipo | Optim | Reglas | MAE | RMSE | R² | Épocas | Tiempo(s) |
|----|:---:|------|-------|:------:|-----|------|-----|--------|-----------|
| 1  | 2 | trimf   | hybrid   | 8  | — | — | — | — | — |
| 2  | 2 | trimf   | gradient | 8  | — | — | — | — | — |
| 3  | 2 | gaussmf | hybrid   | 8  | — | — | — | — | — |
| 4  | 2 | gaussmf | gradient | 8  | — | — | — | — | — |
| 5  | 3 | trimf   | hybrid   | 27 | — | — | — | — | — |
| 6  | 3 | trimf   | gradient | 27 | — | — | — | — | — |
| 7  | 3 | gaussmf | hybrid   | 27 | — | — | — | — | — |
| 8  | 3 | gaussmf | gradient | 27 | — | — | — | — | — |
| 9  | 4 | trimf   | hybrid   | 64 | — | — | — | — | — |
| 10 | 4 | trimf   | gradient | 64 | — | — | — | — | — |
| 11 | 4 | gaussmf | hybrid   | 64 | — | — | — | — | — |
| 12 | 4 | gaussmf | gradient | 64 | — | — | — | — | — |

**★ Mejor configuración:** Config — · — MFs · — · — → MAE = — · R² = — *(pendiente)*

### 6.2 Análisis del efecto de cada factor

> *(Completar con los valores de la tabla anterior)*

| Factor | Nivel | MAE medio | Conclusión |
|--------|-------|-----------|------------|
| Nº MFs | 2 | — | — |
| Nº MFs | 3 | — | — |
| Nº MFs | 4 | — | — |
| Tipo MF | trimf | — | — |
| Tipo MF | gaussmf | — | — |
| Optimización | hybrid | — | — |
| Optimización | gradient | — | — |

### 6.3 Reglas lingüísticas del mejor modelo

> *(Completar tras ejecución — extracto de `outputs/reglas_mejor_anfis.txt`)*

```
REGLAS DIFUSAS — Mejor ANFIS (? MFs, ?, ?)

── Reglas con MENOR ETA (entregas más rápidas) ──
  R???: SI distancia km ES ? Y dias limite envio ES ? Y flete total ES ?  →  ETA ≈ ? días
  ...

── Reglas con MAYOR ETA (entregas más lentas) ──
  R???: SI distancia km ES ? Y dias limite envio ES ? Y flete total ES ?  →  ETA ≈ ? días
  ...
```

### 6.4 Comparativa final — 3 entregas del proyecto

> *(Completar con los valores reales tras ejecución)*

| Entrega | Modelo | Features | MAE (días) | RMSE (días) | R² | Δ MAE vs E1 | Δ R² vs E1 |
|---------|--------|:--------:|:----------:|:-----------:|:--:|:-----------:|:----------:|
| E1 — RN | MLP Baseline (nb08) | 15 | 2.7479 | 4.0278 | 0.2884 | — | — |
| E2 — Evolutivo | GA-REG + MLP (nb10) | 8 | 2.9382 | 4.1116 | 0.2585 | +6.9% ▲ | −10.4% ▼ |
| **E3 — Inc.** | **ANFIS mejor (nb11)** | **3** | **—** | **—** | **—** | **—** | **—** |

### 6.5 Gráficos

Ver:
- `outputs/graficas/anfis_grid_analysis.png` — Heatmap MAE × factores del grid
- `outputs/mfs_mejor_anfis.png` — Funciones de membresía aprendidas
- `outputs/curvas_anfis.png` — Curva de aprendizaje del mejor modelo
- `outputs/comparativo_ANFIS.png` — Barplot comparativo E1 / E2 / E3

---

## 7. Interpretabilidad: ventaja diferencial de ANFIS

A diferencia del MLP (Entrega 1) y del GA-MLP (Entrega 2), el ANFIS produce un sistema **auditables por un humano**:

1. **Funciones de membresía:** se puede visualizar cómo el modelo aprendió los bordes de los conjuntos difusos (ej: a partir de qué distancia considera que una entrega es "Lejana")
2. **Reglas lingüísticas:** cada combinación de conjuntos produce una predicción de ETA explicable en lenguaje natural
3. **Pocos parámetros:** 44–280 parámetros frente a ~100k del MLP — el modelo es completamente inspeccionable

Esta interpretabilidad es el **aporte diferenciador de la Entrega 3**, independientemente de si el MAE supera o no al baseline.

---

## 8. Notebooks

| Notebook | Función |
|----------|---------|
| `11_anfis_razonamiento_incierto.ipynb` | Pipeline completo: datos → 12 experimentos → análisis → comparativa → guardado |

### Artefactos generados

| Archivo | Contenido |
|---------|-----------|
| `outputs/metricas_ANFIS.json` | Resultados del grid completo + mejor config + comparativa 3 entregas |
| `outputs/reglas_mejor_anfis.txt` | Reglas difusas en lenguaje natural del mejor modelo |
| `outputs/mfs_mejor_anfis.png` | Funciones de membresía aprendidas (3 variables lingüísticas) |
| `outputs/curvas_anfis.png` | Curva de aprendizaje del mejor modelo |
| `outputs/comparativo_ANFIS.png` | Barplot comparativo E1 / E2 / E3 (MAE, RMSE, R²) |
| `outputs/graficas/anfis_grid_analysis.png` | Heatmap + barplots del análisis del grid |

---

## 9. Comando de ejecución

```bash
# Con el entorno activado (tf_gpu o base con TF 2.10):
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  "notebooks/11_anfis_razonamiento_incierto.ipynb"
```

**Tiempo estimado en CPU:** 5–15 minutos (12 configs × ~200 épocas máx., convergencia anticipada por early stopping).

### Orden completo del proyecto

```
08_experimentos_final.ipynb   →  outputs/metricas_final.json
        ↓
09_metodos_evolutivos.ipynb   →  outputs/ga_resultados.json
        ↓
10_experimento_ME.ipynb       →  outputs/metricas_ME.json
        ↓
11_anfis_razonamiento_incierto.ipynb  →  outputs/metricas_ANFIS.json
```
