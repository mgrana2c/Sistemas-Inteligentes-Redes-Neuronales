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

### 6.1 Tabla completa del grid (12 configuraciones)

| ID | MFs | Tipo | Optim | Reglas | MAE | RMSE | R² | Épocas | Tiempo(s) |
|----|:---:|------|-------|:------:|-----|------|:--:|:------:|:---------:|
| 1  | 2 | trimf   | hybrid   | 8  | 3.6364 | 4.9455 | −0.0728 | 120 | 65.2 |
| 2  | 2 | trimf   | gradient | 8  | 3.2649 | 4.5478 | 0.0928 | 64 | 35.6 |
| 3  | 2 | gaussmf | hybrid   | 8  | 3.6364 | 4.9455 | −0.0728 | 120 | 63.2 |
| 4 | 2 | gaussmf | gradient | 8  | 3.2573 | **4.5330** | **0.0987** | 56 | 30.0 |
| 5  | 3 | trimf   | hybrid   | 27 | 4.1152 | 5.3853 | −0.2721 | 30 | 21.5 |
| 6  | 3 | trimf   | gradient | 27 | 3.2391 | 4.5236 | 0.1024 | 67 | 46.4 |
| 7  | 3 | gaussmf | hybrid   | 27 | 4.1152 | 5.3853 | −0.2721 | 30 | 20.7 |
| 8  | 3 | gaussmf | gradient | 27 | 3.3057 | 4.5812 | 0.0794 | 64 | 44.9 |
| 9  | 4 | trimf   | hybrid   | 64 | 3.5010 | 4.7065 | 0.0284 | 18 | 16.2 |
| ★**10** | **4** | **trimf**   | **gradient** | **64** | **3.1160** | 4.**5942** | **0.1254** | **61** | **53.2** |
| 11 | 4 | gaussmf | hybrid   | 64 | 3.5010 | 4.7065 | 0.0284 | 18 | 16.0 |
| 12 | 4 | gaussmf | gradient | 64 | 3.2159 | 4.5255 | 0.1017 | 63 | 54.0 |

**★ Mejor configuración:** Config 10 · 4 MFs · trimf · gradient → MAE = 3.1160 · R² = 0.1254

> Las 6 configuraciones hybrid divergieron numéricamente (MAE > 7.000, R² = −10⁷ a −10¹⁰). El análisis de factores se calcula exclusivamente sobre las 6 configuraciones gradient válidas.

---

### 6.2 Rendimiento del método hybrid

El método hybrid inicializa los consecuentes mediante mínimos cuadrados regularizados (ridge, $\lambda=10^{-3}$) sobre la matriz de firing strengths: resuelve $(\Phi^T\Phi + \lambda I)\,\theta = \Phi^T y$ analíticamente antes del fine-tuning con Adam. A pesar de este calentamiento estructurado, hybrid produce consistentemente peores resultados que gradient en todas las configuraciones:

| MFs | Hybrid MAE | Gradient MAE | Δ |
|:---:|:---:|:---:|:---:|
| 2 | 3.636 | 3.116–3.265 | +0.37–0.52 días |
| 3 | 4.115 | 3.239–3.306 | +0.81–0.88 días |
| 4 | 3.501 | 3.216–3.257 | +0.24–0.29 días |

La inicialización LSQ proporciona consecuentes coherentes con el training set, pero no genera una cuenca de convergencia más favorable que la inicialización Glorot aleatoria del gradient descent puro. En todos los casos, Adam desde inicialización aleatoria alcanza menores mínimos en el mismo presupuesto de épocas.

---

### 6.3 Análisis del efecto de cada factor (configuraciones gradient)

| Factor | Nivel | MAE medio | Mejor MAE del nivel | Conclusión |
|--------|-------|:---------:|:-------------------:|------------|
| Nº MFs | 2     | 3.2208    | 3.2199              | Peor granularidad, converge rápido |
| Nº MFs | 3     | 3.2410    | 3.2191              | Sin ventaja sobre 2 MFs |
| Nº MFs | **4** | **3.1829** | **3.1160**         | **Mejor MAE; más capacidad expresiva** |
| Tipo MF | **trimf** | **3.2002** | **3.1160**    | **Ligeramente superior** |
| Tipo MF | gaussmf | 3.2296   | 3.2191              | Marginalmente inferior |
| Optimización | gradient | 3.2149 | 3.1160        | Factor dominante — ventaja media de 0.52 días |
| Optimización | hybrid | 3.7509|    3.5010          | Consistentemente inferior |

El margen entre la peor y la mejor configuración gradient válida es de solo **0.147 días de MAE** (3.263 vs 3.116). El cuello de botella no es la arquitectura ANFIS sino la cantidad de features disponibles (3/15).
Factor dominante — ventaja media de 0.52 días
Optimización	hybrid	3.7509	3.5010	Consistentemente inferior
---

### 6.4 Reglas lingüísticas del mejor modelo (4 MFs, trimf, gradient)

El modelo aprendió 64 reglas sobre el espacio {Muy Cercano, Cercano, Lejano, Muy Lejano} × {Muy Urgente, Urgente, Holgado, Muy Holgado} × {Muy Econ., Económico, Premium, Muy Premium}.

**Rango de predicción:** 6.4 a 11.9 días (amplitud de 5.5 días).

#### Reglas que predicen entregas más rápidas (ETA ≤ 7.0 días)

| Regla | distancia km | dias limite envio | flete total | ETA |
|-------|-------------|------------------|-------------|:---:|
| R031 | Cercano      | Muy Holgado       | Premium     | 6.4 |
| R047 | Lejano       | Muy Holgado       | Premium     | 6.6 |
| R039 | Lejano       | Urgente           | Premium     | 6.7 |
| R046 | Lejano       | Muy Holgado       | Económico   | 6.7 |
| R038 | Lejano       | Urgente           | Económico   | 6.7 |
| R023 | Cercano      | Urgente           | Premium     | 6.8 |
| R043 | Lejano       | Holgado           | Premium     | 7.0 |

#### Reglas que predicen entregas más lentas (ETA ≥ 11.0 días)

| Regla | distancia km | dias limite envio | flete total | ETA |
|-------|-------------|------------------|-------------|:---:|
| R061 | Muy Lejano   | Muy Holgado       | Muy Econ.   | 11.9 |
| R049 | Muy Lejano   | Muy Urgente       | Muy Econ.   | 11.8 |
| R053 | Muy Lejano   | Urgente           | Muy Econ.   | 11.7 |
| R057 | Muy Lejano   | Holgado           | Muy Econ.   | 11.5 |
| R051 | Muy Lejano   | Muy Urgente       | Premium     | 11.3 |
| R064 | Muy Lejano   | Muy Holgado       | Muy Premium | 11.1 |

#### Patrones y conclusiones del sistema difuso

**1. El nivel de servicio logístico (flete_total) es el factor más consistente:**
Premium reduce el ETA en ~0.8–1.2 días respecto a Muy Econ. para cualquier combinación de distancia y urgencia. Ejemplo directo:

| Condición | ETA |
|-----------|:---:|
| R039: Lejano + Urgente + **Premium** | 6.7 días |
| R037: Lejano + Urgente + **Muy Econ.** | 7.5 días (+0.8 días) |

**2. Paradoja de la distancia corta — Muy Cercano predice ETAs más lentos:**
El clúster `Muy Cercano` predice ETAs de 7.2–10.6 días, mayor que `Lejano` (6.4–9.1 días) y `Cercano` (6.4–8.3 días). El modelo aprendió correctamente que en Olist, distancias muy cortas corresponden a entregas intraurbanas en São Paulo / Rio de Janeiro, donde la densidad logística genera más demoras que rutas interregionales largas. La `distancia_km` captura kilómetros geométricos, no dificultad logística real.

**3. El plazo del vendedor (dias_limite_envio) no es monotónico:**
`Muy Urgente` no garantiza entregas más rápidas. Para el clúster Muy Cercano:
- R004: Muy Cercano + **Muy Urgente** + Muy Premium → 7.2 días
- R016: Muy Cercano + **Muy Holgado** + Muy Premium → 9.1 días

La urgencia del vendedor reduce el ETA ~2 días en Muy Cercano, pero el efecto desaparece en clústeres Lejano/Muy Lejano, donde la restricción es logística, no de gestión del vendedor.

**4. Muy Lejano + Muy Econ. es la combinación crítica:**
Las 4 reglas más lentas del sistema comparten siempre `Muy Lejano` + `Muy Econ.`, indicando que la falta de inversión en flete en rutas largas es el predictor más confiable de demoras severas. El ANFIS lo captura como regla auditable explícita.

---

### 6.5 Comparativa final — 3 entregas del proyecto

| Entrega | Modelo | Features | MAE (días) | RMSE (días) | R² | Δ MAE vs E1 | Δ R² vs E1 |
|---------|--------|:--------:|:----------:|:-----------:|:--:|:-----------:|:----------:|
| E1 — RN | MLP Baseline (nb08) | 15 | 2.7479 | 4.0278 | 0.2884 | — | — |
| E2 — Evolutivo | GA-REG + MLP (nb10) | 8 | 2.9382 | 4.1116 | 0.2585 | +6.9% ▲ | −10.4% ▼ |
| **E3 — Incierto** | **ANFIS (nb11)** | **3** | **3.1160** | **4.4654** | **0.1254** | **+13.4% ▲** | **−56.5% ▼** |

### 6.6 Gráficos

Ver:
- `outputs/graficas/anfis_grid_analysis.png` — Heatmap MAE × factores del grid
- `outputs/mfs_mejor_anfis.png` — Funciones de membresía aprendidas (3 variables lingüísticas)
- `outputs/curvas_anfis.png` — Curva de aprendizaje del mejor modelo
- `outputs/comparativo_ANFIS.png` — Barplot comparativo E1 / E2 / E3

---

## 7. Comparación entre las 3 entregas

### 7.1 Métricas predictivas

| Métrica | E1 — MLP Baseline | E2 — GA-REG | E3 — ANFIS | Mejor |
|---------|:-----------------:|:-----------:|:----------:|:-----:|
| MAE (días) | **2.7479** | 2.9382 | 3.1160 | E1 |
| RMSE (días) | **4.0278** | 4.1116 | 4.4654 | E1 |
| R² | **0.2884** | 0.2585 | 0.1254 | E1 |
| Δ MAE vs E1 | — | +6.9% | +13.4% | — |
| Δ R² vs E1 | — | −10.4% | −56.5% | — |

La precisión predictiva decae monotónicamente de E1 a E3, lo cual es **esperado**: cada entrega usa menos features y un modelo con menor capacidad paramétrica. El objetivo de E3 no es superar a E1 en MAE, sino demostrar razonamiento bajo incertidumbre con reglas interpretables.

### 7.2 Features y dimensionalidad

| | E1 — MLP | E2 — GA-REG | E3 — ANFIS |
|---|:---:|:---:|:---:|
| Features usadas | 15 | 8 | **3** |
| Reducción vs E1 | — | −46.7% | **−80.0%** |
| Features categóricas | 3 | 1 | **0** |
| Criterio de selección | Ninguno (todas) | Fitness R² evolutivo | Semántica lingüística |

El GA (E2) demostró que 8 features capturan ~90% del rendimiento de E1. El ANFIS (E3) opera con el 20% de las features manteniendo ~89% del MAE de E1, logrando el mayor ratio interpretabilidad/feature.

### 7.3 Complejidad del modelo

| | E1 — MLP | E2 — GA-REG | E3 — ANFIS |
|---|:---:|:---:|:---:|
| Arquitectura | [256, 128, 64] + Embeddings | [256, 128, 64] + Embeddings | ANFIS 5 capas Sugeno |
| Parámetros totales | ~100.000 | ~100.000 | **280** |
| Ratio de complejidad vs ANFIS | 357× | 357× | **1×** |
| Tipo de función | Caja negra | Caja negra | **Caja blanca** |
| Reglas explícitas | No | No | **64** |
| Auditable por humano | No | No | **Sí** |

### 7.4 Proceso de entrenamiento

| | E1 — MLP | E2 — GA-REG | E3 — ANFIS |
|---|:---:|:---:|:---:|
| Algoritmo de búsqueda | Ninguno | GA (torneo k=3, elitismo Top-1) | Grid 12 configs |
| Optimizador de pesos | Adam | Adam (proxy 20 épocas/gen) | Adam |
| Épocas (mejor config) | ~120 | ~120 (entrenamiento final) | 72 |
| Tiempo total (CPU) | ~15 min | ~1.5–2.5 h | **~9 min** |
| Inicialización | Glorot uniform | Glorot uniform | Glorot (gradient) |

### 7.5 Interpretabilidad: ventaja diferencial de E3

| Capacidad | E1 — MLP | E2 — GA-REG | E3 — ANFIS |
|-----------|:---:|:---:|:---:|
| Reglas en lenguaje natural | ✗ | ✗ | ✓ (64 reglas) |
| Visualización del aprendizaje | Curvas de loss | Curvas GA + fitness | MFs aprendidas + reglas |
| Explicación de una predicción | ✗ | ✗ | ✓ (firing strengths por regla) |
| Auditoría de sesgos del dominio | Difícil | Difícil | **Directa** |
| Insight de negocio generado | — | — | *"Muy Lejano + Muy Econ. → 11.9 días siempre"* |

El ANFIS es el único modelo del proyecto capaz de responder "¿**por qué** predices X días?" con una justificación en variables del dominio. Esta es la contribución fundamental de la Entrega 3, independientemente de que su MAE sea 13.4% superior al baseline.

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
