# Segmentación de Clientes con K-Means — Mall Customers Dataset

## Resumen ejecutivo

Proyecto de minería de datos que segmenta 200 clientes de un centro comercial con el
algoritmo **K-Means (K=5)** sobre dos variables de comportamiento: **ingreso anual (k$)**
y **spending score (1–100)**, previamente estandarizadas con `StandardScaler`.
La evaluación combina el **método del codo** y el **coeficiente de silueta**.
El modelo final alcanza **silueta 0.5547 en K=5** (máximo del barrido K=2…10) e
inercia **65.57**. Incluye aplicación interactiva en **Streamlit** (español) que
explora datos, visualiza la evaluación, muestra los clusters y clasifica clientes nuevos.

## Objetivos

**General:** desarrollar un modelo de segmentación con K-Means, evaluado con el método
del codo y la silueta, para analizar patrones de comportamiento y apoyar decisiones de marketing.

**Específicos:**
- Comprender el funcionamiento de K-Means.
- Aplicar preprocesamiento con `StandardScaler`.
- Determinar el número óptimo de clusters.
- Evaluar la calidad con métricas (inercia, silueta).
- Interpretar los resultados desde el negocio.
- Desplegar una app interactiva.

## Dataset y diccionario

Fuente: [Mall Customers Dataset (Kaggle)](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
(`data/Mall_Customers.csv`, 200 filas × 5 columnas, 0 nulos, 0 duplicados).

| Columna | Tipo | Descripción |
|---|---|---|
| `CustomerID` | int | Identificador. **Se excluye del modelo** (no es comportamiento). |
| `Gender` | object | Género (Female 112 / Male 88). Descriptiva, no entra al modelo. |
| `Age` | int | Edad (media 38.85, rango 18–70). Descriptiva/comparativa. |
| `Annual Income (k$)` | int | Ingreso anual en **miles de USD** (media 60.56). Feature del modelo. |
| `Spending Score (1-100)` | int | Puntaje de gasto 1–100 (media 50.20). Feature del modelo. |

> `Annual Income (k$)` está en miles de dólares: 60 ≈ USD 60 000. No se convierte a
> pesos ni se interpreta como utilidad: ingreso ≠ ganancia.

## Metodología

1. **EDA:** dimensiones, muestra, tipos, descriptivos, nulos, duplicados, distribuciones,
   boxplots (atípicos), correlaciones. Conclusión: ingreso y gasto casi no correlacionan
   (r≈0.01) y forman grupos visibles en el scatter; son las variables idóneas.
   `CustomerID` se excluye por ser identificador. Correlación mide asociación lineal,
   **no causalidad**.
2. **Preprocesamiento:** `StandardScaler` sobre las 2 features (media 0, desv. 1).
   K-Means usa distancia euclidiana: sin escalar, la variable de mayor rango dominaría.
   En predicción se reutiliza el scaler guardado con `transform()` — **nunca** `fit` sobre
   un cliente individual.
3. **Codo (K=1…10):** inercias 400.00, 269.69, 157.70, 108.92, **65.57**, 55.06, 44.86,
   37.23, 32.39, 29.98. La inercia siempre decrece con K; el codo (mejora marginal)
   está en **K=5**. El codo es heurístico y el K de menor inercia no es el mejor por defecto.
4. **Silueta (K=2…10):** 0.3213, 0.4666, 0.4939, **0.5547**, 0.5399, 0.5281, 0.4552,
   0.4571, 0.4432. Máximo en **K=5 (0.5547)**, coherente con el codo.
   Silueta ≈1 bien asignado, ≈0 frontera, <0 probablemente mal asignado.
5. **Modelo final:** `KMeans(n_clusters=5, random_state=42, n_init=10)`.
   Artefactos: `model/kmeans_model.pkl`, `model/scaler.pkl`
   (serializados con scikit-learn 1.6.1; verificados funcionales en 1.9.1 con
   centroides idénticos al reentrenamiento).
6. **Negocio:** perfiles por cluster con estrategias (ver abajo).

## El modelo final (K=5)

Centroides en unidades originales:

| Cluster | n (% ) | Ingreso medio (k$) | Gasto medio | Edad media | Perfil |
|---|---|---|---|---|---|
| 0 | 81 (40.5 %) | 55.30 | 49.52 | 42.7 | Promedio |
| 1 | 39 (19.5 %) | 86.54 | 82.13 | 32.7 | Alto valor |
| 2 | 22 (11.0 %) | 25.73 | 79.36 | 25.3 | Con potencial (jóvenes entusiastas) |
| 3 | 35 (17.5 %) | 88.20 | 17.11 | 41.1 | Cautelosos premium |
| 4 | 23 (11.5 %) | 26.30 | 20.91 | 45.2 | Bajo valor / cautelosos |

**Mayor oportunidad de crecimiento: cluster 3 (cautelosos premium).** Tiene el ingreso
medio más alto (~88 k$) con el gasto más bajo (~17): la brecha entre capacidad de pago
y gasto actual es la mayor palanca incremental. Estrategia: demostrar valor (garantías,
pruebas, asesoría) en lugar de descuentos agresivos. Segunda oportunidad: cluster 2
(jóvenes de bajo ingreso pero alto gasto) con facilidades de pago y fidelización temprana.

**Límites honestos:** el dataset no contiene rentabilidad, frecuencia de compra ni
lifetime value; no se afirma nada sobre ellos. El clustering no supervisado **no tiene
"accuracy"** salvo que exista una verdad externa (ground truth) contra la cual comparar;
inercia y silueta miden compacidad/separación, no clasificación correcta.

## Arquitectura

```text
Taller_final_mineri_De_datos/
├── app.py                  # Panel Streamlit en español (navegación por pestañas)
├── requirements.txt        # pandas, numpy, scikit-learn, joblib, streamlit, plotly, matplotlib, seaborn, pytest
├── README.md
├── .gitignore
├── data/Mall_Customers.csv
├── model/kmeans_model.pkl  # KMeans K=5 (features: ingreso, gasto)
├── model/scaler.pkl        # StandardScaler ajustado al dataset
├── notebooks/
│   ├── Taller_final.ipynb          # Notebook original (preservado y completado)
│   └── Taller_final_backup.ipynb   # Respaldo previo a la mejora
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Carga + validación de esquema
│   ├── model_service.py    # Inferencia con artefactos (sin reentrenar)
│   ├── analysis.py         # Tablas de inercia y silueta
│   └── business_insights.py# Perfiles y recomendaciones
└── tests/
    ├── test_model_service.py   # Inferencia y compatibilidad de artefactos
    ├── test_data_validation.py # Esquema del dataset y equivalencia del modelo
    └── test_evaluation.py      # Regresión de inercias y siluetas verificadas
```

## Instalación (Windows + VS Code)

```powershell
cd "C:\Users\JESUS DAVID\Desktop\Taller_final_mineri_De_datos"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Uso

```powershell
# App (navegador en http://localhost:8501)
streamlit run app.py

# Tests
pytest -q
```

## Reproducibilidad

- Semilla fija (`random_state=42`, `n_init=10`).
- El scaler guardado conserva `mean_=[60.56, 50.2]` del dataset; la app solo aplica `transform()`.
- No se requieren internet, APIs pagas ni servicios en la nube tras instalar dependencias.
- Entorno verificado: scikit-learn 1.9.1 + Streamlit 1.66.1. Los artefactos se
  serializaron con scikit-learn 1.6.1 y emiten un `InconsistentVersionWarning` inocuo
  al cargarse (centroides idénticos al reentrenamiento); por eso se preservan intactos.
- La app usa `width="stretch"` (API vigente) en tablas y gráficos.

## Limitaciones y trabajo futuro

- Solo 200 registros de un único centro comercial; generalización limitada.
- Modelo 2D (ingreso, gasto): Edad y Género solo se usan descriptivamente.
- K-Means asume clusters esféricos y es sensible a la semilla/atípicos.
- Futuro: probar DBSCAN/GMM, incluir edad como feature, validación con datos nuevos,
  panel de seguimiento de campañas por segmento.
