# HepatoLab 🔬

Laboratorio de Machine Learning para Optimización de Modelos de Clasificación en Trasplante Hepático.

Adaptación de [ROC Lab](https://github.com/mcquaas/ROClab) (Gustavo Ross) para el proyecto de investigación:
**"Identificación de combinaciones D-R con mayor probabilidad de éxito en trasplante hepático en México"**

## Instalación rápida

```bash
# 1. Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 2. Instalar dependencias
pip install -r requirements_hepatolab.txt

# 3. Ejecutar
streamlit run HepatoLab.py
```

## Características

### Todo lo de ROC Lab, más:

| Feature | ROC Lab | HepatoLab |
|---------|---------|-----------|
| Carga CSV/Excel | ✅ | ✅ |
| Detección automática de tipos | ✅ | ✅ |
| Análisis exploratorio (dist/corr/pairplot) | ✅ | ✅ |
| Preprocesamiento (normalización, imputación, outliers) | ✅ | ✅ |
| 10 modelos ML con hiperparámetros | ✅ | ✅ |
| Curva ROC + Umbral ajustable | ✅ | ✅ |
| Matriz de confusión | ✅ | ✅ |
| Métricas (Sens, Spec, F1, Precision) | ✅ | ✅ |
| **Calculadora MELD 3.0, BAR, D-MELD, DRI** | ❌ | ✅ |
| **Comparación multi-modelo con overlay ROC** | ❌ | ✅ |
| **Validación cruzada K-Fold** | ❌ | ✅ |
| **Curva de calibración** | ❌ | ✅ |
| **Curva Precision-Recall** | ❌ | ✅ |
| **Brier Score** | ❌ | ✅ |
| **Exportación CSV de comparación** | ❌ | ✅ |
| **Feature importance para todos los modelos** | Parcial | ✅ |
| **Detección automática de variables MELD** | ❌ | ✅ |
| **Contexto clínico de trasplante** | ❌ | ✅ |
| **Hiperparámetros expandidos (subsample, num_leaves, etc.)** | ❌ | ✅ |

## Modelos disponibles

1. Regresión Logística (± LASSO L1)
2. KNN (con weights y métricas configurables)
3. Naive Bayes
4. SVM (kernels: rbf, linear, poly, sigmoid)
5. Árbol de Decisión
6. Random Forest
7. XGBoost
8. Gradient Boosting
9. AdaBoost
10. LightGBM

## Scores de Trasplante integrados

- **MELD 3.0** — Kim et al., Gastroenterology, 2022
- **BAR Score** — Dutkowski et al., Ann Surg, 2011 (cutoff LATAM ≥15)
- **D-MELD** — Halldorson et al., AJT, 2009 (umbral ≥1600)
- **DRI** — Feng et al., AJT, 2006

## Alineación con el protocolo

| Objetivo | Función en HepatoLab |
|----------|---------------------|
| OE-2: Validación de scores | Calculadora + comparación con modelos ML |
| OE-3: Variables predictivas | Feature importance + correlaciones |
| OE-4: Modelo ML D-R | Entrenamiento + comparación + validación cruzada |

## Licencia

GNU GPL v3 — Basado en ROC Lab de Gustavo Ross (gross@funsalud.org.mx)
