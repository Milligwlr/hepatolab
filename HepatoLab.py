#!/usr/bin/env python3
"""
HepatoLab — Laboratorio ML para Trasplante Hepático
====================================================
Adaptación de ROC Lab para el proyecto de investigación:
"Identificación de combinaciones D-R con mayor probabilidad de éxito
 en trasplante hepático en México"

OE-2: Validación de scores (MELD 3.0, BAR, SOFT, D-MELD, DRI)
OE-3: Variables predictivas clave (Cox + LASSO + SHAP)
OE-4: Modelo ML matching D-R

Autor original ROC Lab: Gustavo Ross (gross@funsalud.org.mx)
Adaptación HepatoLab: Proyecto HepatoMatch MX (2026)
Licencia: GNU GPL v3
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import warnings
import os
from datetime import datetime

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, cross_validate
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
)
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score, brier_score_loss,
    calibration_curve, log_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="HepatoLab — ML para Trasplante Hepático",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔬"
)

# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0e7490 100%);
        padding: 20px 30px; border-radius: 12px; margin-bottom: 20px;
        border: 1px solid #1e3a5f;
    }
    .main-header h1 { color: #f0f9ff; margin: 0; font-size: 28px; font-weight: 800; }
    .main-header h1 span { color: #06b6d4; }
    .main-header p { color: #94a3b8; margin: 4px 0 0 0; font-size: 13px; }
    
    .score-card {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border: 1px solid #1e3a5f; border-radius: 10px; padding: 16px;
        text-align: center; margin-bottom: 8px;
    }
    .score-card .label { color: #94a3b8; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }
    .score-card .value { font-size: 32px; font-weight: 800; font-family: 'JetBrains Mono', monospace; margin: 4px 0; }
    .score-card .desc { color: #64748b; font-size: 10px; }
    
    .risk-low { color: #10b981; }
    .risk-moderate { color: #f59e0b; }
    .risk-high { color: #ef4444; }
    
    .metric-box {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 12px; text-align: center;
    }
    .metric-box .metric-label { color: #64748b; font-size: 11px; font-weight: 600; text-transform: uppercase; }
    .metric-box .metric-value { font-size: 24px; font-weight: 800; color: #0f172a; font-family: monospace; }
    
    .warning-banner {
        background: linear-gradient(90deg, #fef3c7, #fff7ed);
        border: 1px solid #f59e0b40; border-left: 4px solid #f59e0b;
        border-radius: 8px; padding: 12px 16px; margin-bottom: 16px;
        font-size: 12px; color: #92400e;
    }
    
    .info-banner {
        background: linear-gradient(90deg, #ecfeff, #f0f9ff);
        border: 1px solid #06b6d440; border-left: 4px solid #06b6d4;
        border-radius: 8px; padding: 12px 16px; margin-bottom: 16px;
        font-size: 12px; color: #164e63;
    }
    
    div.stButton > button {
        background: linear-gradient(135deg, #0e7490, #06b6d4);
        color: white; border: none; border-radius: 8px;
        padding: 8px 20px; font-weight: 700; font-size: 13px;
    }
    div.stButton > button:hover { background: linear-gradient(135deg, #0891b2, #22d3ee); }
    
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0; padding: 8px 16px;
        font-weight: 600; font-size: 13px;
    }
    
    .footer-text { color: #94a3b8; font-size: 11px; text-align: center; margin-top: 40px; padding: 16px; border-top: 1px solid #e2e8f0; }
    
    [data-testid="stSidebar"] { background: #f8fafc; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SCORE CALCULATORS
# ═══════════════════════════════════════════════════════════════

def calc_meld3(bili, cr, inr, na, alb, sex_female, on_dialysis=False):
    """MELD 3.0 (Kim et al., Gastroenterology, 2022)"""
    bili = max(1.0, float(bili))
    cr = max(1.0, min(3.0, float(cr)))
    if on_dialysis:
        cr = 3.0
    inr = max(1.0, float(inr))
    na = min(137, max(125, float(na)))
    alb = min(3.5, max(1.5, float(alb)))
    female = 1 if sex_female else 0

    ln_b = np.log(bili)
    ln_c = np.log(cr)
    ln_i = np.log(inr)
    na_d = 137 - na
    al_d = 3.5 - alb

    meld = (1.33 * female + 4.56 * ln_b + 0.82 * na_d
            - 0.24 * na_d * ln_c + 9.09 * ln_i + 11.14 * ln_c
            + 1.85 * al_d - 1.83 * al_d * ln_c + 6)
    return max(6, min(40, round(meld)))


def calc_bar(r_age, meld, retx, life_support, d_age, cit):
    """BAR Score (Dutkowski et al., Ann Surg, 2011). Cutoff LATAM ≥15"""
    # Recipient age points
    r = float(r_age)
    rp = 0 if r < 40 else 3 if r <= 49 else 4 if r <= 59 else 5 if r <= 69 else 6
    # MELD points
    mp = 0 if meld < 15 else 4 if meld <= 19 else 7 if meld <= 24 else 9 if meld <= 29 else 10 if meld <= 34 else 13
    # Retransplant
    rtp = 4 if retx else 0
    # Life support
    lsp = 5 if life_support else 0
    # Donor age points
    d = float(d_age)
    dp = 0 if d < 40 else 2 if d <= 49 else 3 if d <= 59 else 4 if d <= 69 else 5
    # CIT points
    c = float(cit)
    cp = 0 if c <= 8 else 2 if c <= 11 else 3
    return rp + mp + rtp + lsp + dp + cp


def calc_dmeld(d_age, meld):
    """D-MELD = Donor age × Recipient MELD"""
    return round(float(d_age) * meld)


def calc_dri(d_age, cause_death, dcd, cit):
    """DRI (Feng et al., AJT, 2006)"""
    coef = 0
    age = float(d_age)
    if 40 <= age <= 49: coef += 0.154
    elif 50 <= age <= 59: coef += 0.274
    elif 60 <= age <= 69: coef += 0.424
    elif age >= 70: coef += 0.501

    cod = str(cause_death).lower()
    if 'anox' in cod: coef += 0.079
    elif 'acv' in cod or 'evc' in cod or 'stroke' in cod: coef += 0.145
    elif 'otro' in cod or 'other' in cod: coef += 0.184

    if dcd: coef += 0.411
    c = float(cit)
    if c > 8: coef += 0.066 * (c - 8)
    return round(np.exp(coef), 2)


def try_compute_scores(df):
    """Try to automatically compute transplant scores from column names."""
    scores = {}
    cols_lower = {c.lower().strip(): c for c in df.columns}

    # Check for MELD components
    meld_map = {
        'bilirubin': ['bilirubin', 'bili', 'bilirrubina', 'tbil', 'total_bilirubin'],
        'creatinine': ['creatinine', 'creat', 'creatinina', 'cr', 'scr'],
        'inr': ['inr'],
        'sodium': ['sodium', 'na', 'sodio', 'serum_sodium'],
        'albumin': ['albumin', 'alb', 'albumina'],
    }

    found = {}
    for key, aliases in meld_map.items():
        for alias in aliases:
            if alias in cols_lower:
                found[key] = cols_lower[alias]
                break

    if len(found) >= 3:  # At least bili, cr, inr for basic MELD
        scores['meld_components'] = found

    return scores


def identify_column_types(df):
    """Identify column types for preprocessing."""
    types = {'numeric': [], 'binary': [], 'categorical': [], 'date': [], 'text': []}
    
    for col in df.columns:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            types['categorical'].append(col)
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            types['date'].append(col)
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() <= 2:
                types['binary'].append(col)
            else:
                types['numeric'].append(col)
        elif df[col].dtype == object:
            if df[col].nunique() <= 2:
                types['binary'].append(col)
            elif df[col].nunique() < 20:
                types['categorical'].append(col)
            else:
                types['text'].append(col)
        else:
            types['categorical'].append(col)
    return types


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
@st.cache_data
def load_data(file):
    ext = file.name.split('.')[-1].lower()
    if ext in ['csv', 'txt']:
        for sep in [',', ';', '\t', None]:
            try:
                file.seek(0)
                if sep is None:
                    return pd.read_csv(file, sep=None, engine='python')
                return pd.read_csv(file, sep=sep)
            except:
                continue
    elif ext in ['xlsx', 'xls']:
        return pd.read_excel(file)
    raise ValueError(f"Formato no soportado: {ext}")


# ═══════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════
MODELS = {
    'Regresión Logística': LogisticRegression,
    'KNN': KNeighborsClassifier,
    'Naive Bayes': GaussianNB,
    'SVM': SVC,
    'Árbol de Decisión': DecisionTreeClassifier,
    'Random Forest': RandomForestClassifier,
    'Gradient Boosting': GradientBoostingClassifier,
    'AdaBoost': AdaBoostClassifier,
}
if HAS_XGB:
    MODELS['XGBoost'] = XGBClassifier
if HAS_LGBM:
    MODELS['LightGBM'] = lgb.LGBMClassifier

MODEL_DESCRIPTIONS = {
    'Regresión Logística': 'Modelo estadístico que utiliza función logística para modelar variable dependiente binaria. Robusto y bien calibrado, ideal como baseline. [Kim et al. usaron regresión logística para MELD 3.0]',
    'KNN': 'Algoritmo basado en instancias que clasifica según la mayoría de los K vecinos más cercanos. Sensible a la escala de las variables.',
    'Naive Bayes': 'Familia de algoritmos probabilísticos basados en el teorema de Bayes con asunción de independencia entre features.',
    'SVM': 'Máquinas de vectores de soporte. Encuentra el hiperplano óptimo de separación entre clases. Efectivo en espacios de alta dimensión.',
    'Árbol de Decisión': 'Método no paramétrico que divide los datos en subconjuntos basados en el valor de las features. Interpretable pero propenso a overfitting.',
    'Random Forest': 'Método ensemble que usa múltiples árboles de decisión para mejorar precisión y controlar overfitting. Robusto y versátil.',
    'Gradient Boosting': 'Técnica ensemble que construye modelos secuencialmente, corrigiendo errores de los anteriores. Alto rendimiento en datos tabulares.',
    'AdaBoost': 'Adaptive Boosting: combina clasificadores débiles para crear uno fuerte. Sensible a outliers.',
    'XGBoost': 'Gradient boosting optimizado. Zhang et al. (2022) mostraron mejoras de 6.7-17.4% AUC vs SOFT en datos UNOS. Briceño et al. alcanzaron AUC 0.80-0.94 con ANN en 11 centros españoles.',
    'LightGBM': 'Framework de gradient boosting basado en árboles, diseñado para entrenamiento eficiente y distribuido.',
}

# ═══════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="main-header">
    <h1>🔬 Hepato<span>Lab</span></h1>
    <p>Laboratorio de Machine Learning para Optimización de Modelos de Clasificación en Trasplante Hepático</p>
    <p style="color: #64748b; font-size: 11px; margin-top: 8px;">
        Protocolo: Identificación de combinaciones D-R óptimas | OE-2, OE-3, OE-4 | 
        Marco regulatorio: CENATRA / LGS / Art. 37 Reglamento MT
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    
    st.markdown("""
    <div class="warning-banner">
        <strong>⚠️ Herramienta de investigación</strong><br>
        No para uso clínico. Los modelos requieren validación con datos multicéntricos mexicanos (OE-4).
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📋 Instrucciones", expanded=False):
        st.markdown("""
        1. **Carga un archivo** CSV/Excel con datos estructurados
        2. **Selecciona** la variable objetivo y las features
        3. **Explora** distribuciones y correlaciones
        4. **Prepara** los datos (normalización, imputación, outliers)
        5. **Entrena** modelos y ajusta hiperparámetros
        6. **Evalúa** con curva ROC, matriz de confusión y métricas
        7. **Compara** modelos para encontrar el óptimo
        """)
    
    with st.expander("📊 Datasets sugeridos", expanded=False):
        st.markdown("""
        - **[Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)** — Target: `Outcome`
        - **[Cardiovascular Disease](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)** — Target: `cardio`
        - **[Breast Cancer Wisconsin](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)** — Target: `diagnosis`
        - **[Cardiovascular Risk 10yr](https://www.kaggle.com/code/bansodesandeep/cardiovascular-risk-prediction/input)** — Target: `TenYearCHD`
        - **Tu base multicéntrica D-R** — Target: supervivencia del injerto a 1 año
        """)
    
    with st.expander("📚 Scores de Trasplante", expanded=False):
        st.markdown("""
        **MELD 3.0** (Kim et al., 2022): Bilirrubina, Cr, INR, Na, Albúmina, Sexo  
        **BAR** (Dutkowski, 2011): 6 variables, cutoff LATAM ≥15  
        **D-MELD** (Halldorson, 2009): Edad donante × MELD, umbral ≥1600  
        **DRI** (Feng, 2006): Riesgo del donante, C-stat ~0.60  
        """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; font-size:10px; color:#94a3b8;">
        HepatoLab v1.0 | Basado en ROC Lab (G. Ross)<br>
        GNU GPL v3 | 2026
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# MAIN LAYOUT - Three columns like ROC Lab
# ═══════════════════════════════════════════════════════════════

col_data, col_model, col_perf = st.columns([3, 3, 3])

# ─── COLUMN 1: DATA ───────────────────────────────────────────
with col_data:
    st.subheader("1. 📂 Carga de Datos")
    
    uploaded_file = st.file_uploader(
        "Carga un archivo CSV o Excel con datos estructurados",
        type=['csv', 'xlsx', 'xls', 'txt'],
        help="Sube un archivo con variables numéricas y/o categóricas. El sistema detecta automáticamente los tipos de columnas."
    )
    
    if uploaded_file:
        try:
            if 'df' not in st.session_state or st.session_state.get('last_file') != uploaded_file.name:
                st.session_state.df = load_data(uploaded_file)
                st.session_state.last_file = uploaded_file.name
                # Reset comparison when new file loaded
                if 'model_comparison' in st.session_state:
                    del st.session_state['model_comparison']
            
            df = st.session_state.df.copy()
            
            # File info
            st.success(f"✅ **{len(df)}** registros × **{len(df.columns)}** columnas cargadas")
            
            # Detect column types
            col_types = identify_column_types(df)
            
            type_cols = st.columns(3)
            with type_cols[0]:
                st.caption(f"🔢 Numéricas: {len(col_types['numeric'])}")
            with type_cols[1]:
                st.caption(f"⚖️ Binarias: {len(col_types['binary'])}")
            with type_cols[2]:
                st.caption(f"🔤 Categóricas: {len(col_types['categorical'])}")
            
            # Preview
            with st.expander(f"👁️ Vista previa ({len(df)} registros)", expanded=False):
                st.dataframe(df.head(10), use_container_width=True, height=200)
            
            # Check for transplant scores
            detected_scores = try_compute_scores(df)
            if detected_scores.get('meld_components'):
                comps = detected_scores['meld_components']
                st.markdown(f"""
                <div class="info-banner">
                    <strong>🏥 Variables MELD detectadas:</strong> {', '.join(comps.values())}<br>
                    Si tu dataset contiene datos de trasplante, HepatoLab puede calcular scores automáticamente.
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Target variable selection
            possible_targets = ['target', 'y', 'outcome', 'objective', 'class', 'diagnosis',
                                'result', 'survival', 'graft_survival', 'ead', 'death',
                                'mortalidad', 'supervivencia', 'injerto', 'cardio',
                                'tenyearchd', 'diabetes']
            
            preselected = None
            for c in df.columns:
                if c.lower().strip() in possible_targets:
                    preselected = c
                    break
            if preselected is None:
                for c in df.columns:
                    if df[c].nunique() == 2:
                        preselected = c
                        break
            
            target_idx = df.columns.get_loc(preselected) if preselected and preselected in df.columns else 0
            target_column = st.selectbox(
                "🎯 Variable objetivo (target)",
                df.columns, index=target_idx,
                help="Variable a predecir. En trasplante: supervivencia del injerto, EAD, mortalidad, etc."
            )
            
            # Feature selection
            available_features = [c for c in df.columns if c != target_column]
            with st.expander("🔧 Seleccionar variables predictivas", expanded=False):
                selected_columns = st.multiselect(
                    "Variables para el modelo",
                    available_features,
                    default=available_features,
                    help="Incluye/excluye columnas para observar cómo cambian las métricas. En trasplante: componentes de scores, variables emergentes (sarcopenia, esteatosis, etc.)"
                )
            
            if not selected_columns:
                st.warning("⚠️ Selecciona al menos una variable predictiva.")
                st.stop()
            
            st.markdown("---")
            
            # ─── EXPLORATORY ANALYSIS ─────────────────────────
            st.subheader("2. 🔍 Análisis Exploratorio")
            
            tab_dist, tab_corr, tab_pair = st.tabs(["📊 Distribuciones", "🔗 Correlaciones", "👁️ Pairplot"])
            
            with tab_dist:
                # Statistics
                st.markdown("**Estadísticas descriptivas:**")
                desc_df = df[selected_columns].describe()
                st.dataframe(desc_df, use_container_width=True, height=220)
                
                # Missing values summary
                missing = df[selected_columns].isnull().sum()
                missing_pct = (missing / len(df) * 100).round(1)
                if missing.sum() > 0:
                    miss_df = pd.DataFrame({'Faltantes': missing[missing > 0], '%': missing_pct[missing_pct > 0]})
                    st.warning(f"⚠️ {missing.sum()} valores faltantes detectados")
                    st.dataframe(miss_df, use_container_width=True)
                
                # Distribution plots
                num_cols = [c for c in selected_columns if c in col_types['numeric'] or c in col_types['binary']]
                if num_cols and st.checkbox("Mostrar histogramas", value=False):
                    n_plots = min(len(num_cols), 12)
                    n_cols_grid = 3
                    n_rows = int(np.ceil(n_plots / n_cols_grid))
                    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(12, 3 * n_rows))
                    axes = np.array(axes).flatten() if n_plots > 1 else [axes]
                    for i, col in enumerate(num_cols[:n_plots]):
                        try:
                            df[col].hist(ax=axes[i], bins=30, color='#0e7490', alpha=0.7, edgecolor='white')
                            axes[i].set_title(col, fontsize=9, fontweight='bold')
                            axes[i].tick_params(labelsize=7)
                        except:
                            pass
                    for j in range(i + 1, len(axes)):
                        axes[j].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            
            with tab_corr:
                numeric_for_corr = df[selected_columns].select_dtypes(include=[np.number])
                if len(numeric_for_corr.columns) >= 2:
                    corr = numeric_for_corr.corr()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    mask = np.triu(np.ones_like(corr, dtype=bool))
                    sns.heatmap(corr, mask=mask, annot=True, cmap='RdBu_r', center=0,
                                ax=ax, annot_kws={"size": 7}, fmt='.2f',
                                linewidths=0.5, linecolor='white',
                                vmin=-1, vmax=1)
                    ax.set_title("Matriz de Correlación", fontsize=12, fontweight='bold')
                    ax.tick_params(labelsize=7)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Top correlations with target
                    if target_column in numeric_for_corr.columns:
                        target_corr = corr[target_column].drop(target_column).abs().sort_values(ascending=False)
                        st.markdown("**Top correlaciones con target:**")
                        for feat, val in target_corr.head(5).items():
                            direction = "+" if corr.loc[feat, target_column] > 0 else "−"
                            bar = "█" * int(val * 20)
                            st.caption(f"`{direction}{val:.3f}` {bar} **{feat}**")
                else:
                    st.info("Se necesitan al menos 2 variables numéricas para correlaciones.")
            
            with tab_pair:
                if st.button("🎨 Generar Pairplot", help="Puede tomar unos segundos con muchas variables"):
                    plot_cols = selected_columns[:6]  # Limit to 6 for performance
                    if target_column not in plot_cols:
                        plot_cols_with_target = plot_cols + [target_column]
                    else:
                        plot_cols_with_target = plot_cols
                    
                    numeric_plot = df[plot_cols_with_target].select_dtypes(include=[np.number])
                    if len(numeric_plot.columns) >= 2:
                        with st.spinner("Generando pairplot..."):
                            fig = sns.pairplot(
                                numeric_plot, hue=target_column if target_column in numeric_plot.columns else None,
                                diag_kind='kde', plot_kws={'alpha': 0.5, 's': 15},
                                palette='viridis', height=1.8
                            )
                            st.pyplot(fig)
                            plt.close()
                    else:
                        st.info("Necesitas al menos 2 variables numéricas.")
            
        except Exception as e:
            st.error(f"Error al cargar datos: {e}")
            st.stop()


# ─── COLUMN 2: MODEL TRAINING ─────────────────────────────────
with col_model:
    if uploaded_file and 'df' in st.session_state and selected_columns:
        df = st.session_state.df.copy()
        
        # ─── DATA PREPARATION ─────────────────────────
        st.subheader("3. ⚙️ Preparación de Datos")
        
        # Encode categorical/binary text columns
        processing_log = []
        for col in selected_columns + [target_column]:
            if df[col].dtype == object:
                try:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                    processing_log.append(f"✅ `{col}` codificada: {mapping}")
                except Exception as e:
                    processing_log.append(f"⚠️ Error en `{col}`: {e}")
        
        if processing_log:
            with st.expander("🔄 Preprocesamiento de variables categóricas"):
                for msg in processing_log:
                    st.caption(msg)
        
        features = df[selected_columns]
        target = df[target_column]
        
        # Missing values
        total_missing = features.isnull().sum().sum()
        if total_missing > 0:
            handle_missing = st.selectbox(
                "🩹 Valores faltantes",
                ["Imputar (media/moda)", "Eliminar filas", "No hacer nada"],
                help="Imputar reemplaza NaN con la media (numéricas) o moda (categóricas). Eliminar descarta filas incompletas."
            )
            if handle_missing == "Eliminar filas":
                mask = features.notnull().all(axis=1) & target.notnull()
                features = features[mask]
                target = target[mask]
                st.caption(f"Registros después de eliminar: {len(features)}")
            elif handle_missing == "Imputar (media/moda)":
                num_feats = features.select_dtypes(include=[np.number]).columns
                cat_feats = features.select_dtypes(exclude=[np.number]).columns
                if len(num_feats) > 0:
                    imp_num = SimpleImputer(strategy='mean')
                    features[num_feats] = imp_num.fit_transform(features[num_feats])
                if len(cat_feats) > 0:
                    imp_cat = SimpleImputer(strategy='most_frequent')
                    features[cat_feats] = imp_cat.fit_transform(features[cat_feats])
                st.caption("✅ Valores imputados")
        else:
            st.caption("✅ Sin valores faltantes")
        
        # Options row
        opt_cols = st.columns(3)
        with opt_cols[0]:
            normalize = st.toggle("📐 Normalizar", value=False, help="StandardScaler para modelos sensibles a escala (LR, SVM, KNN)")
        with opt_cols[1]:
            use_lasso = st.toggle("🎯 LASSO (L1)", value=False, help="Regularización L1 para selección de variables (solo Regresión Logística)")
        with opt_cols[2]:
            remove_outliers = st.toggle("🚫 Outliers", value=False, help="Elimina observaciones con Z-score > 3 en variables numéricas")
        
        # Remove outliers
        if remove_outliers:
            from scipy import stats
            num_feats = features.select_dtypes(include=[np.number]).columns
            if len(num_feats) > 0:
                z = np.abs(stats.zscore(features[num_feats], nan_policy='omit'))
                mask = (z < 3).all(axis=1)
                features = features[mask]
                target = target[mask]
                st.caption(f"Registros sin outliers: {len(features)}")
        
        st.markdown("---")
        
        # Sample splitting
        st.markdown("**📊 División de Muestras**")
        train_pct = st.slider("% Entrenamiento", 50, 95, 80, 5, help="Proporción de datos para entrenamiento") / 100.0
        test_pct = 1 - train_pct
        
        n_train = int(len(features) * train_pct)
        n_test = len(features) - n_train
        
        # Visual split bar
        fig_split, ax_split = plt.subplots(figsize=(6, 0.6))
        ax_split.barh(0, n_train, color='#0e7490', height=0.5)
        ax_split.barh(0, n_test, left=n_train, color='#06b6d4', height=0.5)
        ax_split.text(n_train / 2, 0, f"Entrenamiento: {n_train}", ha='center', va='center', color='white', fontsize=9, fontweight='bold')
        ax_split.text(n_train + n_test / 2, 0, f"Prueba: {n_test}", ha='center', va='center', color='white', fontsize=9, fontweight='bold')
        ax_split.axis('off')
        st.pyplot(fig_split)
        plt.close()
        
        n_folds = st.slider("K-Fold Particiones", 2, 10, 5, help="Número de particiones para validación cruzada")
        
        # Classification info
        n_classes = target.nunique()
        class_values = sorted(target.unique())
        st.caption(f"🎯 Clasificación con clases: {class_values}")
        
        if n_classes > 2:
            st.warning("⚠️ Problema multiclase detectado. Algunas métricas se ajustarán automáticamente.")
        
        # Train/test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=test_pct, random_state=42, stratify=target
            )
        except:
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=test_pct, random_state=42
            )
        
        st.markdown("---")
        
        # ─── MODEL TRAINING ───────────────────────────
        st.subheader("4. 🧠 Entrenamiento del Modelo")
        
        model_name = st.selectbox(
            "Selecciona un modelo",
            list(MODELS.keys()),
            help="Prueba diferentes modelos para encontrar la mejor curva ROC"
        )
        
        st.caption(MODEL_DESCRIPTIONS.get(model_name, ""))
        
        ModelClass = MODELS[model_name]
        
        # Hyperparameters
        with st.expander("🎛️ Ajustar hiperparámetros", expanded=True):
            if model_name == 'Regresión Logística':
                C_val = st.slider("Inversa de regularización (C)", 0.01, 10.0, 1.0, 0.01)
                solver = 'saga' if use_lasso else 'lbfgs'
                penalty = 'l1' if use_lasso else 'l2'
                model = ModelClass(C=C_val, penalty=penalty, solver=solver, max_iter=2000, random_state=42)
            
            elif model_name == 'KNN':
                n_neighbors = st.slider("Número de vecinos (K)", 1, 31, 5, 2)
                weights = st.selectbox("Pesos", ['uniform', 'distance'])
                metric = st.selectbox("Métrica", ['minkowski', 'euclidean', 'manhattan'])
                model = ModelClass(n_neighbors=n_neighbors, weights=weights, metric=metric)
            
            elif model_name == 'Naive Bayes':
                st.caption("Sin hiperparámetros configurables.")
                model = ModelClass()
            
            elif model_name == 'SVM':
                C_val = st.slider("Regularización (C)", 0.01, 10.0, 1.0, 0.01)
                kernel = st.selectbox("Kernel", ['rbf', 'linear', 'poly', 'sigmoid'], index=0)
                model = ModelClass(C=C_val, kernel=kernel, probability=True, random_state=42)
            
            elif model_name == 'Árbol de Decisión':
                max_depth = st.slider("Profundidad máxima", 1, 30, 5)
                min_samples_split = st.slider("Min muestras para split", 2, 20, 2)
                model = ModelClass(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
            
            elif model_name == 'Random Forest':
                n_estimators = st.slider("Número de árboles", 10, 500, 100, 10)
                max_depth = st.slider("Profundidad máxima", 1, 30, 5)
                min_samples_split = st.slider("Min muestras para split", 2, 20, 2)
                model = ModelClass(n_estimators=n_estimators, max_depth=max_depth,
                                   min_samples_split=min_samples_split, random_state=42, n_jobs=-1)
            
            elif model_name == 'XGBoost':
                learning_rate = st.slider("Learning rate", 0.001, 0.3, 0.1, 0.005)
                n_estimators = st.slider("Número de estimadores", 10, 500, 100, 10)
                max_depth = st.slider("Profundidad máxima", 1, 15, 5)
                subsample = st.slider("Subsample", 0.5, 1.0, 0.8, 0.05)
                model = ModelClass(learning_rate=learning_rate, n_estimators=n_estimators,
                                   max_depth=max_depth, subsample=subsample,
                                   random_state=42, eval_metric='logloss', use_label_encoder=False)
            
            elif model_name == 'Gradient Boosting':
                learning_rate = st.slider("Learning rate", 0.01, 0.3, 0.1, 0.01)
                n_estimators = st.slider("Número de estimadores", 10, 500, 100, 10)
                max_depth = st.slider("Profundidad máxima", 1, 15, 5)
                model = ModelClass(learning_rate=learning_rate, n_estimators=n_estimators,
                                   max_depth=max_depth, random_state=42)
            
            elif model_name == 'AdaBoost':
                n_estimators = st.slider("Número de estimadores", 10, 500, 50, 10)
                learning_rate = st.slider("Learning rate", 0.01, 2.0, 1.0, 0.01)
                model = ModelClass(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
            
            elif model_name == 'LightGBM':
                learning_rate = st.slider("Learning rate", 0.01, 0.3, 0.1, 0.01)
                n_estimators = st.slider("Número de estimadores", 10, 500, 100, 10)
                max_depth = st.slider("Profundidad máxima (-1=sin límite)", -1, 20, -1)
                num_leaves = st.slider("Número de hojas", 8, 128, 31)
                model = ModelClass(learning_rate=learning_rate, n_estimators=n_estimators,
                                   max_depth=max_depth, num_leaves=num_leaves,
                                   random_state=42, verbose=-1)
        
        # Auto preprocessing
        auto_preprocess = st.toggle("🔄 Preprocesamiento automático", value=True,
                                     help="Normaliza automáticamente para modelos sensibles a escala")
        
        # Apply preprocessing
        X_train_proc = X_train.copy()
        X_test_proc = X_test.copy()
        
        # Ensure numeric
        X_train_proc = X_train_proc.apply(pd.to_numeric, errors='coerce').fillna(0)
        X_test_proc = X_test_proc.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        scaler = None
        if auto_preprocess and model_name in ['Regresión Logística', 'SVM', 'KNN']:
            scaler = StandardScaler()
            X_train_proc = pd.DataFrame(scaler.fit_transform(X_train_proc), columns=X_train_proc.columns, index=X_train_proc.index)
            X_test_proc = pd.DataFrame(scaler.transform(X_test_proc), columns=X_test_proc.columns, index=X_test_proc.index)
            st.caption("✅ Datos normalizados automáticamente (StandardScaler)")
        elif normalize:
            scaler = StandardScaler()
            X_train_proc = pd.DataFrame(scaler.fit_transform(X_train_proc), columns=X_train_proc.columns, index=X_train_proc.index)
            X_test_proc = pd.DataFrame(scaler.transform(X_test_proc), columns=X_test_proc.columns, index=X_test_proc.index)
            st.caption("✅ Normalización manual aplicada")
        
        # Train
        try:
            model.fit(X_train_proc, y_train)
            st.success(f"✅ Modelo **{model_name}** entrenado")
        except Exception as e:
            st.error(f"Error de entrenamiento: {e}")
            st.stop()
        
        # Predictions
        try:
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test_proc)
            elif hasattr(model, "decision_function"):
                scores_raw = model.decision_function(X_test_proc)
                scores_norm = (scores_raw - scores_raw.min()) / (scores_raw.max() - scores_raw.min() + 1e-8)
                if scores_norm.ndim == 1:
                    y_pred_proba = np.vstack([1 - scores_norm, scores_norm]).T
                else:
                    y_pred_proba = scores_norm
            else:
                y_pred = model.predict(X_test_proc)
                y_pred_proba = np.zeros((len(y_pred), n_classes))
                for i, p in enumerate(y_pred):
                    y_pred_proba[i, int(p)] = 1.0
        except Exception as e:
            st.error(f"Error en predicciones: {e}")
            y_pred = model.predict(X_test_proc)
            y_pred_proba = np.zeros((len(y_pred), n_classes))
            for i, p in enumerate(y_pred):
                y_pred_proba[i, int(p)] = 1.0
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=selected_columns).sort_values(ascending=True)
            top_n = min(15, len(feat_imp))
            
            st.markdown("**📊 Importancia de Variables:**")
            fig_imp, ax_imp = plt.subplots(figsize=(5, max(2, top_n * 0.25)))
            feat_imp.tail(top_n).plot(kind='barh', ax=ax_imp, color='#0e7490', edgecolor='white')
            ax_imp.set_xlabel("Importancia", fontsize=9)
            ax_imp.tick_params(labelsize=8)
            ax_imp.set_title("Feature Importance", fontsize=10, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_imp)
            plt.close()
        elif model_name == 'Regresión Logística':
            coefs = pd.Series(model.coef_[0], index=selected_columns).abs().sort_values(ascending=True)
            top_n = min(15, len(coefs))
            st.markdown("**📊 Coeficientes (valor absoluto):**")
            fig_imp, ax_imp = plt.subplots(figsize=(5, max(2, top_n * 0.25)))
            coefs.tail(top_n).plot(kind='barh', ax=ax_imp, color='#0e7490', edgecolor='white')
            ax_imp.tick_params(labelsize=8)
            plt.tight_layout()
            st.pyplot(fig_imp)
            plt.close()


# ─── COLUMN 3: PERFORMANCE ────────────────────────────────────
with col_perf:
    if uploaded_file and 'df' in st.session_state and selected_columns:
        st.subheader("5. 📈 Rendimiento")
        
        try:
            if n_classes == 2:
                # ─── ROC CURVE ─────────────────────────
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                
                fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
                ax_roc.plot(fpr, tpr, color='#0e7490', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
                ax_roc.plot([0, 1], [0, 1], '--', color='#94a3b8', lw=1)
                ax_roc.fill_between(fpr, tpr, alpha=0.1, color='#0e7490')
                ax_roc.set_xlabel('1 - Especificidad (FPR)', fontsize=9)
                ax_roc.set_ylabel('Sensibilidad (TPR)', fontsize=9)
                ax_roc.set_title(f'Curva ROC (AUC = {roc_auc:.3f})', fontsize=11, fontweight='bold')
                ax_roc.legend(fontsize=8)
                ax_roc.tick_params(labelsize=8)
                ax_roc.set_xlim([0, 1])
                ax_roc.set_ylim([0, 1.02])
                ax_roc.grid(True, alpha=0.2)
                plt.tight_layout()
                st.pyplot(fig_roc)
                plt.close()
                
                # Threshold slider
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = float(thresholds[optimal_idx])
                threshold = st.slider(
                    f"Umbral (óptimo: {optimal_threshold:.2f})",
                    0.0, 1.0, optimal_threshold, 0.01,
                    help="Ajusta el umbral para clasificaciones más estrictas o permisivas"
                )
                
                y_pred_adjusted = (y_pred_proba[:, 1] >= threshold).astype(int)
                
                # ─── CONFUSION MATRIX ──────────────────
                cm = confusion_matrix(y_test, y_pred_adjusted)
                fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                            annot_kws={"size": 14, "fontweight": "bold"},
                            linewidths=1, linecolor='white')
                ax_cm.set_xlabel('Predicción', fontsize=9)
                ax_cm.set_ylabel('Real', fontsize=9)
                ax_cm.set_title('Matriz de Confusión', fontsize=10, fontweight='bold')
                ax_cm.tick_params(labelsize=8)
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close()
                
                # ─── METRICS ───────────────────────────
                sensitivity = recall_score(y_test, y_pred_adjusted, zero_division=0)
                specificity = recall_score(y_test, y_pred_adjusted, pos_label=0, zero_division=0)
                precision = precision_score(y_test, y_pred_adjusted, zero_division=0)
                f1 = f1_score(y_test, y_pred_adjusted, zero_division=0)
                accuracy = accuracy_score(y_test, y_pred_adjusted)
                brier = brier_score_loss(y_test, y_pred_proba[:, 1])
                
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Sensibilidad", f"{sensitivity:.3f}")
                    st.metric("Precisión", f"{precision:.3f}")
                    st.metric("AUC-ROC", f"{roc_auc:.3f}")
                with m2:
                    st.metric("Especificidad", f"{specificity:.3f}")
                    st.metric("F1-Score", f"{f1:.3f}")
                    st.metric("Brier Score", f"{brier:.3f}")
                
                # Classification report
                report = classification_report(y_test, y_pred_adjusted, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose()
                st.markdown("**Tabla de Métricas:**")
                st.dataframe(report_df[['precision', 'recall', 'f1-score', 'support']], use_container_width=True)
                
                # ─── CROSS-VALIDATION ──────────────────
                st.markdown("---")
                st.markdown("**🔄 Validación Cruzada:**")
                try:
                    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
                    X_all = features.apply(pd.to_numeric, errors='coerce').fillna(0)
                    if scaler:
                        X_all_scaled = pd.DataFrame(scaler.fit_transform(X_all), columns=X_all.columns, index=X_all.index)
                    else:
                        X_all_scaled = X_all
                    
                    cv_scores = cross_val_score(model, X_all_scaled, target, cv=cv, scoring='roc_auc')
                    st.caption(f"AUC-ROC ({n_folds}-fold): **{cv_scores.mean():.3f}** ± {cv_scores.std():.3f}")
                    st.caption(f"Folds: {[f'{s:.3f}' for s in cv_scores]}")
                except Exception as e:
                    st.caption(f"CV no disponible: {e}")
                
                # ─── CALIBRATION CURVE ─────────────────
                if st.checkbox("📈 Curva de Calibración", value=False):
                    try:
                        prob_true, prob_pred = calibration_curve(y_test, y_pred_proba[:, 1], n_bins=10)
                        fig_cal, ax_cal = plt.subplots(figsize=(4, 3.5))
                        ax_cal.plot(prob_pred, prob_true, 'o-', color='#0e7490', label=model_name, lw=2, markersize=5)
                        ax_cal.plot([0, 1], [0, 1], '--', color='#94a3b8', label='Calibración perfecta')
                        ax_cal.set_xlabel('Probabilidad predicha', fontsize=9)
                        ax_cal.set_ylabel('Proporción real positivos', fontsize=9)
                        ax_cal.set_title('Curva de Calibración', fontsize=10, fontweight='bold')
                        ax_cal.legend(fontsize=8)
                        ax_cal.grid(True, alpha=0.2)
                        plt.tight_layout()
                        st.pyplot(fig_cal)
                        plt.close()
                    except:
                        st.caption("No se pudo generar la curva de calibración.")
                
                # ─── PRECISION-RECALL CURVE ────────────
                if st.checkbox("📉 Curva Precision-Recall", value=False):
                    prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
                    ap = average_precision_score(y_test, y_pred_proba[:, 1])
                    fig_pr, ax_pr = plt.subplots(figsize=(4, 3.5))
                    ax_pr.plot(rec_vals, prec_vals, color='#0e7490', lw=2, label=f'AP = {ap:.3f}')
                    ax_pr.set_xlabel('Recall', fontsize=9)
                    ax_pr.set_ylabel('Precision', fontsize=9)
                    ax_pr.set_title('Precision-Recall', fontsize=10, fontweight='bold')
                    ax_pr.legend(fontsize=8)
                    ax_pr.grid(True, alpha=0.2)
                    plt.tight_layout()
                    st.pyplot(fig_pr)
                    plt.close()
                
                # ─── ADD TO COMPARISON ─────────────────
                st.markdown("---")
                if st.button("➕ Agregar a comparación de modelos"):
                    if 'model_comparison' not in st.session_state:
                        st.session_state.model_comparison = []
                    
                    entry = {
                        'Modelo': model_name,
                        'AUC': round(roc_auc, 4),
                        'Accuracy': round(accuracy, 4),
                        'Sensibilidad': round(sensitivity, 4),
                        'Especificidad': round(specificity, 4),
                        'Precisión': round(precision, 4),
                        'F1': round(f1, 4),
                        'Brier': round(brier, 4),
                        'Umbral': round(threshold, 2),
                        'Timestamp': datetime.now().strftime('%H:%M:%S'),
                    }
                    
                    # Store ROC data for overlay plot
                    entry['_fpr'] = fpr.tolist()
                    entry['_tpr'] = tpr.tolist()
                    
                    st.session_state.model_comparison.append(entry)
                    st.success(f"✅ {model_name} (AUC={roc_auc:.3f}) agregado")
                
            else:
                # Multiclass
                y_pred_adjusted = np.argmax(y_pred_proba, axis=1)
                st.caption("Problema multiclase — ROC OVR")
                try:
                    from sklearn.preprocessing import label_binarize
                    y_test_bin = label_binarize(y_test, classes=class_values)
                    if y_test_bin.shape[1] == y_pred_proba.shape[1]:
                        roc_auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr')
                        st.metric("AUC-ROC (OVR)", f"{roc_auc:.3f}")
                except:
                    pass
                
                cm = confusion_matrix(y_test, y_pred_adjusted)
                fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                ax_cm.set_xlabel('Predicción')
                ax_cm.set_ylabel('Real')
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close()
                
                report = classification_report(y_test, y_pred_adjusted, output_dict=True, zero_division=0)
                st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
                
        except Exception as e:
            st.error(f"Error en rendimiento: {e}")


# ═══════════════════════════════════════════════════════════════
# MODEL COMPARISON TABLE (below main columns)
# ═══════════════════════════════════════════════════════════════
if 'model_comparison' in st.session_state and st.session_state.model_comparison:
    st.markdown("---")
    st.subheader("6. 🏆 Comparación de Modelos")
    
    comp_df = pd.DataFrame([{k: v for k, v in e.items() if not k.startswith('_')} for e in st.session_state.model_comparison])
    
    # Highlight best
    st.dataframe(
        comp_df.style.highlight_max(subset=['AUC', 'Accuracy', 'Sensibilidad', 'Especificidad', 'F1'], color='#d1fae5')
                     .highlight_min(subset=['Brier'], color='#d1fae5'),
        use_container_width=True
    )
    
    # Overlay ROC curves
    comp_col1, comp_col2 = st.columns([1, 1])
    
    with comp_col1:
        fig_comp, ax_comp = plt.subplots(figsize=(6, 5))
        colors = ['#0e7490', '#f59e0b', '#ef4444', '#10b981', '#8b5cf6', '#ec4899', '#6366f1', '#14b8a6', '#f97316', '#a855f7']
        
        for i, entry in enumerate(st.session_state.model_comparison):
            if '_fpr' in entry and '_tpr' in entry:
                color = colors[i % len(colors)]
                ax_comp.plot(entry['_fpr'], entry['_tpr'], color=color, lw=2,
                            label=f"{entry['Modelo']} (AUC={entry['AUC']:.3f})")
        
        ax_comp.plot([0, 1], [0, 1], '--', color='#94a3b8', lw=1)
        ax_comp.set_xlabel('1 - Especificidad', fontsize=10)
        ax_comp.set_ylabel('Sensibilidad', fontsize=10)
        ax_comp.set_title('Comparación de Curvas ROC', fontsize=12, fontweight='bold')
        ax_comp.legend(fontsize=8, loc='lower right')
        ax_comp.grid(True, alpha=0.2)
        ax_comp.set_xlim([0, 1])
        ax_comp.set_ylim([0, 1.02])
        plt.tight_layout()
        st.pyplot(fig_comp)
        plt.close()
    
    with comp_col2:
        # Bar chart comparison
        metrics_to_plot = ['AUC', 'Sensibilidad', 'Especificidad', 'F1']
        plot_data = comp_df[['Modelo'] + metrics_to_plot].melt(id_vars='Modelo', var_name='Métrica', value_name='Valor')
        
        fig_bar, ax_bar = plt.subplots(figsize=(6, 5))
        x = np.arange(len(comp_df))
        width = 0.18
        
        for i, metric in enumerate(metrics_to_plot):
            vals = comp_df[metric].values
            ax_bar.bar(x + i * width, vals, width, label=metric, color=colors[i], alpha=0.85)
        
        ax_bar.set_xticks(x + width * 1.5)
        ax_bar.set_xticklabels(comp_df['Modelo'], rotation=30, ha='right', fontsize=8)
        ax_bar.set_ylabel('Valor', fontsize=10)
        ax_bar.set_title('Métricas por Modelo', fontsize=12, fontweight='bold')
        ax_bar.legend(fontsize=8)
        ax_bar.set_ylim([0, 1.1])
        ax_bar.grid(True, axis='y', alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close()
    
    # Export
    exp_col1, exp_col2 = st.columns([1, 1])
    with exp_col1:
        csv_data = comp_df.to_csv(index=False)
        st.download_button(
            "📥 Exportar comparación (CSV)",
            csv_data, "hepatolab_comparison.csv", "text/csv"
        )
    with exp_col2:
        if st.button("🗑️ Limpiar comparación"):
            del st.session_state['model_comparison']
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# TRANSPLANT SCORE CALCULATOR (expandable section)
# ═══════════════════════════════════════════════════════════════
if uploaded_file:
    st.markdown("---")
    with st.expander("🏥 Calculadora de Scores de Trasplante (MELD 3.0, BAR, D-MELD, DRI)", expanded=False):
        st.markdown("""
        <div class="info-banner">
            Calcula scores clínicos para un par donador-receptor específico. 
            Estos scores pueden agregarse como features al dataset para mejorar la predicción.
        </div>
        """, unsafe_allow_html=True)
        
        sc1, sc2 = st.columns(2)
        
        with sc1:
            st.markdown("**🫀 Receptor**")
            r_bili = st.number_input("Bilirrubina (mg/dL)", 0.1, 50.0, 4.2, 0.1, key="r_bili")
            r_cr = st.number_input("Creatinina (mg/dL)", 0.1, 15.0, 1.3, 0.1, key="r_cr")
            r_inr = st.number_input("INR", 0.5, 10.0, 1.8, 0.1, key="r_inr")
            r_na = st.number_input("Sodio (mEq/L)", 110, 150, 131, key="r_na")
            r_alb = st.number_input("Albúmina (g/dL)", 0.5, 5.5, 2.6, 0.1, key="r_alb")
            r_female = st.checkbox("Sexo femenino", value=True, key="r_fem")
            r_age = st.number_input("Edad receptor", 18, 85, 56, key="r_age")
            r_retx = st.checkbox("Retrasplante", value=False, key="r_retx")
            r_icu = st.checkbox("UCI/Soporte vital", value=False, key="r_icu")
            r_dial = st.checkbox("Diálisis ≥2x/sem", value=False, key="r_dial")
        
        with sc2:
            st.markdown("**🫀 Donante**")
            d_age = st.number_input("Edad donante", 1, 90, 52, key="d_age")
            d_cause = st.selectbox("Causa de muerte", ["Trauma", "ACV (EVC)", "Anoxia", "Otro"], key="d_cause")
            d_dcd = st.checkbox("DCD (vs DBD)", value=False, key="d_dcd")
            d_cit = st.number_input("TIF estimado (horas)", 0.0, 24.0, 9.0, 0.5, key="d_cit")
        
        if st.button("📊 Calcular Scores", key="calc_scores"):
            meld = calc_meld3(r_bili, r_cr, r_inr, r_na, r_alb, r_female, r_dial)
            bar = calc_bar(r_age, meld, r_retx, r_icu, d_age, d_cit)
            dmeld = calc_dmeld(d_age, meld)
            dri = calc_dri(d_age, d_cause, d_dcd, d_cit)
            
            sc_cols = st.columns(4)
            
            with sc_cols[0]:
                meld_class = "risk-high" if meld > 25 else "risk-moderate" if meld > 15 else "risk-low"
                st.markdown(f"""
                <div class="score-card">
                    <div class="label">MELD 3.0</div>
                    <div class="value {meld_class}">{meld}</div>
                    <div class="desc">Kim et al., 2022 | Rango: 6-40</div>
                </div>""", unsafe_allow_html=True)
            
            with sc_cols[1]:
                bar_class = "risk-high" if bar >= 15 else "risk-low"
                st.markdown(f"""
                <div class="score-card">
                    <div class="label">BAR Score</div>
                    <div class="value {bar_class}">{bar}</div>
                    <div class="desc">Cutoff LATAM ≥15 (vs. ≥18 original)</div>
                </div>""", unsafe_allow_html=True)
            
            with sc_cols[2]:
                dm_class = "risk-high" if dmeld >= 1600 else "risk-low"
                st.markdown(f"""
                <div class="score-card">
                    <div class="label">D-MELD</div>
                    <div class="value {dm_class}">{dmeld}</div>
                    <div class="desc">Edad donante × MELD | Umbral ≥1600</div>
                </div>""", unsafe_allow_html=True)
            
            with sc_cols[3]:
                dri_class = "risk-high" if dri > 1.8 else "risk-moderate" if dri > 1.5 else "risk-low"
                st.markdown(f"""
                <div class="score-card">
                    <div class="label">DRI</div>
                    <div class="value {dri_class}">{dri}</div>
                    <div class="desc">Feng et al., 2006 | Baseline: 1.00</div>
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer-text">
    <strong>HepatoLab v1.0</strong> — Laboratorio ML para Trasplante Hepático<br>
    Basado en <a href="https://github.com/mcquaas/ROClab" target="_blank">ROC Lab</a> (Gustavo Ross, GNU GPL v3)<br>
    Adaptado para el protocolo de investigación: Identificación de combinaciones D-R óptimas en TH en México<br>
    <em>Este software es herramienta de investigación. No sustituye la decisión del Comité Interno de Trasplantes (Art. 37, Reglamento LGS-MT).</em>
</div>
""", unsafe_allow_html=True)
