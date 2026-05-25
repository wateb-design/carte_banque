# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from datasets import load_dataset

st.set_page_config(
    page_title="FraudDetect Pro - Détection de fraudes bancaires",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #00a0e9;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-left: 4px solid #ff9800;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation du session state
if 'df' not in st.session_state:
    st.session_state.df = None

# Fonction de chargement des données
@st.cache_data
def charger_donnees():
    with st.spinner("📡 Téléchargement du dataset depuis Hugging Face..."):
        dataset = load_dataset("Wateb26/creditcard", split="train")
        df = dataset.to_pandas()
        return df

# Chargement automatique
if st.session_state.df is None:
    st.session_state.df = charger_donnees()

df = st.session_state.df

# Sidebar
with st.sidebar:
    st.title("🛡️ FraudDetect Pro")
    st.markdown("---")
    
    st.subheader("📊 Dataset")
    st.success(f"✅ {len(df):,} transactions")
    st.info(f"📋 {len(df.columns)} colonnes")
    
    fraud_count = df['Class'].sum()
    fraud_pct = (fraud_count / len(df)) * 100
    st.metric("🚨 Fraudes", f"{fraud_count:,}", delta=f"{fraud_pct:.3f}%")
    
    st.markdown("---")
    
    st.subheader("🧠 Modèles DL")
    st.markdown("""
    - MLP
    - CNN
    - LSTM
    - GRU
    - Autoencodeur
    - CNN-LSTM
    - CNN-GRU
    - CNN-LSTM-GRU
    """)
    
    st.markdown("---")
    st.caption("© 2026 FraudDetect Pro")

# Page d'accueil
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🛡️ FraudDetect Pro</h1>
    <p style="color: #e0e0e0; font-size: 1.2rem; margin-top: 0.5rem;">
        Plateforme avancée de détection de fraudes bancaires par Deep Learning
    </p>
</div>
""", unsafe_allow_html=True)

# Présentation du dataset
st.subheader("📊 Présentation du dataset")

col1, col2 = st.columns([1.5, 1])

with col1:
    fraud_count = df['Class'].sum()
    legit_count = len(df) - fraud_count
    
    st.markdown(f"""
    <div class="info-box">
        <strong>💳 Credit Card Fraud Detection</strong><br><br>
        📦 Total transactions : <strong>{len(df):,}</strong><br>
        📊 Nombre de colonnes : <strong>{len(df.columns)}</strong><br>
        🚨 Transactions frauduleuses : <strong>{fraud_count:,}</strong> ({fraud_count/len(df)*100:.3f}%)<br>
        ✅ Transactions normales : <strong>{legit_count:,}</strong> ({legit_count/len(df)*100:.3f}%)<br>
        💰 Montant moyen (normal) : <strong>{df[df['Class']==0]['Amount'].mean():.2f} €</strong><br>
        💰 Montant moyen (fraude) : <strong>{df[df['Class']==1]['Amount'].mean():.2f} €</strong>
    </div>
    """, unsafe_allow_html=True)

with col2:
    fig = go.Figure(data=[go.Pie(
        labels=['Normales', 'Fraudes'],
        values=[legit_count, fraud_count],
        hole=0.4,
        marker_colors=['#00cc96', '#ef553b']
    )])
    fig.update_layout(height=350, title="Répartition des classes")
    st.plotly_chart(fig, use_container_width=True)

# Avertissement TensorFlow
st.warning("""
⚠️ **Note technique** : Les modèles deep learning (TensorFlow) ne sont pas disponibles 
sur cette version cloud car TensorFlow n'est pas encore compatible avec Python 3.14. 
Pour entraîner les modèles, utilisez un environnement local avec Python 3.11.
""")

# Aperçu des données
with st.expander("🔍 Aperçu des données"):
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("📊 Statistiques descriptives")
    st.dataframe(df.describe(), use_container_width=True)

# Visualisations rapides
st.subheader("📈 Visualisations exploratoires")

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        df.sample(min(50000, len(df))), 
        x='Amount', 
        color='Class',
        nbins=50,
        title="Distribution des montants",
        color_discrete_map={0: '#00cc96', 1: '#ef553b'}
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    v_cols = [f'V{i}' for i in range(1, 11)]
    fraud_means = df[df['Class'] == 1][v_cols].mean()
    normal_means = df[df['Class'] == 0][v_cols].mean()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=v_cols, y=normal_means, name='Normales', marker_color='#00cc96'))
    fig.add_trace(go.Bar(x=v_cols, y=fraud_means, name='Fraudes', marker_color='#ef553b'))
    fig.update_layout(title="Moyennes des composantes ACP (V1-V10)", barmode='group')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("⚠️ Les colonnes V1 à V28 sont des composantes principales anonymisées (ACP)")
