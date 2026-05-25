# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from datasets import load_dataset

# Configuration de la page (DOIT être la première commande Streamlit)
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
    .dataset-card {
        background-color: #f8f9fa;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    .model-badge {
        background-color: #667eea;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        display: inline-block;
        font-size: 0.8rem;
        margin: 0.25rem;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        transition: 0.3s;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialisation du session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'models_status' not in st.session_state:
    st.session_state.models_status = {
        'MLP': 'pending',
        'CNN': 'pending',
        'LSTM': 'pending',
        'GRU': 'pending',
        'Autoencodeur': 'pending',
        'CNN-LSTM': 'pending',
        'CNN-GRU': 'pending',
        'CNN-LSTM-GRU': 'pending'
    }

# Fonction de chargement des données (avec cache)
@st.cache_data
def charger_donnees():
    with st.spinner("📡 Téléchargement du dataset depuis Hugging Face..."):
        dataset = load_dataset("Wateb26/creditcard", split="train")
        df = dataset.to_pandas()
        return df

# CHARGEMENT AUTOMATIQUE DU DATASET
if st.session_state.df is None:
    st.session_state.df = charger_donnees()

df = st.session_state.df

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=80)
    st.title("🛡️ FraudDetect Pro")
    st.markdown("---")
    
    # Informations dataset
    st.subheader("📊 Dataset")
    st.success(f"✅ **{len(df):,}** transactions")
    st.info(f"📋 **{len(df.columns)}** colonnes")
    
    fraud_count = df['Class'].sum()
    fraud_pct = (fraud_count / len(df)) * 100
    st.metric("🚨 Fraudes", f"{fraud_count:,}", delta=f"{fraud_pct:.3f}%")
    
    st.markdown("---")
    
    # Modèles disponibles
    st.subheader("🧠 Modèles DL")
    st.markdown("""
    <div>
        <span class="model-badge">MLP</span>
        <span class="model-badge">CNN</span>
        <span class="model-badge">LSTM</span>
        <span class="model-badge">GRU</span>
        <span class="model-badge">AE</span>
    </div>
    <div style="margin-top: 0.5rem;">
        <span class="model-badge">CNN-LSTM</span>
        <span class="model-badge">CNN-GRU</span>
        <span class="model-badge">CNN-LSTM-GRU</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Progression
    trained = sum(1 for status in st.session_state.models_status.values() if status == 'trained')
    st.progress(trained / 8, text=f"{trained}/8 modèles entraînés")
    
    st.markdown("---")
    st.caption("© 2026 FraudDetect Pro | v1.0")

# PAGE D'ACCUEIL
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🛡️ FraudDetect Pro</h1>
    <p style="color: #e0e0e0; font-size: 1.2rem; margin-top: 0.5rem;">
        Plateforme avancée de détection de fraudes bancaires par Deep Learning
    </p>
</div>
""", unsafe_allow_html=True)

# SECTION PRÉSENTATION DU DATASET
st.subheader("📊 Présentation du dataset")

col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("""
    <div class="dataset-card">
        <h3 style="margin-top: 0;">💳 Credit Card Fraud Detection</h3>
        <p>Dataset de transactions bancaires européennes réalisées en septembre 2013.</p>
        <table style="width: 100%; font-size: 0.9rem;">
            <tr><td><strong>📦 Total transactions</strong></td><td style="text-align: right;"><strong>{:,}</strong></td></tr>
            <tr><td><strong>📊 Nombre de colonnes</strong></td><td style="text-align: right;"><strong>{}</strong></td></tr>
            <tr><td><strong>🚨 Transactions frauduleuses</strong></td><td style="text-align: right;"><strong>{:,}</strong> ({:.3f}%)</td></tr>
            <tr><td><strong>✅ Transactions normales</strong></td><td style="text-align: right;"><strong>{:,}</strong> ({:.3f}%)</td></tr>
            <tr><td><strong>💰 Montant moyen (normal)</strong></td><td style="text-align: right;"><strong>{:.2f} €</strong></td></tr>
            <tr><td><strong>💰 Montant moyen (fraude)</strong></td><td style="text-align: right;"><strong>{:.2f} €</strong></td></tr>
        </table>
    </div>
    """.format(
        len(df),
        len(df.columns),
        df['Class'].sum(),
        (df['Class'].sum() / len(df)) * 100,
        len(df) - df['Class'].sum(),
        ((len(df) - df['Class'].sum()) / len(df)) * 100,
        df[df['Class'] == 0]['Amount'].mean(),
        df[df['Class'] == 1]['Amount'].mean()
    ), unsafe_allow_html=True)

with col2:
    # Petit graphique camembert pour la répartition
    legit_count = len(df) - df['Class'].sum()
    fraud_count = df['Class'].sum()
    
    fig = go.Figure(data=[go.Pie(
        labels=['Transactions normales', 'Transactions frauduleuses'],
        values=[legit_count, fraud_count],
        hole=0.5,
        marker_colors=['#00cc96', '#ef553b'],
        textinfo='label+percent',
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Répartition des classes",
        height=300,
        margin=dict(t=40, b=0, l=0, r=0),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Description des colonnes
with st.expander("📋 Description des colonnes"):
    st.markdown("""
    | Colonne | Description |
    |---------|-------------|
    | **Time** | Secondes écoulées depuis la première transaction |
    | **V1 à V28** | 28 composantes principales issues d'une ACP (anonymisation) |
    | **Amount** | Montant de la transaction (non transformé) |
    | **Class** | Variable cible : 0 = normale, 1 = fraude |
    """)
    
    st.warning("⚠️ **Note importante** : Les colonnes V1 à V28 sont des composantes principales anonymisées. Elles n'ont pas de signification métier directe.")

# Quick stats
st.markdown("---")
st.subheader("📈 Statistiques rapides")

cola, colb, colc, cold = st.columns(4)

with cola:
    st.metric("Montant max", f"{df['Amount'].max():.2f} €")
with colb:
    st.metric("Montant min", f"{df['Amount'].min():.2f} €")
with colc:
    st.metric("Montant moyen", f"{df['Amount'].mean():.2f} €")
with cold:
    st.metric("Écart-type", f"{df['Amount'].std():.2f} €")

# Aperçu des données
with st.expander("🔍 Aperçu des premières lignes"):
    st.dataframe(df.head(10), use_container_width=True)

# Section modèles
st.markdown("---")
st.subheader("🧠 Modèles deep learning disponibles")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-box">
        <strong>📌 Modèles simples</strong><br>
        • MLP (Multilayer Perceptron)<br>
        • CNN (Convolutional Neural Network)<br>
        • LSTM (Long Short Term Memory)<br>
        • GRU (Gated Recurrent Unit)<br>
        • Autoencodeur
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-box">
        <strong>🔗 Modèles hybrides</strong><br>
        • CNN-LSTM<br>
        • CNN-GRU<br>
        • CNN-LSTM-GRU
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-box">
        <strong>🎯 Prochaines étapes</strong><br>
        1️⃣ Aller dans <strong>Entraînement</strong><br>
        2️⃣ Sélectionner un modèle<br>
        3️⃣ Configurer et lancer<br>
        4️⃣ Comparer les résultats
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("⚠️ Les colonnes V1 à V28 sont des composantes principales anonymisées (ACP) - aucune signification métier directe")









 #CODE PYTHON – CNN FRAUDE BANCAIRE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv(r"C:\Users\GENIUS ELECTRONICS\Desktop\dataset1\base\creditcard.csv")
# Mettre le fichier dans le même dossier que le script
print(df.head())
print(df["Class"].value_counts())


X = df.drop("Class", axis=1)
y = df["Class"]

# Normalisation des variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape pour CNN [échantillons, pas de temps, features]
X_scaled = np.expand_dims(X_scaled, axis=2)

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = Sequential([
    Conv1D(32, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(64, kernel_size=2, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=2048,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Métriques principales
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
specificity = tn / (tn + fp)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print("\n===  Évaluation du modèle CNN ===")
print(f"Accuracy       : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"Specificité    : {specificity:.4f}")
print(f"F1 Score       : {f1:.4f}")
print(f"MCC            : {mcc:.4f}")
print(f"ROC AUC        : {roc_auc:.4f}")

print("\nMatrice de Confusion :\n", cm)
print("\nRapport de Classification :\n", classification_report(y_test, y_pred))

# ------------------------------------------------------------
# 8. Matrice de confusion (affichage)
# ------------------------------------------------------------
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
plt.title("Confusion matrix – CNN model")
plt.ylabel("Real")
plt.xlabel("Prediction")
plt.show()


# Courbe ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], 'k--')
plt.title("ROC curve – CNN model")
plt.xlabel("False positive rate (FPR)")
plt.ylabel("True positive rate (TPR)")
plt.legend()
plt.show()

# Courbe de perte
plt.figure(figsize=(6,5))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title("Loss curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Courbe accuracy
plt.figure(figsize=(6,5))
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title("Accuracy curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Graphique séparation normal / fraude
plt.figure(figsize=(7,5))
plt.scatter(range(len(y_test)), y_test, label="Réel", alpha=0.5)
plt.scatter(range(len(y_pred)), y_pred, label="Prédit", alpha=0.5)
plt.title("Comparaison entre classes réelles et prédites (CNN)")
plt.xlabel("Échantillons")
plt.ylabel("Classe (0 = normal, 1 = fraude)")
plt.legend()
plt.show()

# 11. Séparation Normal vs Fraude
# ------------------------------------------------------------
plt.figure(figsize=(7,5))
plt.hist(y_pred_proba[y_test==0], bins=50, alpha=0.6, label="Normal")
plt.hist(y_pred_proba[y_test==1], bins=50, alpha=0.6, label="Fraude")
plt.axvline(0.5, color="red", linestyle="--", label="Seuil = 0.5")
plt.title("Distribution des probabilités prédites (CNN)")
plt.xlabel("Probabilité prédite de fraude")
plt.ylabel("Fréquence")
plt.legend()
plt.show()


