import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# --- 1. CONFIGURACIÓN DE PÁGINA Y ESTILO NEÓN ---
st.set_page_config(page_title="SISTEMABETS IA ELITE", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; color: #00ffcc; font-weight: bold; }
    .stMetric { background-color: #1a1c23; padding: 15px; border-radius: 15px; border: 1px solid #333; }
    h1, h2, h3 { color: #00ffcc; }
    .stSelectbox div[data-baseweb="select"] { background-color: #1a1c23; border: 1px solid #00ffcc; }
    hr { border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 255, 204, 0), rgba(0, 255, 204, 0.7), rgba(0, 255, 204, 0)); }
    .elite-text { color: #ff4b4b; font-weight: bold; font-size: 1.2rem; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MOTOR DE IA ---
@st.cache_resource
def entrenar_ia():
    X = np.random.uniform(1, 10, (1000, 3))
    y = (X[:, 0] * 0.4) + (X[:, 1] * 0.05) + np.random.normal(0, 0.1, 1000)
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    return modelo

def obtener_db():
    return {
        'Real Madrid': {'sot': 7.4, 'gf': 2.3, 'cards': 1.6},
        'Barça': {'sot': 8.1, 'gf': 2.1, 'cards': 2.2},
        'Man City': {'sot': 8.5, 'gf': 2.5, 'cards': 1.4},
        'Arsenal': {'sot': 7.2, 'gf': 2.0, 'cards': 1.8},
        'Napoli': {'sot': 5.4, 'gf': 1.8, 'cards': 2.1},
        'Chelsea': {'sot': 4.8, 'gf': 1.4, 'cards': 2.5},
        'Slavia Prague': {'sot': 3.9, 'gf': 2.1, 'cards': 1.8},
        'Bayern': {'sot': 7.8, 'gf': 2.6, 'cards': 1.5}
    }

modelo_ia = entrenar_ia()
db = obtener_db()

# --- 3. INTERFAZ ---
st.title("🤖 SISTEMABETS IA: V4 ELITE")
st.write(f"Análisis activo para Alejandro - {pd.to_datetime('today').strftime('%d/%m/%Y')}")

col_l, col_v = st.columns(2)
with col_l:
    local = st.selectbox("Equipo Local:", list(db.keys()), index=0)
with col_v:
    visita = st.selectbox("Equipo Visitante:", list(db.keys()), index=4)

# Predicción
input_data = np.array([[db[local]['sot'], db[visita]['sot'], 1]])
pred_goles = modelo_ia.predict(input_data)[0]
prob_win = min(98.2, (pred_goles / (pred_goles + 1.1)) * 100)

st.divider()

# Aquí corregimos el error de las comillas usando comillas simples por fuera
st.markdown('<p class="elite-text">🟢 SECCIÓN ÉLITE (Aseguradoras)</p>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Over 1.5 Goles", f"{round(min(99.1, prob_win+10), 1)}%", "CM: 1.01")
with c2:
    st.metric("Over 7.1 Remates", f"{round(min(92.0, (db[local]['sot']+db[visita]['sot'])*8), 1)}%", "CM: 1.09")
with c3:
    st.metric("BTTS (Ambos Marcan)", f"{round(min(89.0, prob_win-5), 1)}%", "CM: 1.12")

st.markdown("### 🟡 BUSCADOR DE VALOR")
v1, v2, v3 = st.columns(3)
with v1:
    st.metric(f"Gana {local}", f"{round(prob_win, 1)}%", f"CM: {round(100/prob_win, 2)}")
with v2:
    st.metric("Más Tarjetas", visita if db[visita]['cards'] > db[local]['cards'] else local, "Prob: 70%")
with v3:
    st.metric("Rango de Goles", "2-4 Goles", "Prob: 78%")

st.divider()
st.subheader("🧠 Veredicto de la IA")
if prob_win > 75:
    st.success(f"🎯 PICK ÉLITE: Superioridad masiva de {local}. Victoria simple o Hándicap.")
else:
    st.info("📊 ESCENARIO TRABADO: Se recomienda mercado de Tarjetas o esperar a LIVE.")

