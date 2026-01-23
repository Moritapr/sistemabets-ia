import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import requests

# --- CONFIGURACIÓN Y ESTILO ---
st.set_page_config(page_title="SISTEMABETS IA: LIVE SCRAPER", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stMetricValue"] { color: #00ffcc; font-weight: bold; }
    .stMetric { background-color: #1a1c23; padding: 15px; border-radius: 15px; border: 1px solid #333; }
    </style>
    """, unsafe_allow_html=True)

# --- SCRAPER REAL DE FBREF ---
@st.cache_data(ttl=3600) # Guarda los datos por 1 hora para no saturar
def extraer_datos_vivos():
    try:
        # URL de estadísticas de disparos de la Champions 25/26
        url = "https://fbref.com/en/comps/8/shooting/Champions-League-Stats"
        html = requests.get(url).text
        # Leemos las tablas y buscamos la de 'stats_shooting_combined'
        tablas = pd.read_html(html)
        df = tablas[0]
        
        # Limpieza de columnas (FBref usa multinivel)
        df.columns = [' '.join(col).strip() for col in df.columns.values]
        
        # Nos quedamos con: Equipo, Disparos a puerta (SoT) y SoT por 90
        # Los nombres exactos de columnas en FBref suelen ser 'Unnamed: 0_level_0 Squad' y 'Standard SoT/90'
        df_clean = df[['Unnamed: 0_level_0 Squad', 'Standard SoT/90']].copy()
        df_clean.columns = ['Squad', 'SoT90']
        
        # Convertimos a diccionario para la IA
        return df_clean.set_index('Squad')['SoT90'].to_dict()
    except Exception as e:
        st.error(f"Error al conectar con FBref: {e}")
        return {"Error": 0}

# --- PROCESAMIENTO IA ---
data_viva = extraer_datos_vivos()

if "Error" not in data_viva:
    st.title("🤖 IA CONNECTED: FBREF LIVE DATA")
    st.success(f"Se han extraído {len(data_viva)} equipos reales de la Champions 25/26.")
    
    col1, col2 = st.columns(2)
    with col1:
        local = st.selectbox("Selecciona Local:", sorted(data_viva.keys()))
    with col2:
        visita = st.selectbox("Selecciona Visita:", sorted(data_viva.keys()))

    # --- MODELO RANDOM FOREST ---
    # La IA ahora usa el SoT90 real que acaba de leer de la web
    sot_l = float(data_viva[local])
    sot_v = float(data_viva[visita])
    
    # Simulación de entrenamiento flash
    X = np.array([[5, 2], [8, 1], [3, 4], [7, 2]]) # Datos base de entrenamiento
    y = [1.5, 2.8, 0.8, 2.1]
    model = RandomForestRegressor(n_estimators=100).fit(X, y)
    
    pred = model.predict([[sot_l, sot_v]])[0]
    prob = min(99.0, (pred / (pred + 1.2)) * 100)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Prob. {local}", f"{round(prob, 1)}%")
    c2.metric("SoT/90 Real (Web)", sot_l)
    c3.metric("Cuota Sugerida", round(100/prob, 2))
else:
    st.warning("Usando modo offline. Revisa tu conexión o el requirements.txt")

