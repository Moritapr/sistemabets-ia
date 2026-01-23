import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import cloudscraper

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="SISTEMABETS IA: ELITE 25/26", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stMetricValue"] { color: #00ffcc; font-weight: bold; }
    .stMetric { background-color: #1a1c23; padding: 15px; border-radius: 15px; border: 1px solid #333; }
    h1 { color: #00ffcc; }
    </style>
    """, unsafe_allow_html=True)

# --- SCRAPER NIVEL INGENIERO (BYPASS 403) ---
@st.cache_data(ttl=3600)
def get_data_bypass():
    url = "https://fbref.com/en/comps/8/shooting/Champions-League-Stats"
    # Creamos un scraper que salta protecciones de bots
    scraper = cloudscraper.create_scraper()
    try:
        response = scraper.get(url, timeout=20)
        if response.status_code != 200:
            return f"Error {response.status_code}: FBref sigue bloqueando."
        
        # Procesamos la tabla
        tablas = pd.read_html(response.text)
        df = tablas[0]
        
        # Limpieza de columnas multinivel
        df.columns = [' '.join(col).strip() for col in df.columns.values]
        squad_col = [c for c in df.columns if 'Squad' in c][0]
        sot_col = [c for c in df.columns if 'SoT/90' in c][0]
        
        df_final = df[[squad_col, sot_col]].copy()
        df_final.columns = ['Equipo', 'SoT90']
        
        # Limpiar banderas y espacios
        df_final['Equipo'] = df_final['Equipo'].apply(lambda x: ' '.join(x.split()[1:]) if len(x.split()) > 1 else x)
        
        return df_final.dropna()
    except Exception as e:
        return f"Fallo crítico: {str(e)}"

# --- LÓGICA DE LA APP ---
st.title("🤖 SISTEMABETS IA: BYPASS ACTIVO")
st.write(f"Análisis Champions 25/26 - Alejandro")

with st.spinner('Evadiendo bloqueo de servidor...'):
    df_stats = get_data_bypass()

if isinstance(df_stats, str):
    st.error(df_stats)
    st.info("⚠️ El servidor de FBref detectó el tráfico. Dale a 'Reboot App' en el menú de Streamlit para cambiar la IP de salida.")
else:
    st.success(f"✅ ¡DENTRO! Analizando {len(df_stats)} equipos de Champions.")
    
    col1, col2 = st.columns(2)
    with col1:
        local = st.selectbox("Equipo Local:", sorted(df_stats['Equipo'].unique()), index=0)
    with col2:
        visita = st.selectbox("Equipo Visitante:", sorted(df_stats['Equipo'].unique()), index=1)

    # --- IA PROCESANDO ---
    sot_l = float(df_stats[df_stats['Equipo'] == local]['SoT90'].values[0])
    sot_v = float(df_stats[df_stats['Equipo'] == visita]['SoT90'].values[0])
    
    # Modelo Random Forest simplificado para predicción rápida
    X_train = np.array([[2,1], [8,1], [4,4], [6,2]])
    y_train = [1.0, 2.8, 1.4, 2.2]
    model = RandomForestRegressor(n_estimators=50).fit(X_train, y_train)
    
    pred = model.predict([[sot_l, sot_v]])[0]
    prob = min(98.9, (pred / (pred + 1.25)) * 100)

    # --- MÉTRICAS ---
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Prob. {local}", f"{round(prob, 1)}%", f"{sot_l} SOT/90")
    c2.metric(f"Prob. {visita}", f"{round(100-prob, 1)}%", f"{sot_v} SOT/90")
    c3.metric("Cuota Valor", f"{round(100/prob, 2)}")
