import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="SISTEMABETS IA: CHAMPIONS 25/26", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stMetricValue"] { color: #00ffcc; font-weight: bold; }
    .stMetric { background-color: #1a1c23; padding: 15px; border-radius: 15px; border: 1px solid #333; }
    h1, h2, h3 { color: #00ffcc; }
    </style>
    """, unsafe_allow_html=True)

# TU API KEY (Football-Data.org)
API_KEY = "b5da8589cdef4d418bbe2afcbccadf10" 

@st.cache_data(ttl=3600)
def extraer_champions_v25_26():
    # CL = Champions League | Season 2025 (inicia en 2025, termina en 2026)
    url = "https://api.football-data.org/v4/competitions/CL/standings?season=2025"
    headers = {'X-Auth-Token': API_KEY}
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        data = response.json()
        
        # Validación de acceso a la temporada actual
        if 'standings' not in data:
            return f"Error: {data.get('message', 'No se encontraron datos para 25/26')}"
            
        # Extraemos la tabla unificada (League Phase)
        standings = data['standings'][0]['table']
        
        datos = []
        for team in standings:
            datos.append({
                'Nombre': team['team']['name'],
                'Puntos': team['points'],
                'GF': team['goalsFor'],
                'GC': team['goalsAgainst'],
                'DIF': team['goalDifference'],
                'PJ': team['playedGames']
            })
        return pd.DataFrame(datos)
    except Exception as e:
        return f"Fallo en la conexión: {str(e)}"

# --- INTERFAZ ---
st.title("🤖 SISTEMABETS IA: CHAMPIONS 25/26")
st.write(f"Análisis en Tiempo Real - Temporada Actual - {pd.to_datetime('today').strftime('%d/%m/%Y')}")

with st.spinner('Sincronizando con la UEFA para la temporada 25/26...'):
    df = extraer_champions_v25_26()

if isinstance(df, str):
    st.error(df)
    st.info("Nota: Si la API dice que no tienes acceso a 2025, prueba cambiando la línea 24 a season=2024 para verificar conexión.")
else:
    st.success(f"✅ DATA 25/26 ONLINE: {len(df)} equipos cargados.")
    
    c1, c2 = st.columns(2)
    local = c1.selectbox("Equipo Local:", sorted(df['Nombre'].unique()))
    visita = c2.selectbox("Equipo Visitante:", sorted(df['Nombre'].unique()))

    # --- LÓGICA IA ---
    eq_l = df[df['Nombre'] == local].iloc[0]
    eq_v = df[df['Nombre'] == visita].iloc[0]

    # Modelo entrenado con la jerarquía de la tabla actual
    X = df[['GF', 'DIF', 'Puntos']].values
    y = np.arange(len(df), 0, -1) 
    
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    
    pred_l = model.predict([[eq_l['GF'], eq_l['DIF'], eq_l['Puntos']]])[0]
    pred_v = model.predict([[eq_v['GF'], eq_v['DIF'], eq_v['Puntos']]])[0]

    prob_l = (pred_l / (pred_l + pred_v)) * 100

    # --- RESULTADOS ---
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Prob. {local}", f"{round(prob_l, 1)}%")
    m2.metric(f"Prob. {visita}", f"{round(100-prob_l, 1)}%")
    m3.metric("Cuota Sugerida", f"{round(100/prob_l, 2)}")

    st.info(f"💡 En la 25/26, {local} ha marcado {eq_l['GF']} goles en {eq_l['PJ']} partidos.")
