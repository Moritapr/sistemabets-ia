import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor

# --- CONFIGURACIÓN DE LA APP ---
st.set_page_config(page_title="SISTEMABETS IA: CHAMPIONS ELITE", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stMetricValue"] { color: #00ffcc; font-weight: bold; }
    .stMetric { background-color: #1a1c23; padding: 15px; border-radius: 15px; border: 1px solid #333; }
    h1, h2, h3 { color: #00ffcc; }
    </style>
    """, unsafe_allow_html=True)

# --- TU API KEY VALIDADA ---
API_KEY = "1e90385f6e65c6f70e71c8714e76d7d5" 

@st.cache_data(ttl=3600)
def fetch_champions_live():
    # League ID 2 (Champions) | Season 2025
    url = "https://v3.football.api-sports.io/standings?league=2&season=2025"
    headers = {
        'x-rapidapi-host': "v3.football.api-sports.io",
        'x-rapidapi-key': API_KEY
    }
    try:
        response = requests.get(url, headers=headers, timeout=20)
        res_json = response.json()
        
        # Validación de respuesta
        if not res_json.get('response'):
            return f"Error de API: {res_json.get('errors', 'Respuesta vacía')}"
            
        # Extraer la tabla de la fase de liga
        standings = res_json['response'][0]['league']['standings'][0]
        
        datos = []
        for team in standings:
            datos.append({
                'Nombre': team['team']['name'],
                'Puntos': team['points'],
                'GF': team['all']['goals']['for'],
                'GC': team['all']['goals']['against'],
                'DIF': team['goalsDiff'],
                'Forma': team['form'] if team['form'] else "N/A"
            })
        return pd.DataFrame(datos)
    except Exception as e:
        return f"Fallo de conexión: {str(e)}"

# --- INTERFAZ ---
st.title("🤖 SISTEMABETS IA: CHAMPIONS 25/26")
st.write(f"Bregando con datos reales para Alejandro - {pd.to_datetime('today').strftime('%d/%m/%Y')}")

with st.spinner('Sincronizando con los servidores oficiales de la UEFA...'):
    df_champions = fetch_champions_live()

if isinstance(df_champions, str):
    st.error(df_champions)
    st.info("Revisa tu panel de API-Sports para asegurar que el plan 'Free' está activo.")
else:
    st.success(f"✅ DATA FLOW ACTIVO: {len(df_champions)} equipos sincronizados.")
    
    col_l, col_v = st.columns(2)
    local = col_l.selectbox("Selecciona Local:", sorted(df_champions['Nombre'].unique()), index=0)
    visita = col_v.selectbox("Selecciona Visita:", sorted(df_champions['Nombre'].unique()), index=1)

    # --- MOTOR DE IA (Random Forest) ---
    stats_l = df_champions[df_champions['Nombre'] == local].iloc[0]
    stats_v = df_champions[df_champions['Nombre'] == visita].iloc[0]

    # Preparamos el modelo con los datos succionados
    X_train = df_champions[['GF', 'DIF', 'Puntos']].values
    y_train = np.arange(len(df_champions), 0, -1) # Ranking de poder
    
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

    # Proyección de resultados
    pred_l = model.predict([[stats_l['GF'], stats_l['DIF'], stats_l['Puntos']]])[0]
    pred_v = model.predict([[stats_v['GF'], stats_v['DIF'], stats_v['Puntos']]])[0]

    prob_l = (pred_l / (pred_l + pred_v)) * 100

    # --- DASHBOARD FINAL ---
    st.divider()
    r1, r2, r3 = st.columns(3)
    r1.metric(f"Victoria {local}", f"{round(prob_l, 1)}%", f"Forma: {stats_l['Forma']}")
    r2.metric(f"Victoria {visita}", f"{round(100-prob_l, 1)}%", f"Forma: {stats_v['Forma']}")
    r3.metric("Cuota Justa", f"{round(100/prob_l, 2)}")

    st.subheader("📝 Análisis Táctico IA")
    st.write(f"El modelo asigna a **{local}** una probabilidad del **{round(prob_l, 1)}%** basado en su diferencial de **{stats_l['DIF']} goles**.")
