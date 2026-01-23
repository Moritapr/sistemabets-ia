import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor

# --- ESTILO VISUAL ---
st.set_page_config(page_title="SISTEMABETS IA: CHAMPIONS", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stMetricValue"] { color: #00ffcc; font-weight: bold; }
    .stMetric { background-color: #1a1c23; padding: 15px; border-radius: 15px; border: 1px solid #333; }
    h1, h2, h3 { color: #00ffcc; }
    </style>
    """, unsafe_allow_html=True)

# --- TU API KEY CONFIGURADA ---
API_KEY = "1e90385f6e65c6f70e7" 

@st.cache_data(ttl=3600)
def cargar_datos_champions():
    # League ID 2 = Champions League | Season 2025 (Edición actual)
    url = "https://v3.football.api-sports.io/standings?league=2&season=2025"
    headers = {
        'x-rapidapi-host': "v3.football.api-sports.io",
        'x-rapidapi-key': API_KEY
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        res_json = response.json()
        
        # Acceso a la estructura de datos de la League Phase
        tabla_raw = res_json['response'][0]['league']['standings'][0]
        
        datos = []
        for item in tabla_raw:
            datos.append({
                'Nombre': item['team']['name'],
                'Puntos': item['points'],
                'GF': item['all']['goals']['for'],
                'GC': item['all']['goals']['against'],
                'DIF': item['goalsDiff'],
                'Forma': item['form'] if item['form'] else "???"
            })
        return pd.DataFrame(datos)
    except Exception as e:
        return f"Error en conexión: {str(e)}"

# --- INTERFAZ ---
st.title("🤖 SISTEMABETS IA: MODO CHAMPIONS 25/26")
st.write(f"Conexión Directa mediante API - {pd.to_datetime('today').strftime('%d/%m/%Y')}")

with st.spinner('Sincronizando con los servidores de UEFA...'):
    df = cargar_datos_champions()

if isinstance(df, str):
    st.error(f"⚠️ Fallo al obtener datos: {df}")
    st.info("Verifica que tu API Key esté activa en el panel de API-Football.")
else:
    st.success(f"✅ CONECTADO: {len(df)} equipos listos para análisis.")
    
    # Selectores
    c1, c2 = st.columns(2)
    local = c1.selectbox("Local:", sorted(df['Nombre'].unique()), index=0)
    visita = c2.selectbox("Visita:", sorted(df['Nombre'].unique()), index=1)

    # --- MOTOR DE INTELIGENCIA ARTIFICIAL ---
    eq_l = df[df['Nombre'] == local].iloc[0]
    eq_v = df[df['Nombre'] == visita].iloc[0]

    # Modelo Random Forest (Proyección de potencial)
    X_train = df[['GF', 'DIF', 'Puntos']].values
    y_train = np.arange(len(df), 0, -1) # Ranking de poder
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    pred_l = model.predict([[eq_l['GF'], eq_l['DIF'], eq_l['Puntos']]])[0]
    pred_v = model.predict([[eq_v['GF'], eq_v['DIF'], eq_v['Puntos']]])[0]

    # Cálculo de probabilidad softmax
    prob_l = (pred_l / (pred_l + pred_v)) * 100

    # --- RESULTADOS ---
    st.divider()
    r1, r2, r3 = st.columns(3)
    r1.metric(f"Prob. {local}", f"{round(prob_l, 1)}%", f"Forma: {eq_l['Forma']}")
    r2.metric(f"Prob. {visita}", f"{round(100-prob_l, 1)}%", f"Forma: {eq_v['Forma']}")
    r3.metric("Cuota Justa", f"{round(100/prob_l, 2)}")

    st.subheader("📝 Análisis de Datos Reales")
    st.write(f"El sistema analiza que {local} tiene un diferencial de {eq_l['DIF']} goles en la fase de liga, comparado con el diferencial de {eq_v['DIF']} de {visita}.")
