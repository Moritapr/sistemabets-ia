import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import requests
import random

# --- CONFIGURACIÓN DE INTERFAZ ---
st.set_page_config(page_title="SISTEMABETS IA: REAL-TIME", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stMetricValue"] { color: #00ffcc; font-weight: bold; }
    .stMetric { background-color: #1a1c23; padding: 15px; border-radius: 15px; border: 1px solid #333; }
    h1, h3 { color: #00ffcc; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTOR DE SUCCIÓN DE DATOS (ANTI-BLOQUEO) ---
@st.cache_data(ttl=600) # Solo 10 minutos de cache para frescura total
def extraer_datos_vivos():
    # Usamos la Premier League como base de prueba, se puede cambiar a Champions
    url = "https://www.worldfootball.net/team_performance/eng-premier-league-2025-2026/nach-toren/"
    
    # Rotación de identidades para engañar al servidor
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ]
    
    headers = {'User-Agent': random.choice(user_agents)}
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return f"BLOQUEO: Servidor respondió con código {response.status_code}"
        
        # Extraemos tablas con motor bs4 para mayor compatibilidad
        tablas = pd.read_html(response.text, flavor='bs4')
        
        # Buscamos la tabla que contiene los equipos (usualmente la de performance)
        for df in tablas:
            if 'Team' in df.columns or 'team' in str(df.columns).lower():
                # Limpieza rápida: eliminamos filas vacías y columnas basura
                df = df.dropna(axis=1, how='all').dropna(axis=0, how='any')
                return df
                
        return "No se encontraron tablas de datos válidas."
    except Exception as e:
        return f"Fallo de conexión: {str(e)}"

# --- INTERFAZ PRINCIPAL ---
st.title("🤖 SISTEMABETS IA: FLUJO REAL UNLOCKED")
st.write(f"Bregando datos para Alejandro - {pd.to_datetime('today').strftime('%d/%m/%Y')}")

if st.button("🔄 Forzar Nueva Succión de Datos"):
    st.cache_data.clear()
    st.rerun()

with st.spinner('Penetrando en el servidor de estadísticas...'):
    df_vivos = extraer_datos_vivos()

if isinstance(df_vivos, str):
    st.error(f"❌ ERROR DE EXTRACCIÓN: {df_vivos}")
    st.info("Intenta darle al botón de arriba para cambiar el User-Agent.")
else:
    st.success(f"✅ CONECTADO: {len(df_vivos)} equipos succionados en tiempo real.")
    
    # Debug: Mostrar los datos que la IA está leyendo
    with st.expander("Ver Datos Crutos (Raw Data)"):
        st.dataframe(df_vivos.head(10))

    # Identificar columna de equipos y métricas
    col_equipo = [c for c in df_vivos.columns if 'team' in str(c).lower() or 'Team' in str(c)][0]
    # Usamos la columna de goles o puntos (usualmente numéricas)
    cols_num = df_vivos.select_dtypes(include=[np.number]).columns

    col_l, col_v = st.columns(2)
    local = col_l.selectbox("Local:", sorted(df_vivos[col_equipo].unique()))
    visita = col_v.selectbox("Visita:", sorted(df_vivos[col_equipo].unique()))

    # --- PROCESAMIENTO IA (Random Forest) ---
    stats_l = df_vivos[df_vivos[col_equipo] == local][cols_num].values[0]
    stats_v = df_vivos[df_vivos[col_equipo] == visita][cols_num].values[0]

    # La IA entrena con la distribución actual de la tabla
    X_train = df_vivos[cols_num].values
    y_train = np.arange(len(X_train)) # Ranking relativo
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predicción de potencia
    pred_l = model.predict([stats_l])[0]
    pred_v = model.predict([stats_v])[0]
    
    # Convertir a probabilidad (softmax simple)
    exp_l, exp_v = np.exp(-pred_l), np.exp(-pred_v) # Menor ranking = mejor
    prob_l = (exp_l / (exp_l + exp_v)) * 100

    # --- DASHBOARD DE RESULTADOS ---
    st.divider()
    res1, res2, res3 = st.columns(3)
    res1.metric(f"Victoria {local}", f"{round(prob_l, 1)}%")
    res2.metric(f"Victoria {visita}", f"{round(100-prob_l, 1)}%")
    res3.metric("Cuota Justa", f"{round(100/prob_l, 2)}")

    st.subheader("🧠 Veredicto de la IA")
    if prob_l > 65:
        st.success(f"🎯 PICK ELITE: Los datos reales muestran una superioridad táctica de {local}.")
    else:
        st.warning("📊 PARTIDO TRABADO: El algoritmo no detecta un ganador claro basado en el flujo actual.")
