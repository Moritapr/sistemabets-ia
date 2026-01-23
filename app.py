import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# --- CONFIGURACIÓN DEL SISTEMA ---
st.set_page_config(page_title="SISTEMABETS IA: CORE CENTRAL", layout="wide")

# Mapeo de tus links
FUENTES = {
    "Champions League": "https://native-stats.org/competition/CL/standings",
    "Premier League": "https://native-stats.org/competition/PL/standings",
    "La Liga (España)": "https://native-stats.org/competition/PD/standings",
    "Bundesliga": "https://native-stats.org/competition/BL1/standings",
    "Serie A": "https://native-stats.org/competition/SA/standings",
    "Ligue 1": "https://native-stats.org/competition/FL1/standings",
    "Liga Portugal": "https://native-stats.org/competition/PPL/standings",
    "Betting Trends": "https://native-stats.org/betting"
}

@st.cache_data(ttl=1800)
def scraper_inteligente(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')
        tablas = soup.find_all('table')
        
        # Leemos la tabla principal de la liga
        df = pd.read_html(str(tablas[0]))[0]
        
        # Estandarizamos columnas (Pos, Equipo, PJ, Pts, Dif, Goles)
        df.columns = ['Pos', 'Equipo', 'PJ', 'Pts', 'Dif', 'Goles', 'Forma']
        
        # Procesamos la columna Goles (20:5) para IA
        df[['GF', 'GC']] = df['Goles'].str.split(':', expand=True).astype(int)
        
        # Limpieza de nombres de equipos (quitar números de posición)
        df['Equipo'] = df['Equipo'].str.replace(r'\d+', '', regex=True).str.strip()
        
        return df
    except Exception as e:
        return f"Error en fuente: {str(e)}"

# --- INTERFAZ ---
st.sidebar.title("🔍 FUENTES DE DATOS")
seleccion = st.sidebar.selectbox("Selecciona Link para Scrapping:", list(FUENTES.keys()))

st.title(f"🤖 SISTEMABETS IA: MODO {seleccion.upper()}")

df = scraper_inteligente(FUENTES[seleccion])

if isinstance(df, str):
    st.error(df)
else:
    st.success(f"✅ IA conectada exitosamente a: {FUENTES[seleccion]}")
    
    # Análisis comparativo
    c1, c2 = st.columns(2)
    local = c1.selectbox("Local:", df['Equipo'].unique(), index=0)
    visita = c2.selectbox("Visita:", df['Equipo'].unique(), index=1)

    # --- MOTOR DE PREDICCIÓN (IA ACCEDE Y BUSCA) ---
    stats_l = df[df['Equipo'] == local].iloc[0]
    stats_v = df[df['Equipo'] == visita].iloc[0]

    # La IA usa GF, GC, Puntos y Dif para crear el pronóstico
    X = df[['GF', 'GC', 'Pts', 'Dif']].values
    y = np.arange(len(df), 0, -1) # Ranking de poder inverso
    
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    
    # Predicción de cuotas y probabilidades
    p_l = model.predict([[stats_l['GF'], stats_l['GC'], stats_l['Pts'], stats_l['Dif']]])[0]
    p_v = model.predict([[stats_v['GF'], stats_v['GC'], stats_v['Pts'], stats_v['Dif']]])[0]
    
    prob_l = (p_l / (p_l + p_v)) * 100

    # --- DASHBOARD ---
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Victoria {local}", f"{round(prob_l, 1)}%", f"Goles: {stats_l['GF']}")
    m2.metric(f"Victoria {visita}", f"{round(100-prob_l, 1)}%", f"Goles: {stats_v['GF']}")
    m3.metric("Cuota Fair", f"{round(100/prob_l, 2)}")

    # Módulo de "Betting" integrado
    st.subheader("📊 Análisis de Probabilidades (Mercado Over/Under)")
    promedio_goles = (stats_l['GF'] + stats_v['GC']) / (stats_l['PJ'] + 0.1)
    
    if promedio_goles > 2.5:
        st.warning(f"⚠️ ALTA PROBABILIDAD DE OVER 2.5: El sistema detecta un flujo de {round(promedio_goles, 2)} goles.")
    else:
        st.info("💡 TENDENCIA UNDER: Defensas sólidas detectadas por la IA.")
