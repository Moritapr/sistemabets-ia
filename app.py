import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURACIÓN DE ALTA PRECISIÓN ---
st.set_page_config(page_title="SISTEMABETS IA: PRO ANALYST", layout="wide")

FUENTES = {
    "Champions League": "https://native-stats.org/competition/CL/",
    "Premier League": "https://native-stats.org/competition/PL",
    "La Liga": "https://native-stats.org/competition/PD",
    "Bundesliga": "https://native-stats.org/competition/BL1",
    "Serie A": "https://native-stats.org/competition/SA",
    "Ligue 1": "https://native-stats.org/competition/FL1",
    "Liga Portugal": "https://native-stats.org/competition/PPL",
    "Betting Trends": "https://native-stats.org/betting",
    "Real Madrid Profile": "https://native-stats.org/team/81",
    "Home": "https://native-stats.org/"
}

@st.cache_data(ttl=600)
def super_scrapper_v3(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        res = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(res.content, 'html.parser')
        tablas = soup.find_all('table')
        if not tablas: return None
        
        df = pd.read_html(str(max(tablas, key=lambda t: len(t.find_all('tr')))))[0]
        
        # Mapeo posicional inteligente
        df = df.rename(columns={df.columns[0]: 'Pos', df.columns[1]: 'Equipo', df.columns[-1]: 'Pts'})
        df['Equipo'] = df['Equipo'].astype(str).str.replace(r'^\d+\s+', '', regex=True).str.strip()
        
        # Extracción de métricas de mercado (Goles)
        for col in df.columns:
            if ":" in str(df[col].iloc[0]):
                gs = df[col].astype(str).str.split(':', expand=True)
                df['GF'] = pd.to_numeric(gs[0], errors='coerce').fillna(0).astype(int)
                df['GC'] = pd.to_numeric(gs[1], errors='coerce').fillna(0).astype(int)
                break
        
        # Cálculo de Forma (Puntos por partido aproximados)
        df['PJ'] = pd.to_numeric(df.iloc[:, 2], errors='coerce').fillna(1)
        df['Rating_Ataque'] = df['GF'] / df['PJ']
        df['Rating_Defensa'] = df['GC'] / df['PJ']
        
        return df
    except: return None

# --- INTERFAZ DE USUARIO ---
st.title("🏆 SISTEMABETS IA: ESTRATEGIA DE MERCADOS")
sel = st.sidebar.selectbox("Selecciona Competición:", list(FUENTES.keys()))
data = super_scrapper_v3(FUENTES[sel])

if data is not None:
    st.success(f"Sincronizado con data real 25/26")
    
    col1, col2 = st.columns(2)
    l_team = col1.selectbox("Local:", data['Equipo'].unique(), index=2)
    v_team = col2.selectbox("Visitante:", data['Equipo'].unique(), index=0)

    # --- MOTOR DE INFERENCIA ---
    l_stats = data[data['Equipo'] == l_team].iloc[0]
    v_stats = data[data['Equipo'] == v_team].iloc[0]

    # Probabilidades de victoria (Win/Draw/Loss)
    win_prob = (l_stats['Pts'] / (l_stats['Pts'] + v_stats['Pts'] + 0.1)) * 100
    
    # Análisis de Goles (Over/Under 2.5)
    expectativa_goles = (l_stats['Rating_Ataque'] + v_stats['Rating_Ataque']) / 2
    expectativa_recibo = (l_stats['Rating_Defensa'] + v_stats['Rating_Defensa']) / 2
    total_esperado = expectativa_goles + expectativa_recibo

    # --- DASHBOARD DE APUESTAS ---
    st.divider()
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.subheader("🎯 Ganador")
        st.metric(l_team, f"{round(win_prob, 1)}%")
        st.metric(v_team, f"{round(100-win_prob, 1)}%")

    with c2:
        st.subheader("⚽ Goles (O/U)")
        if total_esperado > 2.5:
            st.warning(f"PICK: Over 2.5 ({round(total_esperado, 2)})")
        else:
            st.info(f"PICK: Under 2.5 ({round(total_esperado, 2)})")

    with c3:
        st.subheader("🔥 Ambos Marcan")
        btts = "SÍ" if l_stats['Rating_Ataque'] > 1.2 and v_stats['Rating_Ataque'] > 1.2 else "NO"
        st.subheader(f"VERDICTO: {btts}")

    st.write("---")
    st.caption(f"Análisis basado en el rendimiento actual: {l_team} marca {round(l_stats['Rating_Ataque'],2)} goles/partido vs {v_team}.")

else:
    st.error("Error al conectar. Verifica los links o el tráfico en Native-Stats.")
