import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re

st.set_page_config(page_title="SISTEMABETS IA: DATA REAL", layout="wide")

FUENTES = {
    "Champions League": "https://native-stats.org/competition/CL/",
    "Premier League": "https://native-stats.org/competition/PL",
    "La Liga": "https://native-stats.org/competition/PD",
    "Bundesliga": "https://native-stats.org/competition/BL1",
    "Serie A": "https://native-stats.org/competition/SA",
    "Ligue 1": "https://native-stats.org/competition/FL1",
    "Liga Portugal": "https://native-stats.org/competition/PPL",
    "Betting Trends": "https://native-stats.org/betting",
    "Real Madrid Profile": "https://native-stats.org/team/81"
}

def limpiar_numero(valor):
    """Limpia strings para que Python los entienda como números reales."""
    try:
        # Elimina cualquier cosa que no sea número o punto decimal
        limpio = re.sub(r'[^\d.]+', '', str(valor))
        return float(limpio) if limpio else 0.0
    except:
        return 0.0

@st.cache_data(ttl=600)
def scraper_precision(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        res = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(res.content, 'html.parser')
        tablas = soup.find_all('table')
        if not tablas: return None
        
        # Seleccionamos la tabla de liga (usualmente la que tiene más de 10 filas)
        tabla_principal = max(tablas, key=lambda t: len(t.find_all('tr')))
        df = pd.read_html(str(tabla_principal))[0]
        
        # Mapeo posicional inamovible
        df = df.rename(columns={df.columns[1]: 'Equipo', df.columns[-1]: 'Pts'})
        
        # Limpieza de nombres de equipos
        df['Equipo'] = df['Equipo'].astype(str).str.replace(r'^\d+\s+', '', regex=True).str.strip()
        
        # LIMPIEZA PROFUNDA DE PUNTOS
        df['Pts'] = df['Pts'].apply(limpiar_numero)
        
        # Búsqueda y limpieza de Goles (GF:GC)
        df['GF'], df['GC'] = 0.0, 0.0
        for col in df.columns:
            sample = str(df[col].iloc[0])
            if ":" in sample:
                gs = df[col].astype(str).str.split(':', expand=True)
                df['GF'] = gs[0].apply(limpiar_numero)
                df['GC'] = gs[1].apply(limpiar_numero)
                break
        
        # Partidos Jugados (PJ) - suele ser la 3ra columna
        df['PJ'] = df.iloc[:, 2].apply(limpiar_numero).replace(0, 1)
        
        return df
    except Exception as e:
        st.error(f"Error en el flujo de datos: {e}")
        return None

# --- INTERFAZ ---
st.title("🤖 SISTEMABETS IA: ANÁLISIS DE MERCADOS")
sel = st.sidebar.selectbox("Selecciona Competición:", list(FUENTES.keys()))
data = scraper_precision(FUENTES[sel])

if data is not None:
    st.success(f"Sincronizado con data real 25/26")
    
    # Selector de equipos
    equipos = data['Equipo'].unique()
    c1, c2 = st.columns(2)
    l_team = c1.selectbox("Local:", equipos, index=min(2, len(equipos)-1))
    v_team = c2.selectbox("Visitante:", equipos, index=0)

    # --- LÓGICA DE APUESTAS CON DATA REAL ---
    l_stats = data[data['Equipo'] == l_team].iloc[0]
    v_stats = data[data['Equipo'] == v_team].iloc[0]

    # Probabilidad basada en puntos reales
    prob_l = (l_stats['Pts'] / (l_stats['Pts'] + v_stats['Pts'] + 0.1)) * 100
    
    # Análisis de Over/Under y Ambos Marcan
    ataque_l = l_stats['GF'] / l_stats['PJ']
    defensa_v = v_stats['GC'] / v_stats['PJ']
    expectativa = (ataque_l + defensa_v) # Proyección de goles del local

    # --- PANEL DE VEREDICTOS ---
    st.divider()
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.subheader("🎯 Ganador")
        st.metric(l_team, f"{round(prob_l, 1)}%", f"Pts: {int(l_stats['Pts'])}")
        st.metric(v_team, f"{round(100-prob_l, 1)}%", f"Pts: {int(v_stats['Pts'])}")

    with m2:
        st.subheader("⚽ Over/Under 2.5")
        pick_ou = "OVER" if (ataque_l + (v_stats['GF']/v_stats['PJ'])) > 2.5 else "UNDER"
        st.write(f"Veredicto: **{pick_ou}**")
        st.caption(f"Arsenal promedia {round(data[data['Equipo']=='Arsenal']['GF'].iloc[0]/data[data['Equipo']=='Arsenal']['PJ'].iloc[0], 2)} goles.")

    with m3:
        st.subheader("🔥 Ambos Marcan")
        btts = "SÍ" if ataque_l > 1.1 and (v_stats['GF']/v_stats['PJ']) > 1.1 else "NO"
        st.write(f"Veredicto: **{btts}**")

    st.write("---")
    st.info(f"Análisis IA: El sistema está operando con los {int(l_stats['Pts'])} pts de {l_team} y los {int(v_stats['Pts'])} de {v_team}.")
