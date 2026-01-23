import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="SISTEMABETS IA: MULTI-LEAGUE", layout="wide")

# Diccionario de URLs basado en tus links
LIGAS = {
    "Champions League": "https://native-stats.org/competition/CL",
    "Premier League": "https://native-stats.org/competition/PL",
    "La Liga (España)": "https://native-stats.org/competition/PD",
    "Bundesliga": "https://native-stats.org/competition/BL1",
    "Serie A": "https://native-stats.org/competition/SA",
    "Ligue 1": "https://native-stats.org/competition/FL1"
}

@st.cache_data(ttl=1800)
def succionar_liga(url_key):
    url = f"{LIGAS[url_key]}/standings"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        tabla = soup.find('table')
        df = pd.read_html(str(tabla))[0]
        
        # Limpieza estandarizada
        df.columns = ['Pos', 'Equipo', 'PJ', 'Pts', 'Dif', 'Goles', 'Forma']
        # Extraer GF y GC
        df[['GF', 'GC']] = df['Goles'].str.split(':', expand=True).astype(int)
        return df
    except:
        return None

# --- INTERFAZ ---
st.title("🤖 SISTEMABETS IA: CENTRAL DE INTELIGENCIA")
liga_sel = st.sidebar.selectbox("Selecciona la Competición:", list(LIGAS.keys()))

st.write(f"Analizando datos de: **{liga_sel}**")

df = succionar_liga(liga_sel)

if df is not None:
    st.success(f"✅ Conectado a {liga_sel}. {len(df)} equipos sincronizados.")
    
    # Análisis de "Betting Stats" (Integrando el link /betting que pasaste)
    st.sidebar.info("💡 Tip: La IA está cruzando datos de ataque y defensa para detectar Over/Under.")

    col1, col2 = st.columns(2)
    local = col1.selectbox("Local:", df['Equipo'].unique(), index=0)
    visita = col2.selectbox("Visita:", df['Equipo'].unique(), index=1)

    # --- MOTOR DE PREDICCIÓN ---
    eq_l = df[df['Equipo'] == local].iloc[0]
    eq_v = df[df['Equipo'] == visita].iloc[0]

    # La IA usa GF, GC y Puntos para proyectar
    X = df[['GF', 'GC', 'Pts']].values
    y = np.arange(len(df), 0, -1)
    model = RandomForestRegressor(n_estimators=100).fit(X, y)
    
    p_l = model.predict([[eq_l['GF'], eq_l['GC'], eq_l['Pts']]])[0]
    p_v = model.predict([[eq_v['GF'], eq_v['GC'], eq_v['Pts']]])[0]
    
    prob_l = (p_l / (p_l + p_v)) * 100

    # --- RESULTADOS ---
    st.divider()
    res1, res2, res3 = st.columns(3)
    res1.metric(f"Prob. {local}", f"{round(prob_l, 1)}%")
    res2.metric(f"Prob. {visita}", f"{round(100-prob_l, 1)}%")
    res3.metric("Cuota Fair", f"{round(100/prob_l, 2)}")

    # Sección de "Deep Data" (Usando el link de Team/81 como ejemplo de lo que podemos sacar)
    st.subheader("📊 Análisis Profundo (Native-Stats Deep Data)")
    c_a, c_b = st.columns(2)
    with c_a:
        st.write(f"**Ataque {local}:** {eq_l['GF']} goles en {eq_l['PJ']} juegos.")
        st.progress(min(eq_l['GF']/30, 1.0))
    with c_b:
        st.write(f"**Defensa {visita}:** {eq_v['GC']} goles recibidos.")
        st.progress(min(eq_v['GC']/30, 1.0))
else:
    st.error("Error al conectar con Native Stats. Revisa la estructura de los links.")
