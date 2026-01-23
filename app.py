import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# --- CONFIGURACIÓN CENTRAL ---
st.set_page_config(page_title="SISTEMABETS IA: FULL-DATA", layout="wide")

# MANTENEMOS TODOS TUS LINKS ORIGINALES
FUENTES = {
    "Champions League": "https://native-stats.org/competition/CL/standings",
    "Premier League": "https://native-stats.org/competition/PL/standings",
    "La Liga (España)": "https://native-stats.org/competition/PD/standings",
    "Bundesliga": "https://native-stats.org/competition/BL1/standings",
    "Serie A": "https://native-stats.org/competition/SA/standings",
    "Ligue 1": "https://native-stats.org/competition/FL1/standings",
    "Liga Portugal": "https://native-stats.org/competition/PPL/standings",
    "Betting Trends": "https://native-stats.org/betting",
    "Perfil Equipo (ID:81)": "https://native-stats.org/team/81"
}

@st.cache_data(ttl=1800)
def scraper_multilink(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.content, 'lxml')
        tablas = soup.find_all('table')
        
        if not tablas:
            return "No se detectaron tablas en este link específico."

        # BUSCADOR INTELIGENTE DE TABLAS (Para no errar el índice)
        # Buscamos la tabla que tenga columnas de puntos o estadísticas
        df_final = None
        for t in tablas:
            temp_df = pd.read_html(str(t))[0]
            # Si tiene al menos 5 columnas, es una tabla de datos útil
            if len(temp_df.columns) >= 5:
                df_final = temp_df
                break
        
        if df_final is None: return "Estructura de tabla no compatible."

        # LIMPIEZA QUIRÚRGICA (Evita el 'out of range')
        # Nos quedamos con las columnas que realmente necesitamos
        df_final = df_final.iloc[:, :7]
        df_final.columns = ['Pos', 'Equipo', 'PJ', 'Pts', 'Dif', 'Goles', 'Extra']
        
        # Limpiar nombres (Ej: "1 Arsenal" -> "Arsenal")
        df_final['Equipo'] = df_final['Equipo'].astype(str).str.replace(r'^\d+\s+', '', regex=True).str.strip()
        
        # Procesar Goles "20:2" -> GF=20, GC=2
        # Usamos split expandido con manejo de errores
        goles_split = df_final['Goles'].astype(str).str.split(':', expand=True)
        df_final['GF'] = pd.to_numeric(goles_split[0], errors='coerce').fillna(0).astype(int)
        df_final['GC'] = pd.to_numeric(goles_split[1], errors='coerce').fillna(0).astype(int)
        
        return df_final
    except Exception as e:
        return f"Fallo en conexión: {str(e)}"

# --- INTERFAZ ---
st.sidebar.title("🔍 CONTROL DE FUENTES")
seleccion = st.sidebar.selectbox("Selecciona Link para Scrapping:", list(FUENTES.keys()))

st.title(f"🤖 SISTEMABETS IA: MODO {seleccion.upper()}")
st.write(f"Sincronizando con: {FUENTES[seleccion]}")

df = scraper_multilink(FUENTES[seleccion])

if isinstance(df, str):
    st.error(df)
    st.info("💡 Algunos links como '/betting' tienen formatos distintos. El scraper está optimizado para las tablas de posiciones.")
else:
    st.success(f"✅ IA CONECTADA. {len(df)} registros succionados.")
    
    # Análisis de Duelo
    c1, c2 = st.columns(2)
    with c1:
        local = st.selectbox("Local:", df['Equipo'].unique(), index=2 if len(df)>2 else 0)
    with c2:
        visita = st.selectbox("Visita:", df['Equipo'].unique(), index=0)

    # --- MOTOR IA (Random Forest) ---
    try:
        stats_l = df[df['Equipo'] == local].iloc[0]
        stats_v = df[df['Equipo'] == visita].iloc[0]

        # Entrenamiento basado en la liga actual (Arsenal 21 pts, Madrid 15 pts)
        X = df[['GF', 'GC', 'Pts', 'Dif']].values
        y = np.arange(len(df), 0, -1) # Ranking
        
        model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
        
        # Predicción
        p_l = model.predict([[stats_l['GF'], stats_l['GC'], stats_l['Pts'], stats_l['Dif']]])[0]
        p_v = model.predict([[stats_v['GF'], stats_v['GC'], stats_v['Pts'], stats_v['Dif']]])[0]
        
        prob_l = (p_l / (p_l + p_v)) * 100

        # --- DASHBOARD ---
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Victoria {local}", f"{round(prob_l, 1)}%", f"GF: {stats_l['GF']}")
        m2.metric(f"Victoria {visita}", f"{round(100-prob_l, 1)}%", f"GF: {stats_v['GF']}")
        m3.metric("Cuota Fair", f"{round(100/prob_l, 2)}")
        
        # Módulo de Goles (Mercado)
        st.subheader("📊 Análisis de Probabilidades (Over/Under)")
        promedio = (stats_l['GF'] + stats_v['GC']) / (stats_l['PJ'] + 0.1)
        st.progress(min(promedio/4, 1.0))
        st.write(f"Tendencia de goles: **{round(promedio, 2)}** por partido basándose en data actual.")

    except Exception as e:
        st.warning("Selecciona dos equipos diferentes para procesar el análisis.")
