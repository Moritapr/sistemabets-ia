import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# --- CONFIGURACIÓN DE IDENTIDAD ---
st.set_page_config(page_title="SISTEMABETS IA: UNSTOPPABLE", layout="wide")

# Mapeo exacto de tus links
FUENTES = {
    "Champions League": "https://native-stats.org/competition/CL/standings",
    "Premier League": "https://native-stats.org/competition/PL/standings",
    "La Liga (España)": "https://native-stats.org/competition/PD/standings",
    "Bundesliga": "https://native-stats.org/competition/BL1/standings",
    "Serie A": "https://native-stats.org/competition/SA/standings",
    "Ligue 1": "https://native-stats.org/competition/FL1/standings",
    "Liga Portugal": "https://native-stats.org/competition/PPL/standings",
    "Betting Insights": "https://native-stats.org/betting"
}

@st.cache_data(ttl=600)
def super_scraper(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36',
        'Accept-Language': 'es-ES,es;q=0.9'
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return f"Error de servidor: {response.status_code}"
            
        # Usamos lxml para mayor velocidad y precisión
        soup = BeautifulSoup(response.content, 'lxml')
        tablas = soup.find_all('table')
        
        if not tablas:
            return "No se encontraron tablas en esta URL. Es posible que el sitio use carga dinámica pesada."

        # Buscamos la tabla que tenga más filas (la de posiciones)
        tabla_principal = max(tablas, key=lambda t: len(t.find_all('tr')))
        df = pd.read_html(str(tabla_principal))[0]
        
        # --- LIMPIEZA QUIRÚRGICA ---
        # Si la tabla tiene muchas columnas, nos quedamos con las esenciales
        # Generalmente: Pos, Team, P, W, D, L, Goals, Diff, Pts
        if len(df.columns) >= 7:
            # Intentamos detectar columnas por nombre o posición
            df = df.iloc[:, [0, 1, 2, 3, 4, 5, 6]] # Ajuste estándar
            df.columns = ['Pos', 'Equipo', 'PJ', 'Pts', 'Dif', 'Goles', 'Forma']
        
        # Limpiar nombres de equipos (quitar basura de scraping)
        df['Equipo'] = df['Equipo'].astype(str).str.replace(r'\d+', '', regex=True).str.strip()
        
        # Procesar Goles (Evitar el error de split si la celda está vacía)
        df['Goles'] = df['Goles'].astype(str)
        df[['GF', 'GC']] = df['Goles'].str.split(':', expand=True).iloc[:, :2].fillna(0).astype(int)
        
        return df
    except Exception as e:
        return f"Fallo en la extracción: {str(e)}"

# --- INTERFAZ ---
st.title("🤖 SISTEMABETS IA: CENTRAL DE INTELIGENCIA 25/26")
st.sidebar.header("🔧 Panel de Control")
opcion = st.sidebar.selectbox("Selecciona Competición:", list(FUENTES.keys()))

st.write(f"### Analizando: {opcion}")

with st.spinner('Sincronizando con Native-Stats...'):
    df = super_scraper(FUENTES[opcion])

if isinstance(df, str):
    st.error(df)
    st.info("💡 Alejandro, si el error persiste, intenta refrescar la página de Streamlit. A veces el servidor de la web nos bloquea temporalmente.")
else:
    st.success(f"✅ ¡DENTRO! {len(df)} equipos cargados con éxito.")
    
    col1, col2 = st.columns(2)
    with col1:
        local = st.selectbox("Equipo Local:", df['Equipo'].unique(), index=2 if len(df)>2 else 0)
    with col2:
        visita = st.selectbox("Equipo Visitante:", df['Equipo'].unique(), index=0)

    # --- MOTOR DE IA (Random Forest) ---
    try:
        # Filtramos data
        s_l = df[df['Equipo'] == local].iloc[0]
        s_v = df[df['Equipo'] == visita].iloc[0]

        # Entrenamiento express
        X = df[['GF', 'GC', 'Pts']].values
        y = np.arange(len(df), 0, -1)
        model = RandomForestRegressor(n_estimators=100).fit(X, y)

        # Predicción
        p_l = model.predict([[s_l['GF'], s_l['GC'], s_l['Pts']]])[0]
        p_v = model.predict([[s_v['GF'], s_v['GC'], s_v['Pts']]])[0]
        
        prob_l = (p_l / (p_l + p_v)) * 100

        # --- DASHBOARD ---
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Victoria {local}", f"{round(prob_l, 1)}%", f"GF: {s_l['GF']}")
        m2.metric(f"Victoria {visita}", f"{round(100-prob_l, 1)}%", f"GF: {s_v['GF']}")
        m3.metric("Cuota Fair", f"{round(100/prob_l, 2)}")
        
        st.subheader("📊 Comparativa de Poder")
        st.bar_chart(df.set_index('Equipo')[['GF', 'GC']].loc[[local, visita]])

    except Exception as e:
        st.warning("Selecciona dos equipos distintos para el análisis.")
