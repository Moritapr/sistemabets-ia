import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# --- CONFIGURACIÓN CENTRAL ---
st.set_page_config(page_title="SISTEMABETS IA: FULL-DATA", layout="wide")

# RESPETAMOS TUS LINKS EXACTAMENTE COMO LOS PASASTE
FUENTES = {
    "Champions League": "https://native-stats.org/competition/CL/",
    "Premier League": "https://native-stats.org/competition/PL",
    "La Liga (España)": "https://native-stats.org/competition/PD",
    "Bundesliga": "https://native-stats.org/competition/BL1",
    "Serie A": "https://native-stats.org/competition/SA",
    "Ligue 1": "https://native-stats.org/competition/FL1",
    "Liga Portugal": "https://native-stats.org/competition/PPL",
    "Betting Trends": "https://native-stats.org/betting",
    "Perfil Equipo (ID:81)": "https://native-stats.org/team/81",
    "Home": "https://native-stats.org/"
}

@st.cache_data(ttl=1800)
def scraper_puro(url):
    # Usamos un User-Agent de Chrome real para evitar bloqueos
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    try:
        # Entramos al link tal cual lo pasaste
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Buscamos todas las tablas en la página
        tablas = soup.find_all('table')
        
        if not tablas:
            return "No se detectaron tablas en este link. Verifica si la página cargó correctamente."

        # Buscamos la tabla que parezca una clasificación (la que tenga más filas)
        tabla_principal = max(tablas, key=lambda t: len(t.find_all('tr')))
        df = pd.read_html(str(tabla_principal))[0]

        # Limpieza básica para que la IA no explote
        # Native-stats suele tener: Pos, Team, P, W, D, L, Goals, Diff, Pts
        if len(df.columns) >= 7:
            # Forzamos nombres de columnas estándar para el modelo
            df = df.iloc[:, :9] # Tomamos las primeras 9 columnas si existen
            # Asignamos nombres basados en la estructura común de Native Stats
            # Nota: Esto se adapta si la tabla tiene menos columnas
            cols = ['Pos', 'Equipo', 'PJ', 'V', 'E', 'D', 'Goles', 'Dif', 'Pts']
            df.columns = cols[:len(df.columns)]
        
        # Limpiar nombres (Quitar el número de posición que a veces viene pegado)
        df['Equipo'] = df['Equipo'].astype(str).str.replace(r'^\d+\s+', '', regex=True).str.strip()
        
        # Procesar Goles "20:2" si existe la columna
        if 'Goles' in df.columns:
            goles_split = df['Goles'].astype(str).str.split(':', expand=True)
            df['GF'] = pd.to_numeric(goles_split[0], errors='coerce').fillna(0).astype(int)
            df['GC'] = pd.to_numeric(goles_split[1], errors='coerce').fillna(0).astype(int)
        
        return df
    except Exception as e:
        return f"Error al acceder al link: {str(e)}"

# --- INTERFAZ ---
st.sidebar.title("🔍 CONTROL DE FUENTES")
seleccion = st.sidebar.selectbox("Selecciona Link Real:", list(FUENTES.keys()))

st.title(f"🤖 SISTEMABETS IA: MODO {seleccion.upper()}")
st.info(f"Conectado a: {FUENTES[seleccion]}")

df = scraper_puro(FUENTES[seleccion])

if isinstance(df, str):
    st.error(df)
else:
    st.success(f"✅ ¡DENTRO! {len(df)} registros detectados.")
    
    # Mostrar la tabla para que verifiques que es la correcta
    with st.expander("Ver tabla de datos crudos"):
        st.dataframe(df)

    # Análisis de Duelo si la tabla tiene equipos y puntos
    if 'Equipo' in df.columns and 'Pts' in df.columns:
        c1, c2 = st.columns(2)
        local = c1.selectbox("Local:", df['Equipo'].unique(), index=2 if len(df)>2 else 0)
        visita = c2.selectbox("Visita:", df['Equipo'].unique(), index=0)

        # --- MOTOR IA (Random Forest) ---
        try:
            stats_l = df[df['Equipo'] == local].iloc[0]
            stats_v = df[df['Equipo'] == visita].iloc[0]

            # Si no hay GF/GC (como en el link de /betting), usamos Puntos y Posición
            features = ['Pts']
            if 'GF' in df.columns: features.extend(['GF', 'GC'])
            
            X = df[features].fillna(0).values
            y = np.arange(len(df), 0, -1)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
            
            # Predicción
            pred_l = model.predict([stats_l[features].fillna(0).values])[0]
            pred_v = model.predict([stats_v[features].fillna(0).values])[0]
            
            prob_l = (pred_l / (pred_l + pred_v)) * 100

            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric(f"Victoria {local}", f"{round(prob_l, 1)}%")
            m2.metric(f"Victoria {visita}", f"{round(100-prob_l, 1)}%")
            m3.metric("Probabilidad Empate", f"{round(abs(prob_l - (100-prob_l)), 1)}%")
        except:
            st.warning("Esta tabla no tiene el formato estándar de liga para el análisis comparativo.")
