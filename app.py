import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# --- CONFIGURACIÓN DE PANTALLA ---
st.set_page_config(page_title="SISTEMABETS IA: MULTI-SOURCE", layout="wide")

# Mapeo de tus links a categorías
LINKS_DICT = {
    "Champions League (CL)": "https://native-stats.org/competition/CL/standings",
    "Premier League (PL)": "https://native-stats.org/competition/PL/standings",
    "La Liga (PD)": "https://native-stats.org/competition/PD/standings",
    "Ligue 1 (FL1)": "https://native-stats.org/competition/FL1/standings",
    "Bundesliga (BL1)": "https://native-stats.org/competition/BL1/standings",
    "Serie A (SA)": "https://native-stats.org/competition/SA/standings",
    "Liga Portugal (PPL)": "https://native-stats.org/competition/PPL/standings"
}

@st.cache_data(ttl=1800)
def succión_inteligente(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')
        tabla = soup.find('table')
        df = pd.read_html(str(tabla))[0]
        
        # Estandarización de columnas para la IA
        df.columns = ['Pos', 'Equipo', 'PJ', 'Pts', 'Dif', 'Goles', 'Extra']
        df[['GF', 'GC']] = df['Goles'].str.split(':', expand=True).astype(int)
        df['Equipo'] = df['Equipo'].str.replace(r'[0-9]+', '', regex=True).strip()
        return df
    except Exception as e:
        return f"Error en link: {str(e)}"

# --- INTERFAZ CENTRAL ---
st.title("🤖 IA SISTEMABETS: CENTRAL DE DATOS")
st.sidebar.header("Configuración de Fuentes")
liga_seleccionada = st.sidebar.selectbox("Selecciona Competición:", list(LINKS_DICT.keys()))

st.write(f"### Analizando: {liga_seleccionada}")
url_actual = LINKS_DICT[liga_seleccionada]

df = succión_inteligente(url_actual)

if isinstance(df, str):
    st.error(df)
else:
    st.success(f"✅ Datos sincronizados: {len(df)} equipos detectados.")
    
    # Selectores para el duelo
    c1, c2 = st.columns(2)
    local = c1.selectbox("Equipo Local:", df['Equipo'].unique(), index=0)
    visita = c2.selectbox("Equipo Visitante:", df['Equipo'].unique(), index=1)

    # --- MOTOR DE IA MULTIVARIABLE ---
    # Extraemos filas de datos
    data_l = df[df['Equipo'] == local].iloc[0]
    data_v = df[df['Equipo'] == visita].iloc[0]

    # Preparamos entrenamiento basado en la tabla completa
    X = df[['GF', 'GC', 'Pts', 'Dif']].values
    y = np.arange(len(df), 0, -1) # Ranking de poder
    
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    
    # Predicción de Probabilidades
    pred_l = model.predict([[data_l['GF'], data_l['GC'], data_l['Pts'], data_l['Dif']]])[0]
    pred_v = model.predict([[data_v['GF'], data_v['GC'], data_v['Pts'], data_v['Dif']]])[0]
    
    prob_l = (pred_l / (pred_l + pred_v)) * 100

    # --- DASHBOARD DE RESULTADOS ---
    st.divider()
    res1, res2, res3 = st.columns(3)
    res1.metric(f"Victoria {local}", f"{round(prob_l, 1)}%")
    res2.metric(f"Victoria {visita}", f"{round(100-prob_l, 1)}%")
    res3.metric("Cuota Fair", f"{round(100/prob_l, 2)}")

    # Sección de Betting Stats (Inspirado en tu link /betting)
    st.subheader("📊 Análisis de Mercado (Betting Insights)")
    b1, b2 = st.columns(2)
    
    # Lógica Over/Under básica basada en el promedio de goles
    promedio_goles = (data_l['GF'] + data_v['GC']) / data_l['PJ']
    with b1:
        st.write("**Probabilidad Over 2.5:**")
        st.progress(min(promedio_goles / 4, 1.0))
        st.caption(f"Basado en {data_l['GF']} goles a favor del local.")
    
    with b2:
        st.write("**Probabilidad Ambos Anotan:**")
        anotan = 0.8 if data_l['GF'] > 1 and data_v['GF'] > 1 else 0.4
        st.progress(anotan)

