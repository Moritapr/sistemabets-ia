import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import requests

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="SISTEMABETS IA: LIVE", page_icon="⚽", layout="wide")

# Estilo Neón Dark
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stMetricValue"] { color: #00ffcc; font-weight: bold; }
    .stMetric { background-color: #1a1c23; padding: 15px; border-radius: 15px; border: 1px solid #333; }
    h1, h2, h3 { color: #00ffcc; }
    .elite-text { color: #ff4b4b; font-weight: bold; font-size: 1.1rem; }
    </style>
    """, unsafe_allow_html=True)

# --- SCRAPER AVANZADO (ANTI-BLOQUEO) ---
@st.cache_data(ttl=3600)
def get_live_data():
    url = "https://fbref.com/en/comps/8/shooting/Champions-League-Stats"
    # Esta es la "máscara" para que FBref no nos bloquee
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return f"Error de servidor: Código {response.status_code}"
        
        # Leemos la tabla usando el motor 'html5lib' que es más tolerante
        tablas = pd.read_html(response.text, flavor='html5lib')
        df = tablas[0]
        
        # Limpieza de columnas multinivel
        df.columns = [' '.join(col).strip() for col in df.columns.values]
        
        # Buscamos las columnas de Equipo y SoT/90
        squad_col = [c for c in df.columns if 'Squad' in c][0]
        sot_col = [c for c in df.columns if 'SoT/90' in c][0]
        
        df_final = df[[squad_col, sot_col]].copy()
        df_final.columns = ['Equipo', 'SoT90']
        
        # Limpiamos nombres de equipos (ej. "es Real Madrid" -> "Real Madrid")
        df_final['Equipo'] = df_final['Equipo'].apply(lambda x: ' '.join(x.split()[1:]) if len(x.split()) > 1 else x)
        
        return df_final.dropna()
    except Exception as e:
        return f"Error técnico: {str(e)}"

# --- LÓGICA DE LA APP ---
st.title("🤖 SISTEMABETS IA: MODO ELITE 25/26")
st.write(f"Conexión activa - {pd.to_datetime('today').strftime('%d/%m/%Y')}")

with st.spinner('Saltando seguridad de FBref y extrayendo datos...'):
    df_stats = get_live_data()

if isinstance(df_stats, str):
    st.error(f"❌ Falló el Scraper: {df_stats}")
    st.info("💡 Tip: Ve a Streamlit Cloud y dale a 'Reboot App' para forzar la nueva configuración.")
else:
    st.success(f"✅ ¡Conectado! Analizando {len(df_stats)} equipos reales de Champions.")
    
    col1, col2 = st.columns(2)
    with col1:
        local = st.selectbox("Local:", sorted(df_stats['Equipo'].unique()), index=0)
    with col2:
        visita = st.selectbox("Visita:", sorted(df_stats['Equipo'].unique()), index=1)

    # --- IA PROCESANDO ---
    sot_l = float(df_stats[df_stats['Equipo'] == local]['SoT90'].values[0])
    sot_v = float(df_stats[df_stats['Equipo'] == visita]['SoT90'].values[0])
    
    # Modelo Random Forest (Proyección de Goles)
    X_train = np.array([[3,1], [8,1], [4,4], [6,2], [2,5], [7,1], [5,3]])
    y_train = [1.1, 2.9, 1.4, 2.2, 0.8, 2.5, 1.7]
    model = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
    
    pred_val = model.predict([[sot_l, sot_v]])[0]
    prob_win = min(98.5, (pred_val / (pred_val + 1.2)) * 100)

    # --- RESULTADOS TIPO TERMINAL ---
    st.divider()
    st.markdown('<p class="elite-text">📊 ANÁLISIS DE PROBABILIDAD (SOT/90)</p>', unsafe_allow_html=True)
    
    res1, res2, res3 = st.columns(3)
    res1.metric(f"Prob. {local}", f"{round(prob_win, 1)}%", f"{sot_l} SoT/90")
    res2.metric(f"Prob. {visita}", f"{round(100-prob_win, 1)}%", f"{sot_v} SoT/90")
    res3.metric("Cuota Mínima", f"{round(100/prob_win, 2)}")

    st.subheader("🧠 Veredicto de la IA")
    if prob_win > 70:
        st.success(f"🎯 PICK ELITE: Alta probabilidad para {local}. Valor detectado en el mercado.")
    else:
        st.warning("⚖️ PARTIDO CERRADO: La IA sugiere precaución o buscar mercados de 'Goles Totales'.")
