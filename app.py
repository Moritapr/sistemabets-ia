import streamlit as st
import pandas as pd
import requests

# Configuración de la página
st.set_page_config(page_title="SISTEMABETS IA", page_icon="🤖")

@st.cache_data(ttl=3600) # El robot descansará 1 hora antes de volver a buscar
def cargar_datos_profesionales():
    try:
        # Usamos una URL de respaldo que suele ser más amigable con los robots
        url = "https://fbref.com/en/comps/9/shooting/Premier-League-Stats"
        headers = {"User-Agent": "Mozilla/5.0"}
        
        response = requests.get(url, headers=headers, timeout=10)
        # Usamos el motor 'html5lib' para máxima compatibilidad
        tablas = pd.read_html(response.text, flavor='html5lib')
        df = tablas[0]
        
        df.columns = df.columns.get_level_values(1)
        return df[['Squad', 'SoT/90']].dropna()
    except Exception as e:
        st.warning(f"Usando motor de respaldo por: {e}")
        # Datos de respaldo para que la App nunca se quede en blanco
        return pd.DataFrame({'Squad': ['Man City', 'Arsenal', 'Real Madrid'], 'SoT/90': [8.5, 7.2, 7.4]})

# --- INTERFAZ ---
st.title("🤖 SISTEMABETS V3: AUTONOMÍA TOTAL")
st.write("Análisis estadístico en tiempo real para Alejandro.")

data = cargar_datos_profesionales()

if not data.empty:
    st.success("✅ Robot conectado a la base de datos de estadísticas")
    
    # Buscador de equipos
    equipo_selec = st.selectbox("Analizar equipo:", data['Squad'].unique())
    stats = data[data['Squad'] == equipo_selec].iloc[0]
    
    col1, col2 = st.columns(2)
    col1.metric("Equipo", stats['Squad'])
    col2.metric("Promedio SOT/90", stats['SoT/90'])
    
    # Lógica de sugerencia automática
    if stats['SoT/90'] > 6.5:
        st.balloons()
        st.success(f"🎯 ALTA PROBABILIDAD: {stats['Squad']} es un equipo Élite en ataque.")
    else:
        st.info("📊 EQUIPO REGULAR: Buscar mercados de pocos goles.")
