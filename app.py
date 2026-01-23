import streamlit as st
import pandas as pd

# --- MÓDULO DE INGESTA AUTOMÁTICA ---
@st.cache_data
def cargar_datos_vivos():
    try:
        # Ejemplo: Extraemos la tabla de disparos de FBref para la Premier League
        url = "https://fbref.com/en/comps/9/shooting/Premier-League-Stats"
        # Leemos todas las tablas de la página
        tablas = pd.read_html(url)
        # La tabla principal suele ser la primera [0]
        df = tablas[0]
        
        # Limpiamos los niveles de las columnas (FBref usa índices múltiples)
        df.columns = df.columns.get_level_values(1)
        
        # Filtramos solo lo que nos importa: Equipo y SOT/90
        df_final = df[['Squad', 'SoT/90']].copy()
        df_final.dropna(inplace=True)
        return df_final
    except Exception as e:
        st.error(f"Error al conectar con la fuente de datos: {e}")
        return pd.DataFrame()

# --- ACTUALIZACIÓN DE LA APP ---
st.title("🤖 SISTEMABETS V2: AUTONOMÍA TOTAL")
df_vivo = cargar_datos_vivos()

if not df_vivo.empty:
    st.success("✅ Datos de la Premier League extraídos en tiempo real")
    st.dataframe(df_vivo) # Aquí verás la tabla real sin haber subido nada
else:
    st.warning("No se pudieron extraer datos. Revisa tu conexión o la URL.")