import streamlit as st
import pandas as pd
import requests

@st.cache_data
def cargar_datos_vivos():
    try:
        url = "https://fbref.com/en/comps/9/shooting/Premier-League-Stats"
        # DISFRAZ DE HUMANO: Esto evita el Error 403
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers)
        tablas = pd.read_html(response.text)
        df = tablas[0]
        
        # Limpieza de columnas
        df.columns = df.columns.get_level_values(1)
        df_final = df[['Squad', 'SoT/90']].copy()
        df_final.dropna(inplace=True)
        return df_final
    except Exception as e:
        st.error(f"Error de ingeniería: {e}")
        return pd.DataFrame()
