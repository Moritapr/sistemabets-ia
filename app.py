import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="SISTEMABETS IA: ULTRA-PRECISIÓN", layout="wide")

FUENTES = {
    "Champions League": "https://native-stats.org/competition/CL/",
    "Premier League": "https://native-stats.org/competition/PL",
    "La Liga": "https://native-stats.org/competition/PD",
    "Bundesliga": "https://native-stats.org/competition/BL1",
    "Serie A": "https://native-stats.org/competition/SA",
    "Ligue 1": "https://native-stats.org/competition/FL1",
    "Liga Portugal": "https://native-stats.org/competition/PPL",
    "Betting Trends": "https://native-stats.org/betting",
    "Real Madrid": "https://native-stats.org/team/81"
}

def motor_extraccion_total(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=20)
        soup = BeautifulSoup(response.content, 'html.parser')
        tablas = soup.find_all('table')
        if not tablas: return "ERROR: No hay tablas en la web."
        
        # Seleccionamos la tabla con más datos (Standings)
        df = pd.read_html(str(max(tablas, key=lambda t: len(t.find_all('tr')))))[0]
        
        # 1. NORMALIZACIÓN DE COLUMNAS (Mapeo por palabras clave)
        cols_map = {}
        for i, col in enumerate(df.columns):
            c_name = str(col).upper()
            if any(x in c_name for x in ['TEAM', 'EQUIPO', 'CLUB']): cols_map[col] = 'Equipo'
            elif any(x in c_name for x in ['PTS', 'POINTS', 'PUNTOS']): cols_map[col] = 'Pts'
            elif any(x in c_name for x in ['MP', 'PJ', 'PLAYED', 'MATCHES']): cols_map[col] = 'PJ'
            elif any(x in c_name for x in ['GOALS', 'GOLES', 'G', 'F:A']): cols_map[col] = 'Goles_Raw'
            
        df = df.rename(columns=cols_map)

        # 2. SEGURO CONTRA ERROR 'PJ'
        if 'PJ' not in df.columns:
            # Si no existe la columna, intentamos buscarla por posición (usualmente la 3ra)
            df.rename(columns={df.columns[2]: 'PJ'}, inplace=True)

        # 3. LIMPIEZA DE DATOS CRÍTICOS
        df['Equipo'] = df['Equipo'].astype(str).str.replace(r'^\d+\s+', '', regex=True).str.strip()
        df['Pts'] = pd.to_numeric(df['Pts'], errors='coerce').fillna(0)
        df['PJ'] = pd.to_numeric(df['PJ'], errors='coerce').fillna(1).replace(0, 1)

        # 4. EXTRACCIÓN DE GOLES GF:GC
        if 'Goles_Raw' in df.columns:
            gs = df['Goles_Raw'].astype(str).str.extract(r'(\d+):(\d+)')
            df['GF'] = pd.to_numeric(gs[0]).fillna(0)
            df['GC'] = pd.to_numeric(gs[1]).fillna(0)
        else:
            # Búsqueda manual de la columna con el formato ":"
            for col in df.columns:
                if df[col].astype(str).str.contains(':').any():
                    gs = df[col].astype(str).str.extract(r'(\d+):(\d+)')
                    df['GF'], df['GC'] = pd.to_numeric(gs[0]).fillna(0), pd.to_numeric(gs[1]).fillna(0)
                    break
        
        # Validar que existan goles para no dar datos falsos
        if 'GF' not in df.columns: return "ERROR: Formato de goles no detectado."

        return df
    except Exception as e:
        return f"FALLO TÉCNICO: {str(e)}"

# --- INTERFAZ ---
st.title("🛡️ SISTEMABETS IA: DATA MINING 25/26")
comp = st.sidebar.selectbox("SELECCIONAR LIGA:", list(FUENTES.keys()))
df = motor_extraccion_total(FUENTES[comp])

if isinstance(df, pd.DataFrame):
    st.success(f"Conexión estable. Analizando {len(df)} equipos de {comp}.")
    
    # Análisis de Duelo
    eqs = sorted(df['Equipo'].unique())
    c1, c2 = st.columns(2)
    l_team = c1.selectbox("Local:", eqs, index=min(2, len(eqs)-1))
    v_team = c2.selectbox("Visita:", eqs, index=0)

    # Stats Reales
    sl, sv = df[df['Equipo'] == l_team].iloc[0], df[df['Equipo'] == v_team].iloc[0]

    # CÁLCULO DE PROBABILIDAD (Modelo Poisson)
    # Lambdas (Goles esperados)
    lambda_l = (sl['GF'] / sl['PJ']) * (sv['GC'] / df['GC'].mean())
    lambda_v = (sv['GF'] / sv['PJ']) * (sl['GC'] / df['GC'].mean())

    # Matriz de probabilidades
    m = np.outer(poisson.pmf(range(6), lambda_l), poisson.pmf(range(6), lambda_v))
    p_l, p_e, p_v = np.sum(np.tril(m, -1)), np.sum(np.diag(m)), np.sum(np.triu(m, 1))
    p_over = 1 - (m[0,0] + m[0,1] + m[1,0] + m[1,1] + m[2,0] + m[0,2])

    # --- RESULTADOS ---
    st.divider()
    res1, res2, res3, res4 = st.columns(4)
    res1.metric(f"Gana {l_team}", f"{round(p_l*100,1)}%")
    res2.metric("Empate", f"{round(p_e*100,1)}%")
    res3.metric(f"Gana {v_team}", f"{round(p_v*100,1)}%")
    res4.metric("Prob. Over 2.5", f"{round(p_over*100,1)}%")

    st.subheader("📊 Métricas de Ingeniería vs Rendimiento Real")
    
    st.write(f"**Análisis de Goles:** El sistema proyecta **{round(lambda_l + lambda_v, 2)}** goles totales para este encuentro basándose en los {int(sl['GF'])} goles del local y {int(sv['GF'])} del visitante.")

    with st.expander("Inspeccionar Tabla de Datos Original"):
        st.dataframe(df[['Equipo', 'PJ', 'GF', 'GC', 'Pts']])
else:
    st.error(df)
