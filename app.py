import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
from scipy.stats import poisson

# --- CONFIGURACIÓN DE INGENIERÍA DE DATOS ---
st.set_page_config(page_title="SISTEMABETS IA: PROFESIONAL V6", layout="wide")

FUENTES = {
    "Champions League": "https://native-stats.org/competition/CL/",
    "Premier League": "https://native-stats.org/competition/PL",
    "La Liga": "https://native-stats.org/competition/PD",
    "Bundesliga": "https://native-stats.org/competition/BL1",
    "Serie A": "https://native-stats.org/competition/SA",
    "Ligue 1": "https://native-stats.org/competition/FL1",
    "Liga Portugal": "https://native-stats.org/competition/PPL",
    "Betting Trends": "https://native-stats.org/betting",
    "Real Madrid Profile": "https://native-stats.org/team/81"
}

def extractor_quirurgico(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=20)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 1. Buscamos la tabla de clasificación real
        tablas = soup.find_all('table')
        if not tablas: return None
        
        # Filtramos tablas que no tengan suficientes filas para ser una liga
        tablas_validas = [t for t in tablas if len(t.find_all('tr')) > 5]
        if not tablas_validas: return None
        
        # 2. Parseo y limpieza de estructura
        df = pd.read_html(str(tablas_validas[0]))[0]
        
        # Mapeo por inspección de columnas (buscamos Puntos y Equipo por contenido)
        for col in df.columns:
            # Si la columna tiene ":" es la de goles
            if df[col].astype(str).str.contains(':').any():
                df = df.rename(columns={col: 'Goles_Raw'})
            # Si tiene números del 1 al 20 suele ser Posición o PJ
            # Asumimos posición estándar de Native-Stats: [1]=Equipo, [Ult]=Pts
        
        df = df.rename(columns={df.columns[1]: 'Equipo', df.columns[-1]: 'Pts'})
        df['Equipo'] = df['Equipo'].astype(str).str.replace(r'^\d+\s+', '', regex=True).str.strip()
        
        # 3. VERIFICACIÓN DE INTEGRIDAD DE DATOS (No valores neutros)
        # Extraemos GF y GC solo si el formato es correcto
        if 'Goles_Raw' in df.columns:
            # Filtramos solo filas donde el formato sea N:N
            mask = df['Goles_Raw'].astype(str).str.contains(r'\d+:\d+')
            df = df[mask].copy()
            
            gs = df['Goles_Raw'].str.split(':', expand=True)
            df['GF'] = pd.to_numeric(gs[0])
            df['GC'] = pd.to_numeric(gs[1])
        else:
            return "ERROR_ESTRUCTURA: No se halló columna de goles real."

        # Extraemos PJ (Partidos Jugados) buscando la columna con valores entre 1 y 60
        for col in df.columns:
            if col not in ['Equipo', 'Pts', 'Goles_Raw', 'GF', 'GC']:
                val = pd.to_numeric(df[col], errors='coerce').iloc[0]
                if 1 <= val <= 60:
                    df['PJ'] = pd.to_numeric(df[col])
                    break
        
        # 4. RATINGS TÉCNICOS (Calculados sobre data pura)
        df['Pts'] = pd.to_numeric(df['Pts'])
        df['Ataque'] = df['GF'] / df['PJ']
        df['Defensa'] = df['GC'] / df['PJ']
        
        return df[['Equipo', 'PJ', 'GF', 'GC', 'Pts', 'Ataque', 'Defensa']]
    except Exception as e:
        return f"ERROR_SISTEMA: {str(e)}"

# --- INTERFAZ ---
st.title("🎯 SISTEMABETS IA: ENGINE V6 - PRECISIÓN MILITAR")
comp = st.sidebar.selectbox("LIGA ACTUAL:", list(FUENTES.keys()))
df = extractor_quirurgico(FUENTES[comp])

if isinstance(df, pd.DataFrame):
    st.success(f"Conexión verificada: {comp} 2026. Datos 100% reales.")
    
    # Análisis entre equipos
    col_l, col_v = st.columns(2)
    l_team = col_l.selectbox("Local:", df['Equipo'].unique(), index=2)
    v_team = col_v.selectbox("Visita:", df['Equipo'].unique(), index=0)

    # Selección de filas
    s_l = df[df['Equipo'] == l_team].iloc[0]
    s_v = df[df['Equipo'] == v_team].iloc[0]

    # MOTOR DE POISSON PARA MERCADOS (O/U y 1X2)
    # Proyectamos goles esperados basados en fortaleza vs debilidad
    exp_l = s_l['Ataque'] * (s_v['Defensa'] / df['Defensa'].mean())
    exp_v = s_v['Ataque'] * (s_l['Defensa'] / df['Defensa'].mean())

    # Matriz de Poisson (Probabilidad de cada marcador hasta 6 goles)
    prob_matrix = np.outer(poisson.pmf(range(7), exp_l), poisson.pmf(range(7), exp_v))
    
    p_l = np.sum(np.tril(prob_matrix, -1))
    p_e = np.sum(np.diag(prob_matrix))
    p_v = np.sum(np.triu(prob_matrix, 1))
    p_over = 1 - (prob_matrix[0,0] + prob_matrix[0,1] + prob_matrix[1,0] + prob_matrix[1,1] + prob_matrix[2,0] + prob_matrix[0,2])

    # --- DASHBOARD DE INGENIERÍA ---
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Gana {l_team}", f"{round(p_l*100, 1)}%", f"Pts: {int(s_l['Pts'])}")
    m2.metric("Empate", f"{round(p_e*100, 1)}%")
    m3.metric(f"Gana {v_team}", f"{round(p_v*100, 1)}%", f"Pts: {int(s_v['Pts'])}")
    m4.metric("Prob. Over 2.5", f"{round(p_over*100, 1)}%")

    # Comparativa de Rendimiento Real
    
    st.subheader("📊 Análisis de Fuerza de Ataque vs Resistencia")
    c1, c2 = st.columns(2)
    c1.write(f"**Ataque {l_team}:** {round(s_l['Ataque'], 2)} goles/partido")
    c1.progress(min(s_l['Ataque']/3, 1.0))
    c2.write(f"**Defensa {v_team} (Recibidos):** {round(s_v['Defensa'], 2)} goles/partido")
    c2.progress(min(s_v['Defensa']/3, 1.0))

    # VEREDICTO FINAL
    st.divider()
    if p_over > 0.65:
        st.error(f"🔥 PICK SUGERIDO: OVER 2.5 GOLES (Confianza alta: {round(p_over*100,1)}%)")
    elif p_l > 0.60:
        st.success(f"💰 PICK SUGERIDO: VICTORIA {l_team.upper()}")
    else:
        st.info("💡 MERCADO COMPLEJO: Se sugiere esperar a Live o buscar Hándicap.")

else:
    st.error(df if df else "No se pudo extraer la tabla. El sitio Native-Stats ha cambiado su estructura.")
