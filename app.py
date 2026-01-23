import streamlit as st
import requests
import pandas as pd

# --- CONFIGURACIÓN DE IDENTIDAD ---
st.set_page_config(page_title="SISTEMABETS IA: CHAMPIONS 25/26", layout="wide")

# TU LLAVE MAESTRA DE RAPIDAPI
API_KEY = "97aad21a39msh44116ce32f77720p1489fcjsn7520e33e8485"

@st.cache_data(ttl=3600)
def fetch_champions_2026():
    # Esta API nos da acceso a la temporada actual de UEFA
    url = "https://football-prediction-api.p.rapidapi.com/api/v2/predictions"
    querystring = {"federation":"UEFA", "market":"classic"}
    headers = {
        "X-RapidAPI-Key": API_KEY,
        "X-RapidAPI-Host": "football-prediction-api.p.rapidapi.com"
    }
    
    try:
        response = requests.get(url, headers=headers, params=querystring, timeout=15)
        res_json = response.json()
        
        if not res_json.get('data'):
            return "No hay partidos disponibles hoy o la Key llegó al límite."

        # Procesamos la data de la jornada 25/26
        lista_partidos = []
        for match in res_json['data']:
            lista_partidos.append({
                'Local': match['home_team'],
                'Visita': match['away_team'],
                'Prob_L': float(match['probabilities']['home']),
                'Prob_E': float(match['probabilities']['draw']),
                'Prob_V': float(match['probabilities']['away']),
                'Recomendacion': match['prediction'],
                'Fecha': match['start_date']
            })
        return pd.DataFrame(lista_partidos)
    except Exception as e:
        return f"Error de protocolo: {str(e)}"

# --- INTERFAZ STREAMLIT ---
st.title("🤖 SISTEMABETS IA: CHAMPIONS ELITE")
st.write(f"Análisis basado en datos reales - {pd.to_datetime('today').strftime('%d/%m/%Y')}")

with st.spinner('Sincronizando con el servidor de predicciones UEFA...'):
    df_matches = fetch_champions_2026()

if isinstance(df_matches, str):
    st.error(df_matches)
    st.info("💡 Tip: Verifica en RapidAPI que el plan 'Free' de Football Prediction esté suscrito.")
else:
    st.success(f"✅ ¡CONEXIÓN TOTAL! {len(df_matches)} duelos de Champions analizados.")
    
    # Selector de duelo
    match_sel = st.selectbox("Selecciona el partido para analizar:", 
                            df_matches['Local'] + " vs " + df_matches['Visita'])
    
    # Extraer estadísticas del partido elegido
    data = df_matches[(df_matches['Local'] + " vs " + df_matches['Visita']) == match_sel].iloc[0]

    # --- DASHBOARD DE RESULTADOS ---
    st.divider()
    c1, c2, c3 = st.columns(3)
    
    # Mostramos probabilidades reales 25/26
    c1.metric(f"Gana {data['Local']}", f"{round(data['Prob_L']*100, 1)}%")
    c2.metric("Empate", f"{round(data['Prob_E']*100, 1)}%")
    c3.metric(f"Gana {data['Visita']}", f"{round(data['Prob_V']*100, 1)}%")

    st.subheader("📝 Veredicto de la IA")
    color = "green" if data['Prob_L'] > 0.6 else "orange"
    st.markdown(f"La recomendación para este encuentro es: **:{color}[{data['Recomendacion'].upper()}]**")
    
    st.caption(f"Fecha del encuentro: {data['Fecha']}")
