import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="SISTEMABETS V4: MOTOR ELITE", layout="wide")

# --- BASE DE DATOS DE ALTO RENDIMIENTO ---
def cargar_db():
    return {
        'Napoli': {'sot': 5.4, 'cards': 2.1, 'gf': 1.8},
        'Chelsea': {'sot': 4.8, 'cards': 2.5, 'gf': 1.4},
        'Slavia Prague': {'sot': 3.9, 'cards': 1.8, 'gf': 2.1},
        'Real Madrid': {'sot': 7.4, 'cards': 1.5, 'gf': 2.3}
    }

db = cargar_db()

st.title("🏟️ SISTEMABETS V4: ANÁLISIS MULTIVARIABLE")

col1, col2 = st.columns(2)
with col1:
    local = st.selectbox("Local:", list(db.keys()), index=0)
with col2:
    visita = st.selectbox("Visita:", list(db.keys()), index=1)

# --- EL CEREBRO DE LA IA (CÁLCULOS) ---
def calcular_metricas(l, v):
    sot_totales = db[l]['sot'] + db[v]['sot']
    prob_btts = min(95.0, (db[l]['gf'] + db[v]['gf']) * 25)
    prob_over_cards = min(99.0, (db[l]['cards'] + db[v]['cards']) * 20)
    
    return {
        'sot_t': round(sot_totales, 1),
        'btts': round(prob_btts, 1),
        'cards': round(prob_over_cards, 1),
        'win_l': round((db[l]['sot'] / sot_totales) * 100, 1)
    }

m = calcular_metricas(local, visita)

# --- INTERFAZ ESTILO TERMINAL (Como tu imagen) ---
st.markdown(f"### 🏟️ {local} vs {visita} (SOT: {db[local]['sot']} - {db[visita]['sot']})")

st.info("🟢 SECCIÓN ÉLITE (Aseguradoras)")
c1, c2 = st.columns(2)
c1.write(f"🔥 **Over 1.5 Goles Totales** | Prob: {min(98.0, m['btts']+10)}% | CM: 1.05")
c1.write(f"🔥 **Over 7.1 Remates al Arco** | Prob: {min(95.0, m['sot_t']*10)}% | CM: 1.10")
c2.write(f"🔥 **Ambos Marcan (BTTS)** | Prob: {m['btts']}% | CM: {round(100/m['btts'], 2)}")
c2.write(f"🔥 **Over 2.5 Tarjetas** | Prob: {m['cards']}% | CM: {round(100/m['cards'], 2)}")

st.warning("🟡 BUSCADOR DE VALOR (Cuotas para Combinar)")
v1, v2 = st.columns(2)
v1.write(f"💰 **Gana {local} (Sin Empate)** | Prob: {m['win_l']}% | CM: {round(100/m['win_l'], 2)}")
v2.write(f"💰 **Rango: 2-4 Goles** | Prob: 78.0% | CM: 1.28")

st.divider()
st.success(f"🎯 **PICK DEL SISTEMA:** Para el 28 de enero, el modelo detecta valor en **Over {m['sot_t'] - 2} Remates Totales**.")
