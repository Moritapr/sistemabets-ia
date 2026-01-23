import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SISTEMABETS IA v3.0 - Football Data", layout="wide")

# ============================================================================
# MÓDULO 1: CONECTOR FOOTBALL-DATA.ORG
# ============================================================================

class FootballDataAPI:
    """Conector a Football-Data.org (GRATIS e ILIMITADO)"""
    
    BASE_URL = "https://api.football-data.org/v4"
    
    # IDs de ligas principales
    LIGAS = {
        "Champions League": "CL",
        "Premier League": "PL",
        "La Liga": "PD",
        "Bundesliga": "BL1",
        "Serie A": "SA",
        "Ligue 1": "FL1",
        "Eredivisie": "DED",
        "Championship": "ELC"
    }
    
    def __init__(self, api_key):
        self.headers = {
            "X-Auth-Token": api_key
        }
    
    def obtener_standings(self, liga_code):
        """Obtiene tabla de posiciones"""
        try:
            url = f"{self.BASE_URL}/competitions/{liga_code}/standings"
            
            response = requests.get(url, headers=self.headers, timeout=15)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            if not data.get('standings'):
                return None
            
            standings = data['standings'][0]['table']
            
            # Convertir a DataFrame
            equipos = []
            for team in standings:
                equipos.append({
                    'Equipo': team['team']['name'],
                    'PJ': team['playedGames'],
                    'Victorias': team['won'],
                    'Empates': team['draw'],
                    'Derrotas': team['lost'],
                    'GF': team['goalsFor'],
                    'GC': team['goalsAgainst'],
                    'Pts': team['points'],
                    'Posicion': team['position']
                })
            
            return pd.DataFrame(equipos)
            
        except Exception as e:
            st.error(f"Error API: {str(e)}")
            return None
    
    def obtener_partidos_recientes(self, equipo_nombre, liga_code):
        """Obtiene últimos 5 partidos de un equipo"""
        try:
            url = f"{self.BASE_URL}/competitions/{liga_code}/matches"
            params = {"status": "FINISHED"}
            
            response = requests.get(url, headers=self.headers, params=params, timeout=15)
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            partidos = []
            
            for match in data.get('matches', [])[:50]:  # Últimos 50 partidos
                home = match['homeTeam']['name']
                away = match['awayTeam']['name']
                
                if home == equipo_nombre or away == equipo_nombre:
                    if match['score']['fullTime']['home'] is not None:
                        partidos.append({
                            'local': home,
                            'visitante': away,
                            'goles_local': match['score']['fullTime']['home'],
                            'goles_visitante': match['score']['fullTime']['away'],
                            'fecha': match['utcDate']
                        })
            
            return sorted(partidos, key=lambda x: x['fecha'], reverse=True)[:5]
            
        except Exception as e:
            return []

# ============================================================================
# MÓDULO 2: ANALIZADOR AVANZADO CON FORMA
# ============================================================================

class AnalizadorAvanzado:
    """Motor de análisis mejorado"""
    
    VENTAJA_LOCAL = 1.18
    
    @staticmethod
    def calcular_forma(partidos_recientes, equipo_nombre):
        """Calcula forma reciente (últimos 5 partidos)"""
        if not partidos_recientes:
            return 0.5, ""
        
        puntos = []
        forma_str = ""
        
        for p in partidos_recientes:
            es_local = p['local'] == equipo_nombre
            gf = p['goles_local'] if es_local else p['goles_visitante']
            gc = p['goles_visitante'] if es_local else p['goles_local']
            
            if gf > gc:
                puntos.append(1.0)
                forma_str += "V"
            elif gf == gc:
                puntos.append(0.5)
                forma_str += "E"
            else:
                puntos.append(0.0)
                forma_str += "D"
        
        # Peso exponencial (último partido vale más)
        pesos = [0.1, 0.15, 0.2, 0.25, 0.3]
        score = sum(p * w for p, w in zip(puntos, pesos[:len(puntos)]))
        
        return score, forma_str
    
    @staticmethod
    def calcular_stats_local_visitante(equipo_nombre, partidos_recientes):
        """Calcula stats separadas de local y visitante"""
        stats = {
            'gf_local': 0, 'gc_local': 0, 'pj_local': 0,
            'gf_visitante': 0, 'gc_visitante': 0, 'pj_visitante': 0
        }
        
        for p in partidos_recientes:
            if p['local'] == equipo_nombre:
                stats['gf_local'] += p['goles_local']
                stats['gc_local'] += p['goles_visitante']
                stats['pj_local'] += 1
            elif p['visitante'] == equipo_nombre:
                stats['gf_visitante'] += p['goles_visitante']
                stats['gc_visitante'] += p['goles_local']
                stats['pj_visitante'] += 1
        
        return stats
    
    @staticmethod
    def calcular_lambdas_avanzado(local, visitante, df_liga, partidos_local, partidos_visitante):
        """Calcula lambdas con stats locales/visitantes y forma"""
        
        # Stats básicas de la tabla
        gf_local_total = local['GF'] / max(local['PJ'], 1)
        gc_visitante_total = visitante['GC'] / max(visitante['PJ'], 1)
        gf_visitante_total = visitante['GF'] / max(visitante['PJ'], 1)
        gc_local_total = local['GC'] / max(local['PJ'], 1)
        
        # Stats específicas de últimos partidos
        stats_local = AnalizadorAvanzado.calcular_stats_local_visitante(
            local['Equipo'], partidos_local
        )
        stats_visitante = AnalizadorAvanzado.calcular_stats_local_visitante(
            visitante['Equipo'], partidos_visitante
        )
        
        # Ajustar con forma reciente si hay datos
        if stats_local['pj_local'] > 0:
            gf_local_casa = stats_local['gf_local'] / stats_local['pj_local']
            gc_local_casa = stats_local['gc_local'] / stats_local['pj_local']
        else:
            gf_local_casa = gf_local_total
            gc_local_casa = gc_local_total
        
        if stats_visitante['pj_visitante'] > 0:
            gf_visitante_fuera = stats_visitante['gf_visitante'] / stats_visitante['pj_visitante']
            gc_visitante_fuera = stats_visitante['gc_visitante'] / stats_visitante['pj_visitante']
        else:
            gf_visitante_fuera = gf_visitante_total
            gc_visitante_fuera = gc_visitante_total
        
        # Media de la liga
        media_gf = df_liga['GF'].mean() / df_liga['PJ'].mean()
        
        # Lambdas
        lambda_local = (gf_local_casa / media_gf) * gc_visitante_fuera * AnalizadorAvanzado.VENTAJA_LOCAL
        lambda_visitante = (gf_visitante_fuera / media_gf) * gc_local_casa / AnalizadorAvanzado.VENTAJA_LOCAL
        
        # Ajuste por forma
        forma_local, _ = AnalizadorAvanzado.calcular_forma(partidos_local, local['Equipo'])
        forma_visitante, _ = AnalizadorAvanzado.calcular_forma(partidos_visitante, visitante['Equipo'])
        
        if forma_local > 0.7:
            lambda_local *= 1.15
        elif forma_local < 0.3:
            lambda_local *= 0.85
        
        if forma_visitante > 0.7:
            lambda_visitante *= 1.15
        elif forma_visitante < 0.3:
            lambda_visitante *= 0.85
        
        return lambda_local, lambda_visitante
    
    @staticmethod
    def matriz_probabilidades(lambda_l, lambda_v, max_goles=7):
        """Genera matriz de probabilidades"""
        matriz = np.zeros((max_goles, max_goles))
        
        for i in range(max_goles):
            for j in range(max_goles):
                matriz[i, j] = poisson.pmf(i, lambda_l) * poisson.pmf(j, lambda_v)
        
        return matriz
    
    @staticmethod
    def calcular_mercados(matriz):
        """Calcula todos los mercados principales"""
        
        # 1X2
        p_local = np.sum(np.tril(matriz, -1))
        p_empate = np.sum(np.diag(matriz))
        p_visitante = np.sum(np.triu(matriz, 1))
        
        # Over/Under
        over_under = {}
        for threshold in [0.5, 1.5, 2.5, 3.5, 4.5]:
            total = int(threshold) + 1
            p_under = sum([matriz[i, j] for i in range(matriz.shape[0]) 
                          for j in range(matriz.shape[1]) if i+j < total])
            over_under[f"Over {threshold}"] = 1 - p_under
            over_under[f"Under {threshold}"] = p_under
        
        # BTTS
        p_btts_no = matriz[0,:].sum() + matriz[:,0].sum() - matriz[0,0]
        p_btts_si = 1 - p_btts_no
        
        # Resultado exacto más probable
        idx_max = np.unravel_index(matriz.argmax(), matriz.shape)
        
        return {
            '1X2': {'Local': p_local, 'Empate': p_empate, 'Visitante': p_visitante},
            'Over/Under': over_under,
            'BTTS': {'Si': p_btts_si, 'No': p_btts_no},
            'Resultado_Exacto': f"{idx_max[0]}-{idx_max[1]}",
            'Prob_Exacto': matriz[idx_max[0], idx_max[1]]
        }

# ============================================================================
# MÓDULO 3: BUSCADOR DE VALOR
# ============================================================================

class BuscadorValor:
    """Identifica apuestas con valor esperado positivo"""
    
    @staticmethod
    def calcular_ev(probabilidad, cuota):
        """Valor Esperado = (Prob × Cuota) - 1"""
        return (probabilidad * cuota - 1) * 100
    
    @staticmethod
    def encontrar_oportunidades(mercados, margen_minimo=5):
        """Busca las mejores oportunidades"""
        
        oportunidades = []
        
        # 1X2
        for tipo, prob in mercados['1X2'].items():
            cuota_justa = 1 / prob if prob > 0 else 999
            cuota_mercado = cuota_justa * 0.92  # Margen típico 8%
            ev = BuscadorValor.calcular_ev(prob, cuota_mercado)
            
            if ev >= margen_minimo:
                oportunidades.append({
                    'Mercado': f"1X2 - {tipo}",
                    'Probabilidad': f"{prob*100:.1f}%",
                    'Cuota_Justa': f"{cuota_justa:.2f}",
                    'EV': f"{ev:.1f}%",
                    'Confianza': '🔥' if prob > 0.6 else '⚡' if prob > 0.45 else '💡'
                })
        
        # Over/Under 2.5
        for tipo, prob in mercados['Over/Under'].items():
            if '2.5' in tipo or '1.5' in tipo:
                cuota_justa = 1 / prob if prob > 0 else 999
                cuota_mercado = cuota_justa * 0.92
                ev = BuscadorValor.calcular_ev(prob, cuota_mercado)
                
                if ev >= margen_minimo:
                    oportunidades.append({
                        'Mercado': tipo,
                        'Probabilidad': f"{prob*100:.1f}%",
                        'Cuota_Justa': f"{cuota_justa:.2f}",
                        'EV': f"{ev:.1f}%",
                        'Confianza': '🔥' if prob > 0.6 else '⚡'
                    })
        
        # BTTS
        for tipo, prob in mercados['BTTS'].items():
            cuota_justa = 1 / prob if prob > 0 else 999
            cuota_mercado = cuota_justa * 0.92
            ev = BuscadorValor.calcular_ev(prob, cuota_mercado)
            
            if ev >= margen_minimo:
                oportunidades.append({
                    'Mercado': f"BTTS - {tipo}",
                    'Probabilidad': f"{prob*100:.1f}%",
                    'Cuota_Justa': f"{cuota_justa:.2f}",
                    'EV': f"{ev:.1f}%",
                    'Confianza': '🔥' if prob > 0.55 else '⚡'
                })
        
        return sorted(oportunidades, key=lambda x: float(x['EV'].replace('%','')), reverse=True)

# ============================================================================
# INTERFAZ
# ============================================================================

def main():
    st.title("⚽ SISTEMABETS IA v3.0 - FOOTBALL DATA")
    st.caption("Powered by Football-Data.org | Requests ilimitados ✅")
    
    # Sidebar - Configuración API
    st.sidebar.header("🔑 Configuración")
    
    api_key = st.sidebar.text_input(
        "API Key", 
        value="b5da8589cdef4d418bbe2afcbccadf10",
        type="password",
        help="Tu API key de Football-Data.org"
    )
    
    if not api_key:
        st.warning("⚠️ Ingresa tu API Key")
        return
    
    # Inicializar API
    api = FootballDataAPI(api_key)
    
    # Selección de liga
    liga_nombre = st.sidebar.selectbox("Liga:", list(FootballDataAPI.LIGAS.keys()))
    liga_code = FootballDataAPI.LIGAS[liga_nombre]
    
    # Cargar datos
    with st.spinner(f"Cargando {liga_nombre}..."):
        df = api.obtener_standings(liga_code)
    
    if df is None or df.empty:
        st.error("❌ Error cargando datos. Verifica tu API Key.")
        st.info("Si acabas de crear tu cuenta, espera 1-2 minutos para que se active.")
        return
    
    st.success(f"✅ {len(df)} equipos cargados")
    
    # Selección de equipos
    st.subheader("🎯 Seleccionar Partido")
    col1, col2 = st.columns(2)
    
    equipos = sorted(df['Equipo'].unique())
    
    with col1:
        equipo_local = st.selectbox("🏟️ Local:", equipos, index=0)
    
    with col2:
        equipo_visitante = st.selectbox("✈️ Visitante:", equipos, index=min(1, len(equipos)-1))
    
    if equipo_local == equipo_visitante:
        st.warning("⚠️ Selecciona dos equipos diferentes")
        return
    
    # Obtener stats
    el = df[df['Equipo'] == equipo_local].iloc[0]
    ev = df[df['Equipo'] == equipo_visitante].iloc[0]
    
    # ANÁLISIS
    if st.button("🔍 ANALIZAR PARTIDO", type="primary", use_container_width=True):
        
        with st.spinner("Obteniendo forma reciente..."):
            partidos_local = api.obtener_partidos_recientes(equipo_local, liga_code)
            partidos_visitante = api.obtener_partidos_recientes(equipo_visitante, liga_code)
        
        with st.spinner("Calculando probabilidades..."):
            # Lambdas con forma reciente
            lambda_l, lambda_v = AnalizadorAvanzado.calcular_lambdas_avanzado(
                el, ev, df, partidos_local, partidos_visitante
            )
            
            # Forma
            forma_local, forma_str_local = AnalizadorAvanzado.calcular_forma(partidos_local, equipo_local)
            forma_visitante, forma_str_visitante = AnalizadorAvanzado.calcular_forma(partidos_visitante, equipo_visitante)
            
            # Matriz y mercados
            matriz = AnalizadorAvanzado.matriz_probabilidades(lambda_l, lambda_v)
            mercados = AnalizadorAvanzado.calcular_mercados(matriz)
            
            # Búsqueda de valor
            oportunidades = BuscadorValor.encontrar_oportunidades(mercados)
        
        # ============ RESULTADOS ============
        st.divider()
        st.subheader("📊 ANÁLISIS COMPLETO")
        
        # Métricas principales
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🏠 Gana Local", f"{mercados['1X2']['Local']*100:.1f}%")
        m2.metric("🤝 Empate", f"{mercados['1X2']['Empate']*100:.1f}%")
        m3.metric("✈️ Gana Visitante", f"{mercados['1X2']['Visitante']*100:.1f}%")
        m4.metric("⚽ Goles Esperados", f"{lambda_l + lambda_v:.2f}")
        
        # Over/Under
        st.subheader("📈 Over/Under")
        ou1, ou2, ou3 = st.columns(3)
        ou1.metric("Over 1.5", f"{mercados['Over/Under']['Over 1.5']*100:.1f}%")
        ou2.metric("Over 2.5", f"{mercados['Over/Under']['Over 2.5']*100:.1f}%")
        ou3.metric("Over 3.5", f"{mercados['Over/Under']['Over 3.5']*100:.1f}%")
        
        # BTTS
        st.subheader("🎯 Ambos Marcan")
        bt1, bt2 = st.columns(2)
        bt1.metric("BTTS Sí", f"{mercados['BTTS']['Si']*100:.1f}%")
        bt2.metric("BTTS No", f"{mercados['BTTS']['No']*100:.1f}%")
        
        # Resultado más probable
        st.info(f"🎲 **Resultado más probable:** {mercados['Resultado_Exacto']} ({mercados['Prob_Exacto']*100:.1f}%)")
        
        # Oportunidades
        st.divider()
        st.subheader("💎 MEJORES OPORTUNIDADES")
        
        if oportunidades:
            df_oport = pd.DataFrame(oportunidades)
            st.dataframe(df_oport, use_container_width=True, hide_index=True)
        else:
            st.warning("No hay oportunidades con EV > 5% en este partido")
        
        # Stats técnicas
        with st.expander("🔬 Ver análisis técnico"):
            tc1, tc2 = st.columns(2)
            
            with tc1:
                st.write(f"**{equipo_local} (Local)**")
                st.write(f"- Forma últimos 5: {forma_str_local if forma_str_local else 'N/D'}")
                st.write(f"- Score forma: {forma_local:.2f}")
                st.write(f"- Goles totales: {int(el['GF'])} en {int(el['PJ'])} PJ")
                st.write(f"- Promedio: {el['GF']/max(el['PJ'],1):.2f} goles/partido")
                st.write(f"- Lambda calculado: {lambda_l:.2f}")
            
            with tc2:
                st.write(f"**{equipo_visitante} (Visitante)**")
                st.write(f"- Forma últimos 5: {forma_str_visitante if forma_str_visitante else 'N/D'}")
                st.write(f"- Score forma: {forma_visitante:.2f}")
                st.write(f"- Goles totales: {int(ev['GF'])} en {int(ev['PJ'])} PJ")
                st.write(f"- Promedio: {ev['GF']/max(ev['PJ'],1):.2f} goles/partido")
                st.write(f"- Lambda calculado: {lambda_v:.2f}")
        
        # Últimos partidos
        with st.expander("📋 Ver últimos partidos"):
            c1, c2 = st.columns(2)
            
            with c1:
                st.write(f"**{equipo_local}**")
                if partidos_local:
                    for p in partidos_local[:5]:
                        resultado = f"{p['goles_local']}-{p['goles_visitante']}"
                        st.write(f"• {p['local']} {resultado} {p['visitante']}")
                else:
                    st.write("No hay datos recientes")
            
            with c2:
                st.write(f"**{equipo_visitante}**")
                if partidos_visitante:
                    for p in partidos_visitante[:5]:
                        resultado = f"{p['goles_local']}-{p['goles_visitante']}"
                        st.write(f"• {p['local']} {resultado} {p['visitante']}")
                else:
                    st.write("No hay datos recientes")
        
        # Tabla completa
        with st.expander("🏆 Ver clasificación"):
            st.dataframe(
                df[['Posicion', 'Equipo', 'PJ', 'Victorias', 'Empates', 'Derrotas', 'GF', 'GC', 'Pts']]
                .sort_values('Posicion'),
                use_container_width=True,
                hide_index=True
            )

if __name__ == "__main__":
    main()
