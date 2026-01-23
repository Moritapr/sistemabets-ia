import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SISTEMABETS IA v4.0 PRO", layout="wide")

# ============================================================================
# MÓDULO 1: CONECTOR AVANZADO FOOTBALL-DATA.ORG
# ============================================================================

class FootballDataAPI:
    """Conector profesional con análisis de 20 partidos multi-competición"""
    
    BASE_URL = "https://api.football-data.org/v4"
    
    LIGAS = {
        "Champions League": "CL",
        "Premier League": "PL",
        "La Liga": "PD",
        "Bundesliga": "BL1",
        "Serie A": "SA",
        "Ligue 1": "FL1",
        "Eredivisie": "DED",
        "Championship": "ELC",
        "Liga Portugal": "PPL"
    }
    
    def __init__(self, api_key):
        self.headers = {"X-Auth-Token": api_key}
        self.cache_teams = {}
    
    def obtener_standings(self, liga_code):
        """Obtiene tabla de posiciones y cachea IDs de equipos"""
        try:
            url = f"{self.BASE_URL}/competitions/{liga_code}/standings"
            response = requests.get(url, headers=self.headers, timeout=15)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            if not data.get('standings'):
                return None
            
            standings = data['standings'][0]['table']
            equipos = []
            
            for team in standings:
                team_id = team['team']['id']
                team_name = team['team']['name']
                self.cache_teams[team_name] = team_id
                
                equipos.append({
                    'Equipo': team_name,
                    'ID': team_id,
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
    
    def obtener_ultimos_20_partidos(self, equipo_nombre):
        """
        Obtiene últimos 20 partidos del equipo en TODAS las competiciones
        (Liga, Copa, Europa, etc.) - GARANTIZA SEPARACIÓN CORRECTA DE EQUIPOS
        """
        try:
            team_id = self.cache_teams.get(equipo_nombre)
            
            if not team_id:
                st.warning(f"No se encontró ID para {equipo_nombre}")
                return []
            
            # Endpoint específico del equipo (todas sus competiciones)
            url = f"{self.BASE_URL}/teams/{team_id}/matches"
            params = {
                "status": "FINISHED",
                "limit": 100  # Obtener hasta 100 partidos para asegurar 20 válidos
            }
            
            response = requests.get(url, headers=self.headers, params=params, timeout=15)
            
            if response.status_code != 200:
                st.warning(f"Error {response.status_code} obteniendo partidos de {equipo_nombre}")
                return []
            
            data = response.json()
            partidos = []
            
            for match in data.get('matches', []):
                # Verificar que el partido tenga resultado completo
                if match['score']['fullTime']['home'] is None:
                    continue
                
                home_team = match['homeTeam']['name']
                away_team = match['awayTeam']['name']
                
                # CRÍTICO: Asegurar que el equipo pertenezca a este partido
                if home_team != equipo_nombre and away_team != equipo_nombre:
                    continue
                
                partidos.append({
                    'local': home_team,
                    'visitante': away_team,
                    'goles_local': match['score']['fullTime']['home'],
                    'goles_visitante': match['score']['fullTime']['away'],
                    'fecha': match['utcDate'],
                    'competicion': match['competition']['name']
                })
            
            # Ordenar por fecha DESC y tomar últimos 20
            partidos_ordenados = sorted(partidos, key=lambda x: x['fecha'], reverse=True)
            
            return partidos_ordenados[:20]
            
        except Exception as e:
            st.warning(f"Error obteniendo historial de {equipo_nombre}: {str(e)}")
            return []

# ============================================================================
# MÓDULO 2: MOTOR DE ANÁLISIS PROFUNDO CON IA AVANZADA
# ============================================================================

class AnalizadorProfundo:
    """
    Motor de IA mejorado que analiza 20 partidos con múltiples factores
    GARANTIZA: Nunca probabilidades en 0%, análisis balanceado y realista
    """
    
    # Configuración avanzada
    VENTAJA_LOCAL_BASE = 1.20
    MIN_PROBABILIDAD = 0.05  # Mínimo 5% para cualquier resultado
    
    @staticmethod
    def calcular_forma_20_partidos(partidos, equipo_nombre):
        """
        Calcula forma usando los últimos 20 partidos con pesos decrecientes
        Últimos partidos pesan MÁS que partidos antiguos
        """
        if not partidos:
            return 0.50, "", {}
        
        puntos = []
        forma_visual = ""
        detalles = {
            'victorias': 0,
            'empates': 0,
            'derrotas': 0,
            'gf': 0,
            'gc': 0,
            'racha_actual': ""
        }
        
        for i, partido in enumerate(partidos[:20]):
            es_local = partido['local'] == equipo_nombre
            gf = partido['goles_local'] if es_local else partido['goles_visitante']
            gc = partido['goles_visitante'] if es_local else partido['goles_local']
            
            detalles['gf'] += gf
            detalles['gc'] += gc
            
            # Determinar resultado
            if gf > gc:
                puntos.append(1.0)
                forma_visual += "V"
                detalles['victorias'] += 1
                if i < 5:
                    detalles['racha_actual'] += "✅"
            elif gf == gc:
                puntos.append(0.5)
                forma_visual += "E"
                detalles['empates'] += 1
                if i < 5:
                    detalles['racha_actual'] += "🟰"
            else:
                puntos.append(0.0)
                forma_visual += "D"
                detalles['derrotas'] += 1
                if i < 5:
                    detalles['racha_actual'] += "❌"
        
        # Pesos exponenciales (últimos partidos valen MÁS)
        # Partido 1 (más reciente): 10%, Partido 20 (más viejo): 0.5%
        pesos = np.linspace(0.10, 0.005, len(puntos))
        pesos = pesos / pesos.sum()  # Normalizar para que sumen 1
        
        score_forma = sum(p * w for p, w in zip(puntos, pesos))
        
        return score_forma, forma_visual, detalles
    
    @staticmethod
    def calcular_stats_local_visitante_avanzado(equipo_nombre, partidos):
        """
        Calcula estadísticas separadas jugando LOCAL vs VISITANTE
        Esto es CRÍTICO porque los equipos rinden diferente en casa/fuera
        """
        stats = {
            # Como LOCAL
            'local_pj': 0,
            'local_gf': 0,
            'local_gc': 0,
            'local_victorias': 0,
            'local_empates': 0,
            'local_derrotas': 0,
            
            # Como VISITANTE
            'visitante_pj': 0,
            'visitante_gf': 0,
            'visitante_gc': 0,
            'visitante_victorias': 0,
            'visitante_empates': 0,
            'visitante_derrotas': 0
        }
        
        for p in partidos[:20]:
            if p['local'] == equipo_nombre:
                # Jugó de LOCAL
                stats['local_pj'] += 1
                stats['local_gf'] += p['goles_local']
                stats['local_gc'] += p['goles_visitante']
                
                if p['goles_local'] > p['goles_visitante']:
                    stats['local_victorias'] += 1
                elif p['goles_local'] == p['goles_visitante']:
                    stats['local_empates'] += 1
                else:
                    stats['local_derrotas'] += 1
                    
            elif p['visitante'] == equipo_nombre:
                # Jugó de VISITANTE
                stats['visitante_pj'] += 1
                stats['visitante_gf'] += p['goles_visitante']
                stats['visitante_gc'] += p['goles_local']
                
                if p['goles_visitante'] > p['goles_local']:
                    stats['visitante_victorias'] += 1
                elif p['goles_visitante'] == p['goles_local']:
                    stats['visitante_empates'] += 1
                else:
                    stats['visitante_derrotas'] += 1
        
        return stats
    
    @staticmethod
    def calcular_lambdas_profesional(local_team, visitante_team, df_liga, partidos_local, partidos_visitante):
        """
        Calcula lambdas de Poisson con análisis PROFUNDO de múltiples factores:
        1. Estadísticas de la tabla (temporada completa)
        2. Forma reciente (últimos 20 partidos)
        3. Rendimiento específico Local/Visitante
        4. Ajuste por nivel de competición
        5. Factor psicológico (rachas)
        
        GARANTIZA: Nunca lambda < 0.3 (mínimo realista)
        """
        
        # ===== 1. STATS BÁSICAS DE LA TABLA =====
        gf_local_tabla = local_team['GF'] / max(local_team['PJ'], 1)
        gc_local_tabla = local_team['GC'] / max(local_team['PJ'], 1)
        gf_visitante_tabla = visitante_team['GF'] / max(visitante_team['PJ'], 1)
        gc_visitante_tabla = visitante_team['GC'] / max(visitante_team['PJ'], 1)
        
        # ===== 2. ANÁLISIS PROFUNDO DE FORMA =====
        forma_local, _, detalles_local = AnalizadorProfundo.calcular_forma_20_partidos(
            partidos_local, local_team['Equipo']
        )
        forma_visitante, _, detalles_visitante = AnalizadorProfundo.calcular_forma_20_partidos(
            partidos_visitante, visitante_team['Equipo']
        )
        
        # ===== 3. STATS ESPECÍFICAS LOCAL/VISITANTE =====
        stats_local = AnalizadorProfundo.calcular_stats_local_visitante_avanzado(
            local_team['Equipo'], partidos_local
        )
        stats_visitante = AnalizadorProfundo.calcular_stats_local_visitante_avanzado(
            visitante_team['Equipo'], partidos_visitante
        )
        
        # Promedios específicos
        gf_local_casa = stats_local['local_gf'] / max(stats_local['local_pj'], 1) if stats_local['local_pj'] > 0 else gf_local_tabla
        gc_local_casa = stats_local['local_gc'] / max(stats_local['local_pj'], 1) if stats_local['local_pj'] > 0 else gc_local_tabla
        
        gf_visitante_fuera = stats_visitante['visitante_gf'] / max(stats_visitante['visitante_pj'], 1) if stats_visitante['visitante_pj'] > 0 else gf_visitante_tabla
        gc_visitante_fuera = stats_visitante['visitante_gc'] / max(stats_visitante['visitante_pj'], 1) if stats_visitante['visitante_pj'] > 0 else gc_visitante_tabla
        
        # ===== 4. MEDIA DE LA LIGA =====
        media_goles_liga = df_liga['GF'].sum() / df_liga['PJ'].sum()
        
        # ===== 5. CÁLCULO DE LAMBDAS BASE =====
        # Fuerza ofensiva vs defensiva del rival
        lambda_local = (gf_local_casa / media_goles_liga) * (gc_visitante_fuera / media_goles_liga) * media_goles_liga
        lambda_visitante = (gf_visitante_fuera / media_goles_liga) * (gc_local_casa / media_goles_liga) * media_goles_liga
        
        # ===== 6. AJUSTE POR VENTAJA LOCAL =====
        ventaja_local_dinamica = AnalizadorProfundo.VENTAJA_LOCAL_BASE
        
        # Si el local tiene buena racha en casa, aumentar ventaja
        if stats_local['local_pj'] >= 5:
            winrate_casa = stats_local['local_victorias'] / stats_local['local_pj']
            if winrate_casa > 0.7:
                ventaja_local_dinamica *= 1.08
            elif winrate_casa < 0.3:
                ventaja_local_dinamica *= 0.95
        
        lambda_local *= ventaja_local_dinamica
        lambda_visitante /= (ventaja_local_dinamica * 0.85)
        
        # ===== 7. AJUSTE POR FORMA RECIENTE =====
        # Forma excelente (>75%): +20% ataque
        # Forma mala (<30%): -15% ataque
        if forma_local > 0.75:
            lambda_local *= 1.20
        elif forma_local > 0.60:
            lambda_local *= 1.10
        elif forma_local < 0.30:
            lambda_local *= 0.85
        elif forma_local < 0.40:
            lambda_local *= 0.92
        
        if forma_visitante > 0.75:
            lambda_visitante *= 1.20
        elif forma_visitante > 0.60:
            lambda_visitante *= 1.10
        elif forma_visitante < 0.30:
            lambda_visitante *= 0.85
        elif forma_visitante < 0.40:
            lambda_visitante *= 0.92
        
        # ===== 8. AJUSTE POR DIFERENCIA DE NIVEL =====
        diff_posiciones = abs(local_team['Posicion'] - visitante_team['Posicion'])
        diff_puntos = abs(local_team['Pts'] - visitante_team['Pts'])
        
        if diff_posiciones >= 10 or diff_puntos >= 20:
            # Gran diferencia de nivel
            if local_team['Posicion'] < visitante_team['Posicion']:
                lambda_local *= 1.08
                lambda_visitante *= 0.93
            else:
                lambda_local *= 0.93
                lambda_visitante *= 1.08
        
        # ===== 9. GARANTIZAR MÍNIMOS REALISTAS =====
        # NUNCA un equipo puede tener lambda < 0.3 (implicaría 0% de ganar)
        lambda_local = max(lambda_local, 0.35)
        lambda_visitante = max(lambda_visitante, 0.35)
        
        # Limitar máximos también (partido no es tenis)
        lambda_local = min(lambda_local, 4.0)
        lambda_visitante = min(lambda_visitante, 4.0)
        
        return lambda_local, lambda_visitante, {
            'forma_local': forma_local,
            'forma_visitante': forma_visitante,
            'detalles_local': detalles_local,
            'detalles_visitante': detalles_visitante,
            'stats_local': stats_local,
            'stats_visitante': stats_visitante
        }
    
    @staticmethod
    def matriz_probabilidades(lambda_l, lambda_v, max_goles=8):
        """Genera matriz de probabilidades con Poisson"""
        matriz = np.zeros((max_goles, max_goles))
        
        for i in range(max_goles):
            for j in range(max_goles):
                matriz[i, j] = poisson.pmf(i, lambda_l) * poisson.pmf(j, lambda_v)
        
        return matriz
    
    @staticmethod
    def calcular_mercados_completos(matriz):
        """
        Calcula TODOS los mercados de apuestas
        GARANTIZA: Probabilidades balanceadas, nunca 0%
        """
        
        # ===== 1X2 =====
        p_local = np.sum(np.tril(matriz, -1))  # Local gana
        p_empate = np.sum(np.diag(matriz))      # Empate
        p_visitante = np.sum(np.triu(matriz, 1)) # Visitante gana
        
        # Normalizar para asegurar que sumen 100%
        total_1x2 = p_local + p_empate + p_visitante
        p_local /= total_1x2
        p_empate /= total_1x2
        p_visitante /= total_1x2
        
        # Garantizar mínimos (nunca 0%)
        p_local = max(p_local, AnalizadorProfundo.MIN_PROBABILIDAD)
        p_empate = max(p_empate, AnalizadorProfundo.MIN_PROBABILIDAD)
        p_visitante = max(p_visitante, AnalizadorProfundo.MIN_PROBABILIDAD)
        
        # ===== OVER/UNDER =====
        over_under = {}
        for threshold in [0.5, 1.5, 2.5, 3.5, 4.5]:
            limite = threshold + 0.01
            p_over = sum([matriz[i, j] for i in range(matriz.shape[0]) 
                         for j in range(matriz.shape[1]) if (i+j) > limite])
            p_under = 1 - p_over
            
            over_under[f"Over {threshold}"] = max(p_over, 0.01)
            over_under[f"Under {threshold}"] = max(p_under, 0.01)
        
        # ===== BTTS (Ambos Marcan) =====
        p_btts_no = matriz[0,:].sum() + matriz[:,0].sum() - matriz[0,0]
        p_btts_si = 1 - p_btts_no
        
        # ===== DOBLE OPORTUNIDAD =====
        p_1x = p_local + p_empate  # Local o Empate
        p_12 = p_local + p_visitante  # Local o Visitante
        p_x2 = p_empate + p_visitante  # Empate o Visitante
        
        # ===== RESULTADO EXACTO MÁS PROBABLE =====
        idx_max = np.unravel_index(matriz.argmax(), matriz.shape)
        resultado_probable = f"{idx_max[0]}-{idx_max[1]}"
        prob_resultado = matriz[idx_max[0], idx_max[1]]
        
        # Top 3 resultados más probables
        top_resultados = []
        matriz_flat = matriz.flatten()
        indices_ordenados = np.argsort(matriz_flat)[::-1][:5]
        
        for idx in indices_ordenados:
            i = idx // matriz.shape[1]
            j = idx % matriz.shape[1]
            top_resultados.append({
                'Marcador': f"{i}-{j}",
                'Probabilidad': matriz[i, j]
            })
        
        # ===== GOLES POR EQUIPO =====
        goles_local = {}
        goles_visitante = {}
        
        for goles in range(6):
            goles_local[f"Local {goles}+ goles"] = max(sum(matriz[goles:, :].flatten()), 0.01)
            goles_visitante[f"Visitante {goles}+ goles"] = max(sum(matriz[:, goles:].flatten()), 0.01)
        
        return {
            '1X2': {
                'Local': p_local,
                'Empate': p_empate,
                'Visitante': p_visitante
            },
            'Over/Under': over_under,
            'BTTS': {
                'Si': max(p_btts_si, 0.01),
                'No': max(p_btts_no, 0.01)
            },
            'Doble_Oportunidad': {
                '1X (Local o Empate)': p_1x,
                '12 (Local o Visitante)': p_12,
                'X2 (Empate o Visitante)': p_x2
            },
            'Resultado_Exacto': resultado_probable,
            'Prob_Exacto': prob_resultado,
            'Top_Resultados': top_resultados,
            'Goles_Local': goles_local,
            'Goles_Visitante': goles_visitante
        }

# ============================================================================
# MÓDULO 3: BUSCADOR INTELIGENTE DE VALOR
# ============================================================================

class BuscadorValorPro:
    """Encuentra las mejores oportunidades de apuesta con EV positivo"""
    
    # Márgenes típicos de casas de apuestas por mercado
    MARGENES = {
        '1X2': 0.08,  # 8% margen
        'Over/Under': 0.07,
        'BTTS': 0.06,
        'Doble_Oportunidad': 0.05,
        'Resultado_Exacto': 0.15
    }
    
    @staticmethod
    def calcular_ev(probabilidad, cuota):
        """Valor Esperado: (Prob × Cuota) - 1"""
        return (probabilidad * cuota - 1) * 100
    
    @staticmethod
    def calcular_kelly(probabilidad, cuota, bankroll=100):
        """Criterio de Kelly para gestión de bankroll"""
        if cuota <= 1:
            return 0
        
        q = 1 - probabilidad
        kelly = ((cuota * probabilidad) - q) / (cuota - 1)
        
        # Kelly fraccionario (25% para ser conservador)
        kelly_frac = kelly * 0.25
        
        return max(min(kelly_frac, 0.10), 0) * bankroll  # Máximo 10% del bankroll
    
    @staticmethod
    def encontrar_mejores_apuestas(mercados, umbral_ev=3):
        """
        Analiza todos los mercados y encuentra las mejores oportunidades
        Retorna apuestas ordenadas por EV
        """
        oportunidades = []
        
        # ===== 1X2 =====
        for resultado, prob in mercados['1X2'].items():
            if prob < 0.01:
                continue
                
            cuota_justa = 1 / prob
            cuota_mercado = cuota_justa * (1 - BuscadorValorPro.MARGENES['1X2'])
            ev = BuscadorValorPro.calcular_ev(prob, cuota_mercado)
            
            if ev >= umbral_ev:
                kelly_stake = BuscadorValorPro.calcular_kelly(prob, cuota_mercado)
                
                confianza = '🔥🔥🔥' if prob > 0.65 else '🔥🔥' if prob > 0.50 else '🔥'
                
                oportunidades.append({
                    'Mercado': f"1X2 - {resultado}",
                    'Probabilidad': f"{prob*100:.1f}%",
                    'Cuota_Justa': f"{cuota_justa:.2f}",
                    'Cuota_Estimada': f"{cuota_mercado:.2f}",
                    'EV': f"+{ev:.1f}%",
                    'Kelly_Stake': f"${kelly_stake:.2f}",
                    'Confianza': confianza,
                    'EV_NUM': ev
                })
        
        # ===== OVER/UNDER =====
        for mercado, prob in mercados['Over/Under'].items():
            if '1.5' in mercado or '2.5' in mercado or '3.5' in mercado:
                if prob < 0.01:
                    continue
                    
                cuota_justa = 1 / prob
                cuota_mercado = cuota_justa * (1 - BuscadorValorPro.MARGENES['Over/Under'])
                ev = BuscadorValorPro.calcular_ev(prob, cuota_mercado)
                
                if ev >= umbral_ev:
                    kelly_stake = BuscadorValorPro.calcular_kelly(prob, cuota_mercado)
                    confianza = '🔥🔥🔥' if prob > 0.65 else '🔥🔥' if prob > 0.50 else '🔥'
                    
                    oportunidades.append({
                        'Mercado': mercado,
                        'Probabilidad': f"{prob*100:.1f}%",
                        'Cuota_Justa': f"{cuota_justa:.2f}",
                        'Cuota_Estimada': f"{cuota_mercado:.2f}",
                        'EV': f"+{ev:.1f}%",
                        'Kelly_Stake': f"${kelly_stake:.2f}",
                        'Confianza': confianza,
                        'EV_NUM': ev
                    })
        
        # ===== BTTS =====
        for tipo, prob in mercados['BTTS'].items():
            if prob < 0.01:
                continue
                
            cuota_justa = 1 / prob
            cuota_mercado = cuota_justa * (1 - BuscadorValorPro.MARGENES['BTTS'])
            ev = BuscadorValorPro.calcular_ev(prob, cuota_mercado)
            
            if ev >= umbral_ev:
                kelly_stake = BuscadorValorPro.calcular_kelly(prob, cuota_mercado)
                confianza = '🔥🔥🔥' if prob > 0.60 else '🔥🔥' if prob > 0.50 else '🔥'
                
                oportunidades.append({
                    'Mercado': f"BTTS - {tipo}",
                    'Probabilidad': f"{prob*100:.1f}%",
                    'Cuota_Justa': f"{cuota_justa:.2f}",
                    'Cuota_Estimada': f"{cuota_mercado:.2f}",
                    'EV': f"+{ev:.1f}%",
                    'Kelly_Stake': f"${kelly_stake:.2f}",
                    'Confianza': confianza,
                    'EV_NUM': ev
                })
        
        # ===== DOBLE OPORTUNIDAD =====
        for mercado, prob in mercados['Doble_Oportunidad'].items():
            if prob < 0.01:
                continue
                
            cuota_justa = 1 / prob
            cuota_mercado = cuota_justa * (1 - BuscadorValorPro.MARGENES['Doble_Oportunidad'])
            ev = BuscadorValorPro.calcular_ev(prob, cuota_mercado)
            
            if ev >= umbral_ev:
                kelly_stake = BuscadorValorPro.calcular_kelly(prob, cuota_mercado)
                confianza = '🔥🔥🔥' if prob > 0.75 else '🔥🔥' if prob > 0.65 else '🔥'
                
                oportunidades.append({
                    'Mercado': mercado,
                    'Probabilidad': f"{prob*100:.1f}%",
                    'Cuota_Justa': f"{cuota_justa:.2f}",
                    'Cuota_Estimada': f"{cuota_mercado:.2f}",
                    'EV': f"+{ev:.1f}%",
                    'Kelly_Stake': f"${kelly_stake:.2f}",
                    'Confianza': confianza,
                    'EV_NUM': ev
                })
        
        # Ordenar por EV descendente
        return sorted(oportunidades, key=lambda x: x['EV_NUM'], reverse=True)

# ============================================================================
# INTERFAZ STREAMLIT PROFESIONAL
# ============================================================================

def main():
    
    # Header
    st.title("⚽ SISTEMABETS IA v4.0 PRO")
    st.caption("🚀 Análisis Profundo con 20 Partidos | Motor de IA Avanzado | Powered by Football-Data.org")
    
    # Sidebar
    st.sidebar.header("⚙️ CONFIGURACIÓN")
    
    api_key = st.sidebar.text_input(
        "🔑 API Key",
        value="b5da8589cdef4d418bbe2afcbccadf10",
        type="password",
        help="Tu API key de Football-Data.org"
    )
    
    if not api_key:
        st.warning("⚠️ Por favor ingresa tu API Key")
        return
    
    # Configuración de análisis
    st.sidebar.subheader("📊 Parámetros de Análisis")
    umbral_ev = st.sidebar.slider("EV Mínimo (%)", 1, 15, 3, help="Valor esperado mínimo para mostrar apuestas")
    mostrar_20_partidos = st.sidebar.checkbox("Ver los 20 partidos completos", value=False)
    
    # Inicializar API
    api = FootballDataAPI(api_key)
    
    # Selección de liga
    st.sidebar.subheader("🏆 Seleccionar Liga")
    liga_nombre = st.sidebar.selectbox(
        "Competición:",
        list(FootballDataAPI.LIGAS.keys()),
        index=0
    )
    liga_code = FootballDataAPI.LIGAS[liga_nombre]
    
    # Cargar tabla
    with st.spinner(f"🔄 Cargando datos de {liga_nombre}..."):
        df = api.obtener_standings(liga_code)
    
    if df is None or df.empty:
        st.error("❌ Error al cargar datos. Verifica tu API Key.")
        st.info("💡 Si acabas de crear tu cuenta, espera 1-2 minutos para activación.")
        return
    
    st.sidebar.success(f"✅ {len(df)} equipos cargados")
    
    # ==================================================================
    # SELECCIÓN DE PARTIDO
    # ==================================================================
    
    st.header("🎯 SELECCIONAR PARTIDO")
    
    col1, col2 = st.columns(2)
    
    equipos = sorted(df['Equipo'].unique())
    
    with col1:
        equipo_local = st.selectbox(
            "🏟️ Equipo Local:",
            equipos,
            index=0
        )
    
    with col2:
        equipo_visitante = st.selectbox(
            "✈️ Equipo Visitante:",
            [e for e in equipos if e != equipo_local],
            index=0
        )
    
    if equipo_local == equipo_visitante:
        st.warning("⚠️ Selecciona dos equipos diferentes")
        return
    
    # Obtener datos de los equipos
    local_data = df[df['Equipo'] == equipo_local].iloc[0]
    visitante_data = df[df['Equipo'] == equipo_visitante].iloc[0]
    
    # Mostrar preview de equipos
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.metric(
            label=f"🏟️ {equipo_local}",
            value=f"Pos: {int(local_data['Posicion'])}",
            delta=f"{int(local_data['Pts'])} pts"
        )
    
    with c2:
        st.metric(
            label="VS",
            value="",
            delta=""
        )
    
    with c3:
        st.metric(
            label=f"✈️ {equipo_visitante}",
            value=f"Pos: {int(visitante_data['Posicion'])}",
            delta=f"{int(visitante_data['Pts'])} pts"
        )
    
    # ==================================================================
    # BOTÓN DE ANÁLISIS
    # ==================================================================
    
    st.divider()
    
    if st.button("🔍 ANALIZAR PARTIDO COMPLETO", type="primary", use_container_width=True):
        
        # Paso 1: Obtener últimos 20 partidos
        with st.spinner(f"📥 Obteniendo últimos 20 partidos de {equipo_local}..."):
            partidos_local = api.obtener_ultimos_20_partidos(equipo_local)
        
        with st.spinner(f"📥 Obteniendo últimos 20 partidos de {equipo_visitante}..."):
            partidos_visitante = api.obtener_ultimos_20_partidos(equipo_visitante)
        
        if not partidos_local:
            st.warning(f"⚠️ No se encontraron partidos recientes de {equipo_local}")
        if not partidos_visitante:
            st.warning(f"⚠️ No se encontraron partidos recientes de {equipo_visitante}")
        
        # Paso 2: Análisis profundo
        with st.spinner("🤖 Procesando análisis con IA avanzada..."):
            
            # Calcular lambdas con motor profesional
            lambda_local, lambda_visitante, analisis_detallado = AnalizadorProfundo.calcular_lambdas_profesional(
                local_data,
                visitante_data,
                df,
                partidos_local,
                partidos_visitante
            )
            
            # Generar matriz de probabilidades
            matriz = AnalizadorProfundo.matriz_probabilidades(lambda_local, lambda_visitante)
            
            # Calcular todos los mercados
            mercados = AnalizadorProfundo.calcular_mercados_completos(matriz)
            
            # Buscar oportunidades de valor
            oportunidades = BuscadorValorPro.encontrar_mejores_apuestas(mercados, umbral_ev)
        
        # ==================================================================
        # MOSTRAR RESULTADOS
        # ==================================================================
        
        st.success("✅ Análisis completado con éxito")
        
        st.divider()
        st.header("📊 RESULTADOS DEL ANÁLISIS")
        
        # ===== MÉTRICAS PRINCIPALES =====
        st.subheader("🎲 Probabilidades 1X2")
        
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.metric(
                "🏠 Gana Local",
                f"{mercados['1X2']['Local']*100:.1f}%",
                delta=f"Cuota ~{1/mercados['1X2']['Local']:.2f}" if mercados['1X2']['Local'] > 0 else "N/A"
            )
        
        with m2:
            st.metric(
                "🤝 Empate",
                f"{mercados['1X2']['Empate']*100:.1f}%",
                delta=f"Cuota ~{1/mercados['1X2']['Empate']:.2f}" if mercados['1X2']['Empate'] > 0 else "N/A"
            )
        
        with m3:
            st.metric(
                "✈️ Gana Visitante",
                f"{mercados['1X2']['Visitante']*100:.1f}%",
                delta=f"Cuota ~{1/mercados['1X2']['Visitante']:.2f}" if mercados['1X2']['Visitante'] > 0 else "N/A"
            )
        
        with m4:
            st.metric(
                "⚽ Goles Totales",
                f"{lambda_local + lambda_visitante:.2f}",
                delta=f"L: {lambda_local:.2f} | V: {lambda_visitante:.2f}"
            )
        
        # ===== OVER/UNDER =====
        st.divider()
        st.subheader("📈 Over/Under Goles")
        
        ou1, ou2, ou3 = st.columns(3)
        
        with ou1:
            st.metric("Over 1.5", f"{mercados['Over/Under']['Over 1.5']*100:.1f}%")
            st.metric("Under 1.5", f"{mercados['Over/Under']['Under 1.5']*100:.1f}%")
        
        with ou2:
            st.metric("Over 2.5", f"{mercados['Over/Under']['Over 2.5']*100:.1f}%")
            st.metric("Under 2.5", f"{mercados['Over/Under']['Under 2.5']*100:.1f}%")
        
        with ou3:
            st.metric("Over 3.5", f"{mercados['Over/Under']['Over 3.5']*100:.1f}%")
            st.metric("Under 3.5", f"{mercados['Over/Under']['Under 3.5']*100:.1f}%")
        
        # ===== BTTS Y DOBLE OPORTUNIDAD =====
        st.divider()
        
        col_btts, col_doble = st.columns(2)
        
        with col_btts:
            st.subheader("🎯 Ambos Marcan (BTTS)")
            st.metric("BTTS Sí", f"{mercados['BTTS']['Si']*100:.1f}%")
            st.metric("BTTS No", f"{mercados['BTTS']['No']*100:.1f}%")
        
        with col_doble:
            st.subheader("🔒 Doble Oportunidad")
            for mercado, prob in mercados['Doble_Oportunidad'].items():
                st.metric(mercado, f"{prob*100:.1f}%")
        
        # ===== RESULTADOS EXACTOS =====
        st.divider()
        st.subheader("🎲 Resultados Más Probables")
        
        st.info(f"**Resultado más probable:** {mercados['Resultado_Exacto']} ({mercados['Prob_Exacto']*100:.1f}%)")
        
        # Top 5 resultados
        df_top_resultados = pd.DataFrame(mercados['Top_Resultados'])
        df_top_resultados['Probabilidad'] = df_top_resultados['Probabilidad'].apply(lambda x: f"{x*100:.2f}%")
        
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.dataframe(df_top_resultados, use_container_width=True, hide_index=True)
        
        # ===== MEJORES OPORTUNIDADES =====
        st.divider()
        st.header("💎 MEJORES OPORTUNIDADES DE APUESTA")
        
        if oportunidades:
            st.success(f"✅ Se encontraron {len(oportunidades)} apuestas con valor positivo (EV > {umbral_ev}%)")
            
            # Crear DataFrame y mostrar
            df_oportunidades = pd.DataFrame(oportunidades)
            df_oportunidades = df_oportunidades.drop('EV_NUM', axis=1)
            
            st.dataframe(
                df_oportunidades,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Confianza": st.column_config.TextColumn("🔥", width="small"),
                    "Kelly_Stake": st.column_config.TextColumn("💰 Stake Sugerido", width="medium")
                }
            )
            
            # Explicación de Kelly
            with st.expander("ℹ️ ¿Qué es Kelly Stake?"):
                st.write("""
                **Kelly Stake** es una fórmula matemática de gestión de bankroll que calcula cuánto apostar 
                basándose en la ventaja estadística (EV). 
                
                - Usamos Kelly **fraccionario (25%)** para ser conservadores
                - Asume un bankroll de $100
                - **NUNCA apuestes más del 10% de tu bankroll** en una sola apuesta
                - Esta es solo una guía, ajusta según tu tolerancia al riesgo
                """)
        
        else:
            st.warning(f"⚠️ No se encontraron apuestas con EV superior a {umbral_ev}%")
            st.info("💡 Intenta reducir el umbral de EV en la configuración lateral")
        
        # ===== ANÁLISIS TÉCNICO DETALLADO =====
        st.divider()
        
        with st.expander("🔬 VER ANÁLISIS TÉCNICO COMPLETO"):
            
            col_tec1, col_tec2 = st.columns(2)
            
            detalles_local = analisis_detallado['detalles_local']
            detalles_visitante = analisis_detallado['detalles_visitante']
            stats_local = analisis_detallado['stats_local']
            stats_visitante = analisis_detallado['stats_visitante']
            
            with col_tec1:
                st.markdown(f"### 🏟️ {equipo_local} (LOCAL)")
                st.write(f"**Score de forma:** {analisis_detallado['forma_local']:.3f}")
                st.write(f"**Racha últimos 5:** {detalles_local['racha_actual']}")
                st.write("")
                st.write(f"**Últimos 20 partidos:**")
                st.write(f"- Victorias: {detalles_local['victorias']}")
                st.write(f"- Empates: {detalles_local['empates']}")
                st.write(f"- Derrotas: {detalles_local['derrotas']}")
                st.write(f"- Goles a favor: {detalles_local['gf']}")
                st.write(f"- Goles en contra: {detalles_local['gc']}")
                st.write(f"- Promedio goles: {detalles_local['gf']/20:.2f} por partido")
                st.write("")
                st.write(f"**Jugando de LOCAL (últimos 20):**")
                st.write(f"- Partidos jugados: {stats_local['local_pj']}")
                st.write(f"- Victorias: {stats_local['local_victorias']}")
                st.write(f"- Goles favor: {stats_local['local_gf']} ({stats_local['local_gf']/max(stats_local['local_pj'],1):.2f}/partido)")
                st.write(f"- Goles contra: {stats_local['local_gc']} ({stats_local['local_gc']/max(stats_local['local_pj'],1):.2f}/partido)")
                st.write("")
                st.write(f"**Lambda calculado:** {lambda_local:.3f}")
            
            with col_tec2:
                st.markdown(f"### ✈️ {equipo_visitante} (VISITANTE)")
                st.write(f"**Score de forma:** {analisis_detallado['forma_visitante']:.3f}")
                st.write(f"**Racha últimos 5:** {detalles_visitante['racha_actual']}")
                st.write("")
                st.write(f"**Últimos 20 partidos:**")
                st.write(f"- Victorias: {detalles_visitante['victorias']}")
                st.write(f"- Empates: {detalles_visitante['empates']}")
                st.write(f"- Derrotas: {detalles_visitante['derrotas']}")
                st.write(f"- Goles a favor: {detalles_visitante['gf']}")
                st.write(f"- Goles en contra: {detalles_visitante['gc']}")
                st.write(f"- Promedio goles: {detalles_visitante['gf']/20:.2f} por partido")
                st.write("")
                st.write(f"**Jugando de VISITANTE (últimos 20):**")
                st.write(f"- Partidos jugados: {stats_visitante['visitante_pj']}")
                st.write(f"- Victorias: {stats_visitante['visitante_victorias']}")
                st.write(f"- Goles favor: {stats_visitante['visitante_gf']} ({stats_visitante['visitante_gf']/max(stats_visitante['visitante_pj'],1):.2f}/partido)")
                st.write(f"- Goles contra: {stats_visitante['visitante_gc']} ({stats_visitante['visitante_gc']/max(stats_visitante['visitante_pj'],1):.2f}/partido)")
                st.write("")
                st.write(f"**Lambda calculado:** {lambda_visitante:.3f}")
        
        # ===== HISTORIAL DE PARTIDOS =====
        with st.expander("📋 VER HISTORIAL DE PARTIDOS"):
            
            col_hist1, col_hist2 = st.columns(2)
            
            with col_hist1:
                st.markdown(f"### {equipo_local}")
                
                if partidos_local:
                    num_mostrar = 20 if mostrar_20_partidos else 10
                    
                    for i, p in enumerate(partidos_local[:num_mostrar], 1):
                        es_local = p['local'] == equipo_local
                        gf = p['goles_local'] if es_local else p['goles_visitante']
                        gc = p['goles_visitante'] if es_local else p['goles_local']
                        
                        # Emoji resultado
                        if gf > gc:
                            emoji = "✅"
                        elif gf == gc:
                            emoji = "🟰"
                        else:
                            emoji = "❌"
                        
                        # Formato
                        ubicacion = "🏠" if es_local else "✈️"
                        marcador = f"{p['goles_local']}-{p['goles_visitante']}"
                        
                        st.write(f"{i}. {emoji} {ubicacion} {p['local']} **{marcador}** {p['visitante']}")
                        st.caption(f"   {p['competicion']} | {p['fecha'][:10]}")
                else:
                    st.write("No hay datos de partidos")
            
            with col_hist2:
                st.markdown(f"### {equipo_visitante}")
                
                if partidos_visitante:
                    num_mostrar = 20 if mostrar_20_partidos else 10
                    
                    for i, p in enumerate(partidos_visitante[:num_mostrar], 1):
                        es_local = p['local'] == equipo_visitante
                        gf = p['goles_local'] if es_local else p['goles_visitante']
                        gc = p['goles_visitante'] if es_local else p['goles_local']
                        
                        if gf > gc:
                            emoji = "✅"
                        elif gf == gc:
                            emoji = "🟰"
                        else:
                            emoji = "❌"
                        
                        ubicacion = "🏠" if es_local else "✈️"
                        marcador = f"{p['goles_local']}-{p['goles_visitante']}"
                        
                        st.write(f"{i}. {emoji} {ubicacion} {p['local']} **{marcador}** {p['visitante']}")
                        st.caption(f"   {p['competicion']} | {p['fecha'][:10]}")
                else:
                    st.write("No hay datos de partidos")
        
        # ===== TABLA DE CLASIFICACIÓN =====
        with st.expander("🏆 VER TABLA COMPLETA DE LA LIGA"):
            st.dataframe(
                df[['Posicion', 'Equipo', 'PJ', 'Victorias', 'Empates', 'Derrotas', 'GF', 'GC', 'Pts']]
                .sort_values('Posicion'),
                use_container_width=True,
                hide_index=True
            )
    
    # Footer
    st.divider()
    st.caption("⚽ SISTEMABETS IA v4.0 PRO | Análisis estadístico con 20 partidos | Motor IA Avanzado")
    st.caption("⚠️ Advertencia: Las apuestas implican riesgo. Apuesta responsablemente.")

if __name__ == "__main__":
    main()
