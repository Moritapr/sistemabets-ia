import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SISTEMABETS IA v5.0 ULTRA PRO", layout="wide")

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
        """Obtiene últimos 20 partidos del equipo en TODAS las competiciones"""
        try:
            team_id = self.cache_teams.get(equipo_nombre)
            
            if not team_id:
                st.warning(f"No se encontró ID para {equipo_nombre}")
                return []
            
            url = f"{self.BASE_URL}/teams/{team_id}/matches"
            params = {"status": "FINISHED", "limit": 100}
            
            response = requests.get(url, headers=self.headers, params=params, timeout=15)
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            partidos = []
            
            for match in data.get('matches', []):
                if match['score']['fullTime']['home'] is None:
                    continue
                
                home_team = match['homeTeam']['name']
                away_team = match['awayTeam']['name']
                
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
            
            partidos_ordenados = sorted(partidos, key=lambda x: x['fecha'], reverse=True)
            return partidos_ordenados[:20]
            
        except Exception as e:
            st.warning(f"Error obteniendo historial de {equipo_nombre}: {str(e)}")
            return []
    
    def obtener_enfrentamientos_directos(self, equipo1, equipo2):
        """Obtiene historial de enfrentamientos directos (H2H)"""
        try:
            team_id = self.cache_teams.get(equipo1)
            if not team_id:
                return []
            
            url = f"{self.BASE_URL}/teams/{team_id}/matches"
            params = {"status": "FINISHED", "limit": 100}
            
            response = requests.get(url, headers=self.headers, params=params, timeout=15)
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            h2h = []
            
            for match in data.get('matches', []):
                if match['score']['fullTime']['home'] is None:
                    continue
                
                home = match['homeTeam']['name']
                away = match['awayTeam']['name']
                
                if (home == equipo1 and away == equipo2) or (home == equipo2 and away == equipo1):
                    h2h.append({
                        'local': home,
                        'visitante': away,
                        'goles_local': match['score']['fullTime']['home'],
                        'goles_visitante': match['score']['fullTime']['away'],
                        'fecha': match['utcDate'],
                        'competicion': match['competition']['name']
                    })
            
            return sorted(h2h, key=lambda x: x['fecha'], reverse=True)[:10]
            
        except Exception as e:
            return []

# ============================================================================
# MÓDULO 2: MOTOR DE ANÁLISIS ULTRA PROFUNDO
# ============================================================================

class AnalizadorUltraProfundo:
    """Motor de IA con análisis de múltiples factores y argumentos sólidos"""
    
    VENTAJA_LOCAL_BASE = 1.15
    MIN_PROBABILIDAD_REALISTA = 0.005  # 0.5% mínimo para evitar valores irreales
    
    @staticmethod
    def calcular_forma_20_partidos(partidos, equipo_nombre):
        """Calcula forma usando los últimos 20 partidos con pesos decrecientes"""
        if not partidos:
            return 0.50, "", {}
        
        puntos = []
        forma_visual = ""
        detalles = {
            'victorias': 0, 'empates': 0, 'derrotas': 0,
            'gf': 0, 'gc': 0, 'racha_actual': "",
            'goles_1h': 0, 'goles_2h': 0,
            'corners_promedio': 0, 'tarjetas_promedio': 0
        }
        
        for i, partido in enumerate(partidos[:20]):
            es_local = partido['local'] == equipo_nombre
            gf = partido['goles_local'] if es_local else partido['goles_visitante']
            gc = partido['goles_visitante'] if es_local else partido['goles_local']
            
            detalles['gf'] += gf
            detalles['gc'] += gc
            detalles['goles_1h'] += int(gf * 0.45)
            detalles['goles_2h'] += int(gf * 0.55)
            
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
        
        pesos = np.linspace(0.10, 0.005, len(puntos))
        pesos = pesos / pesos.sum()
        score_forma = sum(p * w for p, w in zip(puntos, pesos))
        
        return score_forma, forma_visual, detalles
    
    @staticmethod
    def calcular_stats_avanzadas(equipo_nombre, partidos):
        """Calcula estadísticas avanzadas: local/visitante, primera mitad, etc."""
        stats = {
            'local_pj': 0, 'local_gf': 0, 'local_gc': 0, 'local_victorias': 0,
            'visitante_pj': 0, 'visitante_gf': 0, 'visitante_gc': 0, 'visitante_victorias': 0,
            'partidos_over25': 0, 'partidos_btts': 0,
            'goles_1h_favor': 0, 'goles_1h_contra': 0,
            'partidos_gol_1h': 0
        }
        
        for p in partidos[:20]:
            total_goles = p['goles_local'] + p['goles_visitante']
            
            if total_goles > 2.5:
                stats['partidos_over25'] += 1
            
            if p['goles_local'] > 0 and p['goles_visitante'] > 0:
                stats['partidos_btts'] += 1
            
            if p['local'] == equipo_nombre:
                stats['local_pj'] += 1
                stats['local_gf'] += p['goles_local']
                stats['local_gc'] += p['goles_visitante']
                if p['goles_local'] > p['goles_visitante']:
                    stats['local_victorias'] += 1
                
                goles_1h = int(p['goles_local'] * 0.45)
                stats['goles_1h_favor'] += goles_1h
                stats['goles_1h_contra'] += int(p['goles_visitante'] * 0.45)
                if goles_1h > 0:
                    stats['partidos_gol_1h'] += 1
                    
            elif p['visitante'] == equipo_nombre:
                stats['visitante_pj'] += 1
                stats['visitante_gf'] += p['goles_visitante']
                stats['visitante_gc'] += p['goles_local']
                if p['goles_visitante'] > p['goles_local']:
                    stats['visitante_victorias'] += 1
                
                goles_1h = int(p['goles_visitante'] * 0.45)
                stats['goles_1h_favor'] += goles_1h
                stats['goles_1h_contra'] += int(p['goles_local'] * 0.45)
                if goles_1h > 0:
                    stats['partidos_gol_1h'] += 1
        
        return stats
    
    @staticmethod
    def calcular_lambdas_y_factores(local_team, visitante_team, df_liga, partidos_local, partidos_visitante, h2h):
        """Calcula lambdas con análisis profundo y genera argumentos"""
        
        gf_local_tabla = local_team['GF'] / max(local_team['PJ'], 1)
        gc_local_tabla = local_team['GC'] / max(local_team['PJ'], 1)
        gf_visitante_tabla = visitante_team['GF'] / max(visitante_team['PJ'], 1)
        gc_visitante_tabla = visitante_team['GC'] / max(visitante_team['PJ'], 1)
        
        forma_local, _, detalles_local = AnalizadorUltraProfundo.calcular_forma_20_partidos(
            partidos_local, local_team['Equipo']
        )
        forma_visitante, _, detalles_visitante = AnalizadorUltraProfundo.calcular_forma_20_partidos(
            partidos_visitante, visitante_team['Equipo']
        )
        
        stats_local = AnalizadorUltraProfundo.calcular_stats_avanzadas(
            local_team['Equipo'], partidos_local
        )
        stats_visitante = AnalizadorUltraProfundo.calcular_stats_avanzadas(
            visitante_team['Equipo'], partidos_visitante
        )
        
        gf_local_casa = stats_local['local_gf'] / max(stats_local['local_pj'], 1) if stats_local['local_pj'] > 0 else gf_local_tabla
        gc_local_casa = stats_local['local_gc'] / max(stats_local['local_pj'], 1) if stats_local['local_pj'] > 0 else gc_local_tabla
        gf_visitante_fuera = stats_visitante['visitante_gf'] / max(stats_visitante['visitante_pj'], 1) if stats_visitante['visitante_pj'] > 0 else gf_visitante_tabla
        gc_visitante_fuera = stats_visitante['visitante_gc'] / max(stats_visitante['visitante_pj'], 1) if stats_visitante['visitante_pj'] > 0 else gc_visitante_tabla
        
        media_goles_liga = df_liga['GF'].sum() / df_liga['PJ'].sum()
        
        lambda_local = (gf_local_casa / media_goles_liga) * (gc_visitante_fuera / media_goles_liga) * media_goles_liga
        lambda_visitante = (gf_visitante_fuera / media_goles_liga) * (gc_local_casa / media_goles_liga) * media_goles_liga
        
        ventaja_local = AnalizadorUltraProfundo.VENTAJA_LOCAL_BASE
        if stats_local['local_pj'] >= 5:
            winrate_casa = stats_local['local_victorias'] / stats_local['local_pj']
            if winrate_casa > 0.7:
                ventaja_local *= 1.10
            elif winrate_casa < 0.3:
                ventaja_local *= 0.92
        
        lambda_local *= ventaja_local
        lambda_visitante /= (ventaja_local * 0.90)
        
        if forma_local > 0.75:
            lambda_local *= 1.25
        elif forma_local > 0.60:
            lambda_local *= 1.12
        elif forma_local < 0.25:
            lambda_local *= 0.80
        elif forma_local < 0.40:
            lambda_local *= 0.90
        
        if forma_visitante > 0.75:
            lambda_visitante *= 1.25
        elif forma_visitante > 0.60:
            lambda_visitante *= 1.12
        elif forma_visitante < 0.25:
            lambda_visitante *= 0.80
        elif forma_visitante < 0.40:
            lambda_visitante *= 0.90
        
        diff_posiciones = abs(local_team['Posicion'] - visitante_team['Posicion'])
        diff_puntos = abs(local_team['Pts'] - visitante_team['Pts'])
        
        factor_calidad = 1.0
        if diff_posiciones >= 8 or diff_puntos >= 15:
            if local_team['Posicion'] < visitante_team['Posicion']:
                factor_calidad = 1.12
                lambda_local *= factor_calidad
                lambda_visitante *= 0.88
            else:
                factor_calidad = 0.88
                lambda_local *= factor_calidad
                lambda_visitante *= 1.12
        
        h2h_factor_local = 1.0
        h2h_factor_visitante = 1.0
        if h2h and len(h2h) >= 3:
            victorias_local_h2h = sum(1 for p in h2h if 
                (p['local'] == local_team['Equipo'] and p['goles_local'] > p['goles_visitante']) or
                (p['visitante'] == local_team['Equipo'] and p['goles_visitante'] > p['goles_local'])
            )
            if victorias_local_h2h / len(h2h) > 0.6:
                h2h_factor_local = 1.08
                h2h_factor_visitante = 0.94
            elif victorias_local_h2h / len(h2h) < 0.3:
                h2h_factor_local = 0.94
                h2h_factor_visitante = 1.08
            
            lambda_local *= h2h_factor_local
            lambda_visitante *= h2h_factor_visitante
        
        lambda_local = max(lambda_local, 0.25)
        lambda_visitante = max(lambda_visitante, 0.25)
        lambda_local = min(lambda_local, 4.5)
        lambda_visitante = min(lambda_visitante, 4.5)
        
        return lambda_local, lambda_visitante, {
            'forma_local': forma_local,
            'forma_visitante': forma_visitante,
            'detalles_local': detalles_local,
            'detalles_visitante': detalles_visitante,
            'stats_local': stats_local,
            'stats_visitante': stats_visitante,
            'ventaja_local': ventaja_local,
            'factor_calidad': factor_calidad,
            'h2h_factor_local': h2h_factor_local
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
    def calcular_mercados_extendidos(matriz, stats_local, stats_visitante, lambda_l, lambda_v):
        """Calcula TODOS los mercados incluyendo handicap, primera mitad, etc."""
        
        # 1X2
        p_local = np.sum(np.tril(matriz, -1))
        p_empate = np.sum(np.diag(matriz))
        p_visitante = np.sum(np.triu(matriz, 1))
        
        total_1x2 = p_local + p_empate + p_visitante
        p_local = max(p_local / total_1x2, AnalizadorUltraProfundo.MIN_PROBABILIDAD_REALISTA)
        p_empate = max(p_empate / total_1x2, AnalizadorUltraProfundo.MIN_PROBABILIDAD_REALISTA)
        p_visitante = max(p_visitante / total_1x2, AnalizadorUltraProfundo.MIN_PROBABILIDAD_REALISTA)
        
        # Over/Under
        over_under = {}
        for threshold in [0.5, 1.5, 2.5, 3.5, 4.5]:
            limite = threshold + 0.01
            p_over = sum([matriz[i, j] for i in range(matriz.shape[0]) 
                         for j in range(matriz.shape[1]) if (i+j) > limite])
            p_under = 1 - p_over
            over_under[f"Over {threshold}"] = max(p_over, AnalizadorUltraProfundo.MIN_PROBABILIDAD_REALISTA)
            over_under[f"Under {threshold}"] = max(p_under, AnalizadorUltraProfundo.MIN_PROBABILIDAD_REALISTA)
        
        # BTTS
        p_btts_no = matriz[0,:].sum() + matriz[:,0].sum() - matriz[0,0]
        p_btts_si = max(1 - p_btts_no, AnalizadorUltraProfundo.MIN_PROBABILIDAD_REALISTA)
        p_btts_no = max(p_btts_no, AnalizadorUltraProfundo.MIN_PROBABILIDAD_REALISTA)
        
        # Doble Oportunidad
        doble_oportunidad = {
            '1X (Local o Empate)': p_local + p_empate,
            '12 (Local o Visitante)': p_local + p_visitante,
            'X2 (Empate o Visitante)': p_empate + p_visitante
        }
        
        # Handicap Asiático
        handicap = {}
        p_local_menos15 = sum([matriz[i, j] for i in range(matriz.shape[0]) 
                              for j in range(matriz.shape[1]) if i - j >= 2])
        handicap['Local -1.5'] = max(p_local_menos15, AnalizadorUltraProfundo.MIN_PROBABILIDAD_REALISTA)
        handicap['Visitante +1.5'] = max(1 - p_local_menos15, AnalizadorUltraProfundo.MIN_PROBABILIDAD_REALISTA)
        
        p_local_menos05 = sum([matriz[i, j] for i in range(matriz.shape[0]) 
                              for j in range(matriz.shape[1]) if i > j])
        handicap['Local -0.5'] = max(p_local_menos05, AnalizadorUltraProfundo.MIN_PROBABILIDAD_REALISTA)
        handicap['Visitante +0.5'] = max(1 - p_local_menos05, AnalizadorUltraProfundo.MIN_PROBABILIDAD_REALISTA)
        
        # Primera Mitad
        goles_1h_local = stats_local['goles_1h_favor'] / 20 if stats_local else lambda_l * 0.45
        goles_1h_visitante = stats_visitante['goles_1h_favor'] / 20 if stats_visitante else lambda_v * 0.45
        
        primera_mitad = {
            'Gol en 1H': max(1 - poisson.pmf(0, goles_1h_local + goles_1h_visitante), AnalizadorUltraProfundo.MIN_PROBABILIDAD_REALISTA),
            'Más de 0.5 goles 1H': max(1 - poisson.pmf(0, goles_1h_local + goles_1h_visitante), AnalizadorUltraProfundo.MIN_PROBABILIDAD_REALISTA),
            'Más de 1.5 goles 1H': max(sum([poisson.pmf(k, goles_1h_local + goles_1h_visitante) for k in range(2, 10)]), AnalizadorUltraProfundo.MIN_PROBABILIDAD_REALISTA)
        }
        
        # Resultado exacto
        idx_max = np.unravel_index(matriz.argmax(), matriz.shape)
        top_resultados = []
        matriz_flat = matriz.flatten()
        indices_ordenados = np.argsort(matriz_flat)[::-1][:5]
        
        for idx in indices_ordenados:
            i = idx // matriz.shape[1]
            j = idx % matriz.shape[1]
            top_resultados.append({
                'Marcador': f"{i}-{j}",
                'Probabilidad': max(matriz[i, j], AnalizadorUltraProfundo.MIN_PROBABILIDAD_REALISTA)
            })
        
        return {
            '1X2': {'Local': p_local, 'Empate': p_empate, 'Visitante': p_visitante},
            'Over/Under': over_under,
            'BTTS': {'Si': p_btts_si, 'No': p_btts_no},
            'Doble_Oportunidad': doble_oportunidad,
            'Handicap': handicap,
            'Primera_Mitad': primera_mitad,
            'Resultado_Exacto': f"{idx_max[0]}-{idx_max[1]}",
            'Prob_Exacto': max(matriz[idx_max[0], idx_max[1]], AnalizadorUltraProfundo.MIN_PROBABILIDAD_REALISTA),
            'Top_Resultados': top_resultados
        }

# ============================================================================
# MÓDULO 3: GENERADOR DE ARGUMENTOS Y RECOMENDACIONES
# ============================================================================

class GeneradorArgumentos:
    """Genera argumentos sólidos para cada apuesta recomendada"""
    
    @staticmethod
    def generar_argumento_1x2(resultado, prob, local_team, visitante_team, analisis):
        """Genera argumento para apuesta 1X2"""
        argumentos = []
        
        forma_local = analisis['forma_local']
        forma_visitante = analisis['forma_visitante']
        detalles_local = analisis['detalles_local']
        detalles_visitante = analisis['detalles_visitante']
        
        if resultado == "Local":
            if forma_local > 0.7:
                argumentos.append(f"✅ {local_team['Equipo']} tiene excelente forma ({forma_local*100:.0f}%) con {detalles_local['victorias']} victorias en últimos 20 partidos")
            
            if local_team['Posicion'] < visitante_team['Posicion'] - 5:
                argumentos.append(f"📊 Diferencia de nivel: {local_team['Equipo']} está {visitante_team['Posicion'] - local_team['Posicion']} posiciones arriba en la tabla")
            
            stats_local = analisis['stats_local']
            if stats_local['local_pj'] >= 5:
                winrate = stats_local['local_victorias'] / stats_local['local_pj']
                if winrate > 0.6:
                    argumentos.append(f"🏠 Fuerte en casa: {stats_local['local_victorias']}/{stats_local['local_pj']} victorias jugando de local ({winrate*100:.0f}%)")
            
            if analisis.get('h2h_factor_local', 1.0) > 1.0:
                argumentos.append(f"🎯 Domina los enfrentamientos directos históricamente")
        
        elif resultado == "Visitante":
            if forma_visitante > 0.7:
                argumentos.append(f"✅ {visitante_team['Equipo']} tiene excelente forma ({forma_visitante*100:.0f}%) con {detalles_visitante['victorias']} victorias recientes")
            
            if visitante_team['Posicion'] < local_team['Posicion'] - 5:
                argumentos.append(f"📊 Equipo superior: {visitante_team['Posicion']} vs {local_team['Posicion']} en la tabla")
            
            stats_visitante = analisis['stats_visitante']
            if stats_visitante['visitante_pj'] >= 5:
                winrate = stats_visitante['visitante_victorias'] / stats_visitante['visitante_pj']
                if winrate > 0.5:
                    argumentos.append(f"✈️ Buen rendimiento fuera: {stats_visitante['visitante_victorias']}/{stats_visitante['visitante_pj']} victorias de visitante ({winrate*100:.0f}%)")
            
            if forma_local < 0.4:
                argumentos.append(f"📉 {local_team['Equipo']} con mala forma reciente ({forma_local*100:.0f}%), solo {detalles_local['victorias']} victorias en 20 partidos")
        
        else:  # Empate
            if abs(forma_local - forma_visitante) < 0.15:
                argumentos.append(f"⚖️ Equipos igualados en forma: {forma_local*100:.0f}% vs {forma_visitante*100:.0f}%")
            
            if abs(local_team['Posicion'] - visitante_team['Posicion']) <= 3:
                argumentos.append(f"📊 Posiciones muy cercanas en la tabla ({local_team['Posicion']} vs {visitante_team['Posicion']})")
            
            empates_local = detalles_local['empates']
            empates_visitante = detalles_visitante['empates']
            if empates_local + empates_visitante >= 8:
                argumentos.append(f"🤝 Alta tendencia al empate: {empates_local} + {empates_visitante} empates en sus últimos 20 partidos")
        
        return argumentos
    
    @staticmethod
    def generar_argumento_over_under(mercado, prob, analisis):
        """Genera argumento para Over/Under"""
        argumentos = []
        
        detalles_local = analisis['detalles_local']
        detalles_visitante = analisis['detalles_visitante']
        stats_local = analisis['stats_local']
        stats_visitante = analisis['stats_visitante']
        
        goles_promedio_local = detalles_local['gf'] / 20
        goles_promedio_visitante = detalles_visitante['gf'] / 20
        
        if "Over" in mercado:
            threshold = float(mercado.split()[1])
            
            if goles_promedio_local + goles_promedio_visitante > threshold:
                argumentos.append(f"⚽ Promedio combinado: {goles_promedio_local:.1f} + {goles_promedio_visitante:.1f} = {goles_promedio_local + goles_promedio_visitante:.1f} goles/partido")
            
            over25_local = stats_local['partidos_over25'] / 20
            over25_visitante = stats_visitante['partidos_over25'] / 20
            if threshold == 2.5 and (over25_local > 0.6 or over25_visitante > 0.6):
                argumentos.append(f"📈 Tendencia alta de goles: {over25_local*100:.0f}% y {over25_visitante*100:.0f}% de sus partidos tienen Over 2.5")
            
            gc_promedio_local = detalles_local['gc'] / 20
            gc_promedio_visitante = detalles_visitante['gc'] / 20
            if gc_promedio_local > 1.2 and gc_promedio_visitante > 1.2:
                argumentos.append(f"🥅 Ambos equipos encajan goles: {gc_promedio_local:.1f} y {gc_promedio_visitante:.1f} promedio")
        
        else:  # Under
            threshold = float(mercado.split()[1])
            
            if goles_promedio_local + goles_promedio_visitante < threshold:
                argumentos.append(f"🔒 Bajo promedio combinado de goles: {goles_promedio_local + goles_promedio_visitante:.1f}")
            
            over25_local = stats_local['partidos_over25'] / 20
            over25_visitante = stats_visitante['partidos_over25'] / 20
            if threshold == 2.5 and (over25_local < 0.4 and over25_visitante < 0.4):
                argumentos.append(f"📉 Pocos partidos con Over 2.5: solo {over25_local*100:.0f}% y {over25_visitante*100:.0f}%")
        
        return argumentos
    
    @staticmethod
    def generar_argumento_btts(si_no, prob, analisis):
        """Genera argumento para BTTS"""
        argumentos = []
        
        stats_local = analisis['stats_local']
        stats_visitante = analisis['stats_visitante']
        detalles_local = analisis['detalles_local']
        detalles_visitante = analisis['detalles_visitante']
        
        if si_no == "Si":
            btts_rate_local = stats_local['partidos_btts'] / 20
            btts_rate_visitante = stats_visitante['partidos_btts'] / 20
            
            if btts_rate_local > 0.55 or btts_rate_visitante > 0.55:
                argumentos.append(f"⚽⚽ Alta frecuencia BTTS: {btts_rate_local*100:.0f}% y {btts_rate_visitante*100:.0f}% en sus últimos 20 partidos")
            
            gf_local = detalles_local['gf'] / 20
            gf_visitante = detalles_visitante['gf'] / 20
            if gf_local > 1.2 and gf_visitante > 1.0:
                argumentos.append(f"🎯 Ambos anotan regularmente: {gf_local:.1f} y {gf_visitante:.1f} goles/partido")
            
            gc_local = detalles_local['gc'] / 20
            gc_visitante = detalles_visitante['gc'] / 20
            if gc_local > 1.0 and gc_visitante > 1.0:
                argumentos.append(f"🥅 Defensas vulnerables: encajan {gc_local:.1f} y {gc_visitante:.1f} goles/partido")
        
        else:  # No
            btts_rate_local = stats_local['partidos_btts'] / 20
            btts_rate_visitante = stats_visitante['partidos_btts'] / 20
            
            if btts_rate_local < 0.4 or btts_rate_visitante < 0.4:
                argumentos.append(f"🔒 Baja frecuencia BTTS: solo {btts_rate_local*100:.0f}% y {btts_rate_visitante*100:.0f}%")
            
            gf_visitante = detalles_visitante['gf'] / 20
            if gf_visitante < 0.8:
                argumentos.append(f"📉 Visitante con bajo poder ofensivo: {gf_visitante:.1f} goles/partido")
        
        return argumentos
    
    @staticmethod
    def generar_argumento_handicap(mercado, prob, analisis, local_team, visitante_team):
        """Genera argumento para Handicap"""
        argumentos = []
        
        if "Local" in mercado:
            diff_posiciones = visitante_team['Posicion'] - local_team['Posicion']
            if diff_posiciones > 8:
                argumentos.append(f"📊 Superioridad clara: {local_team['Equipo']} está {diff_posiciones} posiciones arriba")
            
            detalles_local = analisis['detalles_local']
            gf_local = detalles_local['gf'] / 20
            gc_local = detalles_local['gc'] / 20
            diff_goles = gf_local - gc_local
            if diff_goles > 1.0:
                argumentos.append(f"⚽ Excelente diferencia de goles: +{diff_goles:.1f} promedio")
            
            stats_local = analisis['stats_local']
            if stats_local['local_pj'] >= 5:
                gf_casa = stats_local['local_gf'] / stats_local['local_pj']
                if gf_casa > 2.0:
                    argumentos.append(f"🏠 Dominante en casa: {gf_casa:.1f} goles/partido de local")
        
        else:  # Visitante
            stats_visitante = analisis['stats_visitante']
            if stats_visitante['visitante_pj'] >= 5:
                gc_fuera = stats_visitante['visitante_gc'] / stats_visitante['visitante_pj']
                if gc_fuera < 1.2:
                    argumentos.append(f"🛡️ Sólido defensivamente fuera: solo {gc_fuera:.1f} goles en contra/partido")
        
        return argumentos
    
    @staticmethod
    def generar_argumento_primera_mitad(mercado, prob, analisis):
        """Genera argumento para mercados de Primera Mitad"""
        argumentos = []
        
        stats_local = analisis['stats_local']
        stats_visitante = analisis['stats_visitante']
        
        goles_1h_local = stats_local['goles_1h_favor'] / 20
        goles_1h_visitante = stats_visitante['goles_1h_favor'] / 20
        
        if "Gol" in mercado or "0.5" in mercado:
            tasa_gol_1h_local = stats_local['partidos_gol_1h'] / 20
            tasa_gol_1h_visitante = stats_visitante['partidos_gol_1h'] / 20
            
            if tasa_gol_1h_local > 0.6 or tasa_gol_1h_visitante > 0.6:
                argumentos.append(f"⚡ Equipos que arrancan anotando: {tasa_gol_1h_local*100:.0f}% y {tasa_gol_1h_visitante*100:.0f}% anotan en 1H")
            
            if goles_1h_local + goles_1h_visitante > 0.8:
                argumentos.append(f"🎯 Promedio combinado 1H: {goles_1h_local + goles_1h_visitante:.1f} goles")
        
        elif "1.5" in mercado:
            if goles_1h_local + goles_1h_visitante > 1.0:
                argumentos.append(f"⚽⚽ Ritmo alto en 1H: promedio {goles_1h_local + goles_1h_visitante:.1f} goles")
        
        return argumentos

# ============================================================================
# MÓDULO 4: CALCULADOR DE VALOR Y EV
# ============================================================================

class CalculadorValor:
    """Calcula Expected Value (EV) y encuentra oportunidades de valor"""
    
    @staticmethod
    def calcular_cuota_justa(probabilidad):
        """Calcula la cuota justa sin margen de la casa"""
        return 1 / max(probabilidad, 0.001)
    
    @staticmethod
    def calcular_ev(probabilidad, cuota_mercado):
        """
        Calcula Expected Value (EV)
        EV = (Probabilidad × Cuota) - 1
        EV positivo = apuesta con valor
        """
        return (probabilidad * cuota_mercado) - 1
    
    @staticmethod
    def encontrar_valor(mercados, margen_casa=0.05):
        """
        Encuentra apuestas con valor positivo
        margen_casa: típicamente 5-10% para casas de apuestas
        """
        oportunidades = []
        
        for categoria, opciones in mercados.items():
            if categoria == 'Top_Resultados' or categoria == 'Resultado_Exacto' or categoria == 'Prob_Exacto':
                continue
            
            if isinstance(opciones, dict):
                for mercado, prob in opciones.items():
                    cuota_justa = CalculadorValor.calcular_cuota_justa(prob)
                    cuota_esperada_mercado = cuota_justa * (1 - margen_casa)
                    
                    ev = CalculadorValor.calcular_ev(prob, cuota_esperada_mercado)
                    
                    # Solo incluir si tiene valor positivo significativo
                    if ev > 0.02:  # Mínimo 2% de EV
                        oportunidades.append({
                            'Categoria': categoria,
                            'Mercado': mercado,
                            'Probabilidad': prob,
                            'Cuota_Justa': cuota_justa,
                            'Cuota_Min_Esperada': cuota_esperada_mercado,
                            'EV': ev,
                            'Confianza': CalculadorValor.calcular_confianza(prob, ev)
                        })
        
        # Ordenar por EV descendente
        oportunidades.sort(key=lambda x: x['EV'], reverse=True)
        return oportunidades
    
    @staticmethod
    def calcular_confianza(probabilidad, ev):
        """
        Calcula nivel de confianza basado en probabilidad y EV
        """
        if probabilidad > 0.6 and ev > 0.1:
            return "⭐⭐⭐ MUY ALTA"
        elif probabilidad > 0.5 and ev > 0.08:
            return "⭐⭐ ALTA"
        elif probabilidad > 0.4 and ev > 0.05:
            return "⭐ MEDIA"
        else:
            return "⚠️ BAJA"

# ============================================================================
# INTERFAZ STREAMLIT
# ============================================================================

def main():
    st.title("⚽ SISTEMABETS IA v5.0 ULTRA PRO")
    st.markdown("### Sistema de análisis predictivo con IA para apuestas deportivas")
    
    # Sidebar configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        api_key = st.text_input("API Key Football-Data.org", type="password", 
                               help="Obtén tu clave gratis en https://www.football-data.org/")
        
        if not api_key:
            st.warning("⚠️ Ingresa tu API Key para comenzar")
            st.stop()
        
        st.markdown("---")
        st.markdown("### 📖 Glosario de Términos")
        
        with st.expander("🎲 ¿Qué es Lambda (λ)?"):
            st.markdown("""
            **Lambda** es el parámetro fundamental de la distribución de Poisson.
            
            Representa el **promedio esperado de goles** que un equipo puede marcar.
            
            - **Lambda 2.0** = Se espera que el equipo anote 2 goles
            - **Lambda 0.8** = Se espera que anote menos de 1 gol
            
            Se calcula combinando:
            - Rendimiento ofensivo del equipo
            - Rendimiento defensivo del rival
            - Forma reciente
            - Ventaja de local/visitante
            """)
        
        with st.expander("💰 ¿Qué es el EV (Expected Value)?"):
            st.markdown("""
            **EV (Valor Esperado)** mide si una apuesta tiene valor matemático.
            
            **Fórmula:** EV = (Probabilidad × Cuota) - 1
            
            - **EV > 0** = Apuesta con valor (rentable a largo plazo)
            - **EV = 0** = Cuota justa
            - **EV < 0** = Sin valor (favorece a la casa)
            
            **Ejemplo:**
            - Probabilidad 60% (0.60)
            - Cuota de mercado: 1.80
            - EV = (0.60 × 1.80) - 1 = 0.08 (+8% de valor)
            
            ✅ Un EV de +5% o más indica buena oportunidad
            """)
        
        with st.expander("📊 ¿Por qué ver los 20 partidos?"):
            st.markdown("""
            Ver los últimos 20 partidos te permite:
            
            - ✅ Validar la forma actual del equipo
            - ✅ Identificar patrones (rachas, goles, etc.)
            - ✅ Ver rendimiento local vs visitante
            - ✅ Analizar contra qué rivales jugó
            - ✅ Detectar lesiones o cambios tácticos
            
            La IA usa estos datos para calcular probabilidades, 
            pero TÚ puedes ver el contexto completo.
            """)
    
    # Selección de liga
    st.header("1️⃣ Selecciona la Liga")
    liga_nombre = st.selectbox("Liga", list(FootballDataAPI.LIGAS.keys()))
    liga_code = FootballDataAPI.LIGAS[liga_nombre]
    
    api = FootballDataAPI(api_key)
    
    # Cargar standings
    with st.spinner("Cargando tabla de posiciones..."):
        df_liga = api.obtener_standings(liga_code)
    
    if df_liga is None or df_liga.empty:
        st.error("❌ No se pudo cargar la tabla. Verifica tu API Key o la liga seleccionada.")
        st.stop()
    
    st.success(f"✅ Tabla de {liga_nombre} cargada correctamente")
    
    # Mostrar tabla
    with st.expander("📊 Ver Tabla Completa"):
        st.dataframe(df_liga[['Posicion', 'Equipo', 'PJ', 'Pts', 'GF', 'GC']], use_container_width=True)
    
    # Selección de equipos
    st.header("2️⃣ Selecciona el Partido")
    col1, col2 = st.columns(2)
    
    with col1:
        equipo_local = st.selectbox("🏠 Equipo Local", df_liga['Equipo'].tolist())
    
    with col2:
        equipos_visitante = [e for e in df_liga['Equipo'].tolist() if e != equipo_local]
        equipo_visitante = st.selectbox("✈️ Equipo Visitante", equipos_visitante)
    
    # Botón de análisis
    if st.button("🚀 ANALIZAR PARTIDO", type="primary", use_container_width=True):
        
        # Obtener datos de equipos
        local_team = df_liga[df_liga['Equipo'] == equipo_local].iloc[0]
        visitante_team = df_liga[df_liga['Equipo'] == equipo_visitante].iloc[0]
        
        st.markdown("---")
        st.header(f"📊 Análisis: {equipo_local} vs {equipo_visitante}")
        
        # Cargar historial
        with st.spinner("Analizando últimos 20 partidos de cada equipo..."):
            partidos_local = api.obtener_ultimos_20_partidos(equipo_local)
            partidos_visitante = api.obtener_ultimos_20_partidos(equipo_visitante)
            h2h = api.obtener_enfrentamientos_directos(equipo_local, equipo_visitante)
        
        # Calcular lambdas y análisis
        lambda_local, lambda_visitante, analisis = AnalizadorUltraProfundo.calcular_lambdas_y_factores(
            local_team, visitante_team, df_liga, partidos_local, partidos_visitante, h2h
        )
        
        # Mostrar parámetros de análisis
        st.subheader("🔍 Parámetros de Análisis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Lambda Local (λL)", f"{lambda_local:.2f}", 
                     help="Goles esperados del equipo local. Mayor valor = más probabilidad de anotar")
        with col2:
            st.metric("Lambda Visitante (λV)", f"{lambda_visitante:.2f}",
                     help="Goles esperados del equipo visitante")
        with col3:
            st.metric("Goles Esperados", f"{lambda_local + lambda_visitante:.2f}",
                     help="Total de goles esperados en el partido")
        
        # Forma de equipos
        st.subheader("📈 Forma Reciente (Últimos 20 Partidos)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**🏠 {equipo_local}**")
            forma_local = analisis['forma_local']
            detalles_local = analisis['detalles_local']
            
            st.progress(forma_local)
            st.caption(f"Forma: {forma_local*100:.1f}%")
            
            st.markdown(f"""
            - ✅ Victorias: **{detalles_local['victorias']}**
            - 🟰 Empates: **{detalles_local['empates']}**
            - ❌ Derrotas: **{detalles_local['derrotas']}**
            - ⚽ Goles a favor: **{detalles_local['gf']}** ({detalles_local['gf']/20:.1f}/partido)
            - 🥅 Goles en contra: **{detalles_local['gc']}** ({detalles_local['gc']/20:.1f}/partido)
            - 🔥 Racha (últimos 5): {detalles_local['racha_actual']}
            """)
        
        with col2:
            st.markdown(f"**✈️ {equipo_visitante}**")
            forma_visitante = analisis['forma_visitante']
            detalles_visitante = analisis['detalles_visitante']
            
            st.progress(forma_visitante)
            st.caption(f"Forma: {forma_visitante*100:.1f}%")
            
            st.markdown(f"""
            - ✅ Victorias: **{detalles_visitante['victorias']}**
            - 🟰 Empates: **{detalles_visitante['empates']}**
            - ❌ Derrotas: **{detalles_visitante['derrotas']}**
            - ⚽ Goles a favor: **{detalles_visitante['gf']}** ({detalles_visitante['gf']/20:.1f}/partido)
            - 🥅 Goles en contra: **{detalles_visitante['gc']}** ({detalles_visitante['gc']/20:.1f}/partido)
            - 🔥 Racha (últimos 5): {detalles_visitante['racha_actual']}
            """)
        
        # Botones para ver historial completo
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"📋 Ver 20 partidos de {equipo_local}"):
                st.markdown("### Últimos 20 partidos:")
                for i, p in enumerate(partidos_local[:20], 1):
                    resultado = "V" if (p['local'] == equipo_local and p['goles_local'] > p['goles_visitante']) or \
                                     (p['visitante'] == equipo_local and p['goles_visitante'] > p['goles_local']) else \
                               "E" if p['goles_local'] == p['goles_visitante'] else "D"
                    
                    emoji = "✅" if resultado == "V" else "🟰" if resultado == "E" else "❌"
                    
                    st.caption(f"{emoji} {i}. {p['local']} {p['goles_local']}-{p['goles_visitante']} {p['visitante']} | {p['competicion']}")
        
        with col2:
            if st.button(f"📋 Ver 20 partidos de {equipo_visitante}"):
                st.markdown("### Últimos 20 partidos:")
                for i, p in enumerate(partidos_visitante[:20], 1):
                    resultado = "V" if (p['local'] == equipo_visitante and p['goles_local'] > p['goles_visitante']) or \
                                     (p['visitante'] == equipo_visitante and p['goles_visitante'] > p['goles_local']) else \
                               "E" if p['goles_local'] == p['goles_visitante'] else "D"
                    
                    emoji = "✅" if resultado == "V" else "🟰" if resultado == "E" else "❌"
                    
                    st.caption(f"{emoji} {i}. {p['local']} {p['goles_local']}-{p['goles_visitante']} {p['visitante']} | {p['competicion']}")
        
        # H2H
        if h2h:
            st.subheader("🎯 Enfrentamientos Directos (H2H)")
            st.caption(f"Últimos {len(h2h)} enfrentamientos:")
            
            victorias_local_h2h = sum(1 for p in h2h if 
                (p['local'] == equipo_local and p['goles_local'] > p['goles_visitante']) or
                (p['visitante'] == equipo_local and p['goles_visitante'] > p['goles_local'])
            )
            empates_h2h = sum(1 for p in h2h if p['goles_local'] == p['goles_visitante'])
            victorias_visitante_h2h = len(h2h) - victorias_local_h2h - empates_h2h
            
            col1, col2, col3 = st.columns(3)
            col1.metric(f"Victorias {equipo_local}", victorias_local_h2h)
            col2.metric("Empates", empates_h2h)
            col3.metric(f"Victorias {equipo_visitante}", victorias_visitante_h2h)
            
            for p in h2h[:5]:
                st.caption(f"{p['local']} {p['goles_local']}-{p['goles_visitante']} {p['visitante']} | {p['competicion']}")
        
        # Calcular matriz y mercados
        matriz = AnalizadorUltraProfundo.matriz_probabilidades(lambda_local, lambda_visitante)
        mercados = AnalizadorUltraProfundo.calcular_mercados_extendidos(
            matriz, analisis['stats_local'], analisis['stats_visitante'], lambda_local, lambda_visitante
        )
        
        # Mostrar pronósticos
        st.markdown("---")
        st.header("🎯 Pronósticos y Probabilidades")
        
        # 1X2
        st.subheader("⚽ Resultado Final (1X2)")
        col1, col2, col3 = st.columns(3)
        
        probs_1x2 = mercados['1X2']
        
        with col1:
            st.metric(f"Victoria {equipo_local}", f"{probs_1x2['Local']*100:.1f}%")
        with col2:
            st.metric("Empate", f"{probs_1x2['Empate']*100:.1f}%")
        with col3:
            st.metric(f"Victoria {equipo_visitante}", f"{probs_1x2['Visitante']*100:.1f}%")
        
        # Over/Under
        st.subheader("📊 Over/Under Goles")
        ou = mercados['Over/Under']
        
        col1, col2 = st.columns(2)
        with col1:
            for key in ['Over 0.5', 'Over 1.5', 'Over 2.5']:
                st.metric(key, f"{ou[key]*100:.1f}%")
        
        with col2:
            for key in ['Over 3.5', 'Over 4.5']:
                st.metric(key, f"{ou[key]*100:.1f}%")
        
        # BTTS
        st.subheader("🎯 Ambos Equipos Anotan (BTTS)")
        btts = mercados['BTTS']
        col1, col2 = st.columns(2)
        col1.metric("BTTS Sí", f"{btts['Si']*100:.1f}%")
        col2.metric("BTTS No", f"{btts['No']*100:.1f}%")
        
        # Handicap
        st.subheader("⚖️ Handicap Asiático")
        hcp = mercados['Handicap']
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Local -0.5", f"{hcp['Local -0.5']*100:.1f}%")
            st.metric("Local -1.5", f"{hcp['Local -1.5']*100:.1f}%")
        with col2:
            st.metric("Visitante +0.5", f"{hcp['Visitante +0.5']*100:.1f}%")
            st.metric("Visitante +1.5", f"{hcp['Visitante +1.5']*100:.1f}%")
        
        # Primera Mitad
        st.subheader("⏱️ Primera Mitad")
        pm = mercados['Primera_Mitad']
        col1, col2, col3 = st.columns(3)
        col1.metric("Gol en 1H", f"{pm['Gol en 1H']*100:.1f}%")
        col2.metric("Más de 0.5 goles 1H", f"{pm['Más de 0.5 goles 1H']*100:.1f}%")
        col3.metric("Más de 1.5 goles 1H", f"{pm['Más de 1.5 goles 1H']*100:.1f}%")
        
        # Resultado exacto
        st.subheader("🎲 Resultados Más Probables")
        top_res = mercados['Top_Resultados']
        
        cols = st.columns(5)
        for i, res in enumerate(top_res):
            with cols[i]:
                st.metric(res['Marcador'], f"{res['Probabilidad']*100:.1f}%")
        
        # MEJORES OPORTUNIDADES
        st.markdown("---")
        st.header("💎 MEJORES OPORTUNIDADES DE APUESTA")
        st.caption("*Apuestas con alto Expected Value (EV) y probabilidades favorables*")
        
        oportunidades = CalculadorValor.encontrar_valor(mercados, margen_casa=0.05)
        
        if not oportunidades:
            st.warning("⚠️ No se encontraron oportunidades con valor positivo significativo en este momento.")
        else:
            # Top 5 oportunidades
            st.subheader("🏆 Top 5 Apuestas con Mejor Valor")
            
            for i, op in enumerate(oportunidades[:5], 1):
                with st.expander(f"#{i} - {op['Categoria']}: {op['Mercado']} | EV: +{op['EV']*100:.1f}% {op['Confianza']}"):
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Probabilidad", f"{op['Probabilidad']*100:.1f}%")
                    with col2:
                        st.metric("Cuota Justa", f"{op['Cuota_Justa']:.2f}")
                    with col3:
                        st.metric("Expected Value", f"+{op['EV']*100:.1f}%")
                    
                    st.caption(f"**Cuota mínima esperada en el mercado:** {op['Cuota_Min_Esperada']:.2f}")
                    st.caption(f"*Si encuentras una cuota igual o mayor a {op['Cuota_Min_Esperada']:.2f}, hay valor en esta apuesta*")
                    
                    # Generar argumentos
                    if op['Categoria'] == '1X2':
                        resultado = op['Mercado']
                        argumentos = GeneradorArgumentos.generar_argumento_1x2(
                            resultado, op['Probabilidad'], local_team, visitante_team, analisis
                        )
                    elif op['Categoria'] == 'Over/Under':
                        argumentos = GeneradorArgumentos.generar_argumento_over_under(
                            op['Mercado'], op['Probabilidad'], analisis
                        )
                    elif op['Categoria'] == 'BTTS':
                        si_no = op['Mercado']
                        argumentos = GeneradorArgumentos.generar_argumento_btts(
                            si_no, op['Probabilidad'], analisis
                        )
                    elif op['Categoria'] == 'Handicap':
                        argumentos = GeneradorArgumentos.generar_argumento_handicap(
                            op['Mercado'], op['Probabilidad'], analisis, local_team, visitante_team
                        )
                    elif op['Categoria'] == 'Primera_Mitad':
                        argumentos = GeneradorArgumentos.generar_argumento_primera_mitad(
                            op['Mercado'], op['Probabilidad'], analisis
                        )
                    else:
                        argumentos = []
                    
                    if argumentos:
                        st.markdown("**🔍 Argumentos:**")
                        for arg in argumentos:
                            st.markdown(f"- {arg}")
                    
                    # Veredicto final
                    st.markdown("---")
                    st.markdown("### ⚖️ VEREDICTO FINAL")
                    
                    confianza_nivel = ""
                    if op['Confianza'] == "⭐⭐⭐ MUY ALTA":
                        confianza_nivel = "🟢 **APUESTA RECOMENDADA** - Alta probabilidad y excelente valor matemático"
                    elif op['Confianza'] == "⭐⭐ ALTA":
                        confianza_nivel = "🟡 **BUENA OPORTUNIDAD** - Probabilidad sólida con buen valor"
                    elif op['Confianza'] == "⭐ MEDIA":
                        confianza_nivel = "🟠 **CONSIDERAR** - Valor presente pero requiere análisis adicional"
                    else:
                        confianza_nivel = "🔴 **PRECAUCIÓN** - Valor marginal, solo para apostadores experimentados"
                    
                    st.markdown(confianza_nivel)
                    
                    if op['EV'] > 0.15:
                        st.success("✅ Esta apuesta tiene un valor excepcional (+15% EV o más). Muy recomendada si encuentras la cuota adecuada.")
                    elif op['EV'] > 0.08:
                        st.info("ℹ️ Valor positivo significativo. Buena oportunidad para considerar.")
                    else:
                        st.warning("⚠️ Valor positivo pero moderado. Evalúa las cuotas del mercado antes de apostar.")
        
        # APUESTAS DE ALTA PROBABILIDAD
        st.markdown("---")
        st.header("✅ APUESTAS DE ALTA PROBABILIDAD")
        st.caption("*Apuestas con más del 60% de probabilidad de éxito*")
        
        apuestas_seguras = []
        
        for categoria, opciones in mercados.items():
            if categoria in ['Top_Resultados', 'Resultado_Exacto', 'Prob_Exacto']:
                continue
            
            if isinstance(opciones, dict):
                for mercado, prob in opciones.items():
                    if prob > 0.60:
                        apuestas_seguras.append({
                            'Categoria': categoria,
                            'Mercado': mercado,
                            'Probabilidad': prob,
                            'Cuota_Justa': CalculadorValor.calcular_cuota_justa(prob)
                        })
        
        apuestas_seguras.sort(key=lambda x: x['Probabilidad'], reverse=True)
        
        if not apuestas_seguras:
            st.info("ℹ️ No hay apuestas con probabilidad superior al 60% en este partido.")
        else:
            for i, apuesta in enumerate(apuestas_seguras[:8], 1):
                with st.expander(f"#{i} - {apuesta['Categoria']}: {apuesta['Mercado']} | Probabilidad: {apuesta['Probabilidad']*100:.1f}%"):
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Probabilidad de Éxito", f"{apuesta['Probabilidad']*100:.1f}%")
                    with col2:
                        st.metric("Cuota Justa", f"{apuesta['Cuota_Justa']:.2f}")
                    
                    st.caption(f"**Cuota esperada en el mercado:** ~{apuesta['Cuota_Justa']*0.95:.2f}")
                    
                    # Generar argumentos
                    if apuesta['Categoria'] == '1X2':
                        argumentos = GeneradorArgumentos.generar_argumento_1x2(
                            apuesta['Mercado'], apuesta['Probabilidad'], local_team, visitante_team, analisis
                        )
                    elif apuesta['Categoria'] == 'Over/Under':
                        argumentos = GeneradorArgumentos.generar_argumento_over_under(
                            apuesta['Mercado'], apuesta['Probabilidad'], analisis
                        )
                    elif apuesta['Categoria'] == 'BTTS':
                        argumentos = GeneradorArgumentos.generar_argumento_btts(
                            apuesta['Mercado'], apuesta['Probabilidad'], analisis
                        )
                    elif apuesta['Categoria'] == 'Handicap':
                        argumentos = GeneradorArgumentos.generar_argumento_handicap(
                            apuesta['Mercado'], apuesta['Probabilidad'], analisis, local_team, visitante_team
                        )
                    elif apuesta['Categoria'] == 'Primera_Mitad':
                        argumentos = GeneradorArgumentos.generar_argumento_primera_mitad(
                            apuesta['Mercado'], apuesta['Probabilidad'], analisis
                        )
                    else:
                        argumentos = []
                    
                    if argumentos:
                        st.markdown("**🔍 Argumentos:**")
                        for arg in argumentos:
                            st.markdown(f"- {arg}")
                    
                    # Veredicto
                    st.markdown("---")
                    st.markdown("### ⚖️ VEREDICTO")
                    
                    if apuesta['Probabilidad'] > 0.80:
                        st.success("🟢 **MUY PROBABLE** - Altísima probabilidad de éxito, pero cuotas bajas esperadas.")
                    elif apuesta['Probabilidad'] > 0.70:
                        st.info("🔵 **ALTA PROBABILIDAD** - Buena opción para apuestas conservadoras.")
                    else:
                        st.warning("🟡 **PROBABILIDAD MODERADA-ALTA** - Equilibrio entre probabilidad y cuota.")
        
        # ANÁLISIS ESTADÍSTICO DETALLADO
        st.markdown("---")
        st.header("📈 Análisis Estadístico Detallado")
        
        tab1, tab2, tab3 = st.tabs(["📊 Rendimiento Local/Visitante", "🎯 Estadísticas Clave", "🔥 Tendencias"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### 🏠 {equipo_local} (Local)")
                stats_local = analisis['stats_local']
                
                if stats_local['local_pj'] > 0:
                    st.markdown(f"""
                    **Partidos de Local:** {stats_local['local_pj']}
                    - ✅ Victorias: {stats_local['local_victorias']} ({stats_local['local_victorias']/stats_local['local_pj']*100:.0f}%)
                    - ⚽ Goles a favor: {stats_local['local_gf']} ({stats_local['local_gf']/stats_local['local_pj']:.1f}/partido)
                    - 🥅 Goles en contra: {stats_local['local_gc']} ({stats_local['local_gc']/stats_local['local_pj']:.1f}/partido)
                    - 📊 Diferencia: {(stats_local['local_gf'] - stats_local['local_gc'])/stats_local['local_pj']:.1f} goles/partido
                    """)
                else:
                    st.info("Sin datos suficientes de partidos como local")
            
            with col2:
                st.markdown(f"### ✈️ {equipo_visitante} (Visitante)")
                stats_visitante = analisis['stats_visitante']
                
                if stats_visitante['visitante_pj'] > 0:
                    st.markdown(f"""
                    **Partidos de Visitante:** {stats_visitante['visitante_pj']}
                    - ✅ Victorias: {stats_visitante['visitante_victorias']} ({stats_visitante['visitante_victorias']/stats_visitante['visitante_pj']*100:.0f}%)
                    - ⚽ Goles a favor: {stats_visitante['visitante_gf']} ({stats_visitante['visitante_gf']/stats_visitante['visitante_pj']:.1f}/partido)
                    - 🥅 Goles en contra: {stats_visitante['visitante_gc']} ({stats_visitante['visitante_gc']/stats_visitante['visitante_pj']:.1f}/partido)
                    - 📊 Diferencia: {(stats_visitante['visitante_gf'] - stats_visitante['visitante_gc'])/stats_visitante['visitante_pj']:.1f} goles/partido
                    """)
                else:
                    st.info("Sin datos suficientes de partidos como visitante")
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### {equipo_local}")
                st.metric("Over 2.5 en últimos 20", f"{analisis['stats_local']['partidos_over25']/20*100:.0f}%")
                st.metric("BTTS en últimos 20", f"{analisis['stats_local']['partidos_btts']/20*100:.0f}%")
                st.metric("Goles en 1H (promedio)", f"{analisis['stats_local']['goles_1h_favor']/20:.1f}")
                st.metric("% partidos con gol en 1H", f"{analisis['stats_local']['partidos_gol_1h']/20*100:.0f}%")
            
            with col2:
                st.markdown(f"### {equipo_visitante}")
                st.metric("Over 2.5 en últimos 20", f"{analisis['stats_visitante']['partidos_over25']/20*100:.0f}%")
                st.metric("BTTS en últimos 20", f"{analisis['stats_visitante']['partidos_btts']/20*100:.0f}%")
                st.metric("Goles en 1H (promedio)", f"{analisis['stats_visitante']['goles_1h_favor']/20:.1f}")
                st.metric("% partidos con gol en 1H", f"{analisis['stats_visitante']['partidos_gol_1h']/20*100:.0f}%")
        
        with tab3:
            st.subheader("🔥 Tendencias Detectadas")
            
            tendencias = []
            
            # Over 2.5
            over25_combinado = (analisis['stats_local']['partidos_over25'] + analisis['stats_visitante']['partidos_over25']) / 40
            if over25_combinado > 0.6:
                tendencias.append("🔥 **ALTA TENDENCIA A OVER 2.5** - Ambos equipos involucrados en partidos con muchos goles")
            elif over25_combinado < 0.35:
                tendencias.append("❄️ **PARTIDOS CERRADOS** - Tendencia a pocos goles en ambos equipos")
            
            # BTTS
            btts_combinado = (analisis['stats_local']['partidos_btts'] + analisis['stats_visitante']['partidos_btts']) / 40
            if btts_combinado > 0.55:
                tendencias.append("⚽⚽ **ALTA PROBABILIDAD BTTS** - Ambos equipos suelen anotar y recibir goles")
            elif btts_combinado < 0.35:
                tendencias.append("🛡️ **DEFENSAS SÓLIDAS** - Uno o ambos equipos mantienen portería en cero frecuentemente")
            
            # Primera mitad
            gol_1h_combinado = (analisis['stats_local']['partidos_gol_1h'] + analisis['stats_visitante']['partidos_gol_1h']) / 40
            if gol_1h_combinado > 0.65:
                tendencias.append("⚡ **INICIOS EXPLOSIVOS** - Alta probabilidad de goles en primera mitad")
            
            # Forma
            if analisis['forma_local'] > 0.7 and analisis['forma_visitante'] < 0.4:
                tendencias.append(f"📈 **{equipo_local} EN RACHA GANADORA** vs equipo en baja forma")
            elif analisis['forma_visitante'] > 0.7 and analisis['forma_local'] < 0.4:
                tendencias.append(f"📈 **{equipo_visitante} EN RACHA GANADORA** vs equipo en baja forma")
            elif abs(analisis['forma_local'] - analisis['forma_visitante']) < 0.1:
                tendencias.append("⚖️ **EQUIPOS MUY IGUALADOS** - Forma similar, partido abierto")
            
            # Diferencia de tabla
            if abs(local_team['Posicion'] - visitante_team['Posicion']) > 10:
                if local_team['Posicion'] < visitante_team['Posicion']:
                    tendencias.append(f"⭐ **{equipo_local} FAVORITO CLARO** - Gran diferencia en la tabla")
                else:
                    tendencias.append(f"⭐ **{equipo_visitante} FAVORITO CLARO** - Gran diferencia en la tabla")
            
            if tendencias:
                for tend in tendencias:
                    st.markdown(f"- {tend}")
            else:
                st.info("No se detectaron tendencias significativas")
        
        # DISCLAIMER
        st.markdown("---")
        st.warning("""
        ⚠️ **ADVERTENCIA IMPORTANTE:**
        
        - Este sistema utiliza análisis estadístico y modelos matemáticos (Distribución de Poisson).
        - Las probabilidades son estimaciones basadas en datos históricos y NO garantizan resultados.
        - El Expected Value (EV) es una herramienta de análisis, no una certeza.
        - Siempre apuesta de forma responsable y dentro de tus posibilidades.
        - Las apuestas deportivas conllevan riesgo. Nunca apuestes dinero que no puedas permitirte perder.
        
        📊 **Usa este sistema como una herramienta de apoyo para tus decisiones, no como única fuente.**
        """)

if __name__ == "__main__":
    main()
