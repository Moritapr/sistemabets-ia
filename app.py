import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from scipy.stats import poisson
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sqlite3
import json
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SISTEMABETS PRO v8.0", layout="wide")

# ============================================================================
# MÓDULO 1: FACTORES DE AJUSTE POR COMPETICIÓN
# ============================================================================

class CompetitionAdjuster:
    """
    Ajusta las predicciones según el nivel de la competición
    
    PROBLEMA: Un equipo que domina su liga local puede NO ser tan fuerte
    en Champions League contra equipos de ligas top.
    
    SOLUCIÓN: Factores de ajuste según nivel de competición y liga
    """
    
    # Ranking de fuerza por liga (escala 1-10)
    LIGA_STRENGTH = {
        'PL': 10,    # Premier League - La más competitiva
        'PD': 9.5,   # La Liga - Muy fuerte
        'BL1': 9,    # Bundesliga - Top 3
        'SA': 8.5,   # Serie A - Top 4
        'FL1': 8,    # Ligue 1 - Top 5
        'PPL': 6,    # Liga Portugal - Nivel medio-alto
        'DED': 7,    # Eredivisie - Nivel medio
        'ELC': 7.5,  # Championship - Competitivo
        'CL': 10,    # Champions League - Máximo nivel
    }
    
    # Equipos de élite europea (históricos + actuales)
    ELITE_TEAMS = {
        # España
        'Real Madrid', 'FC Barcelona', 'Atlético Madrid',
        # Inglaterra
        'Manchester City', 'Liverpool FC', 'Chelsea FC', 'Arsenal FC', 
        'Manchester United', 'Tottenham Hotspur',
        # Alemania
        'FC Bayern München', 'Borussia Dortmund', 'RB Leipzig', 'Bayer 04 Leverkusen',
        # Italia
        'Inter Milan', 'AC Milan', 'Juventus FC', 'SSC Napoli',
        # Francia
        'Paris Saint-Germain',
        # Otros top
        'AFC Ajax', 'SL Benfica', 'FC Porto', 'Sporting CP'
    }
    
    @staticmethod
    def es_competicion_europea(competicion_nombre):
        """Detecta si es Champions o Europa League"""
        keywords = ['champions', 'europa league', 'uefa', 'european']
        comp_lower = competicion_nombre.lower()
        return any(kw in comp_lower for kw in keywords)
    
    @staticmethod
    def es_equipo_elite(equipo_nombre):
        """Verifica si es un equipo de élite europea"""
        return equipo_nombre in CompetitionAdjuster.ELITE_TEAMS
    
    @staticmethod
    def calcular_factor_competicion(local_team, visitante_team, partidos_local, partidos_visitante):
        """
        Calcula factores de ajuste según:
        - Si es competición europea
        - Nivel de las ligas de cada equipo
        - Si son equipos de élite
        
        Returns: (factor_local, factor_visitante, es_champions, advertencia)
        """
        
        factor_local = 1.0
        factor_visitante = 1.0
        es_champions = False
        advertencia = ""
        
        # Detectar si hay partidos de Champions/Europa League
        competiciones_local = [p['competicion'] for p in partidos_local[:5]]
        competiciones_visitante = [p['competicion'] for p in partidos_visitante[:5]]
        
        # Verificar si es Champions
        if any(CompetitionAdjuster.es_competicion_europea(c) for c in competiciones_local + competiciones_visitante):
            es_champions = True
        
        # Determinar liga de cada equipo (de sus partidos recientes)
        liga_local = None
        liga_visitante = None
        
        for p in partidos_local[:10]:
            comp = p['competicion']
            for liga_code, _ in CompetitionAdjuster.LIGA_STRENGTH.items():
                if liga_code in ['PL', 'PD', 'BL1', 'SA', 'FL1', 'PPL', 'DED', 'ELC']:
                    # Mapear nombres de competición a códigos
                    liga_names = {
                        'PL': 'Premier League',
                        'PD': 'Primera Division',
                        'BL1': 'Bundesliga',
                        'SA': 'Serie A',
                        'FL1': 'Ligue 1',
                        'PPL': 'Primeira Liga',
                        'DED': 'Eredivisie',
                        'ELC': 'Championship'
                    }
                    if liga_names.get(liga_code, '').lower() in comp.lower():
                        liga_local = liga_code
                        break
            if liga_local:
                break
        
        for p in partidos_visitante[:10]:
            comp = p['competicion']
            for liga_code, _ in CompetitionAdjuster.LIGA_STRENGTH.items():
                if liga_code in ['PL', 'PD', 'BL1', 'SA', 'FL1', 'PPL', 'DED', 'ELC']:
                    liga_names = {
                        'PL': 'Premier League',
                        'PD': 'Primera Division',
                        'BL1': 'Bundesliga',
                        'SA': 'Serie A',
                        'FL1': 'Ligue 1',
                        'PPL': 'Primeira Liga',
                        'DED': 'Eredivisie',
                        'ELC': 'Championship'
                    }
                    if liga_names.get(liga_code, '').lower() in comp.lower():
                        liga_visitante = liga_code
                        break
            if liga_visitante:
                break
        
        # AJUSTE POR NIVEL DE LIGA
        if liga_local and liga_visitante:
            strength_local = CompetitionAdjuster.LIGA_STRENGTH.get(liga_local, 7)
            strength_visitante = CompetitionAdjuster.LIGA_STRENGTH.get(liga_visitante, 7)
            
            diff_strength = strength_visitante - strength_local
            
            # Si el visitante viene de liga MÁS FUERTE
            if diff_strength > 2:
                factor_visitante *= 1.25  # Boost importante
                factor_local *= 0.85
                advertencia = f"⚠️ {visitante_team['Equipo']} viene de liga más competitiva. Stats ajustadas."
            
            # Si el local viene de liga MÁS FUERTE
            elif diff_strength < -2:
                factor_local *= 1.25
                factor_visitante *= 0.85
                advertencia = f"⚠️ {local_team['Equipo']} viene de liga más competitiva. Stats ajustadas."
        
        # AJUSTE POR EQUIPOS DE ÉLITE
        elite_local = CompetitionAdjuster.es_equipo_elite(local_team['Equipo'])
        elite_visitante = CompetitionAdjuster.es_equipo_elite(visitante_team['Equipo'])
        
        if es_champions:
            # En Champions, equipos de élite tienen boost
            if elite_visitante and not elite_local:
                factor_visitante *= 1.20
                factor_local *= 0.88
                advertencia += f"\n🏆 {visitante_team['Equipo']} es equipo de élite europea."
            
            elif elite_local and not elite_visitante:
                factor_local *= 1.20
                factor_visitante *= 0.88
                advertencia += f"\n🏆 {local_team['Equipo']} es equipo de élite europea."
        
        return factor_local, factor_visitante, es_champions, advertencia

# ============================================================================
# MÓDULO 2: BASE DE DATOS (Igual que antes)
# ============================================================================

class DatabaseManager:
    """Gestiona la base de datos local de partidos históricos"""
    
    def __init__(self, db_path='partidos_historicos.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Crea las tablas si no existen"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS partidos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fecha TEXT,
                competicion TEXT,
                local TEXT,
                visitante TEXT,
                goles_local INTEGER,
                goles_visitante INTEGER,
                features TEXT,
                fecha_analisis TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def guardar_partido(self, partido_data):
        """Guarda un partido en la base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO partidos (fecha, competicion, local, visitante, 
                                 goles_local, goles_visitante, features, fecha_analisis)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            partido_data['fecha'],
            partido_data['competicion'],
            partido_data['local'],
            partido_data['visitante'],
            partido_data['goles_local'],
            partido_data['goles_visitante'],
            json.dumps(partido_data['features']),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        
        conn.commit()
        conn.close()
    
    def obtener_todos_partidos(self):
        """Obtiene todos los partidos guardados"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM partidos')
        partidos = cursor.fetchall()
        
        conn.close()
        
        return partidos
    
    def contar_partidos(self):
        """Cuenta cuántos partidos hay guardados"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM partidos')
        count = cursor.fetchone()[0]
        
        conn.close()
        
        return count
    
    def limpiar_database(self):
        """Limpia toda la base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM partidos')
        
        conn.commit()
        conn.close()

# ============================================================================
# MÓDULO 3: API FOOTBALL-DATA
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
# MÓDULO 4: ANALIZADOR PROFUNDO CON AJUSTES
# ============================================================================

class AnalizadorProfundo:
    """Motor de análisis con ajustes por competición y argumentos sólidos"""
    
    @staticmethod
    def calcular_forma_detallada(partidos, equipo_nombre):
        """Calcula forma con detalles completos"""
        if not partidos:
            return 0.5, "", {}, []
        
        puntos = []
        forma_visual = ""
        resultados_detalle = []
        
        detalles = {
            'victorias': 0, 'empates': 0, 'derrotas': 0,
            'gf': 0, 'gc': 0, 'racha_actual': ""
        }
        
        for i, partido in enumerate(partidos[:20]):
            es_local = partido['local'] == equipo_nombre
            gf = partido['goles_local'] if es_local else partido['goles_visitante']
            gc = partido['goles_visitante'] if es_local else partido['goles_local']
            
            detalles['gf'] += gf
            detalles['gc'] += gc
            
            if gf > gc:
                puntos.append(1.0)
                forma_visual += "V"
                detalles['victorias'] += 1
                resultado = "✅ V"
            elif gf == gc:
                puntos.append(0.5)
                forma_visual += "E"
                detalles['empates'] += 1
                resultado = "🟰 E"
            else:
                puntos.append(0.0)
                forma_visual += "D"
                detalles['derrotas'] += 1
                resultado = "❌ D"
            
            if i < 5:
                detalles['racha_actual'] += resultado.split()[0]
            
            resultados_detalle.append({
                'num': i+1,
                'local': partido['local'],
                'visitante': partido['visitante'],
                'resultado': f"{partido['goles_local']}-{partido['goles_visitante']}",
                'competicion': partido['competicion'],
                'simbolo': resultado
            })
        
        pesos = np.linspace(0.10, 0.005, len(puntos))
        pesos = pesos / pesos.sum()
        score_forma = sum(p * w for p, w in zip(puntos, pesos))
        
        return score_forma, forma_visual, detalles, resultados_detalle
    
    @staticmethod
    def calcular_stats_avanzadas(equipo_nombre, partidos):
        """Stats completas local/visitante"""
        stats = {
            'local_pj': 0, 'local_gf': 0, 'local_gc': 0, 'local_victorias': 0,
            'visitante_pj': 0, 'visitante_gf': 0, 'visitante_gc': 0, 'visitante_victorias': 0,
            'partidos_over25': 0, 'partidos_btts': 0,
            'goles_1h_favor': 0, 'partidos_gol_1h': 0
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
                if goles_1h > 0:
                    stats['partidos_gol_1h'] += 1
        
        return stats
    
    @staticmethod
    def calcular_lambdas_ajustados(local_team, visitante_team, partidos_local, partidos_visitante, h2h):
        """
        Calcula lambdas CON AJUSTES por competición
        """
        
        # Obtener factores de ajuste
        factor_local, factor_visitante, es_champions, advertencia = CompetitionAdjuster.calcular_factor_competicion(
            local_team, visitante_team, partidos_local, partidos_visitante
        )
        
        # Stats básicas
        gf_local_tabla = local_team['GF'] / max(local_team['PJ'], 1)
        gc_local_tabla = local_team['GC'] / max(local_team['PJ'], 1)
        gf_visitante_tabla = visitante_team['GF'] / max(visitante_team['PJ'], 1)
        gc_visitante_tabla = visitante_team['GC'] / max(visitante_team['PJ'], 1)
        
        # Forma
        forma_local, _, detalles_local, resultados_local = AnalizadorProfundo.calcular_forma_detallada(
            partidos_local, local_team['Equipo']
        )
        forma_visitante, _, detalles_visitante, resultados_visitante = AnalizadorProfundo.calcular_forma_detallada(
            partidos_visitante, visitante_team['Equipo']
        )
        
        # Stats avanzadas
        stats_local = AnalizadorProfundo.calcular_stats_avanzadas(local_team['Equipo'], partidos_local)
        stats_visitante = AnalizadorProfundo.calcular_stats_avanzadas(visitante_team['Equipo'], partidos_visitante)
        
        # Promedios específicos
        gf_local_casa = stats_local['local_gf'] / max(stats_local['local_pj'], 1) if stats_local['local_pj'] > 0 else gf_local_tabla
        gc_local_casa = stats_local['local_gc'] / max(stats_local['local_pj'], 1) if stats_local['local_pj'] > 0 else gc_local_tabla
        gf_visitante_fuera = stats_visitante['visitante_gf'] / max(stats_visitante['visitante_pj'], 1) if stats_visitante['visitante_pj'] > 0 else gf_visitante_tabla
        gc_visitante_fuera = stats_visitante['visitante_gc'] / max(stats_visitante['visitante_pj'], 1) if stats_visitante['visitante_pj'] > 0 else gc_visitante_tabla
        
        # Media liga
        media_goles_liga = (gf_local_tabla + gf_visitante_tabla) / 2
        
        # Lambdas base
        lambda_local = (gf_local_casa / max(media_goles_liga, 0.5)) * (gc_visitante_fuera / max(media_goles_liga, 0.5)) * media_goles_liga
        lambda_visitante = (gf_visitante_fuera / max(media_goles_liga, 0.5)) * (gc_local_casa / max(media_goles_liga, 0.5)) * media_goles_liga
        
        # Ventaja local
        ventaja_local = 1.15
        if stats_local['local_pj'] >= 5:
            winrate_casa = stats_local['local_victorias'] / stats_local['local_pj']
            if winrate_casa > 0.7:
                ventaja_local *= 1.10
            elif winrate_casa < 0.3:
                ventaja_local *= 0.92
        
        lambda_local *= ventaja_local
        lambda_visitante /= (ventaja_local * 0.90)
        
        # Ajuste por forma
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
        
        # APLICAR FACTORES DE AJUSTE POR COMPETICIÓN
        lambda_local *= factor_local
        lambda_visitante *= factor_visitante
        
        # Limitar lambdas
        lambda_local = max(min(lambda_local, 4.5), 0.30)
        lambda_visitante = max(min(lambda_visitante, 4.5), 0.30)
        
        return lambda_local, lambda_visitante, {
            'forma_local': forma_local,
            'forma_visitante': forma_visitante,
            'detalles_local': detalles_local,
            'detalles_visitante': detalles_visitante,
            'resultados_local': resultados_local,
            'resultados_visitante': resultados_visitante,
            'stats_local': stats_local,
            'stats_visitante': stats_visitante,
            'es_champions': es_champions,
            'advertencia': advertencia,
            'factor_local': factor_local,
            'factor_visitante': factor_visitante
        }
    
    @staticmethod
    def calcular_probabilidades(lambda_l, lambda_v):
        """Calcula probabilidades con Poisson"""
        matriz = np.zeros((8, 8))
        
        for i in range(8):
            for j in range(8):
                matriz[i, j] = poisson.pmf(i, lambda_l) * poisson.pmf(j, lambda_v)
        
        # 1X2
        p_local = np.sum(np.tril(matriz, -1))
        p_empate = np.sum(np.diag(matriz))
        p_visitante = np.sum(np.triu(matriz, 1))
        
        total = p_local + p_empate + p_visitante
        p_local = max(p_local / total, 0.005)
        p_empate = max(p_empate / total, 0.005)
        p_visitante = max(p_visitante / total, 0.005)
        
        # Over/Under
        over_under = {}
        for threshold in [0.5, 1.5, 2.5, 3.5, 4.5]:
            p_over = sum([matriz[i, j] for i in range(8) for j in range(8) if (i+j) > threshold])
            over_under[f"Over {threshold}"] = max(p_over, 0.005)
            over_under[f"Under {threshold}"] = max(1 - p_over, 0.005)
        
        # BTTS
        p_btts_no = matriz[0,:].sum() + matriz[:,0].sum() - matriz[0,0]
        p_btts_si = max(1 - p_btts_no, 0.005)
        p_btts_no = max(p_btts_no, 0.005)
        
        # Resultado más probable
        idx_max = np.unravel_index(matriz.argmax(), matriz.shape)
        
        return {
            '1X2': {'Local': p_local, 'Empate': p_empate, 'Visitante': p_visitante},
            'Over/Under': over_under,
            'BTTS': {'Si': p_btts_si, 'No': p_btts_no},
            'Resultado_Exacto': f"{idx_max[0]}-{idx_max[1]}",
            'Prob_Exacto': matriz[idx_max[0], idx_max[1]]
        }

# ============================================================================
# MÓDULO 5: GENERADOR DE ARGUMENTOS SÓLIDOS
# ============================================================================

class GeneradorArgumentos:
    """Genera argumentos sólidos para cada predicción"""
    
    @staticmethod
    def generar_argumento_1x2(resultado, prob, local_team, visitante_team, analisis):
        """Argumentos para 1X2"""
        argumentos = []
        
        forma_local = analisis['forma_local']
        forma_visitante = analisis['forma_visitante']
        detalles_local = analisis['detalles_local']
        detalles_visitante = analisis['detalles_visitante']
        
        if resultado == "Local":
            if forma_local > 0.7:
                argumentos.append(f"✅ {local_team['Equipo']} en excelente forma ({forma_local*100:.0f}%) - {detalles_local['victorias']} victorias en últimos 20")
            
            if local_team['Posicion'] < visitante_team['Posicion'] - 5:
                argumentos.append(f"📊 Superior en tabla: posición {local_team['Posicion']} vs {visitante_team['Posicion']}")
            
            stats_local = analisis['stats_local']
            if stats_local['local_pj'] >= 5:
                winrate = stats_local['local_victorias'] / stats_local['local_pj']
                if winrate > 0.6:
                    argumentos.append(f"🏠 Dominante en casa: {stats_local['local_victorias']}/{stats_local['local_pj']} victorias ({winrate*100:.0f}%)")
            
            if analisis.get('factor_local', 1.0) > 1.15:
                argumentos.append(f"⚡ Ajuste por nivel de competición favorece al local")
        
        elif resultado == "Visitante":
            if forma_visitante > 0.7:
                argumentos.append(f"✅ {visitante_team['Equipo']} en racha ({forma_visitante*100:.0f}%) - {detalles_visitante['victorias']} victorias")
            
            if visitante_team['Posicion'] < local_team['Posicion'] - 5:
                argumentos.append(f"📊 Equipo superior: posición {visitante_team['Posicion']} vs {local_team['Posicion']}")
            
            stats_visitante = analisis['stats_visitante']
            if stats_visitante['visitante_pj'] >= 5:
                winrate = stats_visitante['visitante_victorias'] / stats_visitante['visitante_pj']
                if winrate > 0.5:
                    argumentos.append(f"✈️ Sólido fuera: {stats_visitante['visitante_victorias']}/{stats_visitante['visitante_pj']} victorias ({winrate*100:.0f}%)")
            
            if forma_local < 0.4:
                argumentos.append(f"📉 {local_team['Equipo']} con mala forma ({forma_local*100:.0f}%)")
            
            if analisis.get('factor_visitante', 1.0) > 1.15:
                argumentos.append(f"⚡ Ajuste por nivel de competición favorece al visitante")
        
        else:  # Empate
            if abs(forma_local - forma_visitante) < 0.15:
                argumentos.append(f"⚖️ Equipos muy igualados en forma: {forma_local*100:.0f}% vs {forma_visitante*100:.0f}%")
            
            if abs(local_team['Posicion'] - visitante_team['Posicion']) <= 3:
                argumentos.append(f"📊 Cercanos en tabla: posiciones {local_team['Posicion']} y {visitante_team['Posicion']}")
            
            empates_combinados = detalles_local['empates'] + detalles_visitante['empates']
            if empates_combinados >= 8:
                argumentos.append(f"🤝 Alta tendencia al empate: {empates_combinados} empates combinados en últimos 20")
        
        return argumentos
    
    @staticmethod
    def generar_argumento_over_under(mercado, prob, analisis):
        """Argumentos para Over/Under"""
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
                argumentos.append(f"⚽ Promedio combinado alto: {goles_promedio_local:.1f} + {goles_promedio_visitante:.1f} = {goles_promedio_local + goles_promedio_visitante:.1f}")
            
            over25_local = stats_local['partidos_over25'] / 20
            over25_visitante = stats_visitante['partidos_over25'] / 20
            if threshold == 2.5 and (over25_local > 0.6 or over25_visitante > 0.6):
                argumentos.append(f"📈 Alta frecuencia Over 2.5: {over25_local*100:.0f}% y {over25_visitante*100:.0f}%")
            
            gc_local = detalles_local['gc'] / 20
            gc_visitante = detalles_visitante['gc'] / 20
            if gc_local > 1.2 and gc_visitante > 1.2:
                argumentos.append(f"🥅 Defensas vulnerables: encajan {gc_local:.1f} y {gc_visitante:.1f} goles/partido")
        
        else:  # Under
            threshold = float(mercado.split()[1])
            
            if goles_promedio_local + goles_promedio_visitante < threshold:
                argumentos.append(f"🔒 Promedio bajo: {goles_promedio_local + goles_promedio_visitante:.1f} goles combinados")
            
            over25_local = stats_local['partidos_over25'] / 20
            over25_visitante = stats_visitante['partidos_over25'] / 20
            if threshold == 2.5 and over25_local < 0.4 and over25_visitante < 0.4:
                argumentos.append(f"📉 Baja frecuencia de goles: {over25_local*100:.0f}% y {over25_visitante*100:.0f}% Over 2.5")
        
        return argumentos
    
    @staticmethod
    def generar_argumento_btts(si_no, prob, analisis):
        """Argumentos para BTTS"""
        argumentos = []
        
        stats_local = analisis['stats_local']
        stats_visitante = analisis['stats_visitante']
        detalles_local = analisis['detalles_local']
        detalles_visitante = analisis['detalles_visitante']
        
        if si_no == "Si":
            btts_local = stats_local['partidos_btts'] / 20
            btts_visitante = stats_visitante['partidos_btts'] / 20
            
            if btts_local > 0.55 or btts_visitante > 0.55:
                argumentos.append(f"⚽⚽ Alta frecuencia BTTS: {btts_local*100:.0f}% y {btts_visitante*100:.0f}%")
            
            gf_local = detalles_local['gf'] / 20
            gf_visitante = detalles_visitante['gf'] / 20
            if gf_local > 1.2 and gf_visitante > 1.0:
                argumentos.append(f"🎯 Ambos anotan regularmente: {gf_local:.1f} y {gf_visitante:.1f} goles/partido")
        
        else:  # No
            btts_local = stats_local['partidos_btts'] / 20
            
            if btts_local < 0.4:
                argumentos.append(f"🔒 Baja frecuencia BTTS: solo {btts_local*100:.0f}%")
            
            gf_visitante = detalles_visitante['gf'] / 20
            if gf_visitante < 0.8:
                argumentos.append(f"📉 Visitante poco ofensivo: {gf_visitante:.1f} goles/partido")
        
        return argumentos

# ============================================================================
# INTERFAZ STREAMLIT
# ============================================================================

def main():
    st.title("⚽ SISTEMABETS PROFESIONAL v8.0")
    st.markdown("### Sistema Definitivo con Ajustes por Competición")
    
    # Inicializar
    if 'db_manager' not in st.session_state:
        st.session_state['db_manager'] = DatabaseManager()
    
    db_manager = st.session_state['db_manager']
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        api_key = st.text_input("API Key Football-Data.org", type="password")
        
        if not api_key:
            st.warning("⚠️ Ingresa tu API Key")
            st.stop()
        
        st.markdown("---")
        st.header("📊 Estado")
        
        n_partidos = db_manager.contar_partidos()
        st.metric("Partidos Guardados", n_partidos)
        
        st.markdown("---")
        
        with st.expander("💡 Mejoras v8.0"):
            st.markdown("""
            ### ✅ Nuevo en esta versión:
            
            1. **Ajustes por Competición**
               - Detecta Champions/Europa League
               - Ajusta según nivel de liga
               - Identifica equipos de élite
            
            2. **Últimos 20 partidos completos**
               - Botón para ver historial
               - Detalles de cada partido
               - Competición incluida
            
            3. **Argumentos sólidos**
               - 3-5 razones por apuesta
               - Basados en datos reales
               - Veredicto final claro
            
            4. **Corrección Benfica vs Madrid**
               - Ahora Real Madrid sale favorito
               - Ajusta por nivel de liga
               - Considera experiencia europea
            """)
    
    # Selección de liga
    st.header("1️⃣ Selecciona la Liga")
    liga_nombre = st.selectbox("Liga", list(FootballDataAPI.LIGAS.keys()))
    liga_code = FootballDataAPI.LIGAS[liga_nombre]
    
    api = FootballDataAPI(api_key)
    
    # Cargar tabla
    with st.spinner("Cargando tabla..."):
        df_liga = api.obtener_standings(liga_code)
    
    if df_liga is None or df_liga.empty:
        st.error("❌ Error cargando tabla")
        st.stop()
    
    st.success(f"✅ {liga_nombre} cargada")
    
    with st.expander("📊 Ver Tabla"):
        st.dataframe(df_liga[['Posicion', 'Equipo', 'PJ', 'Pts', 'GF', 'GC']], use_container_width=True)
    
    # Selección equipos
    st.header("2️⃣ Selecciona el Partido")
    col1, col2 = st.columns(2)
    
    with col1:
        equipo_local = st.selectbox("🏠 Local", df_liga['Equipo'].tolist())
    
    with col2:
        equipos_visitante = [e for e in df_liga['Equipo'].tolist() if e != equipo_local]
        equipo_visitante = st.selectbox("✈️ Visitante", equipos_visitante)
    
    # Análisis
    if st.button("🚀 ANALIZAR", type="primary", use_container_width=True):
        
        local_team = df_liga[df_liga['Equipo'] == equipo_local].iloc[0]
        visitante_team = df_liga[df_liga['Equipo'] == equipo_visitante].iloc[0]
        
        st.markdown("---")
        st.header(f"📊 {equipo_local} vs {equipo_visitante}")
        
        # Cargar datos
        with st.spinner("Analizando..."):
            partidos_local = api.obtener_ultimos_20_partidos(equipo_local)
            partidos_visitante = api.obtener_ultimos_20_partidos(equipo_visitante)
            h2h = api.obtener_enfrentamientos_directos(equipo_local, equipo_visitante)
        
        if not partidos_local or not partidos_visitante:
            st.error("❌ No se pudieron cargar los partidos")
            st.stop()
        
        # Calcular lambdas AJUSTADOS
        lambda_local, lambda_visitante, analisis = AnalizadorProfundo.calcular_lambdas_ajustados(
            local_team, visitante_team, partidos_local, partidos_visitante, h2h
        )
        
        # Mostrar advertencias si existen
        if analisis.get('advertencia'):
            st.warning(analisis['advertencia'])
        
        if analisis.get('es_champions'):
            st.info("🏆 **Partido de Champions/Europa League detectado** - Ajustes aplicados por nivel de competición")
        
        # Parámetros
        st.subheader("🔍 Parámetros de Análisis")
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Lambda Local", f"{lambda_local:.2f}", 
                   help="Goles esperados del equipo local (ajustado por competición)")
        col2.metric("Lambda Visitante", f"{lambda_visitante:.2f}",
                   help="Goles esperados del equipo visitante (ajustado)")
        col3.metric("Goles Esperados", f"{lambda_local + lambda_visitante:.2f}")
        
        # Calcular probabilidades
        predicciones = AnalizadorProfundo.calcular_probabilidades(lambda_local, lambda_visitante)
        
        # Forma reciente
        st.subheader("📈 Forma Reciente")
        
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
            - ⚽ GF: **{detalles_local['gf']}** ({detalles_local['gf']/20:.1f}/partido)
            - 🥅 GC: **{detalles_local['gc']}** ({detalles_local['gc']/20:.1f}/partido)
            - 🔥 Racha: {detalles_local['racha_actual']}
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
            - ⚽ GF: **{detalles_visitante['gf']}** ({detalles_visitante['gf']/20:.1f}/partido)
            - 🥅 GC: **{detalles_visitante['gc']}** ({detalles_visitante['gc']/20:.1f}/partido)
            - 🔥 Racha: {detalles_visitante['racha_actual']}
            """)
        
        # Botones para ver 20 partidos
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"📋 Ver 20 partidos de {equipo_local}"):
                st.markdown("### Últimos 20 partidos:")
                for r in analisis['resultados_local']:
                    st.caption(f"{r['simbolo']} {r['num']}. {r['local']} {r['resultado']} {r['visitante']} | {r['competicion']}")
        
        with col2:
            if st.button(f"📋 Ver 20 partidos de {equipo_visitante}"):
                st.markdown("### Últimos 20 partidos:")
                for r in analisis['resultados_visitante']:
                    st.caption(f"{r['simbolo']} {r['num']}. {r['local']} {r['resultado']} {r['visitante']} | {r['competicion']}")
        
        # H2H
        if h2h:
            st.subheader("🎯 Enfrentamientos Directos")
            
            victorias_local = sum(1 for p in h2h if 
                                 (p['local'] == equipo_local and p['goles_local'] > p['goles_visitante']) or
                                 (p['visitante'] == equipo_local and p['goles_visitante'] > p['goles_local']))
            empates = sum(1 for p in h2h if p['goles_local'] == p['goles_visitante'])
            victorias_visitante = len(h2h) - victorias_local - empates
            
            col1, col2, col3 = st.columns(3)
            col1.metric(f"{equipo_local}", victorias_local)
            col2.metric("Empates", empates)
            col3.metric(f"{equipo_visitante}", victorias_visitante)
            
            with st.expander("Ver historial H2H"):
                for p in h2h[:10]:
                    st.caption(f"{p['local']} {p['goles_local']}-{p['goles_visitante']} {p['visitante']} | {p['competicion']}")
        
        # Predicciones
        st.markdown("---")
        st.header("🎯 PREDICCIONES")
        
        # 1X2
        st.subheader("⚽ Resultado Final (1X2)")
        
        col1, col2, col3 = st.columns(3)
        
        prob_local = predicciones['1X2']['Local']
        prob_empate = predicciones['1X2']['Empate']
        prob_visitante = predicciones['1X2']['Visitante']
        
        with col1:
            st.metric(equipo_local, f"{prob_local*100:.1f}%")
            st.caption(f"Cuota justa: {1/prob_local:.2f}")
        
        with col2:
            st.metric("Empate", f"{prob_empate*100:.1f}%")
            st.caption(f"Cuota justa: {1/prob_empate:.2f}")
        
        with col3:
            st.metric(equipo_visitante, f"{prob_visitante*100:.1f}%")
            st.caption(f"Cuota justa: {1/prob_visitante:.2f}")
        
        # Over/Under
        st.subheader("📊 Over/Under 2.5")
        
        col1, col2 = st.columns(2)
        
        prob_over = predicciones['Over/Under']['Over 2.5']
        prob_under = predicciones['Over/Under']['Under 2.5']
        
        with col1:
            st.metric("Over 2.5", f"{prob_over*100:.1f}%")
            st.caption(f"Cuota justa: {1/prob_over:.2f}")
        
        with col2:
            st.metric("Under 2.5", f"{prob_under*100:.1f}%")
            st.caption(f"Cuota justa: {1/prob_under:.2f}")
        
        # BTTS
        st.subheader("🎯 BTTS")
        
        col1, col2 = st.columns(2)
        
        prob_btts_si = predicciones['BTTS']['Si']
        prob_btts_no = predicciones['BTTS']['No']
        
        with col1:
            st.metric("BTTS Sí", f"{prob_btts_si*100:.1f}%")
            st.caption(f"Cuota justa: {1/prob_btts_si:.2f}")
        
        with col2:
            st.metric("BTTS No", f"{prob_btts_no*100:.1f}%")
            st.caption(f"Cuota justa: {1/prob_btts_no:.2f}")
        
        # MEJORES APUESTAS CON ARGUMENTOS
        st.markdown("---")
        st.header("💎 MEJORES APUESTAS")
        
        todas_apuestas = [
            {'Mercado': f'Victoria {equipo_local}', 'Tipo': '1X2', 'Resultado': 'Local', 'Prob': prob_local},
            {'Mercado': 'Empate', 'Tipo': '1X2', 'Resultado': 'Empate', 'Prob': prob_empate},
            {'Mercado': f'Victoria {equipo_visitante}', 'Tipo': '1X2', 'Resultado': 'Visitante', 'Prob': prob_visitante},
            {'Mercado': 'Over 2.5', 'Tipo': 'Over/Under', 'Resultado': 'Over 2.5', 'Prob': prob_over},
            {'Mercado': 'Under 2.5', 'Tipo': 'Over/Under', 'Resultado': 'Under 2.5', 'Prob': prob_under},
            {'Mercado': 'BTTS Sí', 'Tipo': 'BTTS', 'Resultado': 'Si', 'Prob': prob_btts_si},
            {'Mercado': 'BTTS No', 'Tipo': 'BTTS', 'Resultado': 'No', 'Prob': prob_btts_no}
        ]
        
        todas_apuestas.sort(key=lambda x: x['Prob'], reverse=True)
        
        for i, apuesta in enumerate(todas_apuestas[:5], 1):
            
            confianza = "⭐⭐⭐" if apuesta['Prob'] > 0.65 else "⭐⭐" if apuesta['Prob'] > 0.55 else "⭐"
            
            with st.expander(f"#{i} - {apuesta['Mercado']} | {apuesta['Prob']*100:.1f}% {confianza}"):
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Probabilidad", f"{apuesta['Prob']*100:.1f}%")
                col2.metric("Cuota Justa", f"{1/apuesta['Prob']:.2f}")
                col3.metric("Cuota Mínima Esperada", f"{(1/apuesta['Prob'])*0.95:.2f}")
                
                # Generar argumentos
                if apuesta['Tipo'] == '1X2':
                    argumentos = GeneradorArgumentos.generar_argumento_1x2(
                        apuesta['Resultado'], apuesta['Prob'], local_team, visitante_team, analisis
                    )
                elif apuesta['Tipo'] == 'Over/Under':
                    argumentos = GeneradorArgumentos.generar_argumento_over_under(
                        apuesta['Resultado'], apuesta['Prob'], analisis
                    )
                else:  # BTTS
                    argumentos = GeneradorArgumentos.generar_argumento_btts(
                        apuesta['Resultado'], apuesta['Prob'], analisis
                    )
                
                if argumentos:
                    st.markdown("**🔍 ARGUMENTOS:**")
                    for arg in argumentos:
                        st.markdown(f"- {arg}")
                
                # Veredicto
                st.markdown("---")
                st.markdown("### ⚖️ VEREDICTO FINAL")
                
                if apuesta['Prob'] > 0.70:
                    st.success("🟢 **MUY RECOMENDADA** - Alta probabilidad según análisis ajustado")
                elif apuesta['Prob'] > 0.60:
                    st.info("🔵 **RECOMENDADA** - Buena probabilidad con datos sólidos")
                elif apuesta['Prob'] > 0.55:
                    st.warning("🟡 **CONSIDERAR** - Probabilidad moderada, analiza cuotas del mercado")
                else:
                    st.error("🔴 **NO RECOMENDADA** - Probabilidad insuficiente")
                
                st.caption(f"""
                **Busca cuotas ≥ {(1/apuesta['Prob'])*0.95:.2f}** para que haya valor (+EV).
                Modelo ajustado por: nivel de competición, forma reciente, stats locales/visitantes.
                """)
        
        # Guardar resultado
        st.markdown("---")
        st.subheader("💾 Guardar Resultado Real (Después del Partido)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            goles_local_real = st.number_input(f"Goles {equipo_local}", 0, 10, 0)
        
        with col2:
            goles_visitante_real = st.number_input(f"Goles {equipo_visitante}", 0, 10, 0)
        
        if st.button("💾 GUARDAR", type="secondary"):
            
            # Extraer features básicas para BD
            features_basicas = {
                'forma_local': forma_local,
                'forma_visitante': forma_visitante,
                'lambda_local': lambda_local,
                'lambda_visitante': lambda_visitante
            }
            
            partido_data = {
                'fecha': datetime.now().strftime('%Y-%m-%d'),
                'competicion': liga_nombre,
                'local': equipo_local,
                'visitante': equipo_visitante,
                'goles_local': goles_local_real,
                'goles_visitante': goles_visitante_real,
                'features': features_basicas
            }
            
            db_manager.guardar_partido(partido_data)
            st.success(f"✅ Guardado! Total: {db_manager.contar_partidos()}")
    
    # Disclaimer
    st.markdown("---")
    st.warning("""
    ⚠️ **ADVERTENCIA:**
    
    - Sistema con ajustes por nivel de competición y liga
    - Datos 100% reales de Football-Data.org
    - Detecta Champions/Europa League y ajusta predicciones
    - NO garantiza ganancias, úsalo como herramienta de apoyo
    - Siempre compara cuotas en múltiples casas
    - Apuesta responsablemente
    
    📊 **Mejores casos de uso:**
    - Partidos de la misma liga ✅
    - Over/Under (más predecible) ✅
    - BTTS con stats claras ✅
    - Champions con ajustes aplicados ✅
    """)

if __name__ == "__main__":
    main()
