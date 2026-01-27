import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from scipy.stats import poisson
import sqlite3
import json
import time
import warnings
warnings.filterwarnings('ignore')

# 1. Configuración base
st.set_page_config(
    page_title="SISTEMABETS CON AYUDA DE IA", 
    page_icon="🤖", 
    layout="wide"
)

# 2. INYECCIÓN DE IDENTIDAD PWA (Solo metadatos, sin Style)
URL_LOGO_MEDIEVAL = "https://raw.githubusercontent.com/Moritapr/sistemabets-ia/main/logo.png"

# Usamos triple comilla simple para evitar conflictos con comillas dobles del HTML
pwa_html = f'''
    <head>
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
        <link rel="apple-touch-icon" href="{URL_LOGO_MEDIEVAL}">
    </head>
'''

st.markdown(pwa_html, unsafe_allow_html=True)
# ============================================================================
# BASE DE DATOS
# ============================================================================

class DatabaseManager:
    def __init__(self, db_path='partidos_historicos.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
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
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_equipos ON partidos(local, visitante)')
        conn.commit()
        conn.close()
    
    def guardar_partido(self, partido_data):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id FROM partidos 
            WHERE fecha = ? AND local = ? AND visitante = ?
        ''', (partido_data['fecha'], partido_data['local'], partido_data['visitante']))
        
        if cursor.fetchone():
            conn.close()
            return False
        
        cursor.execute('''
            INSERT INTO partidos (fecha, competicion, local, visitante, 
                                 goles_local, goles_visitante, features, fecha_analisis)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            partido_data['fecha'], partido_data['competicion'],
            partido_data['local'], partido_data['visitante'],
            partido_data['goles_local'], partido_data['goles_visitante'],
            json.dumps(partido_data['features']),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        conn.commit()
        conn.close()
        return True
    
    def contar_partidos(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM partidos')
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def obtener_estadisticas(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM partidos')
        total = cursor.fetchone()[0]
        cursor.execute('SELECT competicion, COUNT(*) FROM partidos GROUP BY competicion')
        por_competicion = dict(cursor.fetchall())
        cursor.execute('SELECT MIN(fecha), MAX(fecha) FROM partidos')
        rango = cursor.fetchone()
        conn.close()
        
        return {
            'total': total,
            'por_competicion': por_competicion,
            'fecha_min': rango[0] if rango[0] else 'N/A',
            'fecha_max': rango[1] if rango[1] else 'N/A'
        }
    
    def limpiar_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM partidos')
        conn.commit()
        conn.close()

# ============================================================================
# API FOOTBALL-DATA
# ============================================================================

class FootballDataAPI:
    BASE_URL = "https://api.football-data.org/v4"
    
    LIGAS = {
        "Champions League": "CL", "Premier League": "PL", "La Liga": "PD",
        "Bundesliga": "BL1", "Serie A": "SA", "Ligue 1": "FL1",
        "Eredivisie": "DED", "Championship": "ELC", "Liga Portugal": "PPL"
    }
    
    LIGA_STRENGTH = {
        'PL': 10, 'PD': 9.5, 'BL1': 9, 'SA': 8.5, 'FL1': 8,
        'PPL': 6, 'DED': 7, 'ELC': 7.5, 'CL': 10
    }
    
    ELITE_TEAMS = {
        'Real Madrid', 'FC Barcelona', 'Atlético Madrid',
        'Manchester City', 'Liverpool FC', 'Chelsea FC', 'Arsenal FC',
        'Manchester United', 'Tottenham Hotspur',
        'FC Bayern München', 'Borussia Dortmund', 'RB Leipzig',
        'Inter Milan', 'AC Milan', 'Juventus FC', 'SSC Napoli',
        'Paris Saint-Germain', 'AFC Ajax', 'SL Benfica', 'FC Porto'
    }
    
    def __init__(self, api_key):
        self.headers = {"X-Auth-Token": api_key}
        self.cache_teams = {}
    
    def obtener_standings(self, liga_code):
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
                self.cache_teams[team['team']['name']] = team['team']['id']
                equipos.append({
                    'Equipo': team['team']['name'],
                    'ID': team['team']['id'],
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
        return self.obtener_ultimos_partidos_extendido(equipo_nombre, 20)
    
    def obtener_ultimos_partidos_extendido(self, equipo_nombre, limit=100):
        try:
            team_id = self.cache_teams.get(equipo_nombre)
            if not team_id:
                return []
            
            url = f"{self.BASE_URL}/teams/{team_id}/matches"
            params = {"status": "FINISHED", "limit": limit}
            response = requests.get(url, headers=self.headers, params=params, timeout=15)
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            partidos = []
            
            for match in data.get('matches', []):
                if match['score']['fullTime']['home'] is None:
                    continue
                
                home = match['homeTeam']['name']
                away = match['awayTeam']['name']
                
                if home != equipo_nombre and away != equipo_nombre:
                    continue
                
                partidos.append({
                    'local': home,
                    'visitante': away,
                    'goles_local': match['score']['fullTime']['home'],
                    'goles_visitante': match['score']['fullTime']['away'],
                    'fecha': match['utcDate'],
                    'competicion': match['competition']['name']
                })
            
            return sorted(partidos, key=lambda x: x['fecha'], reverse=True)
        except:
            return []
    
    def obtener_enfrentamientos_directos_completo(self, equipo1, equipo2):
        try:
            h2h_combinado = []
            partidos_vistos = set()
            
            # Buscar desde ambos equipos para capturar todo el historial
            for equipo_buscar, equipo_rival in [(equipo1, equipo2), (equipo2, equipo1)]:
                team_id = self.cache_teams.get(equipo_buscar)
                if not team_id:
                    continue
                
                url = f"{self.BASE_URL}/teams/{team_id}/matches"
                params = {"status": "FINISHED", "limit": 200}
                response = requests.get(url, headers=self.headers, params=params, timeout=15)
                
                if response.status_code != 200:
                    continue
                
                data = response.json()
                
                for match in data.get('matches', []):
                    if match['score']['fullTime']['home'] is None:
                        continue
                    
                    home = match['homeTeam']['name']
                    away = match['awayTeam']['name']
                    
                    # Verificar que sea enfrentamiento directo
                    if not ((home == equipo1 and away == equipo2) or (home == equipo2 and away == equipo1)):
                        continue
                    
                    # Crear ID único para evitar duplicados
                    match_id = f"{match['utcDate'][:10]}_{home}_{away}"
                    
                    if match_id in partidos_vistos:
                        continue
                    
                    partidos_vistos.add(match_id)
                    
                    h2h_combinado.append({
                        'local': home,
                        'visitante': away,
                        'goles_local': match['score']['fullTime']['home'],
                        'goles_visitante': match['score']['fullTime']['away'],
                        'fecha': match['utcDate'],
                        'competicion': match['competition']['name']
                    })
            
            return sorted(h2h_combinado, key=lambda x: x['fecha'], reverse=True)
        except Exception as e:
            return []


# ============================================================================
# RECOLECTOR AUTOMÁTICO
# ============================================================================

class RecolectorAutomatico:
    def __init__(self, api, db_manager):
        self.api = api
        self.db_manager = db_manager
    
    def recolectar_liga_completa(self, liga_code, liga_nombre, max_partidos=50):
        df_equipos = self.api.obtener_standings(liga_code)
        if df_equipos is None:
            return 0, 0, ["No se pudo cargar la tabla"]
        
        partidos_nuevos = 0
        partidos_duplicados = 0
        errores = []
        
        total_equipos = len(df_equipos)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, equipo_row in df_equipos.iterrows():
            equipo_nombre = equipo_row['Equipo']
            progreso = (idx + 1) / total_equipos
            progress_bar.progress(progreso)
            status_text.text(f"📥 {equipo_nombre}... ({idx + 1}/{total_equipos})")
            
            try:
                partidos = self.api.obtener_ultimos_partidos_extendido(equipo_nombre, max_partidos)
                
                for partido in partidos:
                    partido_data = {
                        'fecha': partido['fecha'][:10],
                        'competicion': partido['competicion'],
                        'local': partido['local'],
                        'visitante': partido['visitante'],
                        'goles_local': partido['goles_local'],
                        'goles_visitante': partido['goles_visitante'],
                        'features': {'goles_local': partido['goles_local'], 'goles_visitante': partido['goles_visitante']}
                    }
                    
                    if self.db_manager.guardar_partido(partido_data):
                        partidos_nuevos += 1
                    else:
                        partidos_duplicados += 1
                
                time.sleep(6)
            except Exception as e:
                errores.append(f"{equipo_nombre}: {str(e)}")
        
        progress_bar.empty()
        status_text.empty()
        
        return partidos_nuevos, partidos_duplicados, errores

# ============================================================================
# ANALIZADOR
# ============================================================================

class AnalizadorExperto:
    @staticmethod
    def calcular_forma_detallada(partidos, equipo_nombre):
        if not partidos:
            return 0.5, "", {}, []
        
        puntos = []
        detalles = {'victorias': 0, 'empates': 0, 'derrotas': 0, 'gf': 0, 'gc': 0, 'racha_actual': ""}
        resultados_detalle = []
        
        for i, p in enumerate(partidos[:20]):
            es_local = p['local'] == equipo_nombre
            gf = p['goles_local'] if es_local else p['goles_visitante']
            gc = p['goles_visitante'] if es_local else p['goles_local']
            
            detalles['gf'] += gf
            detalles['gc'] += gc
            
            if gf > gc:
                puntos.append(1.0)
                detalles['victorias'] += 1
                resultado = "✅"
            elif gf == gc:
                puntos.append(0.5)
                detalles['empates'] += 1
                resultado = "🟰"
            else:
                puntos.append(0.0)
                detalles['derrotas'] += 1
                resultado = "❌"
            
            if i < 5:
                detalles['racha_actual'] += resultado
            
            resultados_detalle.append({
                'simbolo': resultado,
                'local': p['local'],
                'visitante': p['visitante'],
                'resultado': f"{p['goles_local']}-{p['goles_visitante']}",
                'competicion': p['competicion'],
                'fecha': p['fecha'][:10]
            })
        
        pesos = np.linspace(0.10, 0.005, len(puntos))
        pesos = pesos / pesos.sum()
        score_forma = sum(p * w for p, w in zip(puntos, pesos))
        
        return score_forma, "", detalles, resultados_detalle
    
    @staticmethod
    def calcular_stats_avanzadas(equipo_nombre, partidos):
        stats = {
            'local_pj': 0, 'local_gf': 0, 'local_gc': 0, 'local_victorias': 0,
            'visitante_pj': 0, 'visitante_gf': 0, 'visitante_gc': 0, 'visitante_victorias': 0,
            'partidos_over25': 0, 'partidos_btts': 0
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
            elif p['visitante'] == equipo_nombre:
                stats['visitante_pj'] += 1
                stats['visitante_gf'] += p['goles_visitante']
                stats['visitante_gc'] += p['goles_local']
                if p['goles_visitante'] > p['goles_local']:
                    stats['visitante_victorias'] += 1
        
        return stats
    
    @staticmethod
    def calcular_factor_ajuste(local_team, visitante_team, partidos_local, partidos_visitante):
        factor_local = 1.0
        factor_visitante = 1.0
        advertencias = []
        
        # Detectar Champions
        competiciones = [p['competicion'] for p in (partidos_local + partidos_visitante)[:10]]
        es_europea = any('Champions' in c or 'Europa' in c or 'UEFA' in c for c in competiciones)
        
        if es_europea:
            advertencias.append("🏆 Competición europea detectada")
            
            if local_team['Equipo'] in FootballDataAPI.ELITE_TEAMS and visitante_team['Equipo'] not in FootballDataAPI.ELITE_TEAMS:
                factor_local *= 1.15
                factor_visitante *= 0.88
                advertencias.append(f"🌟 {local_team['Equipo']} es equipo de élite")
            elif visitante_team['Equipo'] in FootballDataAPI.ELITE_TEAMS and local_team['Equipo'] not in FootballDataAPI.ELITE_TEAMS:
                factor_visitante *= 1.15
                factor_local *= 0.88
                advertencias.append(f"🌟 {visitante_team['Equipo']} es equipo de élite")
        
        return factor_local, factor_visitante, advertencias
    
    @staticmethod
    def analisis_completo(local_team, visitante_team, partidos_local, partidos_visitante, h2h):
        factor_local, factor_visitante, advertencias = AnalizadorExperto.calcular_factor_ajuste(
            local_team, visitante_team, partidos_local, partidos_visitante
        )
        
        forma_local, _, detalles_local, resultados_local = AnalizadorExperto.calcular_forma_detallada(
            partidos_local, local_team['Equipo']
        )
        forma_visitante, _, detalles_visitante, resultados_visitante = AnalizadorExperto.calcular_forma_detallada(
            partidos_visitante, visitante_team['Equipo']
        )
        
        stats_local = AnalizadorExperto.calcular_stats_avanzadas(local_team['Equipo'], partidos_local)
        stats_visitante = AnalizadorExperto.calcular_stats_avanzadas(visitante_team['Equipo'], partidos_visitante)
        
        gf_local_tabla = local_team['GF'] / max(local_team['PJ'], 1)
        gc_local_tabla = local_team['GC'] / max(local_team['PJ'], 1)
        gf_visitante_tabla = visitante_team['GF'] / max(visitante_team['PJ'], 1)
        gc_visitante_tabla = visitante_team['GC'] / max(visitante_team['PJ'], 1)
        
        gf_local_casa = stats_local['local_gf'] / max(stats_local['local_pj'], 1) if stats_local['local_pj'] > 0 else gf_local_tabla
        gc_local_casa = stats_local['local_gc'] / max(stats_local['local_pj'], 1) if stats_local['local_pj'] > 0 else gc_local_tabla
        gf_visitante_fuera = stats_visitante['visitante_gf'] / max(stats_visitante['visitante_pj'], 1) if stats_visitante['visitante_pj'] > 0 else gf_visitante_tabla
        gc_visitante_fuera = stats_visitante['visitante_gc'] / max(stats_visitante['visitante_pj'], 1) if stats_visitante['visitante_pj'] > 0 else gc_visitante_tabla
        
        media_goles = (gf_local_tabla + gf_visitante_tabla) / 2
        
        lambda_local = (gf_local_casa / max(media_goles, 0.5)) * (gc_visitante_fuera / max(media_goles, 0.5)) * media_goles
        lambda_visitante = (gf_visitante_fuera / max(media_goles, 0.5)) * (gc_local_casa / max(media_goles, 0.5)) * media_goles
        
        lambda_local *= 1.15
        lambda_visitante /= 1.08
        
        if forma_local > 0.75:
            lambda_local *= 1.25
        elif forma_local < 0.35:
            lambda_local *= 0.82
        
        if forma_visitante > 0.75:
            lambda_visitante *= 1.25
        elif forma_visitante < 0.35:
            lambda_visitante *= 0.82
        
        lambda_local *= factor_local
        lambda_visitante *= factor_visitante
        
        lambda_local = max(min(lambda_local, 4.5), 0.30)
        lambda_visitante = max(min(lambda_visitante, 4.5), 0.30)
        
        return {
            'lambda_local': lambda_local,
            'lambda_visitante': lambda_visitante,
            'forma_local': forma_local,
            'forma_visitante': forma_visitante,
            'detalles_local': detalles_local,
            'detalles_visitante': detalles_visitante,
            'resultados_local': resultados_local,
            'resultados_visitante': resultados_visitante,
            'stats_local': stats_local,
            'stats_visitante': stats_visitante,
            'advertencias': advertencias,
            'factor_local': factor_local,
            'factor_visitante': factor_visitante
        }

# ============================================================================
# CALCULADOR DE MERCADOS
# ============================================================================

def calcular_mercados(lambda_l, lambda_v):
    matriz = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            matriz[i, j] = poisson.pmf(i, lambda_l) * poisson.pmf(j, lambda_v)
    
    mercados = {}
    
    # 1X2
    p_local = np.sum(np.tril(matriz, -1))
    p_empate = np.sum(np.diag(matriz))
    p_visitante = np.sum(np.triu(matriz, 1))
    total = p_local + p_empate + p_visitante
    
    mercados['1X2'] = {
        'Local': max(p_local / total, 0.005),
        'Empate': max(p_empate / total, 0.005),
        'Visitante': max(p_visitante / total, 0.005)
    }
    
    # Doble Oportunidad
    mercados['Doble_Oportunidad'] = {
        '1X': mercados['1X2']['Local'] + mercados['1X2']['Empate'],
        '12': mercados['1X2']['Local'] + mercados['1X2']['Visitante'],
        'X2': mercados['1X2']['Empate'] + mercados['1X2']['Visitante']
    }
    
    # Over/Under
    mercados['Over_Under'] = {}
    for threshold in [0.5, 1.5, 2.5, 3.5, 4.5]:
        p_over = sum([matriz[i, j] for i in range(8) for j in range(8) if (i+j) > threshold])
        mercados['Over_Under'][f"Over {threshold}"] = max(p_over, 0.005)
        mercados['Over_Under'][f"Under {threshold}"] = max(1 - p_over, 0.005)
    
    # BTTS
    p_btts_no = matriz[0,:].sum() + matriz[:,0].sum() - matriz[0,0]
    mercados['BTTS'] = {
        'Si': max(1 - p_btts_no, 0.005),
        'No': max(p_btts_no, 0.005)
    }
    
    # Handicap
    mercados['Handicap'] = {}
    p_local_menos15 = sum([matriz[i, j] for i in range(8) for j in range(8) if i - j >= 2])
    mercados['Handicap']['Local -1.5'] = max(p_local_menos15, 0.005)
    mercados['Handicap']['Visitante +1.5'] = max(1 - p_local_menos15, 0.005)
    
    # Primera Mitad
    lambda_1h_local = lambda_l * 0.45
    lambda_1h_visitante = lambda_v * 0.45
    
    matriz_1h = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            matriz_1h[i, j] = poisson.pmf(i, lambda_1h_local) * poisson.pmf(j, lambda_1h_visitante)
    
    mercados['Primera_Mitad'] = {
        'Over 0.5 en 1H': max(1 - matriz_1h[0, 0], 0.005),
        'Over 1.5 en 1H': max(sum([matriz_1h[i, j] for i in range(5) for j in range(5) if (i+j) > 1.5]), 0.005),
        'BTTS en 1H': max(1 - (matriz_1h[0,:].sum() + matriz_1h[:,0].sum() - matriz_1h[0,0]), 0.005)
    }
    
    # Resultado Exacto Top 10
    resultados_exactos = []
    for i in range(8):
        for j in range(8):
            resultados_exactos.append({'Marcador': f"{i}-{j}", 'Probabilidad': max(matriz[i, j], 0.001)})
    
    resultados_exactos.sort(key=lambda x: x['Probabilidad'], reverse=True)
    mercados['Resultado_Exacto'] = resultados_exactos[:10]
    
    return mercados

# ============================================================================
# GENERADOR DE ARGUMENTOS
# ============================================================================

def generar_analisis(mercado, prob, local_team, visitante_team, analisis):
    detalles_local = analisis['detalles_local']
    detalles_visitante = analisis['detalles_visitante']
    
    if 'Local' in mercado and 'Over' not in mercado:
        texto = f"""📝 **ANÁLISIS COMPLETO**

Esta apuesta se considera **{'ALTAMENTE RECOMENDABLE' if prob > 0.65 else 'FAVORABLE'}** con una probabilidad del {prob*100:.1f}%.

🎯 **CAPACIDAD OFENSIVA**

{local_team['Equipo']} presenta {detalles_local['gf']/20:.2f} goles/partido en sus últimos 20 encuentros, acumulando {detalles_local['victorias']} victorias ({analisis['forma_local']*100:.0f}% efectividad). Esto contrasta con los {detalles_visitante['gf']/20:.2f} goles/partido de {visitante_team['Equipo']}.

La racha actual muestra: {detalles_local['racha_actual']}, evidenciando consistencia en el tramo final.

⚖️ **VEREDICTO**

Se recomienda buscar cuotas superiores a {(1/prob)*0.95:.2f} para asegurar valor positivo (+{((prob * (1/prob)*0.95) - 1)*100:.1f}% EV)."""
    
    elif 'Over' in mercado:
        threshold = float(mercado.split()[1])
        goles_promedio = detalles_local['gf']/20 + detalles_visitante['gf']/20
        
        texto = f"""📝 **ANÁLISIS COMPLETO**

Esta apuesta de **Over {threshold}** presenta {prob*100:.1f}% de probabilidad.

⚽ **CAPACIDAD OFENSIVA COMBINADA**

Promedio combinado: {goles_promedio:.2f} goles/partido ({detalles_local['gf']/20:.2f} + {detalles_visitante['gf']/20:.2f}).

{'Supera claramente' if goles_promedio > threshold else 'Se aproxima a'} el umbral de {threshold} goles.

⚖️ **VEREDICTO**

Buscar cuotas mínimas de {(1/prob)*0.95:.2f} para valor positivo."""
    
    elif 'BTTS' in mercado:
        texto = f"""📝 **ANÁLISIS COMPLETO**

BTTS con {prob*100:.1f}% de probabilidad.

Ambos equipos muestran capacidad ofensiva: {detalles_local['gf']/20:.2f} y {detalles_visitante['gf']/20:.2f} goles/partido.

⚖️ **VEREDICTO**

Cuota mínima recomendada: {(1/prob)*0.95:.2f}"""
    
    else:
        texto = f"""📝 **ANÁLISIS**

Probabilidad: {prob*100:.1f}%

Cuota justa: {1/prob:.2f}
Cuota mínima: {(1/prob)*0.95:.2f}"""
    
    return texto

# ============================================================================
# INTERFAZ STREAMLIT
# ============================================================================

def main():
    st.title("⚽ SISTEMABETS CON AYUDA DE IA Y FUNCIONES MATEMATICAS")
    st.markdown("### Sistema enfocado en la ayuda con los analisis deportivos")
    st.markdown("Hecho por Alejandro Mora")
    
    if 'db_manager' not in st.session_state:
        st.session_state['db_manager'] = DatabaseManager()
    
    db_manager = st.session_state['db_manager']
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        api_key = st.text_input("API Key", type="password")
        
        if not api_key:
            st.warning("Ingresa tu API Key")
            st.stop()
        
        st.markdown("---")
        st.header("📊 Base de Datos")
        
        stats_bd = db_manager.obtener_estadisticas()
        st.metric("Partidos", stats_bd['total'])
        
        if stats_bd['total'] > 0:
            with st.expander("Detalles"):
                for comp, count in stats_bd['por_competicion'].items():
                    st.caption(f"{comp}: {count}")
        
        st.markdown("---")
        st.header("🤖 Recolector")
        
        liga_rec = st.selectbox("Liga", list(FootballDataAPI.LIGAS.keys()), key='recolector')
        partidos_eq = st.slider("Partidos/equipo", 20, 50, 30)
        
        if st.button("🚀 RECOLECTAR"):
            api = FootballDataAPI(api_key)
            recolector = RecolectorAutomatico(api, db_manager)
            
            liga_code = FootballDataAPI.LIGAS[liga_rec]
            st.info(f"Recolectando {liga_rec}...")
            st.warning("⏱️ Esto tardará 10-20 minutos")
            
            nuevos, duplicados, errores = recolector.recolectar_liga_completa(
                liga_code, liga_rec, partidos_eq
            )
            
            st.success(f"✅ Completado!")
            st.metric("Nuevos", nuevos)
            st.metric("Duplicados", duplicados)
            
            if errores:
                with st.expander("Errores"):
                    for error in errores:
                        st.caption(error)
        
        st.markdown("---")
        with st.expander("🗄️ Gestión"):
            if st.button("🗑️ Limpiar BD"):
                db_manager.limpiar_database()
                st.success("Limpiado")
                st.rerun()
    
    # Liga
    st.header("1️⃣ Liga")
    liga_nombre = st.selectbox("Selecciona", list(FootballDataAPI.LIGAS.keys()))
    liga_code = FootballDataAPI.LIGAS[liga_nombre]
    
    api = FootballDataAPI(api_key)
    
    with st.spinner("Cargando..."):
        df_liga = api.obtener_standings(liga_code)
    
    if df_liga is None or df_liga.empty:
        st.error("❌ Error")
        st.stop()
    
    st.success(f"✅ {liga_nombre}")
    
    with st.expander("📊 Tabla"):
        st.dataframe(df_liga[['Posicion', 'Equipo', 'PJ', 'Pts', 'GF', 'GC']], use_container_width=True)
    
    # Equipos
    st.header("2️⃣ Partido")
    col1, col2 = st.columns(2)
    
    with col1:
        equipo_local = st.selectbox("🏠 Local", df_liga['Equipo'].tolist())
    
    with col2:
        equipos_visitante = [e for e in df_liga['Equipo'].tolist() if e != equipo_local]
        equipo_visitante = st.selectbox("✈️ Visitante", equipos_visitante)
    
    # ANÁLISIS
    if st.button("🚀 ANALIZAR", type="primary", use_container_width=True):
        
        local_team = df_liga[df_liga['Equipo'] == equipo_local].iloc[0]
        visitante_team = df_liga[df_liga['Equipo'] == equipo_visitante].iloc[0]
        
        st.markdown("---")
        st.header(f"📊 {equipo_local} vs {equipo_visitante}")
        
        with st.spinner("Cargando datos..."):
            partidos_local = api.obtener_ultimos_20_partidos(equipo_local)
            partidos_visitante = api.obtener_ultimos_20_partidos(equipo_visitante)
            h2h = api.obtener_enfrentamientos_directos_completo(equipo_local, equipo_visitante)
            time.sleep(1)
        
        if not partidos_local or not partidos_visitante:
            st.error("❌ No se pudieron cargar partidos")
            st.stop()
        
        # Análisis
        analisis = AnalizadorExperto.analisis_completo(
            local_team, visitante_team, partidos_local, partidos_visitante, h2h
        )
        
        # Advertencias
        if analisis.get('advertencias'):
            for adv in analisis['advertencias']:
                st.warning(adv)
        
        # Parámetros
        st.subheader("🔍 Parámetros")
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Lambda Local", f"{analisis['lambda_local']:.2f}")
        col2.metric("Lambda Visitante", f"{analisis['lambda_visitante']:.2f}")
        col3.metric("Goles Esperados", f"{analisis['lambda_local'] + analisis['lambda_visitante']:.2f}")
        
        # Forma
        st.markdown("---")
        st.subheader("📈 Forma Reciente")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### 🏠 {equipo_local}")
            
            forma_local = analisis['forma_local']
            detalles_local = analisis['detalles_local']
            
            st.progress(forma_local)
            st.caption(f"Forma: {forma_local*100:.1f}%")
            
            st.markdown(f"""
            - ✅ Victorias: **{detalles_local['victorias']}**
            - 🟰 Empates: **{detalles_local['empates']}**
            - ❌ Derrotas: **{detalles_local['derrotas']}**
            - ⚽ GF: **{detalles_local['gf']}** ({detalles_local['gf']/20:.2f}/partido)
            - 🥅 GC: **{detalles_local['gc']}** ({detalles_local['gc']/20:.2f}/partido)
            - 🔥 Racha: {detalles_local['racha_actual']}
            """)
            
            with st.expander(f"📋 Ver 20 partidos de {equipo_local}"):
                df_partidos = pd.DataFrame(analisis['resultados_local'])
                df_partidos = df_partidos[['simbolo', 'local', 'resultado', 'visitante', 'competicion', 'fecha']]
                df_partidos.columns = ['', 'Local', 'Resultado', 'Visitante', 'Competición', 'Fecha']
                st.dataframe(df_partidos, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown(f"### ✈️ {equipo_visitante}")
            
            forma_visitante = analisis['forma_visitante']
            detalles_visitante = analisis['detalles_visitante']
            
            st.progress(forma_visitante)
            st.caption(f"Forma: {forma_visitante*100:.1f}%")
            
            st.markdown(f"""
            - ✅ Victorias: **{detalles_visitante['victorias']}**
            - 🟰 Empates: **{detalles_visitante['empates']}**
            - ❌ Derrotas: **{detalles_visitante['derrotas']}**
            - ⚽ GF: **{detalles_visitante['gf']}** ({detalles_visitante['gf']/20:.2f}/partido)
            - 🥅 GC: **{detalles_visitante['gc']}** ({detalles_visitante['gc']/20:.2f}/partido)
            - 🔥 Racha: {detalles_visitante['racha_actual']}
            """)
            
            with st.expander(f"📋 Ver 20 partidos de {equipo_visitante}"):
                df_partidos = pd.DataFrame(analisis['resultados_visitante'])
                df_partidos = df_partidos[['simbolo', 'local', 'resultado', 'visitante', 'competicion', 'fecha']]
                df_partidos.columns = ['', 'Local', 'Resultado', 'Visitante', 'Competición', 'Fecha']
                st.dataframe(df_partidos, use_container_width=True, hide_index=True)
        
        # H2H
        if h2h:
            st.markdown("---")
            st.subheader(f"🎯 H2H Completo ({len(h2h)} partidos)")
            
            victorias_local = sum(1 for p in h2h if 
                                 (p['local'] == equipo_local and p['goles_local'] > p['goles_visitante']) or
                                 (p['visitante'] == equipo_local and p['goles_visitante'] > p['goles_local']))
            empates = sum(1 for p in h2h if p['goles_local'] == p['goles_visitante'])
            victorias_visitante = len(h2h) - victorias_local - empates
            
            goles_totales = sum(p['goles_local'] + p['goles_visitante'] for p in h2h)
            promedio_goles = goles_totales / len(h2h)
            
            btts_h2h = sum(1 for p in h2h if p['goles_local'] > 0 and p['goles_visitante'] > 0)
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric(f"Victorias {equipo_local}", victorias_local)
            col2.metric("Empates", empates)
            col3.metric(f"Victorias {equipo_visitante}", victorias_visitante)
            col4.metric("Promedio Goles", f"{promedio_goles:.2f}")
            
            st.info(f"📊 **Stats H2H:** BTTS: {btts_h2h}/{len(h2h)} ({btts_h2h/len(h2h)*100:.0f}%) | Over 2.5: {sum(1 for p in h2h if (p['goles_local']+p['goles_visitante'])>2.5)}/{len(h2h)}")
            
            with st.expander(f"📋 Ver TODOS los {len(h2h)} enfrentamientos"):
                df_h2h = pd.DataFrame(h2h)
                df_h2h['Resultado'] = df_h2h.apply(lambda x: f"{x['goles_local']}-{x['goles_visitante']}", axis=1)
                df_h2h['Fecha'] = pd.to_datetime(df_h2h['fecha']).dt.strftime('%Y-%m-%d')
                df_display = df_h2h[['Fecha', 'local', 'Resultado', 'visitante', 'competicion']]
                df_display.columns = ['Fecha', 'Local', 'Resultado', 'Visitante', 'Competición']
                st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # Calcular mercados
        mercados = calcular_mercados(analisis['lambda_local'], analisis['lambda_visitante'])
        
        # Predicciones principales
        st.markdown("---")
        st.header("🎯 PREDICCIONES")
        
        # 1X2
        st.subheader("⚽ 1X2")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(equipo_local, f"{mercados['1X2']['Local']*100:.1f}%")
            st.caption(f"Cuota: {1/mercados['1X2']['Local']:.2f}")
        
        with col2:
            st.metric("Empate", f"{mercados['1X2']['Empate']*100:.1f}%")
            st.caption(f"Cuota: {1/mercados['1X2']['Empate']:.2f}")
        
        with col3:
            st.metric(equipo_visitante, f"{mercados['1X2']['Visitante']*100:.1f}%")
            st.caption(f"Cuota: {1/mercados['1X2']['Visitante']:.2f}")
        
        # Over/Under
        st.subheader("📊 Over/Under")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        col1.metric("Over 0.5", f"{mercados['Over_Under']['Over 0.5']*100:.1f}%")
        col2.metric("Over 1.5", f"{mercados['Over_Under']['Over 1.5']*100:.1f}%")
        col3.metric("Over 2.5", f"{mercados['Over_Under']['Over 2.5']*100:.1f}%")
        col4.metric("Over 3.5", f"{mercados['Over_Under']['Over 3.5']*100:.1f}%")
        col5.metric("Over 4.5", f"{mercados['Over_Under']['Over 4.5']*100:.1f}%")
        
        # BTTS
        st.subheader("🎯 BTTS")
        
        col1, col2 = st.columns(2)
        
        col1.metric("BTTS Sí", f"{mercados['BTTS']['Si']*100:.1f}%")
        col2.metric("BTTS No", f"{mercados['BTTS']['No']*100:.1f}%")
        
        # Mercados adicionales
        st.markdown("---")
        st.header("📊 Mercados Adicionales")
        
        with st.expander("🎲 Doble Oportunidad"):
            for mercado, prob in mercados['Doble_Oportunidad'].items():
                st.metric(mercado, f"{prob*100:.1f}%")
        
        with st.expander("⚖️ Handicap"):
            for mercado, prob in mercados['Handicap'].items():
                st.metric(mercado, f"{prob*100:.1f}%")
        
        with st.expander("⏱️ Primera Mitad"):
            for mercado, prob in mercados['Primera_Mitad'].items():
                st.metric(mercado, f"{prob*100:.1f}%")
        
        with st.expander("🎲 Resultado Exacto Top 10"):
            cols = st.columns(5)
            for i, resultado in enumerate(mercados['Resultado_Exacto'][:10]):
                with cols[i % 5]:
                    st.metric(resultado['Marcador'], f"{resultado['Probabilidad']*100:.1f}%")
        
        # TOP 5 MEJORES APUESTAS
        st.markdown("---")
        st.header("💎 TOP 5 MEJORES APUESTAS")
        
        todas_apuestas = []
        
        # 1X2
        todas_apuestas.append({'Mercado': f'Victoria {equipo_local}', 'Prob': mercados['1X2']['Local']})
        todas_apuestas.append({'Mercado': 'Empate', 'Prob': mercados['1X2']['Empate']})
        todas_apuestas.append({'Mercado': f'Victoria {equipo_visitante}', 'Prob': mercados['1X2']['Visitante']})
        
        # Over/Under
        for threshold in [1.5, 2.5, 3.5]:
            todas_apuestas.append({'Mercado': f'Over {threshold}', 'Prob': mercados['Over_Under'][f'Over {threshold}']})
            todas_apuestas.append({'Mercado': f'Under {threshold}', 'Prob': mercados['Over_Under'][f'Under {threshold}']})
        
        # BTTS
        todas_apuestas.append({'Mercado': 'BTTS Sí', 'Prob': mercados['BTTS']['Si']})
        todas_apuestas.append({'Mercado': 'BTTS No', 'Prob': mercados['BTTS']['No']})
        
        # Doble Oportunidad
        for mercado, prob in mercados['Doble_Oportunidad'].items():
            todas_apuestas.append({'Mercado': mercado, 'Prob': prob})
        
        todas_apuestas.sort(key=lambda x: x['Prob'], reverse=True)
        
        for i, apuesta in enumerate(todas_apuestas[:5], 1):
            
            confianza_estrellas = "⭐⭐⭐" if apuesta['Prob'] > 0.65 else "⭐⭐" if apuesta['Prob'] > 0.55 else "⭐"
            
            with st.expander(f"#{i} - {apuesta['Mercado']} | {apuesta['Prob']*100:.1f}% {confianza_estrellas}"):
                
                col1, col2, col3 = st.columns(3)
                
                col1.metric("Probabilidad", f"{apuesta['Prob']*100:.1f}%")
                col2.metric("Cuota Justa", f"{1/apuesta['Prob']:.2f}")
                col3.metric("Cuota Mínima", f"{(1/apuesta['Prob'])*0.95:.2f}")
                
                st.markdown("---")
                
                narrativa = generar_analisis(apuesta['Mercado'], apuesta['Prob'], local_team, visitante_team, analisis)
                
                st.markdown(narrativa)
                
                st.markdown("---")
                st.markdown("### 🚦 VEREDICTO")
                
                if apuesta['Prob'] > 0.70:
                    st.success("✅ **ALTAMENTE RECOMENDADA**")
                elif apuesta['Prob'] > 0.60:
                    st.info("🔵 **RECOMENDADA**")
                elif apuesta['Prob'] > 0.55:
                    st.warning("🟡 **CONSIDERAR**")
                else:
                    st.error("🔴 **NO RECOMENDADA**")
        
        st.markdown("---")
        st.warning("""
        ⚠️ **ADVERTENCIA:**
        
        - Sistema con ajustes por competición
        - Datos 100% reales de Football-Data.org
        - NO garantiza ganancias
        - Apuesta responsablemente
        
        📊 Análisis de 40+ mercados con argumentos técnicos.
        """)

if __name__ == "__main__":
    main()






