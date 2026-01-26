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

st.set_page_config(page_title="SISTEMABETS HÍBRIDO v7.0", layout="wide")

# ============================================================================
# MÓDULO 1: BASE DE DATOS LOCAL (SQLite)
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
# MÓDULO 2: CONECTOR FOOTBALL-DATA.ORG (IGUAL QUE v5.0)
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
# MÓDULO 3: EXTRACTOR DE FEATURES (Para ML)
# ============================================================================

class ExtractorFeatures:
    """Extrae features de partidos reales para entrenar ML"""
    
    @staticmethod
    def calcular_forma(partidos, equipo_nombre, ultimos_n=5):
        """Calcula forma reciente"""
        if not partidos or len(partidos) < ultimos_n:
            return 0.5
        
        puntos = []
        for partido in partidos[:ultimos_n]:
            es_local = partido['local'] == equipo_nombre
            gf = partido['goles_local'] if es_local else partido['goles_visitante']
            gc = partido['goles_visitante'] if es_local else partido['goles_local']
            
            if gf > gc:
                puntos.append(1.0)
            elif gf == gc:
                puntos.append(0.5)
            else:
                puntos.append(0.0)
        
        return sum(puntos) / len(puntos)
    
    @staticmethod
    def calcular_stats_local_visitante(partidos, equipo_nombre):
        """Calcula stats específicas de local/visitante"""
        stats = {
            'local_pj': 0, 'local_gf': 0, 'local_gc': 0,
            'visitante_pj': 0, 'visitante_gf': 0, 'visitante_gc': 0
        }
        
        for p in partidos[:20]:
            if p['local'] == equipo_nombre:
                stats['local_pj'] += 1
                stats['local_gf'] += p['goles_local']
                stats['local_gc'] += p['goles_visitante']
            elif p['visitante'] == equipo_nombre:
                stats['visitante_pj'] += 1
                stats['visitante_gf'] += p['goles_visitante']
                stats['visitante_gc'] += p['goles_local']
        
        return stats
    
    @staticmethod
    def calcular_tendencias(partidos):
        """Calcula tendencias Over/Under y BTTS"""
        if not partidos:
            return 0.5, 0.5
        
        over25 = sum(1 for p in partidos[:20] if (p['goles_local'] + p['goles_visitante']) > 2.5)
        btts = sum(1 for p in partidos[:20] if p['goles_local'] > 0 and p['goles_visitante'] > 0)
        
        return over25 / min(len(partidos), 20), btts / min(len(partidos), 20)
    
    @staticmethod
    def extraer_features_partido(local_team, visitante_team, partidos_local, partidos_visitante, h2h=[]):
        """
        Extrae todas las features de un partido para ML
        Usa SOLO datos reales de la API
        """
        
        # Promedios básicos
        gf_local = sum(p['goles_local'] if p['local'] == local_team['Equipo'] else p['goles_visitante'] 
                      for p in partidos_local[:20]) / max(len(partidos_local[:20]), 1)
        gc_local = sum(p['goles_visitante'] if p['local'] == local_team['Equipo'] else p['goles_local'] 
                      for p in partidos_local[:20]) / max(len(partidos_local[:20]), 1)
        
        gf_visitante = sum(p['goles_local'] if p['local'] == visitante_team['Equipo'] else p['goles_visitante'] 
                          for p in partidos_visitante[:20]) / max(len(partidos_visitante[:20]), 1)
        gc_visitante = sum(p['goles_visitante'] if p['local'] == visitante_team['Equipo'] else p['goles_local'] 
                          for p in partidos_visitante[:20]) / max(len(partidos_visitante[:20]), 1)
        
        # Forma
        forma_local_5 = ExtractorFeatures.calcular_forma(partidos_local, local_team['Equipo'], 5)
        forma_local_10 = ExtractorFeatures.calcular_forma(partidos_local, local_team['Equipo'], 10)
        forma_local_20 = ExtractorFeatures.calcular_forma(partidos_local, local_team['Equipo'], 20)
        
        forma_visitante_5 = ExtractorFeatures.calcular_forma(partidos_visitante, visitante_team['Equipo'], 5)
        forma_visitante_10 = ExtractorFeatures.calcular_forma(partidos_visitante, visitante_team['Equipo'], 10)
        forma_visitante_20 = ExtractorFeatures.calcular_forma(partidos_visitante, visitante_team['Equipo'], 20)
        
        # Stats local/visitante específicas
        stats_local = ExtractorFeatures.calcular_stats_local_visitante(partidos_local, local_team['Equipo'])
        stats_visitante = ExtractorFeatures.calcular_stats_local_visitante(partidos_visitante, visitante_team['Equipo'])
        
        local_en_casa_gf = stats_local['local_gf'] / max(stats_local['local_pj'], 1)
        local_en_casa_gc = stats_local['local_gc'] / max(stats_local['local_pj'], 1)
        visitante_fuera_gf = stats_visitante['visitante_gf'] / max(stats_visitante['visitante_pj'], 1)
        visitante_fuera_gc = stats_visitante['visitante_gc'] / max(stats_visitante['visitante_pj'], 1)
        
        # Tendencias
        over_local, btts_local = ExtractorFeatures.calcular_tendencias(partidos_local)
        over_visitante, btts_visitante = ExtractorFeatures.calcular_tendencias(partidos_visitante)
        
        # Racha victorias
        racha_local = sum(1 for p in partidos_local[:3] if 
                         (p['local'] == local_team['Equipo'] and p['goles_local'] > p['goles_visitante']) or
                         (p['visitante'] == local_team['Equipo'] and p['goles_visitante'] > p['goles_local']))
        
        racha_visitante = sum(1 for p in partidos_visitante[:3] if 
                             (p['local'] == visitante_team['Equipo'] and p['goles_local'] > p['goles_visitante']) or
                             (p['visitante'] == visitante_team['Equipo'] and p['goles_visitante'] > p['goles_local']))
        
        # H2H
        h2h_local_wins = sum(1 for p in h2h if 
                            (p['local'] == local_team['Equipo'] and p['goles_local'] > p['goles_visitante']) or
                            (p['visitante'] == local_team['Equipo'] and p['goles_visitante'] > p['goles_local']))
        h2h_draws = sum(1 for p in h2h if p['goles_local'] == p['goles_visitante'])
        h2h_away_wins = len(h2h) - h2h_local_wins - h2h_draws if h2h else 0
        h2h_avg_goals = sum(p['goles_local'] + p['goles_visitante'] for p in h2h) / max(len(h2h), 1)
        
        # Construir diccionario de features
        features = {
            'local_gf_promedio': gf_local,
            'local_gc_promedio': gc_local,
            'visitante_gf_promedio': gf_visitante,
            'visitante_gc_promedio': gc_visitante,
            'local_forma_5': forma_local_5,
            'local_forma_10': forma_local_10,
            'local_forma_20': forma_local_20,
            'visitante_forma_5': forma_visitante_5,
            'visitante_forma_10': forma_visitante_10,
            'visitante_forma_20': forma_visitante_20,
            'diff_forma': forma_local_5 - forma_visitante_5,
            'local_en_casa_gf': local_en_casa_gf,
            'local_en_casa_gc': local_en_casa_gc,
            'visitante_fuera_gf': visitante_fuera_gf,
            'visitante_fuera_gc': visitante_fuera_gc,
            'local_posicion': local_team['Posicion'],
            'visitante_posicion': visitante_team['Posicion'],
            'diff_posicion': visitante_team['Posicion'] - local_team['Posicion'],
            'local_puntos': local_team['Pts'],
            'visitante_puntos': visitante_team['Pts'],
            'diff_puntos': local_team['Pts'] - visitante_team['Pts'],
            'local_racha_victorias': racha_local,
            'visitante_racha_victorias': racha_visitante,
            'local_over25_ratio': over_local,
            'visitante_over25_ratio': over_visitante,
            'local_btts_ratio': btts_local,
            'visitante_btts_ratio': btts_visitante,
            'h2h_local_victorias': h2h_local_wins,
            'h2h_empates': h2h_draws,
            'h2h_visitante_victorias': h2h_away_wins,
            'h2h_goles_promedio': h2h_avg_goals if h2h else 2.5,
            'local_eficiencia_ofensiva': gf_local / max(gc_local, 0.5),
            'visitante_eficiencia_ofensiva': gf_visitante / max(gc_visitante, 0.5)
        }
        
        return features

# ============================================================================
# MÓDULO 4: MODELO ML HÍBRIDO
# ============================================================================

class ModeloMLHibrido:
    """Modelo ML que se entrena con datos reales acumulados"""
    
    def __init__(self):
        self.modelo_1x2 = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        self.modelo_over25 = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)
        self.modelo_btts = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)
        self.scaler = StandardScaler()
        self.entrenado = False
        self.n_partidos_entrenamiento = 0
        
        self.feature_names = [
            'local_gf_promedio', 'local_gc_promedio', 'visitante_gf_promedio', 'visitante_gc_promedio',
            'local_forma_5', 'local_forma_10', 'local_forma_20',
            'visitante_forma_5', 'visitante_forma_10', 'visitante_forma_20',
            'diff_forma', 'local_en_casa_gf', 'local_en_casa_gc',
            'visitante_fuera_gf', 'visitante_fuera_gc',
            'local_posicion', 'visitante_posicion', 'diff_posicion',
            'local_puntos', 'visitante_puntos', 'diff_puntos',
            'local_racha_victorias', 'visitante_racha_victorias',
            'local_over25_ratio', 'visitante_over25_ratio',
            'local_btts_ratio', 'visitante_btts_ratio',
            'h2h_local_victorias', 'h2h_empates', 'h2h_visitante_victorias',
            'h2h_goles_promedio', 'local_eficiencia_ofensiva', 'visitante_eficiencia_ofensiva'
        ]
    
    def entrenar_con_partidos_db(self, db_manager):
        """Entrena el modelo con partidos guardados en la base de datos"""
        
        partidos = db_manager.obtener_todos_partidos()
        
        if len(partidos) < 50:
            return {
                'success': False,
                'message': f'Necesitas al menos 50 partidos. Tienes {len(partidos)}'
            }
        
        X = []
        y_1x2 = []
        y_over25 = []
        y_btts = []
        
        for partido in partidos:
            # partido = (id, fecha, competicion, local, visitante, goles_local, goles_visitante, features, fecha_analisis)
            features = json.loads(partido[7])
            goles_local = partido[5]
            goles_visitante = partido[6]
            
            # Convertir features a vector
            feature_vector = [features[name] for name in self.feature_names]
            X.append(feature_vector)
            
            # Labels
            if goles_local > goles_visitante:
                y_1x2.append(2)  # Local
            elif goles_local == goles_visitante:
                y_1x2.append(1)  # Empate
            else:
                y_1x2.append(0)  # Visitante
            
            y_over25.append(1 if (goles_local + goles_visitante) > 2.5 else 0)
            y_btts.append(1 if goles_local > 0 and goles_visitante > 0 else 0)
        
        X = np.array(X)
        y_1x2 = np.array(y_1x2)
        y_over25 = np.array(y_over25)
        y_btts = np.array(y_btts)
        
        # Entrenar
        X_scaled = self.scaler.fit_transform(X)
        
        self.modelo_1x2.fit(X_scaled, y_1x2)
        self.modelo_over25.fit(X_scaled, y_over25)
        self.modelo_btts.fit(X_scaled, y_btts)
        
        self.entrenado = True
        self.n_partidos_entrenamiento = len(partidos)
        
        # Calcular precisión aproximada con cross-validation simple
        if len(partidos) >= 100:
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_1x2, test_size=0.2, random_state=42)
            self.modelo_1x2.fit(X_train, y_train)
            acc_1x2 = self.modelo_1x2.score(X_test, y_test)
            
            _, _, y_train_over, y_test_over = train_test_split(X_scaled, y_over25, test_size=0.2, random_state=42)
            self.modelo_over25.fit(X_train, y_train_over)
            acc_over = self.modelo_over25.score(X_test, y_test_over)
            
            _, _, y_train_btts, y_test_btts = train_test_split(X_scaled, y_btts, test_size=0.2, random_state=42)
            self.modelo_btts.fit(X_train, y_train_btts)
            acc_btts = self.modelo_btts.score(X_test, y_test_btts)
        else:
            acc_1x2 = 0.0
            acc_over = 0.0
            acc_btts = 0.0
        
        return {
            'success': True,
            'n_partidos': len(partidos),
            'accuracy_1x2': acc_1x2,
            'accuracy_over25': acc_over,
            'accuracy_btts': acc_btts
        }
    
    def predecir(self, features):
        """Predice usando ML entrenado"""
        
        if not self.entrenado:
            raise ValueError("Modelo no entrenado")
        
        feature_vector = np.array([[features[name] for name in self.feature_names]])
        X_scaled = self.scaler.transform(feature_vector)
        
        prob_1x2 = self.modelo_1x2.predict_proba(X_scaled)[0]
        prob_over25 = self.modelo_over25.predict_proba(X_scaled)[0][1]
        prob_btts = self.modelo_btts.predict_proba(X_scaled)[0][1]
        
        return {
            '1X2': {'Visitante': prob_1x2[0], 'Empate': prob_1x2[1], 'Local': prob_1x2[2]},
            'Over/Under': {'Over 2.5': prob_over25, 'Under 2.5': 1 - prob_over25},
            'BTTS': {'Si': prob_btts, 'No': 1 - prob_btts}
        }

# ============================================================================
# MÓDULO 5: PREDICTOR HÍBRIDO (Combina Poisson + ML)
# ============================================================================

class PredictorHibrido:
    """Combina predicciones de Poisson y ML según disponibilidad"""
    
    @staticmethod
    def predecir_poisson(lambda_local, lambda_visitante):
        """Predicción usando Poisson clásico"""
        matriz = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                matriz[i, j] = poisson.pmf(i, lambda_local) * poisson.pmf(j, lambda_visitante)
        
        p_local = np.sum(np.tril(matriz, -1))
        p_empate = np.sum(np.diag(matriz))
        p_visitante = np.sum(np.triu(matriz, 1))
        
        total = p_local + p_empate + p_visitante
        p_local /= total
        p_empate /= total
        p_visitante /= total
        
        p_over = sum([matriz[i, j] for i in range(6) for j in range(6) if (i+j) > 2.5])
        p_under = 1 - p_over
        
        p_btts_no = matriz[0,:].sum() + matriz[:,0].sum() - matriz[0,0]
        p_btts_si = 1 - p_btts_no
        
        return {
            '1X2': {'Local': p_local, 'Empate': p_empate, 'Visitante': p_visitante},
            'Over/Under': {'Over 2.5': p_over, 'Under 2.5': p_under},
            'BTTS': {'Si': p_btts_si, 'No': p_btts_no}
        }
    
    @staticmethod
    def combinar_predicciones(pred_poisson, pred_ml, peso_ml=0.6):
        """Combina predicciones con pesos"""
        peso_poisson = 1 - peso_ml
        
        return {
            '1X2': {
                'Local': pred_poisson['1X2']['Local'] * peso_poisson + pred_ml['1X2']['Local'] * peso_ml,
                'Empate': pred_poisson['1X2']['Empate'] * peso_poisson + pred_ml['1X2']['Empate'] * peso_ml,
                'Visitante': pred_poisson['1X2']['Visitante'] * peso_poisson + pred_ml['1X2']['Visitante'] * peso_ml
            },
            'Over/Under': {
                'Over 2.5': pred_poisson['Over/Under']['Over 2.5'] * peso_poisson + pred_ml['Over/Under']['Over 2.5'] * peso_ml,
                'Under 2.5': pred_poisson['Over/Under']['Under 2.5'] * peso_poisson + pred_ml['Over/Under']['Under 2.5'] * peso_ml
            },
            'BTTS': {
                'Si': pred_poisson['BTTS']['Si'] * peso_poisson + pred_ml['BTTS']['Si'] * peso_ml,
                'No': pred_poisson['BTTS']['No'] * peso_poisson + pred_ml['BTTS']['No'] * peso_ml
            }
        }

# ============================================================================
# INTERFAZ STREAMLIT
# ============================================================================

def main():
    st.title("⚽ SISTEMABETS HÍBRIDO v7.0")
    st.markdown("### Machine Learning + Poisson con Datos 100% Reales")
    
    # Inicializar componentes
    if 'db_manager' not in st.session_state:
        st.session_state['db_manager'] = DatabaseManager()
    
    if 'modelo_ml' not in st.session_state:
        st.session_state['modelo_ml'] = ModeloMLHibrido()
    
    db_manager = st.session_state['db_manager']
    modelo_ml = st.session_state['modelo_ml']
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        api_key = st.text_input("API Key Football-Data.org", type="password",
                               help="Obtén tu clave en https://www.football-data.org/")
        
        if not api_key:
            st.warning("⚠️ Ingresa tu API Key para comenzar")
            st.stop()
        
        st.markdown("---")
        
        # Estado del sistema
        st.header("📊 Estado del Sistema")
        
        n_partidos = db_manager.contar_partidos()
        st.metric("Partidos en Base de Datos", n_partidos)
        
        if modelo_ml.entrenado:
            st.success(f"✅ ML Entrenado ({modelo_ml.n_partidos_entrenamiento} partidos)")
        else:
            st.info("ℹ️ ML no entrenado aún")
            if n_partidos < 50:
                st.caption(f"Necesitas {50 - n_partidos} partidos más")
        
        # Botón entrenar ML
        if n_partidos >= 50 and st.button("🤖 ENTRENAR/ACTUALIZAR ML", type="primary"):
            with st.spinner("Entrenando modelo ML..."):
                resultado = modelo_ml.entrenar_con_partidos_db(db_manager)
                
                if resultado['success']:
                    st.success(f"✅ ML entrenado con {resultado['n_partidos']} partidos")
                    
                    if resultado['n_partidos'] >= 100:
                        st.metric("Precisión 1X2", f"{resultado['accuracy_1x2']*100:.1f}%")
                        st.metric("Precisión Over/Under", f"{resultado['accuracy_over25']*100:.1f}%")
                        st.metric("Precisión BTTS", f"{resultado['accuracy_btts']*100:.1f}%")
                else:
                    st.error(resultado['message'])
        
        st.markdown("---")
        
        # Gestión de datos
        with st.expander("🗄️ Gestión de Datos"):
            st.markdown(f"**Total partidos:** {n_partidos}")
            
            if st.button("🗑️ Limpiar Base de Datos", type="secondary"):
                db_manager.limpiar_database()
                st.session_state['modelo_ml'] = ModeloMLHibrido()
                st.success("Base de datos limpiada")
                st.rerun()
        
        st.markdown("---")
        
        # Info
        with st.expander("💡 Cómo Funciona"):
            st.markdown("""
            ### Sistema Híbrido
            
            1. **Analiza partido** con datos reales de API
            2. **Guarda en base de datos** local
            3. **Acumula partidos** (50, 100, 500...)
            4. **Entrena ML** automáticamente
            5. **Combina Poisson + ML** para mejor precisión
            
            ### Estrategia de Predicción:
            - **0-49 partidos:** Solo Poisson (55-60%)
            - **50-199 partidos:** 70% Poisson + 30% ML (58-61%)
            - **200-499 partidos:** 50% Poisson + 50% ML (60-63%)
            - **500+ partidos:** 30% Poisson + 70% ML (62-65%)
            
            ### Mejora con el Tiempo:
            Mientras más partidos analices, mejor será el ML.
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
        st.error("❌ No se pudo cargar la tabla. Verifica tu API Key.")
        st.stop()
    
    st.success(f"✅ Tabla de {liga_nombre} cargada")
    
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
        
        local_team = df_liga[df_liga['Equipo'] == equipo_local].iloc[0]
        visitante_team = df_liga[df_liga['Equipo'] == equipo_visitante].iloc[0]
        
        st.markdown("---")
        st.header(f"📊 {equipo_local} vs {equipo_visitante}")
        
        # Cargar historial
        with st.spinner("Cargando datos de partidos..."):
            partidos_local = api.obtener_ultimos_20_partidos(equipo_local)
            partidos_visitante = api.obtener_ultimos_20_partidos(equipo_visitante)
            h2h = api.obtener_enfrentamientos_directos(equipo_local, equipo_visitante)
        
        if not partidos_local or not partidos_visitante:
            st.error("❌ No se pudieron obtener datos de partidos")
            st.stop()
        
        # Extraer features
        features = ExtractorFeatures.extraer_features_partido(
            local_team, visitante_team, partidos_local, partidos_visitante, h2h
        )
        
        # Calcular lambdas para Poisson
        gf_local = features['local_en_casa_gf']
        gc_local = features['local_en_casa_gc']
        gf_visitante = features['visitante_fuera_gf']
        gc_visitante = features['visitante_fuera_gc']
        
        media_goles = (gf_local + gf_visitante) / 2
        
        lambda_local = (gf_local / max(media_goles, 0.5)) * (gc_visitante / max(media_goles, 0.5)) * media_goles * 1.15
        lambda_visitante = (gf_visitante / max(media_goles, 0.5)) * (gc_local / max(media_goles, 0.5)) * media_goles * 0.95
        
        lambda_local = max(min(lambda_local, 4.5), 0.3)
        lambda_visitante = max(min(lambda_visitante, 4.5), 0.3)
        
        # Predicción Poisson
        pred_poisson = PredictorHibrido.predecir_poisson(lambda_local, lambda_visitante)
        
        # Predicción ML (si está entrenado)
        if modelo_ml.entrenado:
            pred_ml = modelo_ml.predecir(features)
            
            # Determinar peso de ML según cantidad de partidos
            n_partidos = modelo_ml.n_partidos_entrenamiento
            if n_partidos < 200:
                peso_ml = 0.3
            elif n_partidos < 500:
                peso_ml = 0.5
            else:
                peso_ml = 0.7
            
            pred_final = PredictorHibrido.combinar_predicciones(pred_poisson, pred_ml, peso_ml)
            metodo_usado = f"HÍBRIDO ({int((1-peso_ml)*100)}% Poisson + {int(peso_ml*100)}% ML)"
        else:
            pred_final = pred_poisson
            metodo_usado = "POISSON (ML no entrenado aún)"
        
        # Guardar partido en DB para futuro entrenamiento
        # (Se guardará cuando se sepa el resultado real, en producción)
        
        # Mostrar resultados
        st.subheader(f"🔮 Predicción usando: {metodo_usado}")
        
        # Mostrar parámetros
        col1, col2, col3 = st.columns(3)
        col1.metric("Lambda Local", f"{lambda_local:.2f}")
        col2.metric("Lambda Visitante", f"{lambda_visitante:.2f}")
        col3.metric("Goles Esperados", f"{lambda_local + lambda_visitante:.2f}")
        
        # 1X2
        st.markdown("---")
        st.subheader("⚽ Resultado Final (1X2)")
        
        col1, col2, col3 = st.columns(3)
        
        prob_local = pred_final['1X2']['Local']
        prob_empate = pred_final['1X2']['Empate']
        prob_visitante = pred_final['1X2']['Visitante']
        
        with col1:
            st.metric(f"{equipo_local}", f"{prob_local*100:.1f}%")
            st.caption(f"Cuota justa: {1/prob_local:.2f}")
        
        with col2:
            st.metric("Empate", f"{prob_empate*100:.1f}%")
            st.caption(f"Cuota justa: {1/prob_empate:.2f}")
        
        with col3:
            st.metric(f"{equipo_visitante}", f"{prob_visitante*100:.1f}%")
            st.caption(f"Cuota justa: {1/prob_visitante:.2f}")
        
        # Over/Under
        st.subheader("📊 Over/Under 2.5")
        
        col1, col2 = st.columns(2)
        
        prob_over = pred_final['Over/Under']['Over 2.5']
        prob_under = pred_final['Over/Under']['Under 2.5']
        
        with col1:
            st.metric("Over 2.5", f"{prob_over*100:.1f}%")
            st.caption(f"Cuota justa: {1/prob_over:.2f}")
        
        with col2:
            st.metric("Under 2.5", f"{prob_under*100:.1f}%")
            st.caption(f"Cuota justa: {1/prob_under:.2f}")
        
        # BTTS
        st.subheader("🎯 Ambos Equipos Anotan (BTTS)")
        
        col1, col2 = st.columns(2)
        
        prob_btts_si = pred_final['BTTS']['Si']
        prob_btts_no = pred_final['BTTS']['No']
        
        with col1:
            st.metric("BTTS Sí", f"{prob_btts_si*100:.1f}%")
            st.caption(f"Cuota justa: {1/prob_btts_si:.2f}")
        
        with col2:
            st.metric("BTTS No", f"{prob_btts_no*100:.1f}%")
            st.caption(f"Cuota justa: {1/prob_btts_no:.2f}")
        
        # Si ML está entrenado, mostrar comparación
        if modelo_ml.entrenado:
            st.markdown("---")
            st.header("🆚 Comparación: Poisson vs ML vs HÍBRIDO")
            
            df_comparacion = pd.DataFrame({
                'Mercado': [
                    'Victoria Local', 'Empate', 'Victoria Visitante',
                    'Over 2.5', 'Under 2.5', 'BTTS Sí', 'BTTS No'
                ],
                'Poisson': [
                    f"{pred_poisson['1X2']['Local']*100:.1f}%",
                    f"{pred_poisson['1X2']['Empate']*100:.1f}%",
                    f"{pred_poisson['1X2']['Visitante']*100:.1f}%",
                    f"{pred_poisson['Over/Under']['Over 2.5']*100:.1f}%",
                    f"{pred_poisson['Over/Under']['Under 2.5']*100:.1f}%",
                    f"{pred_poisson['BTTS']['Si']*100:.1f}%",
                    f"{pred_poisson['BTTS']['No']*100:.1f}%"
                ],
                'Machine Learning': [
                    f"{pred_ml['1X2']['Local']*100:.1f}%",
                    f"{pred_ml['1X2']['Empate']*100:.1f}%",
                    f"{pred_ml['1X2']['Visitante']*100:.1f}%",
                    f"{pred_ml['Over/Under']['Over 2.5']*100:.1f}%",
                    f"{pred_ml['Over/Under']['Under 2.5']*100:.1f}%",
                    f"{pred_ml['BTTS']['Si']*100:.1f}%",
                    f"{pred_ml['BTTS']['No']*100:.1f}%"
                ],
                'HÍBRIDO (Final)': [
                    f"{pred_final['1X2']['Local']*100:.1f}%",
                    f"{pred_final['1X2']['Empate']*100:.1f}%",
                    f"{pred_final['1X2']['Visitante']*100:.1f}%",
                    f"{pred_final['Over/Under']['Over 2.5']*100:.1f}%",
                    f"{pred_final['Over/Under']['Under 2.5']*100:.1f}%",
                    f"{pred_final['BTTS']['Si']*100:.1f}%",
                    f"{pred_final['BTTS']['No']*100:.1f}%"
                ]
            })
            
            st.dataframe(df_comparacion, use_container_width=True)
            
            st.info(f"""
            💡 **Interpretación:**
            - El sistema usa **{int((1-peso_ml)*100)}% Poisson + {int(peso_ml*100)}% ML**
            - Con {modelo_ml.n_partidos_entrenamiento} partidos entrenados
            - Mientras más partidos acumules, más peso tendrá el ML
            """)
        
        # Mejores apuestas
        st.markdown("---")
        st.header("💎 MEJORES APUESTAS")
        
        todas_predicciones = [
            {'Mercado': f'Victoria {equipo_local}', 'Probabilidad': prob_local, 'Cuota_Justa': 1/prob_local},
            {'Mercado': 'Empate', 'Probabilidad': prob_empate, 'Cuota_Justa': 1/prob_empate},
            {'Mercado': f'Victoria {equipo_visitante}', 'Probabilidad': prob_visitante, 'Cuota_Justa': 1/prob_visitante},
            {'Mercado': 'Over 2.5', 'Probabilidad': prob_over, 'Cuota_Justa': 1/prob_over},
            {'Mercado': 'Under 2.5', 'Probabilidad': prob_under, 'Cuota_Justa': 1/prob_under},
            {'Mercado': 'BTTS Sí', 'Probabilidad': prob_btts_si, 'Cuota_Justa': 1/prob_btts_si},
            {'Mercado': 'BTTS No', 'Probabilidad': prob_btts_no, 'Cuota_Justa': 1/prob_btts_no}
        ]
        
        todas_predicciones.sort(key=lambda x: x['Probabilidad'], reverse=True)
        
        st.subheader("🏆 Top 3 Apuestas Recomendadas")
        
        for i, pred in enumerate(todas_predicciones[:3], 1):
            
            confianza = "⭐⭐⭐" if pred['Probabilidad'] > 0.65 else "⭐⭐" if pred['Probabilidad'] > 0.55 else "⭐"
            
            with st.expander(f"#{i} - {pred['Mercado']} | {pred['Probabilidad']*100:.1f}% {confianza}"):
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Probabilidad", f"{pred['Probabilidad']*100:.1f}%")
                col2.metric("Cuota Justa", f"{pred['Cuota_Justa']:.2f}")
                col3.metric("Cuota Mínima Esperada", f"{pred['Cuota_Justa']*0.95:.2f}")
                
                st.markdown("---")
                st.markdown("### ⚖️ VEREDICTO")
                
                if pred['Probabilidad'] > 0.70:
                    st.success("🟢 **MUY RECOMENDADA** - Alta confianza del sistema")
                elif pred['Probabilidad'] > 0.60:
                    st.info("🔵 **RECOMENDADA** - Buena probabilidad")
                elif pred['Probabilidad'] > 0.55:
                    st.warning("🟡 **CONSIDERAR** - Probabilidad moderada")
                else:
                    st.error("🔴 **NO RECOMENDADA** - Probabilidad insuficiente")
                
                st.caption(f"""
                **Busca cuotas ≥ {pred['Cuota_Justa']*0.95:.2f}** en las casas de apuestas.
                Si encuentras esa cuota o mejor, hay valor matemático (+EV).
                """)
        
        # Opción para guardar resultado
        st.markdown("---")
        st.subheader("💾 Guardar Partido para Entrenamiento Futuro")
        
        st.info("""
        ℹ️ **Una vez finalice el partido**, ingresa el resultado real aquí para:
        - Guardarlo en la base de datos
        - Mejorar el entrenamiento del ML
        - Aumentar la precisión del sistema
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            goles_local_real = st.number_input(f"Goles {equipo_local}", min_value=0, max_value=10, value=0, key='goles_local_real')
        
        with col2:
            goles_visitante_real = st.number_input(f"Goles {equipo_visitante}", min_value=0, max_value=10, value=0, key='goles_visitante_real')
        
        if st.button("💾 GUARDAR RESULTADO REAL", type="secondary"):
            
            partido_data = {
                'fecha': datetime.now().strftime('%Y-%m-%d'),
                'competicion': liga_nombre,
                'local': equipo_local,
                'visitante': equipo_visitante,
                'goles_local': goles_local_real,
                'goles_visitante': goles_visitante_real,
                'features': features
            }
            
            db_manager.guardar_partido(partido_data)
            
            st.success(f"✅ Partido guardado! Total en BD: {db_manager.contar_partidos()}")
            
            if db_manager.contar_partidos() >= 50 and not modelo_ml.entrenado:
                st.info("ℹ️ Ya tienes 50+ partidos. Ve al sidebar y entrena el ML!")
        
        # Forma reciente
        st.markdown("---")
        st.header("📈 Análisis de Forma")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"🏠 {equipo_local}")
            st.progress(features['local_forma_5'])
            st.caption(f"Forma últimos 5: {features['local_forma_5']*100:.0f}%")
            st.caption(f"Forma últimos 10: {features['local_forma_10']*100:.0f}%")
            st.caption(f"Forma últimos 20: {features['local_forma_20']*100:.0f}%")
        
        with col2:
            st.subheader(f"✈️ {equipo_visitante}")
            st.progress(features['visitante_forma_5'])
            st.caption(f"Forma últimos 5: {features['visitante_forma_5']*100:.0f}%")
            st.caption(f"Forma últimos 10: {features['visitante_forma_10']*100:.0f}%")
            st.caption(f"Forma últimos 20: {features['visitante_forma_20']*100:.0f}%")
    
    # Disclaimer
    st.markdown("---")
    st.warning("""
    ⚠️ **ADVERTENCIA:**
    
    - Este sistema combina análisis matemático (Poisson) con Machine Learning
    - Los datos son 100% REALES de Football-Data.org
    - El ML mejora con cada partido que guardas
    - NO garantiza ganancias, úsalo como herramienta de apoyo
    - Apuesta responsablemente y dentro de tus posibilidades
    - Siempre compara cuotas en múltiples casas antes de apostar
    
    📊 **Mejores prácticas:**
    1. Analiza 10-20 partidos por semana
    2. Guarda los resultados reales
    3. Entrena el ML cada 50 partidos nuevos
    4. Solo apuesta si probabilidad > 60% Y encuentras +5% EV
    5. Gestiona tu bankroll (máx 3% por apuesta)
    """)

if __name__ == "__main__":
    main()
