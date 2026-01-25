import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from scipy.stats import poisson
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SISTEMABETS ML v6.0", layout="wide")

# ============================================================================
# MÓDULO 1: RECOLECTOR DE DATOS HISTÓRICOS
# ============================================================================

class RecolectorDatos:
    """Recolecta y prepara datos históricos para entrenar el modelo"""
    
    @staticmethod
    def obtener_features_partido(local_stats, visitante_stats, local_team, visitante_team, h2h_stats=None):
        """
        Extrae features (características) de un partido para ML
        
        FEATURES IMPORTANTES:
        - Forma reciente (últimos 5, 10, 20 partidos)
        - Stats ofensivas/defensivas
        - Diferencia de posición en tabla
        - Rendimiento local/visitante específico
        - Historial H2H
        - Racha actual
        """
        
        features = {}
        
        # 1. STATS BÁSICAS
        features['local_gf_promedio'] = local_stats.get('gf_promedio', 0)
        features['local_gc_promedio'] = local_stats.get('gc_promedio', 0)
        features['visitante_gf_promedio'] = visitante_stats.get('gf_promedio', 0)
        features['visitante_gc_promedio'] = visitante_stats.get('gc_promedio', 0)
        
        # 2. FORMA RECIENTE (ponderada)
        features['local_forma_5'] = local_stats.get('forma_5', 0.5)
        features['local_forma_10'] = local_stats.get('forma_10', 0.5)
        features['local_forma_20'] = local_stats.get('forma_20', 0.5)
        features['visitante_forma_5'] = visitante_stats.get('forma_5', 0.5)
        features['visitante_forma_10'] = visitante_stats.get('forma_10', 0.5)
        features['visitante_forma_20'] = visitante_stats.get('forma_20', 0.5)
        
        # 3. DIFERENCIA DE FORMA
        features['diff_forma'] = features['local_forma_5'] - features['visitante_forma_5']
        
        # 4. STATS ESPECÍFICAS LOCAL/VISITANTE
        features['local_en_casa_gf'] = local_stats.get('local_gf', 0)
        features['local_en_casa_gc'] = local_stats.get('local_gc', 0)
        features['visitante_fuera_gf'] = visitante_stats.get('visitante_gf', 0)
        features['visitante_fuera_gc'] = visitante_stats.get('visitante_gc', 0)
        
        # 5. POSICIÓN EN TABLA
        features['local_posicion'] = local_team.get('Posicion', 10)
        features['visitante_posicion'] = visitante_team.get('Posicion', 10)
        features['diff_posicion'] = features['visitante_posicion'] - features['local_posicion']
        
        # 6. PUNTOS
        features['local_puntos'] = local_team.get('Pts', 0)
        features['visitante_puntos'] = visitante_team.get('Pts', 0)
        features['diff_puntos'] = features['local_puntos'] - features['visitante_puntos']
        
        # 7. RACHA (últimos 3 partidos)
        features['local_racha_victorias'] = local_stats.get('racha_victorias', 0)
        features['visitante_racha_victorias'] = visitante_stats.get('racha_victorias', 0)
        
        # 8. OVER/UNDER TENDENCIAS
        features['local_over25_ratio'] = local_stats.get('over25_ratio', 0.5)
        features['visitante_over25_ratio'] = visitante_stats.get('over25_ratio', 0.5)
        
        # 9. BTTS TENDENCIAS
        features['local_btts_ratio'] = local_stats.get('btts_ratio', 0.5)
        features['visitante_btts_ratio'] = visitante_stats.get('btts_ratio', 0.5)
        
        # 10. H2H (si existe)
        if h2h_stats:
            features['h2h_local_victorias'] = h2h_stats.get('local_wins', 0)
            features['h2h_empates'] = h2h_stats.get('draws', 0)
            features['h2h_visitante_victorias'] = h2h_stats.get('away_wins', 0)
            features['h2h_goles_promedio'] = h2h_stats.get('avg_goals', 2.5)
        else:
            features['h2h_local_victorias'] = 0
            features['h2h_empates'] = 0
            features['h2h_visitante_victorias'] = 0
            features['h2h_goles_promedio'] = 2.5
        
        # 11. EFICIENCIA OFENSIVA/DEFENSIVA
        features['local_eficiencia_ofensiva'] = features['local_gf_promedio'] / max(features['local_gc_promedio'], 0.5)
        features['visitante_eficiencia_ofensiva'] = features['visitante_gf_promedio'] / max(features['visitante_gc_promedio'], 0.5)
        
        return features
    
    @staticmethod
    def crear_dataset_historico(partidos_historicos):
        """
        Convierte partidos históricos en dataset para entrenar
        
        partidos_historicos = [
            {
                'local': 'Team A',
                'visitante': 'Team B',
                'goles_local': 2,
                'goles_visitante': 1,
                'features': {...}  # features calculadas arriba
            },
            ...
        ]
        
        Returns: X (features), y_1x2, y_over25, y_btts
        """
        
        X = []
        y_1x2 = []  # 0=Visitante, 1=Empate, 2=Local
        y_over25 = []  # 0=Under, 1=Over
        y_btts = []  # 0=No, 1=Si
        
        for partido in partidos_historicos:
            features = partido['features']
            goles_local = partido['goles_local']
            goles_visitante = partido['goles_visitante']
            
            # Convertir features dict a lista ordenada
            feature_vector = [
                features['local_gf_promedio'],
                features['local_gc_promedio'],
                features['visitante_gf_promedio'],
                features['visitante_gc_promedio'],
                features['local_forma_5'],
                features['local_forma_10'],
                features['local_forma_20'],
                features['visitante_forma_5'],
                features['visitante_forma_10'],
                features['visitante_forma_20'],
                features['diff_forma'],
                features['local_en_casa_gf'],
                features['local_en_casa_gc'],
                features['visitante_fuera_gf'],
                features['visitante_fuera_gc'],
                features['local_posicion'],
                features['visitante_posicion'],
                features['diff_posicion'],
                features['local_puntos'],
                features['visitante_puntos'],
                features['diff_puntos'],
                features['local_racha_victorias'],
                features['visitante_racha_victorias'],
                features['local_over25_ratio'],
                features['visitante_over25_ratio'],
                features['local_btts_ratio'],
                features['visitante_btts_ratio'],
                features['h2h_local_victorias'],
                features['h2h_empates'],
                features['h2h_visitante_victorias'],
                features['h2h_goles_promedio'],
                features['local_eficiencia_ofensiva'],
                features['visitante_eficiencia_ofensiva']
            ]
            
            X.append(feature_vector)
            
            # Labels (resultados reales)
            if goles_local > goles_visitante:
                y_1x2.append(2)  # Local ganó
            elif goles_local == goles_visitante:
                y_1x2.append(1)  # Empate
            else:
                y_1x2.append(0)  # Visitante ganó
            
            total_goles = goles_local + goles_visitante
            y_over25.append(1 if total_goles > 2.5 else 0)
            y_btts.append(1 if goles_local > 0 and goles_visitante > 0 else 0)
        
        return np.array(X), np.array(y_1x2), np.array(y_over25), np.array(y_btts)

# ============================================================================
# MÓDULO 2: MODELOS DE MACHINE LEARNING
# ============================================================================

class ModeloML:
    """
    Modelo de Machine Learning para predicción de partidos
    
    Usa 3 modelos independientes:
    - Modelo 1X2 (clasificación multiclase)
    - Modelo Over/Under 2.5 (clasificación binaria)
    - Modelo BTTS (clasificación binaria)
    """
    
    def __init__(self):
        # Random Forest: Bueno para capturar relaciones no lineales
        self.modelo_1x2 = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            random_state=42
        )
        
        self.modelo_over25 = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        self.modelo_btts = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        self.scaler = StandardScaler()
        self.entrenado = False
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
    
    def entrenar(self, X, y_1x2, y_over25, y_btts):
        """
        Entrena los 3 modelos con datos históricos
        
        Returns: métricas de precisión
        """
        
        # Normalizar features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split train/test (80/20)
        X_train, X_test, y_1x2_train, y_1x2_test = train_test_split(
            X_scaled, y_1x2, test_size=0.2, random_state=42
        )
        
        _, _, y_over_train, y_over_test = train_test_split(
            X_scaled, y_over25, test_size=0.2, random_state=42
        )
        
        _, _, y_btts_train, y_btts_test = train_test_split(
            X_scaled, y_btts, test_size=0.2, random_state=42
        )
        
        # Entrenar modelos
        self.modelo_1x2.fit(X_train, y_1x2_train)
        self.modelo_over25.fit(X_train, y_over_train)
        self.modelo_btts.fit(X_train, y_btts_train)
        
        # Evaluar
        acc_1x2 = self.modelo_1x2.score(X_test, y_1x2_test)
        acc_over = self.modelo_over25.score(X_test, y_over_test)
        acc_btts = self.modelo_btts.score(X_test, y_btts_test)
        
        self.entrenado = True
        
        return {
            'accuracy_1x2': acc_1x2,
            'accuracy_over25': acc_over,
            'accuracy_btts': acc_btts,
            'n_samples': len(X)
        }
    
    def predecir(self, features):
        """
        Predice resultado de un partido
        
        Returns: probabilidades para cada mercado
        """
        
        if not self.entrenado:
            raise ValueError("El modelo debe ser entrenado primero")
        
        # Convertir features dict a array
        feature_vector = np.array([[
            features['local_gf_promedio'],
            features['local_gc_promedio'],
            features['visitante_gf_promedio'],
            features['visitante_gc_promedio'],
            features['local_forma_5'],
            features['local_forma_10'],
            features['local_forma_20'],
            features['visitante_forma_5'],
            features['visitante_forma_10'],
            features['visitante_forma_20'],
            features['diff_forma'],
            features['local_en_casa_gf'],
            features['local_en_casa_gc'],
            features['visitante_fuera_gf'],
            features['visitante_fuera_gc'],
            features['local_posicion'],
            features['visitante_posicion'],
            features['diff_posicion'],
            features['local_puntos'],
            features['visitante_puntos'],
            features['diff_puntos'],
            features['local_racha_victorias'],
            features['visitante_racha_victorias'],
            features['local_over25_ratio'],
            features['visitante_over25_ratio'],
            features['local_btts_ratio'],
            features['visitante_btts_ratio'],
            features['h2h_local_victorias'],
            features['h2h_empates'],
            features['h2h_visitante_victorias'],
            features['h2h_goles_promedio'],
            features['local_eficiencia_ofensiva'],
            features['visitante_eficiencia_ofensiva']
        ]])
        
        # Normalizar
        X_scaled = self.scaler.transform(feature_vector)
        
        # Predecir probabilidades
        prob_1x2 = self.modelo_1x2.predict_proba(X_scaled)[0]  # [prob_visitante, prob_empate, prob_local]
        prob_over25 = self.modelo_over25.predict_proba(X_scaled)[0][1]  # probabilidad Over
        prob_btts = self.modelo_btts.predict_proba(X_scaled)[0][1]  # probabilidad BTTS Si
        
        return {
            '1X2': {
                'Visitante': prob_1x2[0],
                'Empate': prob_1x2[1],
                'Local': prob_1x2[2]
            },
            'Over/Under': {
                'Over 2.5': prob_over25,
                'Under 2.5': 1 - prob_over25
            },
            'BTTS': {
                'Si': prob_btts,
                'No': 1 - prob_btts
            }
        }
    
    def obtener_importancia_features(self):
        """
        Muestra qué features son más importantes para cada modelo
        """
        
        if not self.entrenado:
            return None
        
        importancia_1x2 = self.modelo_1x2.feature_importances_
        importancia_over = self.modelo_over25.feature_importances_
        importancia_btts = self.modelo_btts.feature_importances_
        
        df_importancia = pd.DataFrame({
            'Feature': self.feature_names,
            'Importancia_1X2': importancia_1x2,
            'Importancia_Over25': importancia_over,
            'Importancia_BTTS': importancia_btts
        })
        
        df_importancia = df_importancia.sort_values('Importancia_1X2', ascending=False)
        
        return df_importancia
    
    def guardar_modelo(self, path='modelo_ml.pkl'):
        """Guarda el modelo entrenado"""
        joblib.dump({
            'modelo_1x2': self.modelo_1x2,
            'modelo_over25': self.modelo_over25,
            'modelo_btts': self.modelo_btts,
            'scaler': self.scaler,
            'entrenado': self.entrenado
        }, path)
    
    def cargar_modelo(self, path='modelo_ml.pkl'):
        """Carga un modelo previamente entrenado"""
        data = joblib.dump(path)
        self.modelo_1x2 = data['modelo_1x2']
        self.modelo_over25 = data['modelo_over25']
        self.modelo_btts = data['modelo_btts']
        self.scaler = data['scaler']
        self.entrenado = data['entrenado']

# ============================================================================
# MÓDULO 3: GENERADOR DE DATOS SINTÉTICOS (PARA DEMO)
# ============================================================================

def generar_datos_sinteticos(n_partidos=500):
    """
    Genera datos sintéticos para demostración
    En producción, usarías datos REALES de una API o base de datos
    """
    
    np.random.seed(42)
    partidos = []
    
    for i in range(n_partidos):
        # Simular equipos con diferentes niveles
        local_nivel = np.random.uniform(0.3, 0.9)
        visitante_nivel = np.random.uniform(0.3, 0.9)
        
        # Features simuladas
        features = {
            'local_gf_promedio': local_nivel * 2.5 + np.random.normal(0, 0.3),
            'local_gc_promedio': (1 - local_nivel) * 2.0 + np.random.normal(0, 0.3),
            'visitante_gf_promedio': visitante_nivel * 2.0 + np.random.normal(0, 0.3),
            'visitante_gc_promedio': (1 - visitante_nivel) * 2.0 + np.random.normal(0, 0.3),
            'local_forma_5': local_nivel + np.random.normal(0, 0.1),
            'local_forma_10': local_nivel + np.random.normal(0, 0.08),
            'local_forma_20': local_nivel + np.random.normal(0, 0.05),
            'visitante_forma_5': visitante_nivel + np.random.normal(0, 0.1),
            'visitante_forma_10': visitante_nivel + np.random.normal(0, 0.08),
            'visitante_forma_20': visitante_nivel + np.random.normal(0, 0.05),
            'diff_forma': 0,
            'local_en_casa_gf': local_nivel * 3.0 + np.random.normal(0, 0.4),
            'local_en_casa_gc': (1 - local_nivel) * 1.8 + np.random.normal(0, 0.3),
            'visitante_fuera_gf': visitante_nivel * 1.5 + np.random.normal(0, 0.3),
            'visitante_fuera_gc': (1 - visitante_nivel) * 2.2 + np.random.normal(0, 0.3),
            'local_posicion': int((1 - local_nivel) * 18 + 1),
            'visitante_posicion': int((1 - visitante_nivel) * 18 + 1),
            'diff_posicion': 0,
            'local_puntos': int(local_nivel * 80),
            'visitante_puntos': int(visitante_nivel * 80),
            'diff_puntos': 0,
            'local_racha_victorias': int(local_nivel * 3),
            'visitante_racha_victorias': int(visitante_nivel * 3),
            'local_over25_ratio': min(local_nivel + 0.2, 0.9),
            'visitante_over25_ratio': min(visitante_nivel + 0.2, 0.9),
            'local_btts_ratio': 0.5 + np.random.uniform(-0.2, 0.2),
            'visitante_btts_ratio': 0.5 + np.random.uniform(-0.2, 0.2),
            'h2h_local_victorias': np.random.randint(0, 5),
            'h2h_empates': np.random.randint(0, 3),
            'h2h_visitante_victorias': np.random.randint(0, 5),
            'h2h_goles_promedio': 2.5 + np.random.normal(0, 0.5),
            'local_eficiencia_ofensiva': 0,
            'visitante_eficiencia_ofensiva': 0
        }
        
        features['diff_forma'] = features['local_forma_5'] - features['visitante_forma_5']
        features['diff_posicion'] = features['visitante_posicion'] - features['local_posicion']
        features['diff_puntos'] = features['local_puntos'] - features['visitante_puntos']
        features['local_eficiencia_ofensiva'] = features['local_gf_promedio'] / max(features['local_gc_promedio'], 0.5)
        features['visitante_eficiencia_ofensiva'] = features['visitante_gf_promedio'] / max(features['visitante_gc_promedio'], 0.5)
        
        # Simular resultado basado en nivel
        prob_local = 0.45 + (local_nivel - visitante_nivel) * 0.3 + 0.15  # ventaja local
        prob_empate = 0.25
        prob_visitante = 1 - prob_local - prob_empate
        
        resultado = np.random.choice([0, 1, 2], p=[prob_visitante, prob_empate, prob_local])
        
        if resultado == 2:  # Local gana
            goles_local = np.random.poisson(features['local_gf_promedio'] * 1.2)
            goles_visitante = np.random.poisson(features['visitante_gf_promedio'] * 0.8)
        elif resultado == 1:  # Empate
            media = (features['local_gf_promedio'] + features['visitante_gf_promedio']) / 2
            goles_local = np.random.poisson(media)
            goles_visitante = goles_local
        else:  # Visitante gana
            goles_local = np.random.poisson(features['local_gf_promedio'] * 0.8)
            goles_visitante = np.random.poisson(features['visitante_gf_promedio'] * 1.2)
        
        partidos.append({
            'local': f'Team_{i}_A',
            'visitante': f'Team_{i}_B',
            'goles_local': int(goles_local),
            'goles_visitante': int(goles_visitante),
            'features': features
        })
    
    return partidos

# ============================================================================
# INTERFAZ STREAMLIT
# ============================================================================

def main():
    st.title("⚽ SISTEMABETS ML v6.0 - Machine Learning REAL")
    st.markdown("### Sistema predictivo con Random Forest y Gradient Boosting")
    
    # Sidebar
    with st.sidebar:
        st.header("🤖 Machine Learning")
        
        st.markdown("""
        ### ¿Cómo funciona?
        
        1. **Recolecta datos** de partidos históricos
        2. **Extrae 33 features** (características)
        3. **Entrena 3 modelos** independientes:
           - Random Forest para 1X2
           - Gradient Boosting para Over/Under
           - Gradient Boosting para BTTS
        4. **Predice** nuevos partidos
        
        ### Ventajas vs Poisson:
        - ✅ Aprende patrones complejos
        - ✅ Considera múltiples variables
        - ✅ Se adapta a diferentes ligas
        - ✅ Mejora con más datos
        """)
        
        st.markdown("---")
        
        # Configuración
        n_partidos_entrenamiento = st.slider(
            "Partidos para entrenar",
            min_value=100,
            max_value=2000,
            value=500,
            step=100,
            help="Más partidos = mejor precisión (pero más lento)"
        )
        
        if st.button("🚀 ENTRENAR MODELO", type="primary"):
            st.session_state['entrenar'] = True
    
    # PASO 1: ENTRENAR MODELO
    if 'modelo_ml' not in st.session_state or st.session_state.get('entrenar', False):
        
        st.header("🔄 Entrenando Modelo de Machine Learning...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Generar datos
        status_text.text("📊 Generando datos históricos...")
        progress_bar.progress(20)
        
        partidos_historicos = generar_datos_sinteticos(n_partidos_entrenamiento)
        
        # Crear dataset
        status_text.text("🔧 Preparando features...")
        progress_bar.progress(40)
        
        X, y_1x2, y_over25, y_btts = RecolectorDatos.crear_dataset_historico(partidos_historicos)
        
        # Entrenar
        status_text.text("🤖 Entrenando modelos...")
        progress_bar.progress(60)
        
        modelo_ml = ModeloML()
        metricas = modelo_ml.entrenar(X, y_1x2, y_over25, y_btts)
        
        progress_bar.progress(100)
        status_text.text("✅ ¡Entrenamiento completado!")
        
        # Guardar en session state
        st.session_state['modelo_ml'] = modelo_ml
        st.session_state['metricas'] = metricas
        st.session_state['entrenar'] = False
        
        # Mostrar métricas
        st.success("✅ Modelo entrenado exitosamente")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Partidos Entrenados", f"{metricas['n_samples']:,}")
        col2.metric("Precisión 1X2", f"{metricas['accuracy_1x2']*100:.1f}%")
        col3.metric("Precisión Over/Under", f"{metricas['accuracy_over25']*100:.1f}%")
        col4.metric("Precisión BTTS", f"{metricas['accuracy_btts']*100:.1f}%")
        
        st.info("""
        💡 **Interpretación de Precisión:**
        - 50-55% = Apenas mejor que azar
        - 55-60% = Precisión aceptable
        - 60-65% = Buena precisión
        - 65%+ = Excelente precisión (¡muy difícil de lograr!)
        """)
        
        # Mostrar importancia de features
        with st.expander("📊 Ver Importancia de Variables"):
            importancia = modelo_ml.obtener_importancia_features()
            st.dataframe(importancia, use_container_width=True)
            
            st.caption("""
            **Las variables más importantes** son las que el modelo usa más para tomar decisiones.
            Por ejemplo, si 'diff_forma' tiene alta importancia, significa que la diferencia de forma
            entre equipos es crucial para predecir el resultado.
            """)
    
    # PASO 2: HACER PREDICCIONES
    if 'modelo_ml' in st.session_state:
        
        st.markdown("---")
        st.header("🎯 Predecir Partido")
        
        st.info("ℹ️ **MODO DEMO:** Ingresa stats manualmente. En producción, esto se obtendría automáticamente de una API.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏠 Equipo Local")
            local_gf = st.number_input("Goles a favor (promedio)", min_value=0.0, max_value=5.0, value=1.8, step=0.1, key='local_gf')
            local_gc = st.number_input("Goles en contra (promedio)", min_value=0.0, max_value=5.0, value=1.2, step=0.1, key='local_gc')
            local_forma = st.slider("Forma reciente (0-1)", 0.0, 1.0, 0.65, 0.05, key='local_forma')
            local_pos = st.number_input("Posición en tabla", min_value=1, max_value=20, value=5, key='local_pos')
            local_pts = st.number_input("Puntos", min_value=0, max_value=100, value=45, key='local_pts')
            local_racha = st.number_input("Racha victorias (últimos 3)", min_value=0, max_value=3, value=2, key='local_racha')
        
        with col2:
            st.subheader("✈️ Equipo Visitante")
            visitante_gf = st.number_input("Goles a favor (promedio)", min_value=0.0, max_value=5.0, value=1.5, step=0.1, key='visit_gf')
            visitante_gc = st.number_input("Goles en contra (promedio)", min_value=0.0, max_value=5.0, value=1.4, step=0.1, key='visit_gc')
            visitante_forma = st.slider("Forma reciente (0-1)", 0.0, 1.0, 0.55, 0.05, key='visit_forma')
            visitante_pos = st.number_input("Posición en tabla", min_value=1, max_value=20, value=8, key='visit_pos')
            visitante_pts = st.number_input("Puntos", min_value=0, max_value=100, value=38, key='visit_pts')
            visitante_racha = st.number_input("Racha victorias (últimos 3)", min_value=0, max_value=3, value=1, key='visit_racha')
        
        if st.button("🔮 PREDECIR RESULTADO", type="primary", use_container_width=True):
            
            # Construir features
            features = {
                'local_gf_promedio': local_gf,
                'local_gc_promedio': local_gc,
                'visitante_gf_promedio': visitante_gf,
                'visitante_gc_promedio': visitante_gc,
                'local_forma_5': local_forma,
                'local_forma_10': local_forma * 0.95,
                'local_forma_20': local_forma * 0.90,
                'visitante_forma_5': visitante_forma,
                'visitante_forma_10': visitante_forma * 0.95,
                'visitante_forma_20': visitante_forma * 0.90,
                'diff_forma': local_forma - visitante_forma,
                'local_en_casa_gf': local_gf * 1.15,
                'local_en_casa_gc': local_gc * 0.95,
                'visitante_fuera_gf': visitante_gf * 0.85,
                'visitante_fuera_gc': visitante_gc * 1.10,
                'local_posicion': local_pos,
                'visitante_posicion': visitante_pos,
                'diff_posicion': visitante_pos - local_pos,
                'local_puntos': local_pts,
                'visitante_puntos': visitante_pts,
                'diff_puntos': local_pts - visitante_pts,
                'local_racha_victorias': local_racha,
                'visitante_racha_victorias': visitante_racha,
                'local_over25_ratio': 0.55,
                'visitante_over25_ratio': 0.50,
                'local_btts_ratio': 0.52,
                'visitante_btts_ratio': 0.48,
                'h2h_local_victorias': 0,
                'h2h_empates': 0,
                'h2h_visitante_victorias': 0,
                'h2h_goles_promedio': 2.5,
                'local_eficiencia_ofensiva': local_gf / max(local_gc, 0.5),
                'visitante_eficiencia_ofensiva': visitante_gf / max(visitante_gc, 0.5)
            }
            
            # Predecir
            modelo_ml = st.session_state['modelo_ml']
            predicciones = modelo_ml.predecir(features)
            
            # Mostrar resultados
            st.markdown("---")
            st.header("🎯 PREDICCIONES DEL MODELO ML")
            
            # 1X2
            st.subheader("⚽ Resultado Final (1X2)")
            col1, col2, col3 = st.columns(3)
            
            prob_local = predicciones['1X2']['Local']
            prob_empate = predicciones['1X2']['Empate']
            prob_visitante = predicciones['1X2']['Visitante']
            
            with col1:
                st.metric("Victoria Local", f"{prob_local*100:.1f}%")
                cuota_local = 1 / prob_local
                st.caption(f"Cuota justa: {cuota_local:.2f}")
            
            with col2:
                st.metric("Empate", f"{prob_empate*100:.1f}%")
                cuota_empate = 1 / prob_empate
                st.caption(f"Cuota justa: {cuota_empate:.2f}")
            
            with col3:
                st.metric("Victoria Visitante", f"{prob_visitante*100:.1f}%")
                cuota_visitante = 1 / prob_visitante
                st.caption(f"Cuota justa: {cuota_visitante:.2f}")
            
            # Recomendación 1X2
            max_prob_1x2 = max(prob_local, prob_empate, prob_visitante)
            if max_prob_1x2 == prob_local:
                resultado_recomendado = "Victoria Local"
                confianza_1x2 = prob_local
            elif max_prob_1x2 == prob_empate:
                resultado_recomendado = "Empate"
                confianza_1x2 = prob_empate
            else:
                resultado_recomendado = "Victoria Visitante"
                confianza_1x2 = prob_visitante
            
            if confianza_1x2 > 0.55:
                st.success(f"✅ **Recomendación ML:** {resultado_recomendado} ({confianza_1x2*100:.1f}% confianza)")
            else:
                st.warning(f"⚠️ **Partido muy igualado** - El modelo no tiene una predicción clara (máx {confianza_1x2*100:.1f}%)")
            
            # Over/Under
            st.subheader("📊 Over/Under 2.5 Goles")
            col1, col2 = st.columns(2)
            
            prob_over = predicciones['Over/Under']['Over 2.5']
            prob_under = predicciones['Over/Under']['Under 2.5']
            
            with col1:
                st.metric("Over 2.5", f"{prob_over*100:.1f}%")
                st.caption(f"Cuota justa: {1/prob_over:.2f}")
            
            with col2:
                st.metric("Under 2.5", f"{prob_under*100:.1f}%")
                st.caption(f"Cuota justa: {1/prob_under:.2f}")
            
            if prob_over > 0.60:
                st.success(f"✅ **Recomendación ML:** Over 2.5 ({prob_over*100:.1f}% confianza)")
            elif prob_under > 0.60:
                st.success(f"✅ **Recomendación ML:** Under 2.5 ({prob_under*100:.1f}% confianza)")
            else:
                st.info(f"ℹ️ **Sin recomendación clara** - Probabilidades balanceadas")
            
            # BTTS
            st.subheader("🎯 Ambos Equipos Anotan (BTTS)")
            col1, col2 = st.columns(2)
            
            prob_btts_si = predicciones['BTTS']['Si']
            prob_btts_no = predicciones['BTTS']['No']
            
            with col1:
                st.metric("BTTS Sí", f"{prob_btts_si*100:.1f}%")
                st.caption(f"Cuota justa: {1/prob_btts_si:.2f}")
            
            with col2:
                st.metric("BTTS No", f"{prob_btts_no*100:.1f}%")
                st.caption(f"Cuota justa: {1/prob_btts_no:.2f}")
            
            if prob_btts_si > 0.60:
                st.success(f"✅ **Recomendación ML:** BTTS Sí ({prob_btts_si*100:.1f}% confianza)")
            elif prob_btts_no > 0.60:
                st.success(f"✅ **Recomendación ML:** BTTS No ({prob_btts_no*100:.1f}% confianza)")
            else:
                st.info(f"ℹ️ **Sin recomendación clara**")
            
            # COMPARACIÓN ML vs POISSON
            st.markdown("---")
            st.header("🆚 Comparación: ML vs Poisson Clásico")
            
            # Calcular Poisson para comparar
            lambda_local = (local_gf * 1.15) * (visitante_gc * 1.10) / ((local_gf + visitante_gf) / 2)
            lambda_visitante = (visitante_gf * 0.85) * (local_gc * 0.95) / ((local_gf + visitante_gf) / 2)
            
            # Matriz Poisson
            matriz = np.zeros((6, 6))
            for i in range(6):
                for j in range(6):
                    matriz[i, j] = poisson.pmf(i, lambda_local) * poisson.pmf(j, lambda_visitante)
            
            prob_local_poisson = np.sum(np.tril(matriz, -1))
            prob_empate_poisson = np.sum(np.diag(matriz))
            prob_visitante_poisson = np.sum(np.triu(matriz, 1))
            
            total = prob_local_poisson + prob_empate_poisson + prob_visitante_poisson
            prob_local_poisson /= total
            prob_empate_poisson /= total
            prob_visitante_poisson /= total
            
            # Over/Under Poisson
            prob_over_poisson = sum([matriz[i, j] for i in range(6) for j in range(6) if (i+j) > 2.5])
            prob_under_poisson = 1 - prob_over_poisson
            
            # BTTS Poisson
            prob_btts_no_poisson = matriz[0,:].sum() + matriz[:,0].sum() - matriz[0,0]
            prob_btts_si_poisson = 1 - prob_btts_no_poisson
            
            # Tabla comparativa
            df_comparacion = pd.DataFrame({
                'Mercado': [
                    'Victoria Local',
                    'Empate',
                    'Victoria Visitante',
                    'Over 2.5',
                    'Under 2.5',
                    'BTTS Sí',
                    'BTTS No'
                ],
                'Machine Learning': [
                    f"{prob_local*100:.1f}%",
                    f"{prob_empate*100:.1f}%",
                    f"{prob_visitante*100:.1f}%",
                    f"{prob_over*100:.1f}%",
                    f"{prob_under*100:.1f}%",
                    f"{prob_btts_si*100:.1f}%",
                    f"{prob_btts_no*100:.1f}%"
                ],
                'Poisson Clásico': [
                    f"{prob_local_poisson*100:.1f}%",
                    f"{prob_empate_poisson*100:.1f}%",
                    f"{prob_visitante_poisson*100:.1f}%",
                    f"{prob_over_poisson*100:.1f}%",
                    f"{prob_under_poisson*100:.1f}%",
                    f"{prob_btts_si_poisson*100:.1f}%",
                    f"{prob_btts_no_poisson*100:.1f}%"
                ],
                'Diferencia': [
                    f"{abs(prob_local - prob_local_poisson)*100:.1f}%",
                    f"{abs(prob_empate - prob_empate_poisson)*100:.1f}%",
                    f"{abs(prob_visitante - prob_visitante_poisson)*100:.1f}%",
                    f"{abs(prob_over - prob_over_poisson)*100:.1f}%",
                    f"{abs(prob_under - prob_under_poisson)*100:.1f}%",
                    f"{abs(prob_btts_si - prob_btts_si_poisson)*100:.1f}%",
                    f"{abs(prob_btts_no - prob_btts_no_poisson)*100:.1f}%"
                ]
            })
            
            st.dataframe(df_comparacion, use_container_width=True)
            
            st.info("""
            💡 **Interpretación:**
            - **Diferencias pequeñas (0-5%):** Ambos modelos coinciden
            - **Diferencias moderadas (5-15%):** ML detectó patrones que Poisson no
            - **Diferencias grandes (15%+):** ML está considerando factores importantes (forma, posición, etc.)
            
            **¿Cuál usar?**
            - Si la diferencia es pequeña → Confía en ambos
            - Si ML tiene mayor confianza (>60%) → Prefiere ML
            - Si ambos son dudosos (<55%) → ¡No apuestes!
            """)
            
            # MEJORES APUESTAS ML
            st.markdown("---")
            st.header("💎 MEJORES APUESTAS SEGÚN ML")
            
            todas_predicciones = []
            
            # 1X2
            todas_predicciones.append({
                'Mercado': 'Victoria Local',
                'Probabilidad': prob_local,
                'Cuota_Justa': 1/prob_local,
                'Confianza': '⭐⭐⭐' if prob_local > 0.65 else '⭐⭐' if prob_local > 0.55 else '⭐'
            })
            todas_predicciones.append({
                'Mercado': 'Empate',
                'Probabilidad': prob_empate,
                'Cuota_Justa': 1/prob_empate,
                'Confianza': '⭐⭐⭐' if prob_empate > 0.65 else '⭐⭐' if prob_empate > 0.55 else '⭐'
            })
            todas_predicciones.append({
                'Mercado': 'Victoria Visitante',
                'Probabilidad': prob_visitante,
                'Cuota_Justa': 1/prob_visitante,
                'Confianza': '⭐⭐⭐' if prob_visitante > 0.65 else '⭐⭐' if prob_visitante > 0.55 else '⭐'
            })
            
            # Over/Under
            todas_predicciones.append({
                'Mercado': 'Over 2.5',
                'Probabilidad': prob_over,
                'Cuota_Justa': 1/prob_over,
                'Confianza': '⭐⭐⭐' if prob_over > 0.65 else '⭐⭐' if prob_over > 0.55 else '⭐'
            })
            todas_predicciones.append({
                'Mercado': 'Under 2.5',
                'Probabilidad': prob_under,
                'Cuota_Justa': 1/prob_under,
                'Confianza': '⭐⭐⭐' if prob_under > 0.65 else '⭐⭐' if prob_under > 0.55 else '⭐'
            })
            
            # BTTS
            todas_predicciones.append({
                'Mercado': 'BTTS Sí',
                'Probabilidad': prob_btts_si,
                'Cuota_Justa': 1/prob_btts_si,
                'Confianza': '⭐⭐⭐' if prob_btts_si > 0.65 else '⭐⭐' if prob_btts_si > 0.55 else '⭐'
            })
            todas_predicciones.append({
                'Mercado': 'BTTS No',
                'Probabilidad': prob_btts_no,
                'Cuota_Justa': 1/prob_btts_no,
                'Confianza': '⭐⭐⭐' if prob_btts_no > 0.65 else '⭐⭐' if prob_btts_no > 0.55 else '⭐'
            })
            
            # Ordenar por probabilidad
            todas_predicciones.sort(key=lambda x: x['Probabilidad'], reverse=True)
            
            # Mostrar top 3
            st.subheader("🏆 Top 3 Apuestas ML")
            
            for i, pred in enumerate(todas_predicciones[:3], 1):
                with st.expander(f"#{i} - {pred['Mercado']} | {pred['Probabilidad']*100:.1f}% {pred['Confianza']}"):
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Probabilidad", f"{pred['Probabilidad']*100:.1f}%")
                    col2.metric("Cuota Justa", f"{pred['Cuota_Justa']:.2f}")
                    col3.metric("Cuota Mínima Esperada", f"{pred['Cuota_Justa']*0.95:.2f}")
                    
                    st.markdown("---")
                    st.markdown("### ⚖️ VEREDICTO ML")
                    
                    if pred['Probabilidad'] > 0.70:
                        st.success("🟢 **MUY RECOMENDADA** - El modelo tiene alta confianza en esta predicción")
                    elif pred['Probabilidad'] > 0.60:
                        st.info("🔵 **RECOMENDADA** - Buena probabilidad según el modelo")
                    elif pred['Probabilidad'] > 0.55:
                        st.warning("🟡 **CONSIDERAR** - Probabilidad moderada, analiza más factores")
                    else:
                        st.error("🔴 **NO RECOMENDADA** - Probabilidad insuficiente")
                    
                    st.caption(f"""
                    **Cómo interpretar:**
                    - Si encuentras cuota ≥ {pred['Cuota_Justa']*0.95:.2f} → **HAY VALOR**
                    - El modelo entrenó con {st.session_state['metricas']['n_samples']} partidos
                    - Precisión del modelo: {st.session_state['metricas']['accuracy_1x2']*100:.0f}%
                    """)
    
    # INFO ADICIONAL
    st.markdown("---")
    st.header("📚 Cómo Mejorar el Modelo")
    
    with st.expander("🚀 Pasos para un Modelo Profesional"):
        st.markdown("""
        ### 1️⃣ **Conseguir Datos Reales**
        ```python
        # APIs profesionales:
        - football-data.org (gratis, limitada)
        - api-football.com (€€, muy completa)
        - sofascore.com (scraping)
        - transfermarkt (valores de jugadores)
        ```
        
        ### 2️⃣ **Agregar Más Features**
        - xG (goles esperados) - CRÍTICO
        - Posesión promedio
        - Tiros a puerta
        - Corners
        - Tarjetas
        - Lesiones de jugadores clave
        - Valor de mercado del equipo
        - Días de descanso
        - Importancia del partido
        
        ### 3️⃣ **Entrenar con MÁS Datos**
        - Mínimo 2,000 partidos
        - Ideal: 5,000-10,000 partidos
        - Múltiples temporadas
        - Múltiples ligas
        
        ### 4️⃣ **Probar Otros Algoritmos**
        ```python
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from sklearn.neural_network import MLPClassifier
        
        # XGBoost suele dar mejores resultados
        modelo = XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05
        )
        ```
        
        ### 5️⃣ **Validación Temporal**
        - NO usar split aleatorio
        - Entrenar con temporadas pasadas
        - Testear con temporada actual
        - Esto simula predicción real
        
        ### 6️⃣ **Comparar con Cuotas Reales**
        ```python
        # Scraping de cuotas
        import requests
        from bs4 import BeautifulSoup
        
        # Comparar vs Bet365, Betfair, etc.
        # Solo apostar si EV > 5%
        ```
        
        ### 7️⃣ **Backtesting**
        - Simular apuestas en partidos pasados
        - Calcular ROI real
        - Ajustar estrategia de bankroll
        
        ### 8️⃣ **Ensemble de Modelos**
        ```python
        # Combinar predicciones
        pred_final = (
            pred_rf * 0.3 +
            pred_xgb * 0.4 +
            pred_poisson * 0.3
        )
        ```
        """)
    
    with st.expander("💡 ¿Por Qué ML es Mejor que Poisson?"):
        st.markdown("""
        ### Poisson Solo Usa:
        - Promedio de goles
        - Ventaja de local (fija)
        
        ### Machine Learning Usa:
        - ✅ 33+ variables simultáneas
        - ✅ Aprende relaciones complejas
        - ✅ Se adapta a diferentes ligas
        - ✅ Detecta patrones ocultos
        - ✅ Mejora con más datos
        
        ### Ejemplo Real:
        **Situación:** Equipo grande (1° en tabla) visita equipo débil (18°)
        
        **Poisson dice:** Local 25%, Empate 25%, Visitante 50%
        → Solo ve promedios de goles
        
        **ML dice:** Local 15%, Empate 20%, Visitante 65%
        → Considera: posición, forma, stats de visitante, H2H, etc.
        
        **ML es MÁS PRECISO** porque entiende el contexto completo.
        """)
    
    # DISCLAIMER
    st.markdown("---")
    st.warning("""
    ⚠️ **ADVERTENCIA IMPORTANTE:**
    
    - Este es un **DEMO educativo** con datos sintéticos
    - En producción necesitas: datos reales + API de cuotas + más features
    - Machine Learning NO garantiza ganancias
    - Incluso con 65% de precisión puedes perder dinero
    - Siempre apuesta responsablemente
    - Usa esto como herramienta de APOYO, no como única fuente
    
    📊 **Para ganar dinero necesitas:**
    1. Modelo con 58%+ precisión consistente
    2. Encontrar cuotas con +5% EV
    3. Gestión estricta de bankroll
    4. Disciplina absoluta
    5. Miles de apuestas (ley de grandes números)
    """)

if __name__ == "__main__":
    main()
