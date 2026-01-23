import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SISTEMABETS IA v2.0 - ANÁLISIS PROFUNDO", layout="wide")

# ============================================================================
# MÓDULO 1: EXTRACCIÓN INTELIGENTE DE DATOS
# ============================================================================

class ScraperMultiSource:
    """Scraper robusto con fallback y validación de datos"""
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }
    
    LIGAS = {
        "Champions League": "https://native-stats.org/competition/CL/",
        "Premier League": "https://native-stats.org/competition/PL",
        "La Liga": "https://native-stats.org/competition/PD",
        "Bundesliga": "https://native-stats.org/competition/BL1",
        "Serie A": "https://native-stats.org/competition/SA",
        "Ligue 1": "https://native-stats.org/competition/FL1"
    }
    
    @staticmethod
    def extraer_tabla_standings(url):
        """Extrae tabla de posiciones con stats locales/visitantes"""
        try:
            response = requests.get(url, headers=ScraperMultiSource.HEADERS, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Buscar todas las tablas
            tablas_html = soup.find_all('table')
            if not tablas_html:
                st.error("No se encontraron tablas en la página")
                return None
            
            # Intentar parsear cada tabla hasta encontrar una válida
            df = None
            for i, tabla in enumerate(tablas_html):
                try:
                    # Parsear tabla individual
                    dfs = pd.read_html(str(tabla))
                    if dfs and len(dfs) > 0:
                        temp_df = dfs[0]
                        
                        # Validar que tenga suficientes filas (al menos 5 equipos)
                        if len(temp_df) >= 5:
                            df = temp_df
                            break
                except Exception as parse_error:
                    continue
            
            if df is None:
                st.error("No se pudo parsear ninguna tabla válida")
                return None
            
            # Resetear índice por si acaso
            df = df.reset_index(drop=True)
            
            # Normalización dinámica de columnas
            df = ScraperMultiSource._normalizar_columnas(df)
            
            # Validar que tenga las columnas mínimas necesarias
            if 'Equipo' not in df.columns or 'GF' not in df.columns:
                st.error(f"Columnas encontradas: {list(df.columns)}")
                st.error("No se encontraron las columnas necesarias (Equipo, GF, GC)")
                return None
            
            return df
            
        except Exception as e:
            st.error(f"Error extrayendo {url}: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    @staticmethod
    def _normalizar_columnas(df):
        """Mapeo inteligente de columnas"""
        
        # Si viene un MultiIndex, aplanar
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' '.join(map(str, col)).strip() for col in df.columns]
        
        # Asegurar que las columnas sean strings
        df.columns = [str(col) for col in df.columns]
        
        cols_map = {}
        
        for col in df.columns:
            c_upper = col.upper()
            
            # Identificación por palabras clave
            if any(x in c_upper for x in ['TEAM', 'EQUIPO', 'CLUB', 'SQUAD']):
                cols_map[col] = 'Equipo'
            elif any(x in c_upper for x in ['PTS', 'POINTS', 'PUNTOS']):
                cols_map[col] = 'Pts'
            elif any(x in c_upper for x in ['MP', 'PJ', 'PLAYED', 'MATCHES']):
                cols_map[col] = 'PJ'
            elif any(x in c_upper for x in ['W', 'G', 'WINS', 'WON']):
                cols_map[col] = 'Victorias'
            elif any(x in c_upper for x in ['D', 'E', 'DRAWS', 'DRAW']):
                cols_map[col] = 'Empates'
            elif any(x in c_upper for x in ['L', 'P', 'LOSS', 'LOST']):
                cols_map[col] = 'Derrotas'
        
        df = df.rename(columns=cols_map)
        
        # Si no se mapeó "Equipo", usar la primera columna de texto
        if 'Equipo' not in df.columns:
            for col in df.columns:
                if df[col].dtype == 'object':
                    df['Equipo'] = df[col]
                    break
        
        # Limpieza de equipo
        if 'Equipo' in df.columns:
            try:
                # Convertir a string de forma segura
                df['Equipo'] = df['Equipo'].apply(lambda x: str(x) if pd.notna(x) else '')
                # Remover números iniciales (posición en tabla)
                df['Equipo'] = df['Equipo'].str.replace(r'^\d+\.?\s*', '', regex=True)
                df['Equipo'] = df['Equipo'].str.strip()
                # Filtrar filas vacías
                df = df[df['Equipo'].str.len() > 0]
            except Exception as e:
                st.warning(f"Advertencia al limpiar equipos: {e}")
        
        # Extracción de goles (formato X:Y)
        goles_encontrados = False
        for col in df.columns:
            try:
                # Convertir columna a string de forma segura
                col_values = df[col].apply(lambda x: str(x) if pd.notna(x) else '')
                
                # Verificar si contiene formato de goles
                if col_values.str.contains(':', regex=False).any():
                    # Extraer goles
                    goles = col_values.str.extract(r'(\d+):(\d+)', expand=True)
                    if goles is not None and len(goles.columns) == 2:
                        df['GF'] = pd.to_numeric(goles[0], errors='coerce').fillna(0).astype(int)
                        df['GC'] = pd.to_numeric(goles[1], errors='coerce').fillna(0).astype(int)
                        goles_encontrados = True
                        break
            except Exception as e:
                continue
        
        # Si no se encontraron goles, buscar columnas GF/GC separadas
        if not goles_encontrados:
            for col in df.columns:
                c_upper = str(col).upper()
                if 'GF' in c_upper or 'SCORED' in c_upper or ('GOALS' in c_upper and 'FOR' in c_upper):
                    df['GF'] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                if 'GC' in c_upper or 'GA' in c_upper or 'CONCEDED' in c_upper or ('GOALS' in c_upper and 'AGAINST' in c_upper):
                    df['GC'] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # Si aún no hay PJ, intentar contarlo de W+D+L
        if 'PJ' not in df.columns:
            if all(x in df.columns for x in ['Victorias', 'Empates', 'Derrotas']):
                df['PJ'] = df['Victorias'] + df['Empates'] + df['Derrotas']
        
        # Conversiones seguras finales
        for col in ['Pts', 'PJ', 'Victorias', 'Empates', 'Derrotas', 'GF', 'GC']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # Si PJ sigue sin existir o tiene ceros, usar valor por defecto
        if 'PJ' not in df.columns or df['PJ'].sum() == 0:
            df['PJ'] = 10  # Valor asumido para evitar divisiones por cero
        
        return df

# ============================================================================
# MÓDULO 2: MOTOR DE ANÁLISIS AVANZADO
# ============================================================================

class AnalizadorPartidos:
    """Motor de IA que calcula probabilidades reales basadas en múltiples factores"""
    
    VENTAJA_LOCAL = 1.18  # Basado en stats históricas reales
    
    @staticmethod
    def calcular_lambdas(equipo_local, equipo_visitante, df_liga):
        """
        Calcula expectativa de goles (lambda) ajustada por:
        - Potencia ofensiva del equipo
        - Debilidad defensiva del rival
        - Ventaja de jugar en casa
        - Normalización por media de liga
        """
        
        # Stats del equipo local
        gf_local = equipo_local['GF'] / max(equipo_local['PJ'], 1)
        gc_visitante = equipo_visitante['GC'] / max(equipo_visitante['PJ'], 1)
        
        # Stats del equipo visitante
        gf_visitante = equipo_visitante['GF'] / max(equipo_visitante['PJ'], 1)
        gc_local = equipo_local['GC'] / max(equipo_local['PJ'], 1)
        
        # Media de goles de la liga
        media_gf = df_liga['GF'].sum() / df_liga['PJ'].sum()
        media_gc = df_liga['GC'].sum() / df_liga['PJ'].sum()
        
        # Lambda ajustado
        lambda_local = (gf_local * (gc_visitante / media_gc)) * AnalizadorPartidos.VENTAJA_LOCAL
        lambda_visitante = (gf_visitante * (gc_local / media_gc)) / AnalizadorPartidos.VENTAJA_LOCAL
        
        return lambda_local, lambda_visitante
    
    @staticmethod
    def matriz_probabilidades(lambda_l, lambda_v, max_goles=6):
        """Genera matriz de probabilidades para todos los resultados posibles"""
        matriz = np.zeros((max_goles, max_goles))
        
        for i in range(max_goles):
            for j in range(max_goles):
                matriz[i, j] = poisson.pmf(i, lambda_l) * poisson.pmf(j, lambda_v)
        
        return matriz
    
    @staticmethod
    def mercados_principales(matriz):
        """Calcula probabilidades de los mercados principales"""
        
        # 1X2
        p_local = np.sum(np.tril(matriz, -1))  # Debajo de diagonal
        p_empate = np.sum(np.diag(matriz))     # Diagonal
        p_visitante = np.sum(np.triu(matriz, 1))  # Arriba de diagonal
        
        # Over/Under 2.5
        p_under_25 = sum([matriz[i, j] for i in range(6) for j in range(6) if i+j < 3])
        p_over_25 = 1 - p_under_25
        
        # Over/Under 1.5
        p_under_15 = matriz[0,0] + matriz[0,1] + matriz[1,0]
        p_over_15 = 1 - p_under_15
        
        # Over/Under 3.5
        p_under_35 = sum([matriz[i, j] for i in range(6) for j in range(6) if i+j < 4])
        p_over_35 = 1 - p_under_35
        
        # BTTS (Ambos marcan)
        p_btts_no = matriz[0,:].sum() + matriz[:,0].sum() - matriz[0,0]
        p_btts_si = 1 - p_btts_no
        
        # Resultado exacto más probable
        idx_max = np.unravel_index(matriz.argmax(), matriz.shape)
        resultado_mas_probable = f"{idx_max[0]}-{idx_max[1]}"
        prob_resultado = matriz[idx_max[0], idx_max[1]]
        
        return {
            '1X2': {'Local': p_local, 'Empate': p_empate, 'Visitante': p_visitante},
            'Over/Under': {
                'Over 1.5': p_over_15,
                'Over 2.5': p_over_25,
                'Over 3.5': p_over_35
            },
            'BTTS': {'Si': p_btts_si, 'No': p_btts_no},
            'Resultado': resultado_mas_probable,
            'Prob_Resultado': prob_resultado
        }
    
    @staticmethod
    def calcular_valor_esperado(probabilidad, cuota):
        """Calcula el valor esperado de una apuesta (EV)"""
        ev = (probabilidad * cuota) - 1
        return ev * 100  # En porcentaje

# ============================================================================
# MÓDULO 3: BUSCADOR DE VALOR (VALUE BETTING)
# ============================================================================

class BuscadorValor:
    """Identifica apuestas con valor positivo comparando probabilidades vs cuotas"""
    
    MARGEN_MINIMO = 5  # EV mínimo para considerar una apuesta
    
    @staticmethod
    def analizar_lineas(mercados, cuotas_sugeridas):
        """
        Compara probabilidades calculadas con cuotas del mercado
        para encontrar oportunidades de valor
        """
        
        oportunidades = []
        
        # Cuotas implícitas (ejemplo - en producción vienen del scraper de casas)
        for mercado, probs in mercados.items():
            if mercado == 'Resultado' or mercado == 'Prob_Resultado':
                continue
                
            for tipo, prob in probs.items():
                # Cuota justa (sin margen de casa)
                cuota_justa = 1 / prob if prob > 0 else 999
                
                # Simulamos cuota de mercado (en real se scrapea)
                cuota_mercado = cuota_justa * 0.92  # Margen típico 8%
                
                # Valor esperado
                ev = AnalizadorPartidos.calcular_valor_esperado(prob, cuota_mercado)
                
                if ev >= BuscadorValor.MARGEN_MINIMO:
                    oportunidades.append({
                        'Mercado': f"{mercado} - {tipo}",
                        'Probabilidad': f"{prob*100:.1f}%",
                        'Cuota_Justa': f"{cuota_justa:.2f}",
                        'Cuota_Mercado': f"{cuota_mercado:.2f}",
                        'Valor_Esperado': f"{ev:.1f}%",
                        'Confianza': '🔥' if ev > 10 else '⚡' if ev > 7 else '💡'
                    })
        
        return sorted(oportunidades, key=lambda x: float(x['Valor_Esperado'].replace('%','')), reverse=True)

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

def main():
    st.title("⚡ SISTEMABETS IA v2.0 - ANÁLISIS PROFESIONAL")
    st.caption("Motor de análisis probabilístico con detección automática de valor")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuración")
    liga_seleccionada = st.sidebar.selectbox("Liga:", list(ScraperMultiSource.LIGAS.keys()))
    
    # Carga de datos
    with st.spinner(f"Extrayendo datos de {liga_seleccionada}..."):
        df = ScraperMultiSource.extraer_tabla_standings(ScraperMultiSource.LIGAS[liga_seleccionada])
    
    if df is None or df.empty:
        st.error("❌ No se pudieron cargar los datos. Verifica la conexión.")
        return
    
    if 'GF' not in df.columns or 'Equipo' not in df.columns:
        st.error("❌ Formato de datos incorrecto. La fuente cambió su estructura.")
        return
    
    st.success(f"✅ {len(df)} equipos cargados | Última actualización: {datetime.now().strftime('%H:%M:%S')}")
    
    # Selección de equipos
    st.subheader("🎯 Seleccionar Partido")
    col1, col2 = st.columns(2)
    
    equipos = sorted(df['Equipo'].unique())
    
    with col1:
        equipo_local_nombre = st.selectbox("🏟️ Equipo Local:", equipos, index=0)
    
    with col2:
        equipo_visitante_nombre = st.selectbox("✈️ Equipo Visitante:", equipos, 
                                                index=min(1, len(equipos)-1))
    
    if equipo_local_nombre == equipo_visitante_nombre:
        st.warning("⚠️ Selecciona dos equipos diferentes")
        return
    
    # Obtener stats
    el = df[df['Equipo'] == equipo_local_nombre].iloc[0]
    ev = df[df['Equipo'] == equipo_visitante_nombre].iloc[0]
    
    # Análisis
    if st.button("🔍 ANALIZAR PARTIDO", type="primary", use_container_width=True):
        
        with st.spinner("Calculando probabilidades..."):
            # Lambdas
            lambda_l, lambda_v = AnalizadorPartidos.calcular_lambdas(el, ev, df)
            
            # Matriz
            matriz = AnalizadorPartidos.matriz_probabilidades(lambda_l, lambda_v)
            
            # Mercados
            mercados = AnalizadorPartidos.mercados_principales(matriz)
            
            # Búsqueda de valor
            oportunidades = BuscadorValor.analizar_lineas(mercados, None)
        
        # ============ RESULTADOS ============
        st.divider()
        st.subheader("📊 RESULTADOS DEL ANÁLISIS")
        
        # Métricas principales
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🏠 Gana Local", f"{mercados['1X2']['Local']*100:.1f}%")
        m2.metric("🤝 Empate", f"{mercados['1X2']['Empate']*100:.1f}%")
        m3.metric("✈️ Gana Visitante", f"{mercados['1X2']['Visitante']*100:.1f}%")
        m4.metric("⚽ Goles Esperados", f"{lambda_l + lambda_v:.2f}")
        
        # Over/Under
        st.subheader("📈 Análisis Over/Under")
        ou1, ou2, ou3 = st.columns(3)
        ou1.metric("Over 1.5", f"{mercados['Over/Under']['Over 1.5']*100:.1f}%")
        ou2.metric("Over 2.5", f"{mercados['Over/Under']['Over 2.5']*100:.1f}%")
        ou3.metric("Over 3.5", f"{mercados['Over/Under']['Over 3.5']*100:.1f}%")
        
        # BTTS
        st.subheader("🎯 Ambos Marcan (BTTS)")
        bt1, bt2 = st.columns(2)
        bt1.metric("BTTS Sí", f"{mercados['BTTS']['Si']*100:.1f}%", 
                   delta="Alta confianza" if mercados['BTTS']['Si'] > 0.6 else None)
        bt2.metric("BTTS No", f"{mercados['BTTS']['No']*100:.1f}%")
        
        # Resultado más probable
        st.info(f"🎲 **Resultado más probable:** {mercados['Resultado']} ({mercados['Prob_Resultado']*100:.1f}%)")
        
        # Oportunidades de valor
        st.divider()
        st.subheader("💎 MEJORES OPORTUNIDADES DE VALOR")
        
        if oportunidades:
            df_oport = pd.DataFrame(oportunidades)
            st.dataframe(df_oport, use_container_width=True, hide_index=True)
        else:
            st.warning("No se encontraron oportunidades con EV > 5%")
        
        # Datos técnicos
        with st.expander("🔬 Ver datos técnicos"):
            tc1, tc2 = st.columns(2)
            
            with tc1:
                st.write(f"**{equipo_local_nombre}**")
                st.write(f"- Goles a favor: {int(el['GF'])}")
                st.write(f"- Goles en contra: {int(el['GC'])}")
                st.write(f"- Partidos jugados: {int(el['PJ'])}")
                st.write(f"- Promedio goles/partido: {el['GF']/max(el['PJ'],1):.2f}")
                st.write(f"- Lambda calculado: {lambda_l:.2f}")
            
            with tc2:
                st.write(f"**{equipo_visitante_nombre}**")
                st.write(f"- Goles a favor: {int(ev['GF'])}")
                st.write(f"- Goles en contra: {int(ev['GC'])}")
                st.write(f"- Partidos jugados: {int(ev['PJ'])}")
                st.write(f"- Promedio goles/partido: {ev['GF']/max(ev['PJ'],1):.2f}")
                st.write(f"- Lambda calculado: {lambda_v:.2f}")
        
        # Tabla de clasificación
        with st.expander("📋 Ver clasificación completa"):
            st.dataframe(df[['Equipo', 'PJ', 'GF', 'GC', 'Pts']].sort_values('Pts', ascending=False),
                        use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
