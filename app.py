import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SISTEMABETS IA v3.0 - API PRO", layout="wide")

# ============================================================================
# MÓDULO 1: CONECTOR API-FOOTBALL
# ============================================================================

class APIFootball:
    """Conector profesional a API-Football (RapidAPI)"""
    
    BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"
    
    # IDs de ligas principales
    LIGAS = {
        "Champions League": 2,
        "Premier League": 39,
        "La Liga": 140,
        "Bundesliga": 78,
        "Serie A": 135,
        "Ligue 1": 61,
        "Liga Portugal": 94
    }
    
    def __init__(self, api_key):
        self.headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }
    
    def obtener_standings(self, liga_id, temporada=2024):
        """Obtiene tabla de posiciones con stats completas"""
        try:
            url = f"{self.BASE_URL}/standings"
            params = {"league": liga_id, "season": temporada}
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            data = response.json()
            
            if not data.get('response'):
                return None
            
            standings = data['response'][0]['league']['standings'][0]
            
            # Convertir a DataFrame
            equipos = []
            for team in standings:
                equipos.append({
                    'Equipo': team['team']['name'],
                    'PJ': team['all']['played'],
                    'Victorias': team['all']['win'],
                    'Empates': team['all']['draw'],
                    'Derrotas': team['all']['lose'],
                    'GF': team['all']['goals']['for'],
                    'GC': team['all']['goals']['against'],
                    'Pts': team['points'],
                    # Stats locales
                    'PJ_Local': team['home']['played'],
                    'GF_Local': team['home']['goals']['for'],
                    'GC_Local': team['home']['goals']['against'],
                    'Victorias_Local': team['home']['win'],
                    # Stats visitante
                    'PJ_Visitante': team['away']['played'],
                    'GF_Visitante': team['away']['goals']['for'],
                    'GC_Visitante': team['away']['goals']['against'],
                    'Victorias_Visitante': team['away']['win'],
                    # Forma
                    'Forma': team['form']
                })
            
            return pd.DataFrame(equipos)
            
        except Exception as e:
            st.error(f"Error API: {str(e)}")
            return None
    
    def obtener_pronosticos(self, fixture_id):
        """Obtiene pronósticos de la API para un partido"""
        try:
            url = f"{self.BASE_URL}/predictions"
            params = {"fixture": fixture_id}
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            data = response.json()
            
            if data.get('response'):
                return data['response'][0]['predictions']
            return None
            
        except Exception as e:
            return None
    
    def obtener_cuotas(self, fixture_id):
        """Obtiene cuotas de casas de apuestas para un partido"""
        try:
            url = f"{self.BASE_URL}/odds"
            params = {"fixture": fixture_id}
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            data = response.json()
            
            if data.get('response'):
                return data['response'][0]['bookmakers']
            return None
            
        except Exception as e:
            return None

# ============================================================================
# MÓDULO 2: ANALIZADOR AVANZADO
# ============================================================================

class AnalizadorAvanzado:
    """Motor de análisis con stats locales/visitantes separadas"""
    
    VENTAJA_LOCAL = 1.18
    
    @staticmethod
    def calcular_lambdas_avanzado(local, visitante, df_liga):
        """
        Calcula lambdas usando stats específicas de local/visitante
        Esto es MUCHO más preciso que usar stats generales
        """
        
        # Potencia ofensiva del local EN CASA
        gf_local_casa = local['GF_Local'] / max(local['PJ_Local'], 1)
        
        # Debilidad defensiva del visitante FUERA
        gc_visitante_fuera = visitante['GC_Visitante'] / max(visitante['PJ_Visitante'], 1)
        
        # Potencia ofensiva del visitante FUERA
        gf_visitante_fuera = visitante['GF_Visitante'] / max(visitante['PJ_Visitante'], 1)
        
        # Debilidad defensiva del local EN CASA
        gc_local_casa = local['GC_Local'] / max(local['PJ_Local'], 1)
        
        # Medias de la liga
        media_gf = df_liga['GF'].mean() / df_liga['PJ'].mean()
        
        # Lambdas ajustados
        lambda_local = (gf_local_casa / media_gf) * gc_visitante_fuera * AnalizadorAvanzado.VENTAJA_LOCAL
        lambda_visitante = (gf_visitante_fuera / media_gf) * gc_local_casa / AnalizadorAvanzado.VENTAJA_LOCAL
        
        return lambda_local, lambda_visitante
    
    @staticmethod
    def analizar_forma(forma_str):
        """Analiza la racha reciente (formato: 'WWDLW')"""
        if not forma_str or forma_str == 'None':
            return 0.5
        
        puntos = {'W': 1.0, 'D': 0.5, 'L': 0.0}
        ultimos = forma_str[-5:]  # Últimos 5 partidos
        
        # Peso exponencial: último partido vale más
        pesos = [0.1, 0.15, 0.2, 0.25, 0.3]
        
        score = sum(puntos.get(r, 0.5) * w for r, w in zip(ultimos, pesos))
        return score
    
    @staticmethod
    def ajustar_por_forma(lambda_val, forma):
        """Ajusta lambda según la forma reciente"""
        forma_score = AnalizadorAvanzado.analizar_forma(forma)
        
        # Ajuste: forma excelente +15%, forma mala -15%
        if forma_score > 0.7:
            return lambda_val * 1.15
        elif forma_score < 0.3:
            return lambda_val * 0.85
        else:
            return lambda_val
    
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
    st.title("⚡ SISTEMABETS IA v3.0 - ANÁLISIS PROFESIONAL")
    st.caption("Powered by API-Football | Datos en tiempo real")
    
    # Sidebar - Configuración API
    st.sidebar.header("🔑 Configuración API")
    
    api_key = st.sidebar.text_input(
        "API Key (RapidAPI)", 
        type="password",
        help="Obtén tu key gratis en: https://rapidapi.com/api-sports/api/api-football"
    )
    
    if not api_key:
        st.warning("⚠️ **Necesitas una API Key para empezar**")
        st.info("""
        ### Cómo obtener tu API Key GRATIS:
        
        1. Ve a https://rapidapi.com/api-sports/api/api-football
        2. Crea una cuenta (gratis)
        3. Suscríbete al plan GRATUITO (100 requests/día)
        4. Copia tu API Key y pégala arriba
        
        **Sin tarjeta de crédito. 100% gratis.**
        """)
        
        st.markdown("---")
        st.subheader("🎯 ¿Por qué usar esta API?")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("✅ Datos de 1000+ ligas")
            st.write("✅ Stats locales/visitantes")
            st.write("✅ Forma reciente de equipos")
            st.write("✅ Head-to-Head histórico")
        
        with col2:
            st.write("✅ Cuotas en tiempo real")
            st.write("✅ Lesiones y suspensiones")
            st.write("✅ Pronósticos profesionales")
            st.write("✅ JSON limpio (sin scraping)")
        
        return
    
    # Inicializar API
    api = APIFootball(api_key)
    
    # Selección de liga
    liga_nombre = st.sidebar.selectbox("Liga:", list(APIFootball.LIGAS.keys()))
    liga_id = APIFootball.LIGAS[liga_nombre]
    temporada = st.sidebar.selectbox("Temporada:", [2024, 2023], index=0)
    
    # Cargar datos
    with st.spinner(f"Cargando datos de {liga_nombre}..."):
        df = api.obtener_standings(liga_id, temporada)
    
    if df is None or df.empty:
        st.error("❌ Error cargando datos. Verifica tu API Key y límite de requests.")
        return
    
    st.success(f"✅ {len(df)} equipos cargados | Última actualización: {datetime.now().strftime('%H:%M:%S')}")
    
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
        
        with st.spinner("Calculando probabilidades..."):
            # Lambdas con stats locales/visitantes
            lambda_l, lambda_v = AnalizadorAvanzado.calcular_lambdas_avanzado(el, ev, df)
            
            # Ajuste por forma
            lambda_l = AnalizadorAvanzado.ajustar_por_forma(lambda_l, el['Forma'])
            lambda_v = AnalizadorAvanzado.ajustar_por_forma(lambda_v, ev['Forma'])
            
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
                st.write(f"- Forma: {el['Forma']}")
                st.write(f"- Goles en casa: {int(el['GF_Local'])} en {int(el['PJ_Local'])} PJ")
                st.write(f"- Promedio casa: {el['GF_Local']/max(el['PJ_Local'],1):.2f} goles/partido")
                st.write(f"- Lambda calculado: {lambda_l:.2f}")
            
            with tc2:
                st.write(f"**{equipo_visitante} (Visitante)**")
                st.write(f"- Forma: {ev['Forma']}")
                st.write(f"- Goles fuera: {int(ev['GF_Visitante'])} en {int(ev['PJ_Visitante'])} PJ")
                st.write(f"- Promedio fuera: {ev['GF_Visitante']/max(ev['PJ_Visitante'],1):.2f} goles/partido")
                st.write(f"- Lambda calculado: {lambda_v:.2f}")
        
        # Tabla completa
        with st.expander("📋 Ver clasificación"):
            st.dataframe(
                df[['Equipo', 'PJ', 'Victorias', 'Empates', 'Derrotas', 'GF', 'GC', 'Pts', 'Forma']]
                .sort_values('Pts', ascending=False),
                use_container_width=True,
                hide_index=True
            )

if __name__ == "__main__":
    main()
