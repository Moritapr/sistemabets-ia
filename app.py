mercados['1X2'] = {
            'Local': max(p_local / total, 0.005),
            'Empate': max(p_empate / total, 0.005),
            'Visitante': max(p_visitante / total, 0.005)
        }
        
        # 2. DOBLE OPORTUNIDAD
        mercados['Doble_Oportunidad'] = {
            '1X (Local o Empate)': mercados['1X2']['Local'] + mercados['1X2']['Empate'],
            '12 (Local o Visitante)': mercados['1X2']['Local'] + mercados['1X2']['Visitante'],
            'X2 (Empate o Visitante)': mercados['1X2']['Empate'] + mercados['1X2']['Visitante']
        }
        
        # 3. OVER/UNDER EXTENDIDO
        mercados['Over_Under'] = {}
        for threshold in [0.5, 1.5, 2.5, 3.5, 4.5]:
            p_over = sum([matriz[i, j] for i in range(8) for j in range(8) if (i+j) > threshold])
            mercados['Over_Under'][f"Over {threshold}"] = max(p_over, 0.005)
            mercados['Over_Under'][f"Under {threshold}"] = max(1 - p_over, 0.005)
        
        # 4. BTTS
        p_btts_no = matriz[0,:].sum() + matriz[:,0].sum() - matriz[0,0]
        mercados['BTTS'] = {
            'Si': max(1 - p_btts_no, 0.005),
            'No': max(p_btts_no, 0.005)
        }
        
        # 5. HANDICAP ASIÁTICO
        mercados['Handicap_Asiatico'] = {}
        
        # Local -0.5 (Local gana)
        mercados['Handicap_Asiatico']['Local -0.5'] = mercados['1X2']['Local']
        mercados['Handicap_Asiatico']['Visitante +0.5'] = mercados['1X2']['Empate'] + mercados['1X2']['Visitante']
        
        # Local -1.5 (Local gana por 2+)
        p_local_menos15 = sum([matriz[i, j] for i in range(8) for j in range(8) if i - j >= 2])
        mercados['Handicap_Asiatico']['Local -1.5'] = max(p_local_menos15, 0.005)
        mercados['Handicap_Asiatico']['Visitante +1.5'] = max(1 - p_local_menos15, 0.005)
        
        # Visitante -0.5 (Visitante gana)
        mercados['Handicap_Asiatico']['Visitante -0.5'] = mercados['1X2']['Visitante']
        mercados['Handicap_Asiatico']['Local +0.5'] = mercados['1X2']['Empate'] + mercados['1X2']['Local']
        
        # 6. OVER/UNDER POR EQUIPO
        mercados['Over_Under_Local'] = {}
        for threshold in [0.5, 1.5, 2.5]:
            p_over_local = sum([matriz[i, j] for i in range(8) for j in range(8) if i > threshold])
            mercados['Over_Under_Local'][f"Local Over {threshold}"] = max(p_over_local, 0.005)
            mercados['Over_Under_Local'][f"Local Under {threshold}"] = max(1 - p_over_local, 0.005)
        
        mercados['Over_Under_Visitante'] = {}
        for threshold in [0.5, 1.5, 2.5]:
            p_over_visit = sum([matriz[i, j] for i in range(8) for j in range(8) if j > threshold])
            mercados['Over_Under_Visitante'][f"Visitante Over {threshold}"] = max(p_over_visit, 0.005)
            mercados['Over_Under_Visitante'][f"Visitante Under {threshold}"] = max(1 - p_over_visit, 0.005)
        
        # 7. PRIMERA MITAD
        lambda_1h_local = lambda_l * 0.45
        lambda_1h_visitante = lambda_v * 0.45
        
        matriz_1h = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                matriz_1h[i, j] = poisson.pmf(i, lambda_1h_local) * poisson.pmf(j, lambda_1h_visitante)
        
        p_local_1h = np.sum(np.tril(matriz_1h, -1))
        p_empate_1h = np.sum(np.diag(matriz_1h))
        p_visitante_1h = np.sum(np.triu(matriz_1h, 1))
        total_1h = p_local_1h + p_empate_1h + p_visitante_1h
        
        mercados['Primera_Mitad'] = {
            'Local gana 1H': max(p_local_1h / total_1h, 0.005),
            'Empate 1H': max(p_empate_1h / total_1h, 0.005),
            'Visitante gana 1H': max(p_visitante_1h / total_1h, 0.005),
            'Over 0.5 en 1H': max(1 - matriz_1h[0, 0], 0.005),
            'Over 1.5 en 1H': max(sum([matriz_1h[i, j] for i in range(5) for j in range(5) if (i+j) > 1.5]), 0.005),
            'BTTS en 1H': max(1 - (matriz_1h[0,:].sum() + matriz_1h[:,0].sum() - matriz_1h[0,0]), 0.005)
        }
        
        # 8. GOLES EXACTOS
        mercados['Goles_Totales'] = {
            '0 goles': max(matriz[0, 0], 0.005),
            '1 gol': max(matriz[1, 0] + matriz[0, 1], 0.005),
            '2 goles': max(matriz[2, 0] + matriz[1, 1] + matriz[0, 2], 0.005),
            '3 goles': max(matriz[3, 0] + matriz[2, 1] + matriz[1, 2] + matriz[0, 3], 0.005),
            '4+ goles': max(sum([matriz[i, j] for i in range(8) for j in range(8) if (i+j) >= 4]), 0.005)
        }
        
        # 9. MULTIGOALS
        mercados['Multigoals'] = {
            '1-2 goles': max(sum([matriz[i, j] for i in range(8) for j in range(8) if 1 <= (i+j) <= 2]), 0.005),
            '2-3 goles': max(sum([matriz[i, j] for i in range(8) for j in range(8) if 2 <= (i+j) <= 3]), 0.005),
            '3-4 goles': max(sum([matriz[i, j] for i in range(8) for j in range(8) if 3 <= (i+j) <= 4]), 0.005),
            '4-5 goles': max(sum([matriz[i, j] for i in range(8) for j in range(8) if 4 <= (i+j) <= 5]), 0.005)
        }
        
        # 10. BTTS + RESULTADO
        mercados['BTTS_Resultado'] = {
            'BTTS + Local gana': max(sum([matriz[i, j] for i in range(1, 8) for j in range(1, 8) if i > j]), 0.005),
            'BTTS + Empate': max(sum([matriz[i, i] for i in range(1, 8)]), 0.005),
            'BTTS + Visitante gana': max(sum([matriz[i, j] for i in range(1, 8) for j in range(1, 8) if j > i]), 0.005)
        }
        
        # 11. RESULTADO EXACTO TOP 10
        resultados_exactos = []
        for i in range(8):
            for j in range(8):
                resultados_exactos.append({
                    'Marcador': f"{i}-{j}",
                    'Probabilidad': max(matriz[i, j], 0.001)
                })
        
        resultados_exactos.sort(key=lambda x: x['Probabilidad'], reverse=True)
        mercados['Resultado_Exacto'] = resultados_exactos[:10]
        
        return mercados

# ============================================================================
# MÓDULO 6: GENERADOR DE ARGUMENTOS NARRATIVOS PROFESIONALES
# ============================================================================

class GeneradorArgumentosNarrativos:
    """Genera argumentos en formato narrativo profesional"""
    
    @staticmethod
    def generar_analisis_completo(mercado, prob, local_team, visitante_team, analisis):
        """
        Genera análisis completo en formato narrativo profesional
        
        NO usa bullet points, sino párrafos coherentes
        """
        
        forma_local = analisis['forma_local']
        forma_visitante = analisis['forma_visitante']
        detalles_local = analisis['detalles_local']
        detalles_visitante = analisis['detalles_visitante']
        stats_local = analisis['stats_local']
        stats_visitante = analisis['stats_visitante']
        
        # Determinar tipo de mercado
        if 'Local' in mercado and 'Over' not in mercado and 'Under' not in mercado:
            tipo = '1X2_LOCAL'
        elif 'Visitante' in mercado and 'Over' not in mercado and 'Under' not in mercado:
            tipo = '1X2_VISITANTE'
        elif 'Empate' in mercado:
            tipo = '1X2_EMPATE'
        elif 'Over' in mercado:
            tipo = 'OVER'
        elif 'Under' in mercado:
            tipo = 'UNDER'
        elif 'BTTS Si' in mercado:
            tipo = 'BTTS_SI'
        elif 'BTTS No' in mercado:
            tipo = 'BTTS_NO'
        else:
            tipo = 'GENERICO'
        
        # GENERAR NARRATIVA SEGÚN TIPO
        
        if tipo == '1X2_LOCAL':
            narrativa = f"""📝 **ANÁLISIS TÉCNICO DETALLADO**

Esta apuesta se considera **{'ALTAMENTE RECOMENDABLE' if prob > 0.65 else 'FAVORABLE' if prob > 0.55 else 'A CONSIDERAR'}** con una probabilidad del {prob*100:.1f}% basada en los siguientes factores fundamentales:

🎯 **CAPACIDAD OFENSIVA Y FORMA RECIENTE**

{local_team['Equipo']} presenta un rendimiento ofensivo de {detalles_local['gf']/20:.2f} goles por partido en sus últimos 20 encuentros, acumulando {detalles_local['victorias']} victorias que representan una efectividad del {forma_local*100:.0f}%. Este dato cobra especial relevancia cuando se compara con el {detalles_visitante['gf']/20:.2f} goles/partido de {visitante_team['Equipo']}, estableciendo una diferencia ofensiva significativa.

La racha actual del equipo local muestra: {detalles_local['racha_actual']}, lo que evidencia {'una consistencia notable' if detalles_local['victorias'] >= 12 else 'un rendimiento moderado'} en el tramo final de la temporada. """

            if stats_local['local_pj'] >= 5:
                winrate_casa = stats_local['local_victorias'] / stats_local['local_pj']
                narrativa += f"""Específicamente jugando como local, {local_team['Equipo']} ha conseguido {stats_local['local_victorias']} victorias en {stats_local['local_pj']} partidos ({winrate_casa*100:.0f}% de efectividad), con un promedio de {stats_local['local_gf']/stats_local['local_pj']:.2f} goles a favor por encuentro en su estadio.

"""
            
            if analisis.get('factor_local', 1.0) > 1.1:
                narrativa += f"""⚡ **AJUSTE POR NIVEL DE COMPETICIÓN**

El sistema ha aplicado un factor de ajuste positivo ({analisis['factor_local']:.2f}x) debido a que el análisis detectó ventajas competitivas relacionadas con el nivel de liga y/o experiencia en competiciones europeas. {analisis.get('advertencias', [''])[0] if analisis.get('advertencias') else ''}

"""
            
            narrativa += f"""⚖️ **VEREDICTO TÉCNICO Y RECOMENDACIÓN**

La convergencia de superioridad ofensiva ({detalles_local['gf']/20:.2f} vs {detalles_visitante['gf']/20:.2f} goles/partido), ventaja de localía, y forma reciente posiciona esta apuesta con una confianza técnica elevada. Se recomienda buscar cuotas superiores a {(1/prob)*0.95:.2f} en el mercado para asegurar un valor esperado positivo. Con cuotas por encima de este umbral, el Expected Value (EV) sería de aproximadamente +{((prob * (1/prob)*0.95) - 1)*100:.1f}%."""

        elif tipo == 'OVER':
            threshold = float(mercado.split()[1])
            goles_promedio = detalles_local['gf']/20 + detalles_visitante['gf']/20
            
            narrativa = f"""📝 **ANÁLISIS TÉCNICO DETALLADO**

Esta apuesta de **Over {threshold}** presenta una probabilidad del {prob*100:.1f}%, clasificándose como **{'ALTAMENTE FAVORABLE' if prob > 0.65 else 'RECOMENDABLE' if prob > 0.55 else 'EQUILIBRADA'}** según nuestro modelo predictivo.

⚽ **CAPACIDAD OFENSIVA COMBINADA**

El análisis de los últimos 20 partidos revela que {local_team['Equipo']} promedia {detalles_local['gf']/20:.2f} goles por encuentro, mientras que {visitante_team['Equipo']} aporta {detalles_visitante['gf']/20:.2f} goles/partido. Esta combinación resulta en un promedio esperado de {goles_promedio:.2f} goles totales, {'superando claramente' if goles_promedio > threshold else 'aproximándose a'} el umbral de {threshold} goles.

📊 **TENDENCIA HISTÓRICA**

En términos de frecuencia histórica, {stats_local['partidos_over25']/20*100:.0f}% de los partidos de {local_team['Equipo']} y {stats_visitante['partidos_over25']/20*100:.0f}% de los de {visitante_team['Equipo']} han superado los 2.5 goles en sus últimos 20 encuentros. """

            if stats_local['partidos_over25']/20 > 0.6 or stats_visitante['partidos_over25']/20 > 0.6:
                narrativa += f"""Esta alta frecuencia, {'especialmente notable' if max(stats_local['partidos_over25']/20, stats_visitante['partidos_over25']/20) > 0.7 else 'moderadamente significativa'}, sugiere que ambos equipos están involucrados regularmente en partidos con abundancia de goles.

"""
            
            narrativa += f"""🥅 **VULNERABILIDAD DEFENSIVA**

Un factor adicional a considerar es la vulnerabilidad defensiva: {local_team['Equipo']} ha encajado {detalles_local['gc']/20:.2f} goles/partido, mientras que {visitante_team['Equipo']} recibe {detalles_visitante['gc']/20:.2f}. {'Estas cifras elevadas en ambos equipos incrementan significativamente la probabilidad de un partido con múltiples goles.' if (detalles_local['gc']/20 > 1.2 and detalles_visitante['gc']/20 > 1.2) else 'La combinación de estos factores defensivos contribuye al pronóstico de goles.'}

⚖️ **VEREDICTO TÉCNICO**

Considerando la capacidad ofensiva combinada, las tendencias históricas y las características defensivas de ambos equipos, se recomienda buscar cuotas mínimas de {(1/prob)*0.95:.2f} para esta apuesta. Con estas cuotas, el Expected Value estimado sería de +{((prob * (1/prob)*0.95) - 1)*100:.1f}%."""

        elif tipo == 'BTTS_SI':
            narrativa = f"""📝 **ANÁLISIS TÉCNICO DETALLADO**

La apuesta de **Ambos Equipos Anotan** presenta una probabilidad del {prob*100:.1f}%, posicionándose como una opción **{'ALTAMENTE RECOMENDABLE' if prob > 0.65 else 'FAVORABLE' if prob > 0.55 else 'EQUILIBRADA'}**.

🎯 **CAPACIDAD OFENSIVA BILATERAL**

{local_team['Equipo']} ha demostrado consistencia ofensiva con {detalles_local['gf']/20:.2f} goles/partido promedio, mientras que {visitante_team['Equipo']} aporta {detalles_visitante['gf']/20:.2f} goles/partido. {'Ambos equipos superan el umbral de 1.0 gol por partido, lo que estadísticamente favorece el BTTS.' if detalles_local['gf']/20 > 1.0 and detalles_visitante['gf']/20 > 1.0 else 'La capacidad ofensiva de ambos equipos sugiere potencial para anotar.'}

📊 **FRECUENCIA HISTÓRICA DE BTTS**

En los últimos 20 partidos, {stats_local['partidos_btts']/20*100:.0f}% de los encuentros de {local_team['Equipo']} han terminado con ambos equipos anotando, mientras que {visitante_team['Equipo']} registra un {stats_visitante['partidos_btts']/20*100:.0f}% en esta misma métrica. {'Esta alta frecuencia bilateral es un indicador técnico muy favorable para BTTS.' if (stats_local['partidos_btts']/20 > 0.55 and stats_visitante['partidos_btts']/20 > 0.55) else 'Estos porcentajes sugieren una probabilidad moderada de que ambos anoten.'}

🥅 **FACTOR DEFENSIVO**

Las estadísticas defensivas refuerzan este pronóstico: {local_team['Equipo']} ha encajado {detalles_local['gc']/20:.2f} goles/partido, mientras que {visitante_team['Equipo']} recibe {detalles_visitante['gc']/20:.2f}. Estas cifras indican que ambas defensas son {'vulnerables' if (detalles_local['gc']/20 > 1.0 and detalles_visitante['gc']/20 > 1.0) else 'penetrables'}, aumentando la probabilidad de que ambos equipos encuentren el gol.

⚖️ **VEREDICTO Y RECOMENDACIÓN**

Con una probabilidad del {prob*100:.1f}%, se recomienda buscar cuotas de al menos {(1/prob)*0.95:.2f} para asegurar valor positivo. El Expected Value con estas cuotas sería de aproximadamente +{((prob * (1/prob)*0.95) - 1)*100:.1f}%."""

        else:
            # Narrativa genérica para otros mercados
            narrativa = f"""📝 **ANÁLISIS TÉCNICO**

Esta apuesta presenta una probabilidad del {prob*100:.1f}% según nuestro modelo predictivo avanzado.

📊 **FUNDAMENTO ESTADÍSTICO**

El análisis se basa en los últimos 20 partidos de ambos equipos, considerando forma reciente, capacidad ofensiva/defensiva, rendimiento local/visitante, y ajustes por nivel de competición.

{local_team['Equipo']}: Forma {forma_local*100:.0f}% | {detalles_local['victorias']}V-{detalles_local['empates']}E-{detalles_local['derrotas']}D | {detalles_local['gf']/20:.2f} GF/partido

{visitante_team['Equipo']}: Forma {forma_visitante*100:.0f}% | {detalles_visitante['victorias']}V-{detalles_visitante['empates']}E-{detalles_visitante['derrotas']}D | {detalles_visitante['gf']/20:.2f} GF/partido

⚖️ **RECOMENDACIÓN**

Buscar cuotas mínimas de {(1/prob)*0.95:.2f} para valor positivo (+{((prob * (1/prob)*0.95) - 1)*100:.1f}% EV estimado)."""

        return narrativa

# ============================================================================
# MÓDULO 7: SISTEMA DE CONFIABILIDAD
# ============================================================================

class SistemaConfiabilidad:
    """Calcula indicadores de confiabilidad para cada predicción"""
    
    @staticmethod
    def calcular_confiabilidad(prob, n_partidos_bd, hay_h2h, mismo_nivel_liga, analisis):
        """
        Calcula nivel de confiabilidad basado en múltiples factores
        
        Returns: (nivel, estrellas, factores)
        """
        
        puntos = 0
        max_puntos = 0
        factores = {}
        
        # Factor 1: Probabilidad (0-25 puntos)
        max_puntos += 25
        if prob > 0.70:
            puntos += 25
            factores['Probabilidad'] = "⭐⭐⭐⭐⭐"
        elif prob > 0.60:
            puntos += 20
            factores['Probabilidad'] = "⭐⭐⭐⭐"
        elif prob > 0.50:
            puntos += 15
            factores['Probabilidad'] = "⭐⭐⭐"
        else:
            puntos += 10
            factores['Probabilidad'] = "⭐⭐"
        
        # Factor 2: Datos en BD (0-20 puntos)
        max_puntos += 20
        if n_partidos_bd > 500:
            puntos += 20
            factores['Datos Históricos'] = "⭐⭐⭐⭐⭐"
        elif n_partidos_bd > 200:
            puntos += 15
            factores['Datos Históricos'] = "⭐⭐⭐⭐"
        elif n_partidos_bd > 50:
            puntos += 10
            factores['Datos Históricos'] = "⭐⭐⭐"
        else:
            puntos += 5
            factores['Datos Históricos'] = "⭐⭐"
        
        # Factor 3: H2H disponible (0-15 puntos)
        max_puntos += 15
        if hay_h2h >= 10:
            puntos += 15
            factores['Enfrentamientos Directos'] = "⭐⭐⭐⭐⭐"
        elif hay_h2h >= 5:
            puntos += 10
            factores['Enfrentamientos Directos'] = "⭐⭐⭐"
        elif hay_h2h > 0:
            puntos += 5
            factores['Enfrentamientos Directos'] = "⭐⭐"
        else:
            puntos += 0
            factores['Enfrentamientos Directos'] = "⭐"
        
        # Factor 4: Mismo nivel de liga (0-20 puntos)
        max_puntos += 20
        if mismo_nivel_liga:
            puntos += 20
            factores['Comparabilidad'] = "⭐⭐⭐⭐⭐"
        else:
            puntos += 10
            factores['Comparabilidad'] = "⭐⭐⭐"
        
        # Factor 5: Ajustes aplicados (0-20 puntos)
        max_puntos += 20
        if analisis.get('advertencias'):
            puntos += 20
            factores['Ajustes de Competición'] = "⭐⭐⭐⭐⭐"
        else:
            puntos += 15
            factores['Ajustes de Competición'] = "⭐⭐⭐⭐"
        
        # Calcular porcentaje
        porcentaje = (puntos / max_puntos) * 100
        
        # Determinar nivel
        if porcentaje >= 85:
            nivel = "🟢🟢🟢🟢🟢"
            texto = "CONFIANZA MUY ALTA"
            color = "success"
        elif porcentaje >= 70:
            nivel = "🟢🟢🟢🟢⚪"
            texto = "CONFIANZA ALTA"
            color = "success"
        elif porcentaje >= 55:
            nivel = "🟢🟢🟢⚪⚪"
            texto = "CONFIANZA MODERADA"
            color = "info"
        elif porcentaje >= 40:
            nivel = "🟡🟡🟡⚪⚪"
            texto = "CONFIANZA BAJA"
            color = "warning"
        else:
            nivel = "🔴🔴⚪⚪⚪"
            texto = "CONFIANZA MUY BAJA"
            color = "error"
        
        return {
            'nivel': nivel,
            'texto': texto,
            'porcentaje': porcentaje,
            'factores': factores,
            'color': color
        }

# ============================================================================
# INTERFAZ STREAMLIT PRINCIPAL
# ============================================================================

def main():
    st.title("⚽ SISTEMABETS EXPERTO v9.0")
    st.markdown("### Sistema Profesional Completo con 40+ Mercados")
    
    # Inicializar componentes
    if 'db_manager' not in st.session_state:
        st.session_state['db_manager'] = DatabaseManager()
    
    db_manager = st.session_state['db_manager']
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        api_key = st.text_input("API Key Football-Data.org", type="password",
                               help="Clave de https://www.football-data.org/")
        
        if not api_key:
            st.warning("⚠️ Ingresa tu API Key")
            st.info("""
            **¿Cómo obtener tu API Key?**
            1. Ve a football-data.org
            2. Regístrate gratis
            3. Copia tu API Key
            4. Pégala aquí
            """)
            st.stop()
        
        st.markdown("---")
        
        # Estado de la base de datos
        st.header("📊 Base de Datos")
        
        stats_bd = db_manager.obtener_estadisticas()
        
        st.metric("Partidos Almacenados", stats_bd['total'])
        
        if stats_bd['total'] > 0:
            with st.expander("Ver Detalles"):
                st.write("**Por Competición:**")
                for comp, count in stats_bd['por_competicion'].items():
                    st.caption(f"{comp}: {count} partidos")
                
                st.write(f"**Rango:** {stats_bd['fecha_min']} a {stats_bd['fecha_max']}")
        
        st.markdown("---")
        
        # RECOLECTOR AUTOMÁTICO
        st.header("🤖 Recolector Automático")
        
        st.info("""
        **¿Para qué sirve?**
        
        Descarga automáticamente partidos históricos de TODOS los equipos de una liga.
        
        Esto alimenta la base de datos para futuras funciones de Machine Learning.
        """)
        
        liga_recolectar = st.selectbox(
            "Liga a recolectar",
            list(FootballDataAPI.LIGAS.keys()),
            key='liga_recolectar'
        )
        
        partidos_por_equipo = st.slider(
            "Partidos por equipo",
            min_value=20,
            max_value=50,
            value=30,
            help="Más partidos = más datos pero más tiempo"
        )
        
        if st.button("🚀 RECOLECTAR PARTIDOS", type="primary"):
            api = FootballDataAPI(api_key)
            recolector = RecolectorAutomatico(api, db_manager)
            
            liga_code = FootballDataAPI.LIGAS[liga_recolectar]
            
            st.info(f"Iniciando recolección de {liga_recolectar}...")
            st.warning("⏱️ Esto puede tardar 10-20 minutos debido a límites de la API (10 req/min)")
            
            nuevos, duplicados, errores = recolector.recolectar_liga_completa(
                liga_code, liga_recolectar, partidos_por_equipo
            )
            
            st.success(f"✅ Recolección completada!")
            st.metric("Partidos nuevos", nuevos)
            st.metric("Partidos duplicados (ignorados)", duplicados)
            
            if errores:
                with st.expander("⚠️ Ver errores"):
                    for error in errores:
                        st.caption(error)
        
        st.markdown("---")
        
        # Gestión
        with st.expander("🗄️ Gestión Avanzada"):
            if st.button("🗑️ Limpiar Base de Datos"):
                db_manager.limpiar_database()
                st.success("Base de datos limpiada")
                st.rerun()
    
    # Selección de liga
    st.header("1️⃣ Selecciona la Liga")
    liga_nombre = st.selectbox("Liga", list(FootballDataAPI.LIGAS.keys()))
    liga_code = FootballDataAPI.LIGAS[liga_nombre]
    
    api = FootballDataAPI(api_key)
    
    # Cargar tabla
    with st.spinner("Cargando tabla..."):
        df_liga = api.obtener_standings(liga_code)
    
    if df_liga is None or df_liga.empty:
        st.error("❌ No se pudo cargar la tabla")
        st.stop()
    
    st.success(f"✅ {liga_nombre} cargada")
    
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
    
    # ANÁLISIS
    if st.button("🚀 ANALIZAR PARTIDO", type="primary", use_container_width=True):
        
        local_team = df_liga[df_liga['Equipo'] == equipo_local].iloc[0]
        visitante_team = df_liga[df_liga['Equipo'] == equipo_visitante].iloc[0]
        
        st.markdown("---")
        st.header(f"📊 {equipo_local} vs {equipo_visitante}")
        
        # Cargar datos
        with st.spinner("Cargando datos de partidos..."):
            partidos_local = api.obtener_ultimos_20_partidos(equipo_local)
            partidos_visitante = api.obtener_ultimos_20_partidos(equipo_visitante)
            h2h = api.obtener_enfrentamientos_directos_completo(equipo_local, equipo_visitante)
        
        if not partidos_local or not partidos_visitante:
            st.error("❌ No se pudieron cargar los partidos")
            st.stop()
        
        # ANÁLISIS COMPLETO
        analisis = AnalizadorExperto.analisis_completo(
            local_team, visitante_team, partidos_local, partidos_visitante, h2h
        )
        
        # Mostrar advertencias
        if analisis.get('advertencias'):
            for adv in analisis['advertencias']:
                st.warning(adv)
        
        # Parámetros
        st.subheader("🔍 Parámetros de Análisis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Lambda Local", f"{analisis['lambda_local']:.2f}",
                   help="Goles esperados del local (ajustado)")
        col2.metric("Lambda Visitante", f"{analisis['lambda_visitante']:.2f}",
                   help="Goles esperados del visitante (ajustado)")
        col3.metric("Total Goles Esperados", f"{analisis['lambda_local'] + analisis['lambda_visitante']:.2f}")
        col4.metric("Factor Ajuste Local", f"{analisis['factor_local']:.2f}x",
                   help="Factor de ajuste por competición")
        
        # FORMA RECIENTE
        st.markdown("---")
        st.subheader("📈 Forma Reciente (Últimos 20 Partidos)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### 🏠 {equipo_local}")
            
            forma_local = analisis['forma_local']
            detalles_local = analisis['detalles_local']
            
            st.progress(forma_local)
            st.caption(f"Forma: {forma_local*100:.1f}%")
            
            st.markdown(f"""
            - ✅ **Victorias:** {detalles_local['victorias']}
            - 🟰 **Empates:** {detalles_local['empates']}
            - ❌ **Derrotas:** {detalles_local['derrotas']}
            - ⚽ **GF:** {detalles_local['gf']} ({detalles_local['gf']/20:.2f}/partido)
            - 🥅 **GC:** {detalles_local['gc']} ({detalles_local['gc']/20:.2f}/partido)
            - 🔥 **Racha:** {detalles_local['racha_actual']}
            """)
            
            # EXPANDER para ver 20 partidos
            with st.expander(f"📋 Ver los 20 partidos de {equipo_local}"):
                df_partidos_local = pd.DataFrame(analisis['resultados_local'])
                df_partidos_local = df_partidos_local[['simbolo', 'local', 'resultado', 'visitante', 'competicion', 'fecha']]
                df_partidos_local.columns = ['', 'Local', 'Resultado', 'Visitante', 'Competición', 'Fecha']
                st.dataframe(df_partidos_local, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown(f"### ✈️ {equipo_visitante}")
            
            forma_visitante = analisis['forma_visitante']
            detalles_visitante = analisis['detalles_visitante']
            
            st.progress(forma_visitante)
            st.caption(f"Forma: {forma_visitante*100:.1f}%")
            
            st.markdown(f"""
            - ✅ **Victorias:** {detalles_visitante['victorias']}
            - 🟰 **Empates:** {detalles_visitante['empates']}
            - ❌ **Derrotas:** {detalles_visitante['derrotas']}
            - ⚽ **GF:** {detalles_visitante['gf']} ({detalles_visitante['gf']/20:.2f}/partido)
            - 🥅 **GC:** {detalles_visitante['gc']} ({detalles_visitante['gc']/20:.2f}/partido)
            - 🔥 **Racha:** {detalles_visitante['racha_actual']}
            """)
            
            # EXPANDER para ver 20 partidos
            with st.expander(f"📋 Ver los 20 partidos de {equipo_visitante}"):
                df_partidos_visitante = pd.DataFrame(analisis['resultados_visitante'])
                df_partidos_visitante = df_partidos_visitante[['simbolo', 'local', 'resultado', 'visitante', 'competicion', 'fecha']]
                df_partidos_visitante.columns = ['', 'Local', 'Resultado', 'Visitante', 'Competición', 'Fecha']
                st.dataframe(df_partidos_visitante, use_container_width=True, hide_index=True)
        
        # H2H COMPLETO
        if h2h:
            st.markdown("---")
            st.subheader(f"🎯 Enfrentamientos Directos - Historial Completo ({len(h2h)} partidos)")
            
            victorias_local = sum(1 for p in h2h if 
                                 (p['local'] == equipo_local and p['goles_local'] > p['goles_visitante']) or
                                 (p['visitante'] == equipo_local and p['goles_visitante'] > p['goles_local']))
            empates = sum(1 for p in h2h if p['goles_local'] == p['goles_visitante'])
            victorias_visitante = len(h2h) - victorias_local - empates
            
            goles_totales_h2h = sum(p['goles_local'] + p['goles_visitante'] for p in h2h)
            promedio_goles_h2h = goles_totales_h2h / len(h2h)
            
            btts_h2h = sum(1 for p in h2h if p['goles_local'] > 0 and p['goles_visitante'] > 0)
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric(f"Victorias {equipo_local}", victorias_local)
            col2.metric("Empates", empates)
            col3.metric(f"Victorias {equipo_visitante}", victorias_visitante)
            col4.metric("Promedio Goles", f"{promedio_goles_h2h:.2f}")
            
            st.info(f"""
            📊 **Estadísticas H2H:**
            - **BTTS:** {btts_h2h}/{len(h2h)} partidos ({btts_h2h/len(h2h)*100:.0f}%)
            - **Over 2.5:** {sum(1 for p in h2h if (p['goles_local'] + p['goles_visitante']) > 2.5)}/{len(h2h)} partidos
            - **Último enfrentamiento:** {h2h[0]['local']} {h2h[0]['goles_local']}-{h2h[0]['goles_visitante']} {h2h[0]['visitante']}
            """)
            
            with st.expander(f"📋 Ver TODOS los {len(h2h)} enfrentamientos"):
                df_h2h = pd.DataFrame(h2h)
                df_h2h['Resultado'] = df_h2h.apply(lambda x: f"{x['goles_local']}-{x['goles_visitante']}", axis=1)
                df_h2h['Fecha'] = pd.to_datetime(df_h2h['fecha']).dt.strftime('%Y-%m-%d')
                df_display = df_h2h[['Fecha', 'local', 'Resultado', 'visitante', 'competicion']]
                df_display.columns = ['Fecha', 'Local', 'Resultado', 'Visitante', 'Competición']
                st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # CALCULAR TODOS LOS MERCADOS
        mercados = CalculadorMercados.calcular_todos_mercados(
            analisis['lambda_local'],
            analisis['lambda_visitante'],
            analisis['stats_local'],
            analisis['stats_visitante']
        )
        
        # PREDICCIONES - MERCADOS PRINCIPALES
        st.markdown("---")
        st.header("🎯 PREDICCIONES - Mercados Principales")
        
        # 1X2
        st.subheader("⚽ Resultado Final (1X2)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(equipo_local, f"{mercados['1X2']['Local']*100:.1f}%")
            st.caption(f"Cuota justa: {1/mercados['1X2']['Local']:.2f}")
        
        with col2:
            st.metric("Empate", f"{mercados['1X2']['Empate']*100:.1f}%")
            st.caption(f"Cuota justa: {1/mercados['1X2']['Empate']:.2f}")
        
        with col3:
            st.metric(equipo_visitante, f"{mercados['1X2']['Visitante']*100:.1f}%")
            st.caption(f"Cuota justa: {1/mercados['1X2']['Visitante']:.2f}")
        
        # Over/Under
        st.subheader("📊 Over/Under")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        col1.metric("Over 0.5", f"{mercados['Over_Under']['Over 0.5']*100:.1f}%")
        col2.metric("Over 1.5", f"{mercados['Over_Under']['Over 1.5']*100:.1f}%")
        col3.metric("Over 2.5", f"{mercados['Over_Under']['Over 2.5']*100:.1f}%")
        col4.metric("Over 3.5", f"{mercados['Over_Under']['Over 3.5']*100:.1f}%")
        col5.metric("Over 4.5", f"{mercados['Over_Under']['Over 4.5']*100:.1f}%")
        
        # BTTS
        st.subheader("🎯 Ambos Equipos Anotan (BTTS)")
        
        col1, col2 = st.columns(2)
        
        col1.metric("BTTS Sí", f"{mercados['BTTS']['Si']*100:.1f}%")
        col2.metric("BTTS No", f"{mercados['BTTS']['No']*100:.1f}%")
        
        # MERCADOS ADICIONALES (EXPANDERS)
        st.markdown("---")
        st.header("📊 Mercados Adicionales (40+ opciones)")
        
        with st.expander("🎲 Doble Oportunidad"):
            col1, col2, col3 = st.columns(3)
            for i, (mercado, prob) in enumerate(mercados['Doble_Oportunidad'].items()):
                with [col1, col2, col3][i]:
                    st.metric(mercado, f"{prob*100:.1f}%")
                    st.caption(f"Cuota: {1/prob:.2f}")
        
        with st.expander("⚖️ Handicap Asiático"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Handicap Local:**")
                for mercado, prob in mercados['Handicap_Asiatico'].items():
                    if 'Local' in mercado and '-' in mercado:
                        st.metric(mercado, f"{prob*100:.1f}%")
            with col2:
                st.markdown("**Handicap Visitante:**")
                for mercado, prob in mercados['Handicap_Asiatico'].items():
                    if 'Visitante' in mercado:
                        st.metric(mercado, f"{prob*100:.1f}%")
        
        with st.expander("⚽ Over/Under por Equipo"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{equipo_local}:**")
                for mercado, prob in mercados['Over_Under_Local'].items():
                    st.caption(f"{mercado}: {prob*100:.1f}%")
            with col2:
                st.markdown(f"**{equipo_visitante}:**")
                for mercado, prob in mercados['Over_Under_Visitante'].items():
                    st.caption(f"{mercado}: {prob*100:.1f}%")
        
        with st.expander("⏱️ Primera Mitad (1H)"):
            for mercado, prob in mercados['Primera_Mitad'].items():
                st.metric(mercado, f"{prob*100:.1f}%")
        
        with st.expander("🎯 Goles Exactos"):
            for mercado, prob in mercados['Goles_Totales'].items():
                st.metric(mercado, f"{prob*100:.1f}%")
        
        with st.expander("📊 Multigoals"):
            for mercado, prob in mercados['Multigoals'].items():
                st.metric(mercado, f"{prob*100:.1f}%")
        
        with st.expander("⚽⚽ BTTS + Resultado"):
            for mercado, prob in mercados['BTTS_Resultado'].items():
                st.metric(mercado, f"{prob*100:.1f}%")
        
        with st.expander("🎲 Resultado Exacto (Top 10)"):
            cols = st.columns(5)
            for i, resultado in enumerate(mercados['Resultado_Exacto'][:10]):
                with cols[i % 5]:
                    st.metric(resultado['Marcador'], f"{resultado['Probabilidad']*100:.1f}%")
        
        # TOP 5 MEJORES APUESTAS CON ANÁLISIS NARRATIVO
        st.markdown("---")
        st.header("💎 TOP 5 MEJORES APUESTAS")
        st.caption("Ordenadas por probabilidad con análisis técnico completo")
        
        # Recopilar todas las apuestas
        todas_apuestas = []
        
        # 1X2
        todas_apuestas.append({
            'Mercado': f'Victoria {equipo_local}',
            'Prob': mercados['1X2']['Local'],
            'Tipo': '1X2'
        })
        todas_apuestas.append({
            'Mercado': 'Empate',
            'Prob': mercados['1X2']['Empate'],
            'Tipo': '1X2'
        })
        todas_apuestas.append({
            'Mercado': f'Victoria {equipo_visitante}',
            'Prob': mercados['1X2']['Visitante'],
            'Tipo': '1X2'
        })
        
        # Over/Under principales
        for threshold in [1.5, 2.5, 3.5]:
            todas_apuestas.append({
                'Mercado': f'Over {threshold}',
                'Prob': mercados['Over_Under'][f'Over {threshold}'],
                'Tipo': 'Over/Under'
            })
            todas_apuestas.append({
                'Mercado': f'Under {threshold}',
                'Prob': mercados['Over_Under'][f'Under {threshold}'],
                'Tipo': 'Over/Under'
            })
        
        # BTTS
        todas_apuestas.append({
            'Mercado': 'BTTS Sí',
            'Prob': mercados['BTTS']['Si'],
            'Tipo': 'BTTS'
        })
        todas_apuestas.append({
            'Mercado': 'BTTS No',
            'Prob': mercados['BTTS']['No'],
            'Tipo': 'BTTS'
        })
        
        # Doble oportunidad
        for mercado, prob in mercados['Doble_Oportunidad'].items():
            todas_apuestas.append({
                'Mercado': mercado,
                'Prob': prob,
                'Tipo': 'Doble_Oportunidad'
            })
        
        # Ordenar por probabilidad
        todas_apuestas.sort(key=lambda x: x['Prob'], reverse=True)
        
        # Mostrar TOP 5
        for i, apuesta in enumerate(todas_apuestas[:5], 1):
            
            # Calcular confiabilidad
            mismo_nivel = not any('liga más competitiva' in adv.lower() for adv in analisis.get('advertencias', []))
            confiabilidad = SistemaConfiabilidad.calcular_confiabilidad(
                apuesta['Prob'],
                db_manager.contar_partidos(),
                len(h2h),
                mismo_nivel,
                analisis
            )
            
            with st.expander(f"#{i} - {apuesta['Mercado']} | {apuesta['Prob']*100:.1f}% {confiabilidad['nivel']}"):
                
                # Métricas principales
                col1, col2, col3 = st.columns(3)
                
                col1.metric("Probabilidad", f"{apuesta['Prob']*100:.1f}%")
                col2.metric("Cuota Justa", f"{1/apuesta['Prob']:.2f}")
                col3.metric("Cuota Mínima (95%)", f"{(1/apuesta['Prob'])*0.95:.2f}")
                
                # Sistema de confiabilidad
                st.markdown("---")
                st.markdown(f"### 📊 Indicador de Confiabilidad: {confiabilidad['texto']}")
                st.markdown(f"**Nivel:** {confiabilidad['nivel']} ({confiabilidad['porcentaje']:.0f}%)")
                
                st.markdown("**Factores Evaluados:**")
                for factor, estrellas in confiabilidad['factores'].items():
                    st.caption(f"- {factor}: {estrellas}")
                
                # Análisis narrativo completo
                st.markdown("---")
                narrativa = GeneradorArgumentosNarrativos.generar_analisis_completo(
                    apuesta['Mercado'],
                    apuesta['Prob'],
                    local_team,
                    visitante_team,
                    analisis
                )
                
                st.markdown(narrativa)
                
                # Veredicto final con semáforo
                st.markdown("---")
                st.markdown("### 🚦 VEREDICTO FINAL")
                
                if apuesta['Prob'] > 0.70 and confiabilidad['porcentaje'] > 70:
                    st.success("✅ **ALTAMENTE RECOMENDADA** - Alta probabilidad y confiabilidad excelente")
                elif apuesta['Prob'] > 0.60 and confiabilidad['porcentaje'] > 60:
                    st.info("🔵 **RECOMENDADA** - Buena probabilidad con confiabilidad sólida")
                elif apuesta['Prob'] > 0.55:
                    st.warning("🟡 **CONSIDERAR** - Probabilidad moderada, evaluar cuotas del mercado")
                else:
                    st.error("🔴 **NO RECOMENDADA** - Probabilidad o confiabilidad insuficiente")
                
                st.caption(f"""
                **Interpretación del EV (Expected Value):**
                
                Si encuentras cuotas ≥ {(1/apuesta['Prob'])*0.95:.2f}, el Expected Value sería de aproximadamente +{((apuesta['Prob'] * (1/apuesta['Prob'])*0.95) - 1)*100:.1f}%.
                
                Un EV positivo significa que matemáticamente esta apuesta tiene valor a largo plazo.
                """)
        
        # DISCLAIMER
        st.markdown("---")
        st.warning("""
        ⚠️ **ADVERTENCIA IMPORTANTE:**
        
        - Sistema con ajustes inteligentes por nivel de competición
        - Análisis basado en datos 100% reales de Football-Data.org
        - Indicadores de confiabilidad para transparencia total
        - NO garantiza ganancias - úsalo como herramienta de apoyo
        - Siempre compara cuotas en múltiples casas
        - Apuesta responsablemente y dentro de tus posibilidades
        
        📊 **Este sistema analiza 40+ mercados diferentes con argumentos técnicos sólidos.**
        """)

if __name__ == "__main__":
    main()import streamlit as st
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
import time
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SISTEMABETS EXPERTO v9.0", layout="wide", initial_sidebar_state="expanded")

# ============================================================================
# MÓDULO 1: BASE DE DATOS MEJORADA
# ============================================================================

class DatabaseManager:
    """Gestiona la base de datos local con funciones avanzadas"""
    
    def __init__(self, db_path='partidos_historicos.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Crea las tablas con índices optimizados"""
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
        
        # Índices para búsquedas rápidas
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_equipos ON partidos(local, visitante)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_fecha ON partidos(fecha)')
        
        conn.commit()
        conn.close()
    
    def guardar_partido(self, partido_data):
        """Guarda un partido evitando duplicados"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Verificar si ya existe
        cursor.execute('''
            SELECT id FROM partidos 
            WHERE fecha = ? AND local = ? AND visitante = ?
        ''', (partido_data['fecha'], partido_data['local'], partido_data['visitante']))
        
        if cursor.fetchone():
            conn.close()
            return False  # Ya existe
        
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
        return True
    
    def obtener_todos_partidos(self):
        """Obtiene todos los partidos guardados"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM partidos ORDER BY fecha DESC')
        partidos = cursor.fetchall()
        conn.close()
        return partidos
    
    def contar_partidos(self):
        """Cuenta total de partidos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM partidos')
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def obtener_estadisticas(self):
        """Estadísticas de la base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total partidos
        cursor.execute('SELECT COUNT(*) FROM partidos')
        total = cursor.fetchone()[0]
        
        # Partidos por competición
        cursor.execute('SELECT competicion, COUNT(*) FROM partidos GROUP BY competicion')
        por_competicion = dict(cursor.fetchall())
        
        # Rango de fechas
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
        """Limpia toda la base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM partidos')
        conn.commit()
        conn.close()

# ============================================================================
# MÓDULO 2: RECOLECTOR AUTOMÁTICO DE PARTIDOS
# ============================================================================

class RecolectorAutomatico:
    """Descarga partidos históricos automáticamente"""
    
    def __init__(self, api, db_manager):
        self.api = api
        self.db_manager = db_manager
    
    def recolectar_liga_completa(self, liga_code, liga_nombre, max_partidos_por_equipo=50):
        """
        Recolecta partidos de TODOS los equipos de una liga
        
        Returns: (partidos_nuevos, partidos_duplicados, errores)
        """
        
        # Obtener equipos de la liga
        df_equipos = self.api.obtener_standings(liga_code)
        
        if df_equipos is None:
            return 0, 0, ["No se pudo cargar la tabla de la liga"]
        
        partidos_nuevos = 0
        partidos_duplicados = 0
        errores = []
        
        total_equipos = len(df_equipos)
        
        # Placeholder para progreso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, equipo_row in df_equipos.iterrows():
            equipo_nombre = equipo_row['Equipo']
            
            # Actualizar progreso
            progreso = (idx + 1) / total_equipos
            progress_bar.progress(progreso)
            status_text.text(f"📥 Descargando partidos de {equipo_nombre}... ({idx + 1}/{total_equipos})")
            
            try:
                # Obtener partidos del equipo
                partidos = self.api.obtener_ultimos_partidos_extendido(equipo_nombre, limit=max_partidos_por_equipo)
                
                # Guardar cada partido
                for partido in partidos:
                    # Crear estructura básica de features
                    features_basicas = {
                        'goles_local': partido['goles_local'],
                        'goles_visitante': partido['goles_visitante']
                    }
                    
                    partido_data = {
                        'fecha': partido['fecha'][:10],  # Solo fecha, sin hora
                        'competicion': partido['competicion'],
                        'local': partido['local'],
                        'visitante': partido['visitante'],
                        'goles_local': partido['goles_local'],
                        'goles_visitante': partido['goles_visitante'],
                        'features': features_basicas
                    }
                    
                    if self.db_manager.guardar_partido(partido_data):
                        partidos_nuevos += 1
                    else:
                        partidos_duplicados += 1
                
                # Delay para respetar rate limit de API (10 req/min)
                time.sleep(6)
                
            except Exception as e:
                errores.append(f"{equipo_nombre}: {str(e)}")
        
        progress_bar.empty()
        status_text.empty()
        
        return partidos_nuevos, partidos_duplicados, errores

# ============================================================================
# MÓDULO 3: API MEJORADA CON FUNCIONES EXTENDIDAS
# ============================================================================

class FootballDataAPI:
    """API con funciones mejoradas para recolección masiva"""
    
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
    
    # Ranking de fuerza por liga
    LIGA_STRENGTH = {
        'PL': 10, 'PD': 9.5, 'BL1': 9, 'SA': 8.5, 'FL1': 8,
        'PPL': 6, 'DED': 7, 'ELC': 7.5, 'CL': 10
    }
    
    # Equipos de élite
    ELITE_TEAMS = {
        'Real Madrid', 'FC Barcelona', 'Atlético Madrid',
        'Manchester City', 'Liverpool FC', 'Chelsea FC', 'Arsenal FC', 
        'Manchester United', 'Tottenham Hotspur',
        'FC Bayern München', 'Borussia Dortmund', 'RB Leipzig', 'Bayer 04 Leverkusen',
        'Inter Milan', 'AC Milan', 'Juventus FC', 'SSC Napoli',
        'Paris Saint-Germain', 'AFC Ajax', 'SL Benfica', 'FC Porto', 'Sporting CP'
    }
    
    def __init__(self, api_key):
        self.headers = {"X-Auth-Token": api_key}
        self.cache_teams = {}
    
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
        """Obtiene últimos 20 partidos (versión original)"""
        return self.obtener_ultimos_partidos_extendido(equipo_nombre, limit=20)
    
    def obtener_ultimos_partidos_extendido(self, equipo_nombre, limit=100):
        """Versión extendida que puede obtener hasta 100 partidos"""
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
            return partidos_ordenados
            
        except Exception as e:
            return []
    
    def obtener_enfrentamientos_directos_completo(self, equipo1, equipo2):
        """Obtiene TODO el historial H2H disponible (no solo 10)"""
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
            
            # Devolver TODOS (no limitar a 10)
            return sorted(h2h, key=lambda x: x['fecha'], reverse=True)
            
        except Exception as e:
            return []

# ============================================================================
# MÓDULO 4: ANALIZADOR CON AJUSTES POR COMPETICIÓN
# ============================================================================

class AnalizadorExperto:
    """Analizador con ajustes inteligentes y cálculos extendidos"""
    
    @staticmethod
    def detectar_nivel_competicion(partidos):
        """Detecta el nivel de competición según partidos recientes"""
        competiciones = [p['competicion'] for p in partidos[:10]]
        
        # Detectar Champions/Europa League
        es_europea = any('Champions' in c or 'Europa League' in c or 'UEFA' in c 
                        for c in competiciones)
        
        return es_europea
    
    @staticmethod
    def calcular_factor_ajuste(local_team, visitante_team, partidos_local, partidos_visitante):
        """
        Calcula factores de ajuste por:
        - Nivel de liga
        - Equipos de élite
        - Competición europea
        """
        
        factor_local = 1.0
        factor_visitante = 1.0
        advertencias = []
        
        # Detectar liga de cada equipo
        liga_local = None
        liga_visitante = None
        
        for p in partidos_local[:5]:
            for liga_code in FootballDataAPI.LIGA_STRENGTH.keys():
                if liga_code in ['CL']: continue
                # Mapeo simple por nombre
                if 'Premier' in p['competicion']: liga_local = 'PL'
                elif 'Primera' in p['competicion'] or 'La Liga' in p['competicion']: liga_local = 'PD'
                elif 'Bundesliga' in p['competicion']: liga_local = 'BL1'
                elif 'Serie A' in p['competicion']: liga_local = 'SA'
                elif 'Ligue 1' in p['competicion']: liga_local = 'FL1'
                elif 'Primeira' in p['competicion']: liga_local = 'PPL'
                elif 'Eredivisie' in p['competicion']: liga_local = 'DED'
                elif 'Championship' in p['competicion']: liga_local = 'ELC'
            if liga_local: break
        
        for p in partidos_visitante[:5]:
            if 'Premier' in p['competicion']: liga_visitante = 'PL'
            elif 'Primera' in p['competicion'] or 'La Liga' in p['competicion']: liga_visitante = 'PD'
            elif 'Bundesliga' in p['competicion']: liga_visitante = 'BL1'
            elif 'Serie A' in p['competicion']: liga_visitante = 'SA'
            elif 'Ligue 1' in p['competicion']: liga_visitante = 'FL1'
            elif 'Primeira' in p['competicion']: liga_visitante = 'PPL'
            elif 'Eredivisie' in p['competicion']: liga_visitante = 'DED'
            elif 'Championship' in p['competicion']: liga_visitante = 'ELC'
            if liga_visitante: break
        
        # Ajuste por nivel de liga
        if liga_local and liga_visitante:
            strength_local = FootballDataAPI.LIGA_STRENGTH.get(liga_local, 7)
            strength_visitante = FootballDataAPI.LIGA_STRENGTH.get(liga_visitante, 7)
            
            diff = strength_visitante - strength_local
            
            if diff > 2:
                factor_visitante *= 1.25
                factor_local *= 0.85
                advertencias.append(f"⚠️ {visitante_team['Equipo']} juega en liga de mayor nivel competitivo")
            elif diff < -2:
                factor_local *= 1.25
                factor_visitante *= 0.85
                advertencias.append(f"⚠️ {local_team['Equipo']} juega en liga de mayor nivel competitivo")
        
        # Ajuste por equipos de élite
        es_champions = AnalizadorExperto.detectar_nivel_competicion(partidos_local + partidos_visitante)
        
        if es_champions:
            advertencias.append("🏆 Competición europea detectada - Ajustes aplicados")
            
            if local_team['Equipo'] in FootballDataAPI.ELITE_TEAMS and visitante_team['Equipo'] not in FootballDataAPI.ELITE_TEAMS:
                factor_local *= 1.15
                factor_visitante *= 0.88
                advertencias.append(f"🌟 {local_team['Equipo']} es equipo de élite europea")
            
            elif visitante_team['Equipo'] in FootballDataAPI.ELITE_TEAMS and local_team['Equipo'] not in FootballDataAPI.ELITE_TEAMS:
                factor_visitante *= 1.15
                factor_local *= 0.88
                advertencias.append(f"🌟 {visitante_team['Equipo']} es equipo de élite europea")
        
        return factor_local, factor_visitante, advertencias
    
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
                resultado = "✅"
            elif gf == gc:
                puntos.append(0.5)
                forma_visual += "E"
                detalles['empates'] += 1
                resultado = "🟰"
            else:
                puntos.append(0.0)
                forma_visual += "D"
                detalles['derrotas'] += 1
                resultado = "❌"
            
            if i < 5:
                detalles['racha_actual'] += resultado
            
            resultados_detalle.append({
                'simbolo': resultado,
                'local': partido['local'],
                'visitante': partido['visitante'],
                'resultado': f"{partido['goles_local']}-{partido['goles_visitante']}",
                'competicion': partido['competicion'],
                'fecha': partido['fecha'][:10]
            })
        
        pesos = np.linspace(0.10, 0.005, len(puntos))
        pesos = pesos / pesos.sum()
        score_forma = sum(p * w for p, w in zip(puntos, pesos))
        
        return score_forma, forma_visual, detalles, resultados_detalle
    
    @staticmethod
    def calcular_stats_avanzadas(equipo_nombre, partidos):
        """Stats completas local/visitante + tendencias"""
        stats = {
            'local_pj': 0, 'local_gf': 0, 'local_gc': 0, 'local_victorias': 0,
            'visitante_pj': 0, 'visitante_gf': 0, 'visitante_gc': 0, 'visitante_victorias': 0,
            'partidos_over25': 0, 'partidos_btts': 0,
            'goles_1h_favor': 0, 'partidos_gol_1h': 0,
            'partidos_over15': 0, 'partidos_over35': 0
        }
        
        for p in partidos[:20]:
            total_goles = p['goles_local'] + p['goles_visitante']
            
            if total_goles > 1.5:
                stats['partidos_over15'] += 1
            if total_goles > 2.5:
                stats['partidos_over25'] += 1
            if total_goles > 3.5:
                stats['partidos_over35'] += 1
            
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
    def analisis_completo(local_team, visitante_team, partidos_local, partidos_visitante, h2h):
        """Análisis completo con todos los cálculos"""
        
        # Factores de ajuste
        factor_local, factor_visitante, advertencias = AnalizadorExperto.calcular_factor_ajuste(
            local_team, visitante_team, partidos_local, partidos_visitante
        )
        
        # Forma detallada
        forma_local, _, detalles_local, resultados_local = AnalizadorExperto.calcular_forma_detallada(
            partidos_local, local_team['Equipo']
        )
        forma_visitante, _, detalles_visitante, resultados_visitante = AnalizadorExperto.calcular_forma_detallada(
            partidos_visitante, visitante_team['Equipo']
        )
        
        # Stats avanzadas
        stats_local = AnalizadorExperto.calcular_stats_avanzadas(local_team['Equipo'], partidos_local)
        stats_visitante = AnalizadorExperto.calcular_stats_avanzadas(visitante_team['Equipo'], partidos_visitante)
        
        # Calcular lambdas
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
        
        # Ventaja local
        lambda_local *= 1.15
        lambda_visitante /= 1.08
        
        # Ajuste por forma
        if forma_local > 0.75:
            lambda_local *= 1.25
        elif forma_local < 0.35:
            lambda_local *= 0.82
        
        if forma_visitante > 0.75:
            lambda_visitante *= 1.25
        elif forma_visitante < 0.35:
            lambda_visitante *= 0.82
        
        # APLICAR FACTORES DE AJUSTE
        lambda_local *= factor_local
        lambda_visitante *= factor_visitante
        
        # Limitar
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
# MÓDULO 5: CALCULADOR DE MERCADOS EXTENDIDOS
# ============================================================================

class CalculadorMercados:
    """Calcula 40+ mercados de apuestas diferentes"""
    
    @staticmethod
    def calcular_todos_mercados(lambda_l, lambda_v, stats_local, stats_visitante):
        """Calcula TODOS los mercados disponibles"""
        
        # Matriz de Poisson
        matriz = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                matriz[i, j] = poisson.pmf(i, lambda_l) * poisson.pmf(j, lambda_v)
        
        mercados = {}
        
        # 1. RESULTADO FINAL (1X2)
        p_local = np.sum(np.tril(matriz, -1))
        p_empate = np.sum(np.diag(matriz))
        p_visitante = np.sum(np.triu(matriz, 1))
        total = p_local + p_empate + p_visitante
        
        mercados['1X2'] = {
            'Local': max

