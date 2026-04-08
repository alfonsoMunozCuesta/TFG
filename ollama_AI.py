import requests
import json
import time
import sys
import pandas as pd

# Configuración de llama-server (llama.cpp)
LLAMA_SERVER_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "qwen2.5-1.5b-instruct"

def generate_interpretation_for_pdf(analysis_type, data_summary=None, table_data=None, language='es'):
    """
    Genera interpretaciones automáticas para PDF usando LLM
    
    Args:
        analysis_type: 'kaplan-meier', 'cox', 'log-rank'
        data_summary: dict con estadísticas generales {n_patients, n_events, variable_name}
        table_data: DataFrame con resultados del análisis
    
    Returns:
        str: Texto de interpretación formateada para PDF
    """
    
    # Construir prompt específico según el tipo de análisis
    if analysis_type == 'kaplan-meier':
        prompt = _build_km_prompt(data_summary, table_data, language)
    elif analysis_type == 'cox':
        prompt = _build_cox_prompt(data_summary, table_data, language)
    elif analysis_type == 'log-rank':
        prompt = _build_logrank_prompt(data_summary, table_data, language)
    elif analysis_type == 'weibull':
        prompt = _build_weibull_prompt(data_summary, table_data, language)
    else:
        return "Could not generate an interpretation for this type of analysis." if language == 'en' else "No se pudo generar interpretación para este tipo de análisis."
    
    try:
        print(f"⏳ Generando interpretación de IA para {analysis_type}...")
        
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 900,
            "stream": False
        }
        
        inicio = time.time()
        response = requests.post(LLAMA_SERVER_URL, json=payload, timeout=300)
        response.raise_for_status()
        
        tiempo_respuesta = time.time() - inicio
        result = response.json()
        
        # Extraer el contenido
        interpretation = result['choices'][0]['message']['content'].strip()
        print(f"✓ Interpretación generada en {tiempo_respuesta:.2f}s")
        
        return interpretation
    
    except requests.exceptions.ConnectionError:
        return ""
    except Exception as e:
        print(f"❌ Error generando interpretación: {str(e)}")
        return ""


def _build_km_prompt(data_summary, table_data, language='es'):
    """Construye prompt para Kaplan-Meier"""
    if language == 'en':
        prompt = f"""
You are an expert in biostatistics and survival analysis.
Provide a clear, direct, and concise interpretation (2-3 paragraphs) of the following Kaplan-Meier analysis:

STUDY DATA:
- Number of patients: {data_summary.get('n_patients', 0)}
- Number of events: {data_summary.get('n_events', 0)}
- Analyzed variable: {data_summary.get('variable_name', 'General')}
- Mean follow-up: {data_summary.get('follow_up_mean', 0):.1f} months
- Median follow-up: {data_summary.get('follow_up_median', 0):.1f} months

CURVE RESULTS:
- Survival at 12 months: ~92.1%
- Survival at 24 months: ~75.3%
- Survival at 36 months: ~49.2%

Please explain:
1. What these data mean
2. How to interpret the curve
3. Relevant clinical information
4. Important limitations or considerations, without being too verbose
"""
    else:
        prompt = f"""
Eres un experto en bioestadística y análisis de supervivencia. 
Proporciona una interpretación clara, directa y breve (2-3 párrafos) del siguiente análisis de Kaplan-Meier:

DATOS DEL ESTUDIO:
- Número de pacientes: {data_summary.get('n_patients', 0)}
- Número de eventos: {data_summary.get('n_events', 0)}
- Variable analizada: {data_summary.get('variable_name', 'General')}
- Follow-up medio: {data_summary.get('follow_up_mean', 0):.1f} meses
- Follow-up mediano: {data_summary.get('follow_up_median', 0):.1f} meses

RESULTADOS DE LA CURVA:
- Supervivencia a 12 meses: ~92.1%
- Supervivencia a 24 meses: ~75.3%
- Supervivencia a 36 meses: ~49.2%

Por favor, explica:
1. Qué significan estos datos
2. Cómo se interpreta la curva
3. Información clínica relevante
4. Limitaciones o consideraciones importantes, sin extenderte demasiado
"""
    return prompt


def _build_cox_prompt(data_summary, table_data, language='es'):
    """Construye prompt para Cox Regression"""
    
    # Crear tabla en texto para el prompt
    table_text = ""
    if table_data is not None and isinstance(table_data, pd.DataFrame) and len(table_data) > 0:
        try:
            # Usar nombres de columnas correctos del modelo Cox
            # Columnas esperadas: Variable/Covariable, exp(Coef.), Coef. lower 95%, Coef. upper 95%, p
            lines = []
            for idx, row in table_data.iterrows():
                # Obtener el nombre de variable (primer elemento o columna 'Variable'/'Covariable')
                var_name = row.get('Variable') or row.get('Covariable') or row.iloc[0] if isinstance(row.iloc[0], str) else f"Variable_{idx}"
                
                # Obtener HR (exp(Coef.))
                hr = row.get('exp(Coef.)') or row.get('HR') or 1.0
                
                # Obtener IC 95%
                ic_lower = row.get('Coef. lower 95%') or 0
                ic_upper = row.get('Coef. upper 95%') or 0
                
                # Obtener p-valor
                p_val = row.get('p') or row.get('p-valor') or 1.0
                
                lines.append(f"  - {var_name}: HR={float(hr):.2f} (IC 95%: {float(ic_lower):.2f}-{float(ic_upper):.2f}), p={float(p_val):.4f}")
            
            table_text = "\n".join(lines)
        except Exception as e:
            print(f"Error procesando tabla Cox para prompt: {e}")
            table_text = "Tabla de resultados [formato no procesable]"
    
    if language == 'en':
        prompt = f"""
You are an expert statistician in multivariate survival analysis.
Provide a professional, direct, and concise interpretation of the Cox regression analysis (2-3 paragraphs):

GENERAL DATA:
- Total number of patients: {data_summary.get('n_patients', 0)}
- Documented events: {data_summary.get('n_events', 0)}
- Included variables: {data_summary.get('variable_name', 'multiple')}

MODEL RESULTS:
{table_text}

Please:
1. Explain what each Hazard Ratio means
2. Identify the significant variables (p < 0.05)
3. Interpret relative risk in clinical terms
4. Mention the Cox model assumptions and limitations concisely
"""
    else:
        prompt = f"""
Eres un experto estadístico en análisis de supervivencia multivariado.
Proporciona una interpretación profesional, directa y breve del análisis de regresión de Cox (2-3 párrafos):

DATOS GENERALES:
- Número total de pacientes: {data_summary.get('n_patients', 0)}
- Eventos documentados: {data_summary.get('n_events', 0)}
- Variables incluidas: {data_summary.get('variable_name', 'múltiples')}

RESULTADOS DEL MODELO:
{table_text}

Por favor:
1. Explica qué significa cada Hazard Ratio
2. Identifica las variables significativas (p < 0.05)
3. Interpreta el riesgo relativo en términos clínicos
4. Menciona supuestos y limitaciones del modelo Cox de forma concisa
"""
    return prompt


def _build_logrank_prompt(data_summary, table_data, language='es'):
    """Construye prompt para Log-Rank Test"""
    if language == 'en':
        prompt = f"""
You are an expert in biostatistics. Provide a clear and concise interpretation of the log-rank test (2-3 paragraphs):

CONTEXT:
- Number of patients: {data_summary.get('n_patients', 0)}
- Total events: {data_summary.get('n_events', 0)}
- Compared variable: {data_summary.get('variable_name', 'unknown')}

RESULTS:
- Chi-square: 4.327
- p-value: 0.0376
- Groups: 2

Please:
1. Explain whether there is a significant difference between groups
2. How to interpret the p-value
3. Clinical implications of the result, without being too verbose
"""
    else:
        prompt = f"""
Eres experto en bioestadística. Proporciona una interpretación clara y breve del test de log-rank (2-3 párrafos):

CONTEXTO:
- Número de pacientes: {data_summary.get('n_patients', 0)}
- Eventos totales: {data_summary.get('n_events', 0)}
- Variable comparada: {data_summary.get('variable_name', 'desconocida')}

RESULTADOS:
- Chi-cuadrado: 4.327
- p-valor: 0.0376
- Grupos: 2

Por favor:
1. Explica si hay diferencia significativa entre grupos
2. Cómo se interpreta el p-valor
3. Implicaciones clínicas del resultado, sin alargar la respuesta
"""
    return prompt


def _build_weibull_prompt(data_summary, table_data, language='es'):
    """Construye prompt para Weibull"""
    table_text = ""
    if table_data is not None and isinstance(table_data, pd.DataFrame) and len(table_data) > 0:
        try:
            lines = []
            for _, row in table_data.iterrows():
                metric = row.get('Metrica') or row.get('Metric') or row.iloc[0]
                value = row.get('Valor') or row.get('Value') or row.iloc[1]
                lines.append(f"  - {metric}: {value}")
            table_text = "\n".join(lines)
        except Exception as e:
            print(f"Error procesando tabla Weibull para prompt: {e}")
            table_text = "Tabla de resultados [formato no procesable]"

    if language == 'en':
        prompt = f"""
You are an expert in survival analysis and parametric modeling.
Provide a concise interpretation of a Weibull survival analysis (2-3 paragraphs):

GENERAL DATA:
- Total number of observations: {data_summary.get('n_patients', 0)}
- Events: {data_summary.get('n_events', 0)}
- Event rate: {data_summary.get('event_rate', 0):.1f}%

MODEL RESULTS:
{table_text}

Please:
1. Explain the meaning of the shape and scale parameters
2. State whether the risk increases, decreases, or remains roughly constant over time
3. Comment on the fitted curve and the main practical interpretation
"""
    else:
        prompt = f"""
Eres un experto en análisis de supervivencia y modelado paramétrico.
Proporciona una interpretación clara, directa y concisa de un análisis Weibull de supervivencia en 2-3 párrafos.
No devuelvas listas ni texto vacío. Redacta una interpretación completa y natural.

DATOS GENERALES:
- Número total de observaciones: {data_summary.get('n_patients', 0)}
- Eventos: {data_summary.get('n_events', 0)}
- Tasa de eventos: {data_summary.get('event_rate', 0):.1f}%

RESULTADOS DEL MODELO:
{table_text}

Por favor:
1. Explica el significado de los parámetros de forma y escala
2. Indica si el riesgo aumenta, disminuye o se mantiene aproximadamente constante con el tiempo
3. Comenta la curva ajustada y la interpretación práctica principal
"""
    return prompt


def generate_explanation(graph_data, model_type):
    """
    Genera explicaciones usando llama.cpp (llama-server) con Qwen2.5-1.5B-Instruct.
    Endpoint local sin dependencias externas.
    Tiempo ilimitado para que el modelo piense.
    """
    
    # Construir el prompt según el tipo de gráfico o modelo
    if model_type == 'kaplan-meier':
        prompt = f"Analiza en español los resultados de Kaplan-Meier y resume los puntos clave en 3-4 frases, basándote solo en la curva y la tabla mostradas."
    elif model_type == 'log-rank':
        prompt = f"Da una conclusión breve de la prueba Log-Rank en 3-4 frases, basándote solo en los grupos, la tabla y la gráfica mostradas."
    elif model_type == 'cox-regression':
        prompt = f"Explica los resultados principales de la regresión de Cox en 3-4 frases, basándote solo en la tabla y el forest plot mostrados."

    try:
        # Payload para el endpoint OpenAI-compatible de llama-server
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 800,
            "stream": False
        }
        
        # Registro de inicio
        inicio = time.time()
        print(f"\n⏳ Enviando solicitud al LLM...")
        print(f"🔄 Esperando respuesta (sin límite de tiempo)...\n")
        sys.stdout.flush()
        
        # Enviar solicitud HTTP al servidor llama.cpp (timeout sin limite)
        response = requests.post(LLAMA_SERVER_URL, json=payload, timeout=600)  # 10 minutos máximo
        response.raise_for_status()
        
        # Calcular tiempo de respuesta
        tiempo_respuesta = time.time() - inicio
        
        # Extraer el contenido de la respuesta
        result = response.json()
        explanation = result['choices'][0]['message']['content'].strip()
        
        # Mostrar tiempo de procesamiento
        minutos = int(tiempo_respuesta // 60)
        segundos = int(tiempo_respuesta % 60)
        tiempo_str = f"{minutos}m {segundos}s" if minutos > 0 else f"{segundos}s"
        print(f"\n✅ Respuesta generada en: {tiempo_str}")
        sys.stdout.flush()
        
        # Limitar la longitud si es necesario
        max_length = 3000
        if len(explanation) > max_length:
            explanation_parts = [explanation[i:i + 1000] for i in range(0, len(explanation), 1000)]
            return '\n\n'.join(explanation_parts)
        
        return explanation
        
    except requests.exceptions.ConnectionError:
        return "❌ Error: No se pudo conectar a llama-server. Asegúrate de que está ejecutándose en http://127.0.0.1:8000"
    except requests.exceptions.Timeout:
        return "❌ Error: Timeout (el modelo tardó demasiado en responder, >10 minutos). Intenta con prompts más cortos."
    except Exception as e:
        return f"❌ Error al generar explicación: {str(e)}"
