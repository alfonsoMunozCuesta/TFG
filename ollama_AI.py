import requests
import json
import time
import sys
import pandas as pd
from config import LLAMA_SERVER_URL, MODEL_NAME

DEFAULT_TEMPERATURE = 0.15
PDF_MAX_TOKENS = 480
EXPLAIN_MAX_TOKENS = 260
REQUEST_TIMEOUT_S = 240


def _looks_like_list_output(text):
    if not text:
        return False
    stripped = text.strip()
    if stripped.startswith('-') or stripped.startswith('*'):
        return True
    return any(marker in stripped for marker in ('1)', '2)', '3)', '1.', '2.', '3.'))


def _academic_system_prompt(language='es'):
    if language == 'en':
        return (
            "You are a survival-analysis assistant for an academic thesis. "
            "Write in formal, clear, evidence-based prose using only provided results. "
            "Return 2-3 short paragraphs with no bullet points and no numbered lists."
        )
    return (
        "Eres un asistente de análisis de supervivencia para un TFG. "
        "Redacta en prosa académica clara y basada en evidencias, usando solo resultados proporcionados. "
        "Devuelve 2-3 párrafos breves, sin viñetas ni listas numeradas."
    )


def _rewrite_to_prose_if_needed(content, max_tokens, language='es', timeout=REQUEST_TIMEOUT_S):
    if not _looks_like_list_output(content):
        return content

    rewrite_payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": _academic_system_prompt(language)},
            {
                "role": "user",
                "content": (
                    "Reescribe este contenido en prosa académica manteniendo exactamente los hechos, "
                    "sin viñetas ni numeración:\n\n" + content
                ),
            },
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "stream": False,
    }
    response = requests.post(LLAMA_SERVER_URL, json=rewrite_payload, timeout=timeout)
    response.raise_for_status()
    result = response.json()
    rewritten = result['choices'][0]['message']['content'].strip()
    return rewritten or content


def _call_llm(prompt, max_tokens, temperature=DEFAULT_TEMPERATURE, timeout=REQUEST_TIMEOUT_S, language='es'):
    """Invoca llama-server con parámetros homogéneos y salida académica consistente."""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": _academic_system_prompt(language)},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    response = requests.post(LLAMA_SERVER_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    result = response.json()
    content = result['choices'][0]['message']['content'].strip()
    return _rewrite_to_prose_if_needed(content, max_tokens=max_tokens, language=language, timeout=timeout)

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
    elif analysis_type == 'exponential':
        prompt = _build_exponential_prompt(data_summary, table_data, language)
    else:
        return "Could not generate an interpretation for this type of analysis." if language == 'en' else "No se pudo generar interpretación para este tipo de análisis."
    
    try:
        print(f"⏳ Generando interpretación de IA para {analysis_type}...")

        inicio = time.time()
        interpretation = _call_llm(prompt, max_tokens=PDF_MAX_TOKENS, language=language)
        tiempo_respuesta = time.time() - inicio
        print(f"✓ Interpretación generada en {tiempo_respuesta:.2f}s")
        
        return interpretation
    
    except requests.exceptions.ConnectionError:
        return ""
    except Exception as e:
        print(f"❌ Error generando interpretación: {str(e)}")
        return ""


def _build_km_prompt(data_summary, table_data, language='es'):
    """Construye prompt para Kaplan-Meier"""
    km_lines = []
    if table_data is not None and isinstance(table_data, pd.DataFrame) and len(table_data) > 0:
        try:
            sample = table_data.head(8)
            col_t = sample.columns[0]
            col_s = sample.columns[1] if len(sample.columns) > 1 else None
            for _, row in sample.iterrows():
                t_val = row.get(col_t, '')
                s_val = row.get(col_s, '') if col_s else ''
                if col_s:
                    km_lines.append(f"- t={t_val}, S(t)={s_val}")
                else:
                    km_lines.append(f"- {t_val}")
        except Exception:
            km_lines.append("- Tabla KM disponible pero no parseable")

    km_text = "\n".join(km_lines) if km_lines else "- No hay tabla KM disponible"

    if language == 'en':
        prompt = f"""
You are an expert in biostatistics and survival analysis.
Provide a concise interpretation (max 140 words) of this Kaplan-Meier analysis.
Use only the information provided and do not invent values.

STUDY DATA:
- Number of patients: {data_summary.get('n_patients', 0)}
- Number of events: {data_summary.get('n_events', 0)}
- Analyzed variable: {data_summary.get('variable_name', 'General')}
- Mean follow-up: {data_summary.get('follow_up_mean', 0):.1f} months
- Median follow-up: {data_summary.get('follow_up_median', 0):.1f} months

KM TABLE (sample):
{km_text}

Return exactly 3 short paragraphs:
1) curve behavior and risk trend
2) practical meaning
3) one limitation
"""
    else:
        prompt = f"""
Eres experto en bioestadística y supervivencia.
Da una interpretación concisa (máximo 140 palabras) del análisis Kaplan-Meier.
Usa solo los datos disponibles y no inventes valores.

DATOS DEL ESTUDIO:
- Número de pacientes: {data_summary.get('n_patients', 0)}
- Número de eventos: {data_summary.get('n_events', 0)}
- Variable analizada: {data_summary.get('variable_name', 'General')}
- Follow-up medio: {data_summary.get('follow_up_mean', 0):.1f} meses
- Follow-up mediano: {data_summary.get('follow_up_median', 0):.1f} meses

TABLA KM (muestra):
{km_text}

Devuelve exactamente 3 párrafos cortos:
1) comportamiento de la curva y tendencia de riesgo
2) interpretación práctica
3) una limitación
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
    logrank_text = "- No hay tabla de Log-Rank disponible"
    if table_data is not None and isinstance(table_data, pd.DataFrame) and len(table_data) > 0:
        try:
            rows = []
            for _, row in table_data.head(8).iterrows():
                cov = row.get('Covariable', row.get('Variable', 'N/A'))
                chi2 = row.get('test_statistic', row.get('chi2', 'N/A'))
                pval = row.get('p_value', row.get('p', 'N/A'))
                rows.append(f"- {cov}: chi2={chi2}, p={pval}")
            logrank_text = "\n".join(rows)
        except Exception:
            logrank_text = "- Tabla de Log-Rank disponible pero no parseable"

    if language == 'en':
        prompt = f"""
You are an expert in biostatistics.
Provide a concise interpretation of this log-rank test (max 130 words).
Do not invent any values.

CONTEXT:
- Number of patients: {data_summary.get('n_patients', 0)}
- Total events: {data_summary.get('n_events', 0)}
- Compared variable: {data_summary.get('variable_name', 'unknown')}

RESULTS:
{logrank_text}

Return exactly 2 paragraphs:
1) significance and p-value interpretation
2) practical implication and one caveat
"""
    else:
        prompt = f"""
Eres experto en bioestadística.
Da una interpretación breve del test Log-Rank (máximo 130 palabras).
No inventes datos.

CONTEXTO:
- Número de pacientes: {data_summary.get('n_patients', 0)}
- Eventos totales: {data_summary.get('n_events', 0)}
- Variable comparada: {data_summary.get('variable_name', 'desconocida')}

RESULTADOS:
{logrank_text}

Devuelve exactamente 2 párrafos:
1) significación e interpretación del p-valor
2) implicación práctica y una limitación
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


def _build_exponential_prompt(data_summary, table_data, language='es'):
    """Construye prompt para Exponential Survival."""
    table_text = ""
    if table_data is not None and isinstance(table_data, pd.DataFrame) and len(table_data) > 0:
        try:
            lines = []
            for _, row in table_data.iterrows():
                metric = row.get('Metrica') or row.get('Metric') or row.iloc[0]
                value = row.get('Valor') or row.get('Value') or row.iloc[1]
                lines.append(f"- {metric}: {value}")
            table_text = "\n".join(lines)
        except Exception:
            table_text = "- Tabla de resultados no parseable"
    else:
        table_text = "- No hay tabla de resultados disponible"

    if language == 'en':
        return f"""
You are an expert in survival analysis.
Interpret the Exponential model shown in the dashboard using only the provided table/curve information.

DATA:
- Observations: {data_summary.get('n_patients', 0)}
- Events: {data_summary.get('n_events', 0)}
- Event rate: {data_summary.get('event_rate', 0):.1f}%

TABLE SUMMARY:
{table_text}

Return exactly 3 short paragraphs:
1) what lambda implies (constant hazard over time)
2) how the exponential fit compares conceptually to empirical KM curve
3) one practical implication and one limitation
"""

    return f"""
Eres experto en análisis de supervivencia.
Interpreta el modelo Exponencial mostrado en el dashboard usando solo la información de tabla y curva.

DATOS:
- Observaciones: {data_summary.get('n_patients', 0)}
- Eventos: {data_summary.get('n_events', 0)}
- Tasa de eventos: {data_summary.get('event_rate', 0):.1f}%

TABLA RESUMEN:
{table_text}

Devuelve exactamente 3 párrafos cortos:
1) qué implica lambda (riesgo constante en el tiempo)
2) cómo encaja el modelo exponencial frente a la curva KM empírica
3) una implicación práctica y una limitación
"""


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
        # Registro de inicio
        inicio = time.time()
        print(f"\n⏳ Enviando solicitud al LLM...")
        print(f"🔄 Esperando respuesta (sin límite de tiempo)...\n")
        sys.stdout.flush()
        explanation = _call_llm(prompt, max_tokens=EXPLAIN_MAX_TOKENS, timeout=REQUEST_TIMEOUT_S, language='es')
        
        # Calcular tiempo de respuesta
        tiempo_respuesta = time.time() - inicio
        
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
