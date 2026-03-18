import requests
import json
import time
import sys

# Configuración de llama-server (llama.cpp)
LLAMA_SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL_NAME = "qwen2.5-1.5b-instruct"

def generate_explanation(graph_data, model_type):
    """
    Genera explicaciones usando llama.cpp (llama-server) con Qwen2.5-1.5B-Instruct.
    Endpoint local sin dependencias externas.
    Tiempo ilimitado para que el modelo piense.
    """
    
    # Construir el prompt según el tipo de gráfico o modelo
    if model_type == 'kaplan-meier':
        prompt = f"Por favor, explica en español esta gráfica de Kaplan-Meier: {kaplan_img}."
    elif model_type == 'log-rank':
        prompt = f"Me puedes dar una conclusión breve de los resultados obtenidos en Log Rank: {logrank_table}."
    elif model_type == 'cox-regression':
        prompt = f"Explica los siguientes resultados de la regresión de Cox: {cox_table}."

    try:
        # Payload para el endpoint OpenAI-compatible de llama-server
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 3000,
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
