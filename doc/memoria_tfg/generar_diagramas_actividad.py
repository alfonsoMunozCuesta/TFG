from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Imagenes"

W, H = 1400, 1900
BG = "white"
BLUE = (221, 235, 247)
BLUE_BORDER = (47, 84, 150)
GREEN = (226, 239, 218)
GREEN_BORDER = (84, 130, 53)
YELLOW = (255, 242, 204)
YELLOW_BORDER = (191, 144, 0)
RED = (244, 204, 204)
RED_BORDER = (153, 0, 0)
TEXT = (20, 20, 20)


def font(size=30, bold=False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size)


FONT = font(30)
FONT_BOLD = font(34, True)
FONT_SMALL = font(25)


def text_size(draw, text, fnt):
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def draw_wrapped(draw, box, text, fnt=FONT, fill=TEXT, align="center"):
    x1, y1, x2, y2 = box
    max_chars = max(16, int((x2 - x1) / (fnt.size * 0.48)))
    lines = []
    for part in text.split("\n"):
        lines.extend(wrap(part, max_chars) or [""])
    line_h = fnt.size + 7
    total_h = len(lines) * line_h
    y = y1 + ((y2 - y1) - total_h) / 2
    for line in lines:
        tw, _ = text_size(draw, line, fnt)
        if align == "center":
            x = x1 + ((x2 - x1) - tw) / 2
        else:
            x = x1 + 18
        draw.text((x, y), line, font=fnt, fill=fill)
        y += line_h


def arrow(draw, start, end):
    draw.line([start, end], fill=BLUE_BORDER, width=4)
    x1, y1 = start
    x2, y2 = end
    if abs(y2 - y1) >= abs(x2 - x1):
        sign = 1 if y2 > y1 else -1
        pts = [(x2, y2), (x2 - 12, y2 - 22 * sign), (x2 + 12, y2 - 22 * sign)]
    else:
        sign = 1 if x2 > x1 else -1
        pts = [(x2, y2), (x2 - 22 * sign, y2 - 12), (x2 - 22 * sign, y2 + 12)]
    draw.polygon(pts, fill=BLUE_BORDER)


def rounded(draw, cx, y, w, h, text, fill=BLUE, outline=BLUE_BORDER, fnt=FONT):
    box = (cx - w // 2, y, cx + w // 2, y + h)
    draw.rounded_rectangle(box, radius=22, fill=fill, outline=outline, width=4)
    draw_wrapped(draw, box, text, fnt=fnt)
    return box


def diamond(draw, cx, y, w, h, text):
    pts = [(cx, y), (cx + w // 2, y + h // 2), (cx, y + h), (cx - w // 2, y + h // 2)]
    draw.polygon(pts, fill=YELLOW, outline=YELLOW_BORDER)
    draw.line(pts + [pts[0]], fill=YELLOW_BORDER, width=4)
    draw_wrapped(draw, (cx - w // 2 + 42, y + 18, cx + w // 2 - 42, y + h - 18), text, fnt=FONT_SMALL)
    return (cx - w // 2, y, cx + w // 2, y + h)


def label(draw, xy, text):
    draw.text(xy, text, font=FONT_SMALL, fill=TEXT)


def diagram(filename, title, steps, decision=None, alternatives=None):
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    cx = W // 2

    draw.text((70, 45), title, font=FONT_BOLD, fill=BLUE_BORDER)
    draw.line((70, 95, W - 70, 95), fill=BLUE_BORDER, width=3)

    y = 140
    draw.ellipse((cx - 35, y, cx + 35, y + 70), fill=GREEN_BORDER, outline=GREEN_BORDER)
    prev = (cx, y + 70)
    y += 115

    for step in steps:
        box = rounded(draw, cx, y, 720, 115, step)
        arrow(draw, prev, (cx, y))
        prev = (cx, y + 115)
        y += 165

    if decision:
        dbox = diamond(draw, cx, y, 620, 150, decision)
        arrow(draw, prev, (cx, y))
        prev = (cx, y + 150)
        label(draw, (cx + 330, y + 48), "Sí")
        y += 205

    if alternatives:
        alt_x = 1030
        alt_y = 285
        for alt in alternatives:
            rounded(draw, alt_x, alt_y, 520, 105, alt, fill=RED, outline=RED_BORDER, fnt=FONT_SMALL)
            alt_y += 150
        draw.text((alt_x - 250, 220), "Flujos alternativos", font=FONT_BOLD, fill=RED_BORDER)
        draw.line((alt_x - 255, 260, alt_x + 255, 260), fill=RED_BORDER, width=3)
        label(draw, (cx - 355, 620), "No / error")
        arrow(draw, (cx - 260, 615), (alt_x - 260, 340))

    end_box = rounded(draw, cx, y, 720, 115, "Resultados disponibles / estado actualizado", fill=GREEN, outline=GREEN_BORDER)
    arrow(draw, prev, (cx, y))
    prev = (cx, y + 115)
    y += 165
    draw.ellipse((cx - 38, y, cx + 38, y + 76), fill="white", outline=GREEN_BORDER, width=5)
    draw.ellipse((cx - 24, y + 14, cx + 24, y + 62), fill=GREEN_BORDER, outline=GREEN_BORDER)
    arrow(draw, prev, (cx, y))

    img.save(OUT / filename)


diagram(
    "act_cu1.png",
    "CU-1: Ejecutar modelo exponencial",
    [
        "Usuario accede a Análisis de Supervivencia y selecciona Análisis Exponencial",
        "Sistema carga la vista del modelo exponencial",
        "Sistema verifica que existe un dataset cargado y preprocesado",
        "Sistema calcula automáticamente el modelo",
        "Sistema muestra tabla resumen, gráfica y comparativa de ajuste",
        "Usuario elige Explicar, Exportar a PDF o Informe combinado",
    ],
    "¿Datos suficientes y compatibles?",
    ["FA1: no se puede ejecutar el modelo; se muestra mensaje de error"],
)

diagram(
    "act_cu2.png",
    "CU-2: Ejecutar modelo Weibull",
    [
        "Usuario accede a Análisis de Supervivencia y selecciona Análisis Weibull",
        "Sistema carga la vista del modelo de Weibull",
        "Sistema verifica que existe un dataset cargado y preprocesado",
        "Sistema calcula automáticamente el modelo",
        "Sistema muestra tabla resumen, gráfica y comparativa de ajuste",
        "Usuario elige Explicar, Exportar a PDF o Informe combinado",
    ],
    "¿Datos y parámetros válidos?",
    ["FA1: error en datos o parámetros; se informa al usuario"],
)

diagram(
    "act_cu3.png",
    "CU-3: Ejecutar Random Survival Forest",
    [
        "Usuario selecciona Random Survival Forest",
        "Sistema ejecuta el análisis con el dataset en sesión",
        "Sistema muestra tabla resumen, curvas por riesgo e importancia de variables",
        "Sistema muestra panel de simulación de perfil con valores por defecto",
        "Usuario modifica valores y pulsa Simular perfil si lo desea",
        "Sistema recalcula curva, métricas e interpretación del perfil",
        "Usuario puede solicitar explicación o exportar a PDF",
    ],
    "¿Entrenamiento correcto?",
    ["FA1: no se completa el entrenamiento o ejecución; se muestra error"],
)

diagram(
    "act_cu4.png",
    "CU-4: Consultar comparación de técnicas",
    [
        "Usuario accede a la página Comparación de técnicas",
        "Sistema valida que exista un dataset disponible en sesión",
        "Sistema carga tabla comparativa de técnicas",
        "Sistema genera gráfico comparativo de capacidades",
        "Sistema genera bloque explicativo mediante IA",
        "Usuario consulta la información como apoyo al análisis",
    ],
    "¿Página, gráfico e IA disponibles?",
    [
        "FA1: no carga la página; se muestra error",
        "FA2: no se genera gráfico; se mantiene tabla y explicación",
        "FA3: no se genera explicación; se muestra aviso",
    ],
)

diagram(
    "act_cu5.png",
    "CU-5: Exportar informes y resultados",
    [
        "Usuario pulsa Exportar a PDF en el módulo correspondiente",
        "Sistema abre el modal de exportación",
        "Usuario introduce nombre opcional y selecciona contenido",
        "Usuario confirma con Descargar PDF",
        "Sistema valida que existan datos o resultados suficientes",
        "Sistema genera PDF con resumen, tabla, gráfica e interpretación",
        "Sistema entrega el archivo generado",
    ],
    "¿Puede generarse el archivo?",
    ["FA1: no se puede generar el archivo; se muestra error"],
)

diagram(
    "act_cu6.png",
    "CU-6: Cambiar idioma de la interfaz",
    [
        "Usuario accede al selector de idioma de la barra de navegación",
        "Sistema muestra los idiomas disponibles",
        "Usuario selecciona el idioma deseado",
        "Sistema procesa la selección",
        "Sistema recarga los textos visibles de la interfaz",
    ],
    "¿Idioma aplicado correctamente?",
    ["FA1: no se puede aplicar el idioma; se mantiene el anterior y se avisa"],
)

diagram(
    "act_cu7.png",
    "CU-7: Analizar variables no binarias",
    [
        "Usuario carga un CSV y pulsa Preprocesar CSV",
        "Sistema valida la estructura del dataset",
        "Sistema conserva variables no binarias soportadas por el pipeline",
        "Usuario selecciona una variable no binaria disponible",
        "Sistema transforma la variable según la técnica aplicada",
        "Sistema ejecuta el análisis correspondiente",
        "Sistema muestra tablas, métricas y gráficas actualizadas",
    ],
    "¿Variable compatible?",
    ["FA1: variable no compatible; se informa al usuario"],
)

diagram(
    "act_cu8.png",
    "CU-8: Mejorar interpretación con IA",
    [
        "Usuario solicita una interpretación de resultados",
        "Sistema recopila la información relevante del análisis",
        "Sistema envía la información al módulo de IA",
        "La IA procesa los datos y genera una explicación",
        "Sistema recibe la respuesta",
        "Sistema presenta la explicación en la interfaz",
    ],
    "¿Respuesta adecuada?",
    ["FA1: la IA no genera una respuesta adecuada; se muestra error"],
)

diagram(
    "act_cu9.png",
    "CU-9: Ampliar visualizaciones Cox y Log-Rank",
    [
        "Usuario accede a Regresión de Cox o Test de Log-Rank",
        "Sistema muestra las opciones de visualización disponibles",
        "Usuario selecciona la visualización deseada",
        "Sistema obtiene resultados de la configuración activa",
        "Sistema procesa la información",
        "Sistema genera y muestra la representación gráfica",
    ],
    "¿Existen resultados previos?",
    ["FA1: no existen resultados previos; se indica que debe ejecutar el análisis"],
)
