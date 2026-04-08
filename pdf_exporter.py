"""
Módulo para exportar análisis de supervivencia a PDF
Soporta Kaplan-Meier, Cox Regression, Log-Rank Test y Weibull
"""

import io
import json
import re
from pathlib import Path
import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import plotly.graph_objects as go
import plotly.io as pio
from translations import get_translation


BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "logo_uco.png"
LOGO_RIGHT_PATH = BASE_DIR / "assets" / "logo_espc.png"


class SurvivalAnalysisPDFExporter:
    """Clase para generar informes PDF de análisis de supervivencia"""
    
    def __init__(self, filename, language='es', has_graph=False):
        self.filename = filename
        self.language = language
        self.has_graph = has_graph  # Para saber si necesitamos landscape
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        self.elements = []
        
    def _setup_custom_styles(self):
        """Configura estilos personalizados para el PDF con Times New Roman"""
        # Título personalizado
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=6,
            spaceBefore=2,
            alignment=1,  # Center
            fontName='Times-Bold',
            leading=28
        ))
        
        # Subtítulo
        self.styles.add(ParagraphStyle(
            name='Subtitle',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.HexColor('#555555'),
            spaceAfter=20,
            spaceBefore=10,
            alignment=1,  # Center
            fontName='Times-Roman',
            leading=16
        ))
        
        # Título de sección - Centrado y Negrita
        self.styles.add(ParagraphStyle(
            name='SectionTitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=8,
            spaceBefore=12,
            fontName='Times-Bold',
            bold=True,
            alignment=1,  # Center
            leading=16
        ))
        
        # Texto normal con sangrías y espaciado
        self.styles.add(ParagraphStyle(
            name='NormalText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            spaceBefore=6,
            fontName='Times-Roman',
            alignment=4,  # Justified
            leading=16,  # Espaciado entre líneas
            firstLineIndent=18,  # Sangría de 0.25 pulgadas
            leftIndent=18,
            rightIndent=18
        ))
        
        # Texto de interpretación/conclusión con párrafos bien formateados
        self.styles.add(ParagraphStyle(
            name='InterpretationText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            spaceBefore=0,
            fontName='Times-Roman',
            alignment=4,  # Justified
            leading=17,  # Espaciado generoso entre líneas
            firstLineIndent=18,  # Sangría de primera línea
            leftIndent=18,  # Margen izquierdo
            rightIndent=18  # Margen derecho
        ))

        self.styles.add(ParagraphStyle(
            name='InterpretationHeadingText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=4,
            spaceBefore=6,
            fontName='Times-Bold',
            alignment=0,
            leading=14,
            leftIndent=18,
            rightIndent=18
        ))

        self.styles.add(ParagraphStyle(
            name='InterpretationMainHeadingText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=3,
            spaceBefore=7,
            fontName='Times-Bold',
            alignment=0,
            leading=14,
            leftIndent=18,
            rightIndent=18
        ))

        self.styles.add(ParagraphStyle(
            name='InterpretationUnderlineText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=2,
            spaceBefore=2,
            fontName='Times-Roman',
            alignment=0,
            leading=14,
            leftIndent=18,
            rightIndent=18
        ))
        
        # Texto de descripción/introducción de análisis
        self.styles.add(ParagraphStyle(
            name='DescriptionText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            spaceBefore=6,
            fontName='Times-Roman',
            alignment=4,  # Justified
            leading=16,
            firstLineIndent=18,
            leftIndent=18,
            rightIndent=18,
            textColor=colors.HexColor('#34495e')
        ))

    def _is_english(self):
        return self.language == 'en'
    
    def _add_header(self, title, subtitle=""):
        """Agrega portada/header al documento"""
        self.elements.append(Spacer(1, 0.5*inch))
        
        title_para = Paragraph(title, self.styles['CustomTitle'])
        self.elements.append(title_para)
        
        if subtitle:
            subtitle_para = Paragraph(subtitle, self.styles['NormalText'])
            self.elements.append(subtitle_para)
        
        self.elements.append(Spacer(1, 0.3*inch))

    def _add_paragraph_block(self, text, style_name='InterpretationText'):
        """Agrega un bloque de texto respetando párrafos y saltos internos."""
        if not text:
            return

        paragraphs = [part.strip() for part in re.split(r'\n\s*\n+', str(text).strip()) if part.strip()]
        underline_targets = (
            'follow up medio',
            'follow up mediano',
            'nivel de eventos',
            'seguimiento',
            'variable de edad'
        )

        def _clean_text(value):
            return re.sub(r'[^a-z0-9 ]+', '', value.lower()).strip()

        def _looks_like_subheading(value):
            cleaned = _clean_text(value)
            if not cleaned:
                return False
            if cleaned.startswith(underline_targets):
                return True
            if re.match(r'^(?:[-•*]|\d+(?:\.\d+)*)\s+', value.strip()):
                return True
            words = cleaned.split()
            return (
                len(words) <= 5 and
                len(cleaned) <= 45 and
                not re.search(r'[.;!?]$', value.strip())
            )

        for index, paragraph_text in enumerate(paragraphs):
            normalized_text = ' '.join(line.strip() for line in paragraph_text.splitlines() if line.strip())
            numbered_heading_match = re.match(r'^(\d+(?:\.\d+)*)\s+([^:]+?):\s*(.*)$', normalized_text)
            if numbered_heading_match:
                number_part = numbered_heading_match.group(1)
                title_part = numbered_heading_match.group(2).strip()
                remainder_part = numbered_heading_match.group(3).strip()
                rendered_parts = [f"<b>{number_part}. {title_part}</b>:"]
                if remainder_part:
                    rendered_parts.append(remainder_part)
                self.elements.append(Paragraph(' '.join(rendered_parts), self.styles['InterpretationText']))
            elif _looks_like_subheading(normalized_text):
                self.elements.append(Paragraph(f"<u>{normalized_text}</u>", self.styles['InterpretationUnderlineText']))
                if index < len(paragraphs) - 1:
                    self.elements.append(Spacer(1, 0.02 * inch))
            else:
                self.elements.append(Paragraph(normalized_text, self.styles[style_name]))
                if index < len(paragraphs) - 1:
                    self.elements.append(Spacer(1, 0.04 * inch))
    
    def _add_section_title(self, title):
        """Agrega título de sección"""
        self.elements.append(Paragraph(title, self.styles['SectionTitle']))
        underline = Table([[""]], colWidths=[7.6 * inch])
        underline.setStyle(TableStyle([
            ('LINEBELOW', (0, 0), (-1, -1), 1, colors.HexColor('#1f77b4')),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ]))
        self.elements.append(underline)
        self.elements.append(Spacer(1, 0.12*inch))
    
    def add_summary_section(self, n_patients, n_events, follow_up_mean, follow_up_median):
        """Agrega sección de resumen general con estilo profesional y sin solapamiento"""
        self._add_section_title("1. SUMMARY" if self._is_english() else "1. RESUMEN GENERAL")
        
        summary_data = [
            ["Metric" if self._is_english() else "Métrica", "Value" if self._is_english() else "Valor"],
            ["Number of patients" if self._is_english() else "Número de pacientes", str(n_patients)],
            ["Number of events" if self._is_english() else "Número de eventos", str(n_events)],
            ["Event rate (%)" if self._is_english() else "Tasa de eventos (%)", f"{(n_events/n_patients*100):.1f}%" if n_patients > 0 else "N/A"],
            ["Average follow-up (months)" if self._is_english() else "Follow-up medio (meses)", f"{follow_up_mean:.2f}" if follow_up_mean > 0 else "N/A"],
            ["Median follow-up (months)" if self._is_english() else "Follow-up mediano (meses)", f"{follow_up_median:.2f}" if follow_up_median > 0 else "N/A"]
        ]
        
        table = Table(summary_data, colWidths=[4.5*inch, 2.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f0f0')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 10),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f8f8')])
        ]))

        self.elements.append(table)
        self.elements.append(Spacer(1, 0.3*inch))
    
    def add_kaplan_meier_section(self, variable_name="", survival_table_data=None):
        """Agrega sección de Kaplan-Meier - 2. TABLA DE RESULTADOS"""
        subtitle = f" - {variable_name}" if variable_name else " - General"
        self._add_section_title(("2. RESULTS TABLE" if self._is_english() else "2. TABLA DE RESULTADOS") + subtitle)
        
        # Descripción
        intro_text = (
            "Kaplan-Meier analysis is a non-parametric statistical method that estimates the survival function from time-to-event data. This section presents descriptive survival statistics."
            if self._is_english() else
            "El análisis de Kaplan-Meier es un método estadístico no paramétrico que estima la función de supervivencia a partir de datos de tiempo hasta evento. Esta sección presenta las estadísticas descriptivas de supervivencia."
        )
        self.elements.append(Paragraph(intro_text, self.styles['DescriptionText']))
        self.elements.append(Spacer(1, 0.15*inch))
        
        # Tabla de supervivencia si existe
        if survival_table_data is not None and isinstance(survival_table_data, pd.DataFrame) and len(survival_table_data) > 0:
            self.elements.append(Spacer(1, 0.1*inch))
            
            # Convertir DataFrame a lista de listas para la tabla
            table_data = [list(survival_table_data.columns)]
            for row in survival_table_data.values:
                table_data.append([str(val) for val in row])
            
            # Calcular ancho de columnas dinámicamente
            num_cols = len(survival_table_data.columns)
            total_width = 8.2*inch
            col_widths = [total_width / num_cols] * num_cols
            
            table = Table(table_data, colWidths=col_widths)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')])
            ]))

            self.elements.append(table)
            self.elements.append(Spacer(1, 0.2*inch))
            print(f"  ✓ Tabla KM agregada: {len(table_data)} rows")
        else:
            self.elements.append(Paragraph(
                ("Survival Table<br/>" if self._is_english() else "Tabla de Supervivencia<br/>")
                + ("Survival parameters are computed from the data analysis." if self._is_english() else "Los parámetros de supervivencia se calculan a partir del análisis de los datos."),
                self.styles['NormalText']
            ))
        
        self.elements.append(Spacer(1, 0.3*inch))
    
    def add_cox_regression_section(self, coefficients_table=None):
        """Agrega sección de Cox Regression - 2. TABLA DE RESULTADOS"""
        self._add_section_title("2. RESULTS TABLE" if self._is_english() else "2. TABLA DE RESULTADOS")
        
        # Descripción
        intro_text = (
            "Cox Regression is a semiparametric model used to assess the effect of multiple covariates on time-to-event. Model coefficients represent the change in log-hazard per unit change in the covariate. exp(coefficient) is the Hazard Ratio (HR), which interprets relative risk."
            if self._is_english() else
            "La Regresión de Cox es un modelo semiparamétrico que permite evaluar el efecto de múltiples covariables en el tiempo hasta evento. Los coeficientes del modelo representan el cambio en el log-hazard por unidad de cambio en la covariable. El exp(coeficiente) es el Hazard Ratio (HR), que interpreta el riesgo relativo."
        )
        self.elements.append(Paragraph(intro_text, self.styles['DescriptionText']))
        self.elements.append(Spacer(1, 0.15*inch))
        
        # Tabla de coeficientes
        if coefficients_table is not None and isinstance(coefficients_table, pd.DataFrame) and len(coefficients_table) > 0:
            self.elements.append(Spacer(1, 0.1*inch))
            
            # Convertir DataFrame a lista de listas para la tabla
            table_data = [list(coefficients_table.columns)]
            for row in coefficients_table.values:
                # Redondear números para que ocupen menor espacio
                row_str = []
                for val in row:
                    if isinstance(val, float):
                        row_str.append(f"{val:.4f}")
                    else:
                        row_str.append(str(val))
                table_data.append(row_str)
            
            # Calcular ancho de columnas de forma inteligente
            num_cols = len(coefficients_table.columns)
            # Ancho total disponible
            total_width = 8.2*inch
            # Distribuir proporcionalmente - primera columna más ancha para nombres
            col_widths = [2.5*inch] + [(total_width - 2.5*inch) / (num_cols - 1)] * (num_cols - 1) if num_cols > 1 else [total_width]
            
            table = Table(table_data, colWidths=col_widths)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f9f9f9')),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')])
            ]))

            self.elements.append(table)
            self.elements.append(Spacer(1, 0.2*inch))
            print(f"  ✓ Tabla Cox agregada: {len(table_data)} rows")
        else:
            self.elements.append(Paragraph(
                ("Cox Model Coefficients<br/>" if self._is_english() else "Coeficientes del Modelo Cox<br/>")
                + ("Coefficients are estimated by partial maximum likelihood." if self._is_english() else "Los coeficientes se estiman mediante máxima verosimilitud parcial."),
                self.styles['NormalText']
            ))
        
        self.elements.append(Spacer(1, 0.3*inch))
    
    def add_log_rank_section(self, test_results=None):
        """Agrega sección de Log-Rank Test - 2. TABLA DE RESULTADOS"""
        self._add_section_title("2. RESULTS TABLE" if self._is_english() else "2. TABLA DE RESULTADOS")
        
        # Descripción
        intro_text = (
            "The Log-Rank Test is a non-parametric statistical test that compares survival curves between two or more groups. The null hypothesis states that there is no difference in survival functions between groups. A p-value < 0.05 indicates significant differences."
            if self._is_english() else
            "El Test de Log-Rank es una prueba estadística no paramétrica que compara las curvas de supervivencia entre dos o más grupos. La hipótesis nula es que no hay diferencia en las funciones de supervivencia entre grupos. Un p-valor < 0.05 indica diferencias significativas."
        )
        self.elements.append(Paragraph(intro_text, self.styles['DescriptionText']))
        self.elements.append(Spacer(1, 0.15*inch))
        
        if test_results is not None:
            # Si es dict
            if isinstance(test_results, dict):
                test_data = [
                    ["Statistic" if self._is_english() else "Estadístico", "Value" if self._is_english() else "Valor"],
                    ["Chi-square" if self._is_english() else "Chi-cuadrado", f"{test_results.get('test_statistic', 'N/A'):.4f}"],
                    ["p-value" if self._is_english() else "p-valor", f"{test_results.get('p_value', 'N/A'):.4f}"],
                    ["Degrees of freedom" if self._is_english() else "Grados de libertad", str(test_results.get('df', 'N/A'))]
                ]
                
                table = Table(test_data, colWidths=[4.4*inch, 3.8*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
                    ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
                    ('FONTSIZE', (0, 0), (-1, 0), 11),
                    ('FONTSIZE', (0, 1), (-1, -1), 10),
                    ('TOPPADDING', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                    ('LEFTPADDING', (0, 0), (-1, -1), 8),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f8f0')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f8f0')])
                ]))

                self.elements.append(table)
                self.elements.append(Spacer(1, 0.2*inch))
                print(f"  ✓ Tabla Log-Rank agregada (dict)")
                
                # Interpretación
                p_val = test_results.get('p_value', 1)
                if p_val < 0.05:
                    interpretation = "✓ There is a SIGNIFICANT difference between groups (p < 0.05). The null hypothesis is rejected." if self._is_english() else "✓ Hay diferencia SIGNIFICATIVA entre grupos (p < 0.05). Se rechaza la hipótesis nula."
                    color = '#27ae60'
                else:
                    interpretation = "✗ There is NO significant difference between groups (p ≥ 0.05). The null hypothesis is not rejected." if self._is_english() else "✗ NO hay diferencia significativa entre grupos (p ≥ 0.05). No se rechaza la hipótesis nula."
                    color = '#e67e22'
                
                interp_label = "Interpretation" if self._is_english() else "Interpretación"
                interp_para = Paragraph(
                    f"<font color='{color}'><b>{interp_label}: {interpretation}</b></font>",
                    self.styles['NormalText']
                )
                self.elements.append(interp_para)
            
            # Si es DataFrame
            elif isinstance(test_results, pd.DataFrame) and len(test_results) > 0:
                # Convertir DataFrame a lista de listas para la tabla
                table_data = [list(test_results.columns)]
                for row in test_results.values:
                    # Redondear números para ocupar menos espacio
                    row_str = []
                    for val in row:
                        if isinstance(val, float):
                            row_str.append(f"{val:.4f}")
                        else:
                            row_str.append(str(val)[:15])  # Limitar a 15 caracteres
                    table_data.append(row_str)
                
                # Calcular ancho de columnas con mejor distribución
                num_cols = len(test_results.columns)
                total_width = 8.2*inch
                # Primera columna más ancha para nombres de variables
                col_widths = [2.5*inch] + [(total_width - 2.5*inch) / (num_cols - 1)] * (num_cols - 1) if num_cols > 1 else [total_width]
                
                # Crear tabla
                table = Table(table_data, colWidths=col_widths, repeatRows=1)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
                    ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
                    ('FONTSIZE', (0, 0), (-1, 0), 8),
                    ('FONTSIZE', (0, 1), (-1, -1), 7),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('LEFTPADDING', (0, 0), (-1, -1), 4),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f8f0')),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f8f0')])
                ]))

                self.elements.append(table)
                self.elements.append(Spacer(1, 0.2*inch))
                print(f"  ✓ Tabla Log-Rank agregada: {len(table_data)} rows x {num_cols} cols")
        else:
            self.elements.append(Paragraph(
                ("Log-Rank Test Results<br/>" if self._is_english() else "Resultados del Test de Log-Rank<br/>")
                + ("The Log-Rank Test results are shown in the table above." if self._is_english() else "Los resultados del Test de Log-Rank se presentan en la tabla anterior."),
                self.styles['NormalText']
            ))
        
        self.elements.append(Spacer(1, 0.3*inch))

    def add_weibull_section(self, weibull_table=None):
        """Agrega sección de Weibull - 2. TABLA DE RESULTADOS"""
        self._add_section_title("2. RESULTS TABLE" if self._is_english() else "2. TABLA DE RESULTADOS")

        intro_text = (
            "The Weibull model is a parametric survival model that estimates how the risk changes over time using a shape and a scale parameter. The table below summarizes the fitted model and its main statistics."
            if self._is_english() else
            "El modelo Weibull es un modelo paramétrico de supervivencia que estima cómo cambia el riesgo con el tiempo usando un parámetro de forma y otro de escala. La tabla siguiente resume el ajuste y sus estadísticas principales."
        )
        self.elements.append(Paragraph(intro_text, self.styles['DescriptionText']))
        self.elements.append(Spacer(1, 0.15*inch))

        if weibull_table is not None and isinstance(weibull_table, pd.DataFrame) and len(weibull_table) > 0:
            self.elements.append(Spacer(1, 0.1*inch))

            table_data = [list(weibull_table.columns)]
            for row in weibull_table.values:
                row_str = []
                for val in row:
                    if isinstance(val, float):
                        row_str.append(f"{val:.4f}")
                    else:
                        row_str.append(str(val))
                table_data.append(row_str)

            num_cols = len(weibull_table.columns)
            total_width = 8.2 * inch
            col_widths = [2.5 * inch] + [(total_width - 2.5 * inch) / (num_cols - 1)] * (num_cols - 1) if num_cols > 1 else [total_width]

            table = Table(table_data, colWidths=col_widths)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fdf3f2')),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fdf3f2')])
            ]))

            self.elements.append(table)
            self.elements.append(Spacer(1, 0.2 * inch))
            print(f"  ✓ Tabla Weibull agregada: {len(table_data)} rows")
        else:
            self.elements.append(Paragraph(
                ("Weibull Fit Summary<br/>" if self._is_english() else "Resumen del Ajuste Weibull<br/>")
                + ("The Weibull fit is summarized in the table above." if self._is_english() else "El ajuste Weibull se resume en la tabla anterior."),
                self.styles['NormalText']
            ))

        self.elements.append(Spacer(1, 0.3 * inch))
    
    def add_conclusions_section(self, conclusions_text=None):
        """Agrega sección de conclusiones - 4. CONCLUSIÓN"""
        self._add_section_title("4. CONCLUSION" if self._is_english() else "4. CONCLUSIÓN")
        
        if conclusions_text:
            self._add_paragraph_block(conclusions_text, 'InterpretationText')
        else:
            default_conclusion = (
                "This report presents a complete survival analysis of the provided data. The results are based on robust statistical methods. Specialist review is recommended for clinical interpretation."
                if self._is_english() else
                "Este informe presenta un análisis completo de supervivencia de los datos proporcionados. Los resultados se basan en métodos estadísticos robustos. Se recomienda la consulta con especialistas para la interpretación clínica de los resultados."
            )
            self._add_paragraph_block(default_conclusion, 'InterpretationText')
        
        self.elements.append(Spacer(1, 0.3*inch))
    
    def add_graph_section(self, title="3. GRÁFICA"):
        """Inicia sección de gráfica"""
        self.elements.append(Spacer(1, 0.2*inch))
        self._add_section_title(title)
        self.elements.append(Spacer(1, 0.2*inch))
    
    def _plotly_to_image(self, fig):
        """Convierte figura Plotly a imagen (BytesIO) con alta resolución"""
        try:
            img_bytes = pio.to_image(fig, format='png', width=1400, height=700, scale=2)
            return io.BytesIO(img_bytes)
        except Exception as e:
            print(f"Error convertiendo gráfica: {str(e)}")
            return None
    
    def add_plotly_figure(self, fig, caption=""):
        """Agrega una figura Plotly al PDF - GRANDE en LANDSCAPE"""
        if fig is None:
            return
        
        try:
            # Si fig es string JSON, convertir a objeto Plotly
            if isinstance(fig, str):
                try:
                    fig_dict = json.loads(fig)
                    fig = go.Figure(fig_dict)
                except:
                    print(f"  ✗ No se pudo convertir JSON a figura Plotly")
                    return
            
            # Convertir con dimensiones ALTAS para mejor resolución
            # 1400x700 (proporción 2:1) para buena resolución
            img_bytes = pio.to_image(fig, format='png', width=1400, height=700, scale=2)
            img_io = io.BytesIO(img_bytes)
            
            # Tamaño en PDF según orientación
            if self.has_graph:
                # LANDSCAPE: mucho más espacio disponible
                # Con márgenes de 0.3" a cada lado, ancho disponible es ~9.4"
                # Usamos 9.0" x 4.5" para máxima visibilidad
                img = Image(img_io, width=9.0*inch, height=4.5*inch)
            else:
                # PORTRAIT: espacio normal
                img = Image(img_io, width=8.0*inch, height=4.0*inch)
            
            # Agregar con poco espaciado
            self.elements.append(Spacer(1, 0.1*inch))
            self.elements.append(img)
            self.elements.append(Spacer(1, 0.15*inch))
            
            if caption:
                caption_para = Paragraph(f"<i><b>{caption}</b></i>", self.styles['NormalText'])
                self.elements.append(caption_para)
                self.elements.append(Spacer(1, 0.1*inch))
            
            print(f"  ✓ Figura Plotly agregada al PDF en LANDSCAPE (9.0\" x 4.5\")")
        except Exception as e:
            print(f"  ✗ Error al agregar figura Plotly: {str(e)}")
    
    def generate(self, title="INFORME DE ANÁLISIS DE SUPERVIVENCIA"):
        """Genera el PDF final en landscape si hay gráficas"""
        # Seleccionar orientación basada en si hay gráficas
        if self.has_graph:
            pagesize = landscape(letter)
            # Márgenes para landscape
            right_margin = 0.3*inch
            left_margin = 0.3*inch
            top_margin = 0.5*inch
            bottom_margin = 0.5*inch
        else:
            pagesize = letter
            # Márgenes para portrait
            right_margin = 0.35*inch
            left_margin = 0.35*inch
            top_margin = 0.6*inch
            bottom_margin = 0.6*inch
        
        doc = SimpleDocTemplate(
            self.filename,
            pagesize=pagesize,
            rightMargin=right_margin,
            leftMargin=left_margin,
            topMargin=top_margin,
            bottomMargin=bottom_margin
        )
        
        # Construir PDF con los elementos que ya están en self.elements
        def _draw_page_decorations(canvas_obj, doc_obj):
            canvas_obj.saveState()
            page_width, page_height = doc_obj.pagesize

            # Línea superior
            canvas_obj.setStrokeColor(colors.HexColor('#d9e5f3'))
            canvas_obj.setLineWidth(0.75)
            canvas_obj.line(doc_obj.leftMargin, page_height - 0.66 * inch, page_width - doc_obj.rightMargin, page_height - 0.66 * inch)

            # Logos
            logo_width = 0.48 * inch
            logo_height = 0.48 * inch
            if LOGO_PATH.exists():
                try:
                    canvas_obj.drawImage(
                        ImageReader(str(LOGO_PATH)),
                        doc_obj.leftMargin,
                        page_height - 0.58 * inch,
                        width=logo_width,
                        height=logo_height,
                        preserveAspectRatio=True,
                        mask='auto'
                    )
                except Exception:
                    pass

            if LOGO_RIGHT_PATH.exists():
                try:
                    canvas_obj.drawImage(
                        ImageReader(str(LOGO_RIGHT_PATH)),
                        page_width - doc_obj.rightMargin - logo_width,
                        page_height - 0.58 * inch,
                        width=logo_width,
                        height=logo_height,
                        preserveAspectRatio=True,
                        mask='auto'
                    )
                except Exception:
                    pass

            # Título
            canvas_obj.setFont('Times-Bold', 10.5)
            canvas_obj.setFillColor(colors.HexColor('#1f77b4'))
            canvas_obj.drawCentredString(page_width / 2, page_height - 0.34 * inch, title)

            # Pie
            canvas_obj.setStrokeColor(colors.HexColor('#d9e5f3'))
            canvas_obj.setLineWidth(0.8)
            canvas_obj.line(doc_obj.leftMargin, 0.55 * inch, page_width - doc_obj.rightMargin, 0.55 * inch)
            canvas_obj.setFont('Times-Roman', 9)
            canvas_obj.setFillColor(colors.HexColor('#666666'))
            footer_label = "Generated" if self._is_english() else "Generado"
            page_label = "Page" if self._is_english() else "Página"
            canvas_obj.drawString(doc_obj.leftMargin, 0.34 * inch, f"{footer_label}: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
            canvas_obj.drawRightString(page_width - doc_obj.rightMargin, 0.34 * inch, f"{page_label} {canvas_obj.getPageNumber()}")

            canvas_obj.restoreState()

        doc.build(self.elements, onFirstPage=_draw_page_decorations, onLaterPages=_draw_page_decorations)
        print(f"✓ PDF generado exitosamente: {self.filename}")


# Función utilitaria para exporte rápido
def export_survival_analysis_to_pdf(
    filename,
    title="INFORME DE ANÁLISIS DE SUPERVIVENCIA",
    include_summary=False,
    include_km=False,
    km_figure=None,
    km_table=None,
    include_cox=False,
    cox_table=None,
    forest_figure=None,
    include_logrank=False,
    logrank_figure=None,
    logrank_results=None,
    include_weibull=False,
    weibull_figure=None,
    weibull_table=None,
    include_ai_interpretation=False,
    ai_text="",
    summary_stats=None,
    language='es',
    landscape_tables=None,
    include_graph=None
):
    """
    Función helper para exportar análisis a PDF de forma simple.
    Solo incluye lo que el usuario marcó (include_summary, include_km, include_cox, include_ai_interpretation).
    
    Args:
        include_summary: Si mostrar la sección de resumen general
        include_km: Si mostrar análisis Kaplan-Meier
        include_cox: Si mostrar análisis Cox Regression
        include_ai_interpretation: Si mostrar interpretación de IA
        landscape_tables: Lista de nombres de tablas que deben estar en horizontal ['logrank', 'cox', etc]
        include_graph: Si FORCE landscape (None=auto-detect, True/False=explicit)
    """
    # Detectar si hay gráficas para usar landscape
    if include_graph is None:
        # Auto-detect: landscape si hay ALGUNA figura
        has_graph = (include_km and km_figure is not None) or \
                    (include_cox and forest_figure is not None) or \
                    (include_logrank and logrank_figure is not None) or \
                    (include_weibull and weibull_figure is not None)
    else:
        # Usar valor explícito (True = forzar landscape, False = forzar portrait)
        has_graph = include_graph
    
    exporter = SurvivalAnalysisPDFExporter(filename, language, has_graph=has_graph)
    if landscape_tables is None:
        landscape_tables = []
    
    # ENCABEZADO SIMPLE (solo título y fecha) - MINIMAL SPACING
    exporter.elements.append(Spacer(1, 0.12*inch))
    title_para = Paragraph(title, exporter.styles['CustomTitle'])
    exporter.elements.append(title_para)
    exporter.elements.append(Spacer(1, 0.08*inch))
    
    print(f"[PDF BUILDER] Elementos iniciales (header): {len(exporter.elements)}")
    
    # ESTRUCTURA CONDICIONAL: Solo agrega lo que fue marcado
    numero_seccion = 1
    primera_seccion = True

    def _start_new_section():
        nonlocal primera_seccion
        if primera_seccion:
            primera_seccion = False
            return
        exporter.elements.append(PageBreak())
    
    # 1. RESUMEN GENERAL
    if summary_stats:
        _start_new_section()
        exporter.add_summary_section(
            summary_stats.get('n_patients', 0),
            summary_stats.get('n_events', 0),
            summary_stats.get('follow_up_mean', 0),
            summary_stats.get('follow_up_median', 0)
        )
        numero_seccion += 1
    
    # 2. TABLA DE RESULTADOS (omitido para Kaplan-Meier)
    tabla_agregada = False
    # La tabla de KM se omite por completo (no aparece en opciones del usuario)
    if include_cox:
        _start_new_section()
        exporter.add_cox_regression_section(cox_table)
        tabla_agregada = True
    elif include_logrank:
        _start_new_section()
        exporter.add_log_rank_section(logrank_results)
        tabla_agregada = True
    elif include_weibull:
        _start_new_section()
        exporter.add_weibull_section(weibull_table)
        tabla_agregada = True
    
    if tabla_agregada:
        numero_seccion += 1
    
    # 3. GRÁFICA
    grafica_agregada = False
    if include_km and km_figure is not None:
        _start_new_section()
        exporter.add_graph_section(f"{numero_seccion}. GRAPH" if exporter._is_english() else f"{numero_seccion}. GRÁFICA")
        exporter.add_plotly_figure(km_figure, "Kaplan-Meier survival curve" if language == 'en' else "Curva de supervivencia Kaplan-Meier")
        grafica_agregada = True
    elif include_cox and forest_figure is not None:
        _start_new_section()
        exporter.add_graph_section(f"{numero_seccion}. GRAPH" if exporter._is_english() else f"{numero_seccion}. GRÁFICA")
        exporter.add_plotly_figure(forest_figure, "Hazard Ratio forest plot" if language == 'en' else "Forest plot de Hazard Ratios")
        grafica_agregada = True
    elif include_logrank and logrank_figure is not None:
        _start_new_section()
        exporter.add_graph_section(f"{numero_seccion}. GRAPH" if exporter._is_english() else f"{numero_seccion}. GRÁFICA")
        exporter.add_plotly_figure(logrank_figure, "Survival curves by groups (Log-Rank Test)" if language == 'en' else "Curvas de supervivencia por grupos (Log-Rank Test)")
        grafica_agregada = True
    elif include_weibull and weibull_figure is not None:
        _start_new_section()
        exporter.add_graph_section(f"{numero_seccion}. GRAPH" if exporter._is_english() else f"{numero_seccion}. GRÁFICA")
        exporter.add_plotly_figure(weibull_figure, "Weibull fitted curve" if language == 'en' else "Curva ajustada Weibull")
        grafica_agregada = True
    
    if grafica_agregada:
        numero_seccion += 1
    
    # 4. CONCLUSIÓN - SOLO si fue marcada en las opciones
    if include_ai_interpretation and ai_text:
        _start_new_section()
        exporter.elements.append(Spacer(1, 0.14 * inch))
        exporter._add_section_title(f"{numero_seccion}. CONCLUSION" if exporter._is_english() else f"{numero_seccion}. CONCLUSIÓN")
        exporter._add_paragraph_block(ai_text, 'InterpretationText')
    
    print(f"[PDF BUILDER] Elementos antes de generate(): {len(exporter.elements)}")
    
    # Generar PDF
    exporter.generate(title)
