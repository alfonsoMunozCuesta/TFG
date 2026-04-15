"""
Callbacks para exportación de análisis a PDF
Genera PDFs profesionales con datos reales de los análisis
"""

from dash import callback_context, dcc, html, Output, Input, State
from dash.exceptions import PreventUpdate
import json
import os
from datetime import datetime
from pdf_exporter import export_survival_analysis_to_pdf, export_weibull_exponential_combined_pdf
from ollama_AI import generate_interpretation_for_pdf
from translations import get_translation
import pandas as pd
import traceback
import re
from weibull import build_weibull_analysis
from exponential import build_exponential_analysis
from rsf import build_rsf_analysis, build_rsf_profile_analysis


def clean_markdown_text(text):
    """Limpia markdown del texto para renderizarlo correctamente en ReportLab"""
    if not text:
        return text
    
    # Remover headers markdown (### ..., ## ..., # ...)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Convertir markdown bold (**text**) a HTML <b></b>
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    
    # Convertir markdown italic (*text*) a HTML <i></i> (excepto ** ya procesado)
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    
    # Limpiar múltiples espacios/saltos de línea
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text


def _validate_ai_explanation(ai_requested, ai_text, analysis_label, language='es'):
    """Exige explicación solo si el usuario pidió IA y no hay texto real."""
    if not ai_requested:
        return None

    normalized_text = (ai_text or "").strip()
    if not normalized_text or normalized_text == "La respuesta es...":
        if language == 'en':
            return f"⚠️ Before exporting the {analysis_label} PDF, generate an AI explanation first."
        return f"⚠️ Antes de exportar el PDF de {analysis_label}, genera primero una explicación de IA."

    return None


def _validate_ai_language(ai_requested, generated_language, current_language):
    """Valida que la explicación IA se haya generado en el idioma activo."""
    if not ai_requested:
        return None

    if not generated_language:
        return None

    if generated_language != current_language:
        if current_language == 'en':
            return "⚠️ The AI explanation was generated in another language. Please click Explain again before exporting."
        return "⚠️ La explicación IA se generó en otro idioma. Pulsa de nuevo Explicar antes de exportar."

    return None


def _get_report_name_from_filename(filename):
    """Obtiene un nombre de informe limpio a partir del nombre de archivo PDF."""
    base_name = os.path.splitext(os.path.basename(filename or ""))[0].strip()
    return base_name or "informe"


def register_pdf_export_callbacks(app):
    """
    Registra todos los callbacks para la funcionalidad de exportación a PDF
    Incluye modales para Kaplan-Meier, Cox Regression, Log-Rank Test, Weibull y RSF
    """
    
    # ===== KAPLAN-MEIER PDF CALLBACKS =====
    
    @app.callback(
        [Output('km-pdf-modal-overlay', 'style'),
         Output('km-pdf-modal-container', 'style')],
        [Input('export-km-btn', 'n_clicks'),
         Input('km-pdf-modal-close-btn', 'n_clicks'),
         Input('km-pdf-modal-cancel-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def toggle_km_pdf_modal(export_clicks, close_clicks, cancel_clicks):
        """Abre/cierra el modal de exportación Kaplan-Meier"""
        if not callback_context.triggered:
            return {'display': 'none'}, {'display': 'none'}
        
        triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
        
        if triggered_id == 'export-km-btn' and export_clicks:
            return _get_modal_styles(True)
        
        return {'display': 'none'}, {'display': 'none'}
    
    
    @app.callback(
        [Output('km-pdf-modal-download', 'data'),
         Output('km-pdf-modal-error', 'children'),
         Output('km-pdf-modal-error', 'style')],
        Input('km-pdf-modal-download-btn', 'n_clicks'),
        [State('km-pdf-modal-filename', 'value'),
         State('km-pdf-modal-checklist-content', 'value'),
         State('km-current-variable', 'data'),
         State('openai-answer-kaplan', 'value'),
         State('df-store', 'data'),
         State('language-store', 'data')],
        prevent_initial_call=True
    )
    def download_km_pdf(n_clicks, filename, options, current_variable, ai_text_from_page, df_json, language):
        """Genera PDF de Kaplan-Meier con datos REALES del análisis seleccionado"""
        try:
            # Generar nombre de archivo
            if not filename or filename.strip() == '':
                var_name = current_variable if current_variable else 'general'
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"km_{var_name}_{timestamp}.pdf"
            else:
                if not filename.endswith('.pdf'):
                    filename += '.pdf'
            
            os.makedirs("downloads", exist_ok=True)
            pdf_path = f"downloads/{filename}"
            report_name = _get_report_name_from_filename(filename)
            
            options = options or ['summary', 'graph']
            ai_error = _validate_ai_explanation('ai_interpretation' in options, ai_text_from_page, 'Kaplan-Meier', language)
            if ai_error:
                return None, ai_error, {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }
            
            # CAPTURAR DATOS REALES DEL ANÁLISIS
            df = None
            if df_json:
                try:
                    df = pd.read_json(df_json, orient='split')
                except:
                    df = None
            
            # CALCULAR ESTADÍSTICAS REALES DEL ANÁLISIS ACTUAL
            n_patients = len(df) if df is not None else 0
            n_events = int(df['final_result'].sum()) if df is not None and 'final_result' in df.columns else 0
            follow_up_mean = float(df['date'].mean()) if df is not None and 'date' in df.columns else 0
            follow_up_median = float(df['date'].median()) if df is not None and 'date' in df.columns else 0
            
            summary_stats = {
                'n_patients': n_patients,
                'n_events': n_events,
                'follow_up_mean': follow_up_mean,
                'follow_up_median': follow_up_median,
                'variable_name': current_variable if current_variable else 'General'
            }
            
            # GENERAR TABLA DE SUPERVIVENCIA REAL
            km_table = None
            if df is not None and n_patients > 0:
                try:
                    from lifelines import KaplanMeierFitter
                    kmf = KaplanMeierFitter()
                    kmf.fit(df['date'], event_observed=df['final_result'])
                    km_table = kmf.survival_function_.reset_index()
                    km_table.columns = ['Tiempo', 'Supervivencia']
                except Exception as e:
                    print(f"[KM PDF] Error generando tabla KM: {e}")
                    km_table = None
            
            # GENERAR GRÁFICA DE KM SI SE SOLICITA
            km_figure = None
            if 'graph' in options and df is not None and current_variable:
                try:
                    print(f"[KM PDF] Regenerando gráfica de KM para {current_variable}...")
                    from kaplan_meier import _create_km_figure
                    
                    # Mapear variable a nombre de covariable correcto
                    covariate_map = {
                        'gender': 'gender_F',
                        'disability': 'disability_N',
                        'age': 'age_band',
                        'education': 'highest_education',
                        'credit': 'studied_credits'
                    }
                    
                    # Buscar qué covariable usar
                    covariate_col = current_variable
                    for key, val in covariate_map.items():
                        if key.lower() in current_variable.lower():
                            covariate_col = val
                            break
                    
                    #Crear figura Plotly directamente
                    km_figure = _create_km_figure(df, covariate_col)
                    print(f"[KM PDF] ✓ Gráfica KM regenerada para {covariate_col}")
                    
                except Exception as e:
                    print(f"[KM PDF] Error regenerando gráfica: {e}")
                    traceback.print_exc()
                    km_figure = None
            
            # GENERAR INTERPRETACIÓN DE IA SI SE SOLICITA
            ai_text = ""
            if 'ai_interpretation' in options:
                print(f"[KM PDF] Generando interpretación de IA para {current_variable}...")
                ai_text = generate_interpretation_for_pdf('kaplan-meier', summary_stats, km_table, language=language)
                
                # Limpiar markdown del texto
                ai_text = clean_markdown_text(ai_text)
            
            print(f"[KM PDF] Generando PDF: {filename}")
            print(f"[KM PDF] Variable: {current_variable} | Pacientes: {n_patients} | Eventos: {n_events}")
            print(f"[KM PDF] Opciones marcadas: {options}")
            
            # GENERAR PDF CON DATOS REALES - SOLO LO QUE SE MARCÓ
            report_title = f"KAPLAN-MEIER REPORT: {current_variable.upper() if current_variable else 'GENERAL'}"
            export_survival_analysis_to_pdf(
                filename=pdf_path,
                title=report_title,
                report_name=report_name,
                include_summary='summary' in options,  # Mostrar resumen SOLO si se marcó
                include_km='table' in options or 'graph' in options,  # Mostrar KM si marcó tabla o gráfica
                km_figure=km_figure if 'graph' in options else None,
                km_table=km_table if 'table' in options else None,
                include_cox=False,
                include_logrank=False,
                include_ai_interpretation='ai_interpretation' in options,  # Mostrar IA SOLO si se marcó
                ai_text=ai_text,
                summary_stats=summary_stats if 'summary' in options else None,
                language=language
            )
            
            print(f"[KM PDF] ✓ PDF generado en: {pdf_path}")
            return dcc.send_file(pdf_path), "", {'display': 'none'}
        
        except Exception as e:
            print(f"ERROR en download_km_pdf: {str(e)}")
            print(traceback.format_exc())
            error_label = 'Kaplan-Meier' if language == 'es' else 'Kaplan-Meier'
            return None, (f"❌ Error al generar el PDF de Kaplan-Meier: {str(e)}" if language == 'es' else f"❌ Error generating the Kaplan-Meier PDF: {str(e)}"), {
                'display': 'block',
                'marginTop': '15px',
                'padding': '10px 12px',
                'borderRadius': '6px',
                'backgroundColor': '#fff1f0',
                'color': '#c0392b',
                'border': '1px solid #f5c6cb',
                'fontSize': '14px',
                'fontWeight': 'bold'
            }
    
    
    # ===== COX REGRESSION PDF CALLBACKS =====
    
    @app.callback(
        [Output('cox-pdf-modal-overlay', 'style'),
         Output('cox-pdf-modal-container', 'style')],
        [Input('export-cox-btn', 'n_clicks'),
         Input('cox-pdf-modal-close-btn', 'n_clicks'),
         Input('cox-pdf-modal-cancel-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def toggle_cox_pdf_modal(export_clicks, close_clicks, cancel_clicks):
        """Abre/cierra el modal de exportación Cox Regression"""
        if not callback_context.triggered:
            return {'display': 'none'}, {'display': 'none'}
        
        triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
        
        if triggered_id == 'export-cox-btn' and export_clicks:
            return _get_modal_styles(True)
        
        return {'display': 'none'}, {'display': 'none'}
    
    
    @app.callback(
        [Output('cox-pdf-modal-download', 'data'),
         Output('cox-pdf-modal-error', 'children'),
         Output('cox-pdf-modal-error', 'style')],
        Input('cox-pdf-modal-download-btn', 'n_clicks'),
        [State('cox-pdf-modal-filename', 'value'),
         State('cox-pdf-modal-checklist-content', 'value'),
         State('cox-current-variables', 'data'),
         State('cox-selected-covariables', 'data'),
         State('openai-answer-cox', 'value'),
         State('df-store', 'data'),
         State('language-store', 'data')],
        prevent_initial_call=True
    )
    def download_cox_pdf(n_clicks, filename, options, cox_variables_text, cox_covariables, ai_text_from_page, df_json, language):
        """Genera PDF de Cox Regression con datos REALES del análisis actualizado"""
        try:
            # Generar nombre de archivo
            if not filename or filename.strip() == '':
                vars_str = ','.join(cox_covariables[:2]) if cox_covariables else 'general'
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"cox_{vars_str}_{timestamp}.pdf"
            else:
                if not filename.endswith('.pdf'):
                    filename += '.pdf'
            
            os.makedirs("downloads", exist_ok=True)
            pdf_path = f"downloads/{filename}"
            report_name = _get_report_name_from_filename(filename)
            
            options = options or ['summary', 'graph']
            ai_error = _validate_ai_explanation('ai_interpretation' in options, ai_text_from_page, 'Cox Regression', language)
            if ai_error:
                return None, ai_error, {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }
            
            print(f"\n[COX PDF] ========== INICIANDO EXPORTACIÓN ==========")
            print(f"[COX PDF] Opciones marcadas: {options}")
            print(f"[COX PDF] Variables seleccionadas: {cox_covariables}")
            
            # CAPTURAR DATOS REALES DEL ANÁLISIS ACTUAL
            df = None
            cox_table = None
            
            if df_json and cox_covariables:
                try:
                    df = pd.read_json(df_json, orient='split')
                    print(f"[COX PDF] DataFrame cargado: {len(df)} rows, {len(df.columns)} cols")
                    
                    # RECALCULAR COX REGRESSION CON LAS VARIABLES SELECCIONADAS
                    from cox_regression import run_cox_regression
                    print(f"[COX PDF] Ejecutando run_cox_regression({cox_covariables})...")
                    
                    summary, cox_table_html = run_cox_regression(df, cox_covariables)
                    print(f"[COX PDF] Summary retornado: empty={summary.empty}, shape={summary.shape}")
                    print(f"[COX PDF] Summary columns: {list(summary.columns) if not summary.empty else 'N/A'}")
                    
                    # Convertir el resumen a tabla limpia
                    if not summary.empty:
                        print(f"[COX PDF] Creando tabla de coeficientes...")
                        try:
                            # Verificar si contiene ERROR
                            if 'ERROR' in summary.get('Covariable', summary.get(summary.columns[0], [])).values:
                                print(f"[COX PDF] Error detectado en Cox: {summary}")
                                cox_table = None
                            else:
                                # Usar los nombres de columnas ya formateados que devuelve run_cox_regression
                                # Los nombres son: Covariable, Coef., exp(Coef.), SE(Coef.), Coef. lower 95%, etc.
                                cols_disponibles = [c for c in ['Covariable', 'Coef.', 'exp(Coef.)', 'Coef. lower 95%', 'Coef. upper 95%', 'p'] 
                                                   if c in summary.columns]
                                
                                if not cols_disponibles or len(cols_disponibles) < 2:
                                    print(f"[COX PDF] Columnas disponibles insuficientes: {list(summary.columns)}")
                                    cox_table = None
                                else:
                                    cox_table = summary[cols_disponibles].copy()
                                    if 'Covariable' in cox_table.columns:
                                        cox_table.columns = ['Variable'] + [c for c in summary.columns[1:] if c in cols_disponibles[1:]]
                                    print(f"[COX PDF] ✓ Tabla creada: {cox_table.shape[0]} variables, {len(cols_disponibles)} columnas")
                                    print(f"[COX PDF] Primeras filas:\n{cox_table.head()}")
                        except Exception as e:
                            print(f"[COX PDF] ✗ Error creando tabla: {e}")
                            traceback.print_exc()
                            cox_table = None
                    
                    print(f"[COX PDF] Análisis recalculado para variables: {cox_covariables}")
                
                except Exception as e:
                    print(f"[COX PDF] ✗ Error recalculando Cox: {e}")
                    traceback.print_exc()
                    cox_table = None
            
            # CALCULAR ESTADÍSTICAS DE RESUMEN REALES
            n_patients = len(df) if df is not None else 0
            n_events = int(df['final_result'].sum()) if df is not None and 'final_result' in df.columns else 0
            follow_up_mean = float(df['date'].mean()) if df is not None and 'date' in df.columns else 0
            follow_up_median = float(df['date'].median()) if df is not None and 'date' in df.columns else 0
            
            summary_stats = {
                'n_patients': n_patients,
                'n_events': n_events,
                'follow_up_mean': follow_up_mean,
                'follow_up_median': follow_up_median,
                'variable_name': ', '.join(cox_covariables) if cox_covariables else 'N/A'
            }
            
            # GENERAR INTERPRETACIÓN DE IA SI SE SOLICITA
            ai_text = ""
            if 'ai_interpretation' in options:
                print(f"[COX PDF] Generando interpretación de IA...")
                ai_text = generate_interpretation_for_pdf('cox', summary_stats, cox_table, language=language)
                
                # Limpiar markdown del texto
                ai_text_cleaned = clean_markdown_text(ai_text)
                print(f"[COX PDF] ✓ Interpretación generada ({len(ai_text)} chars → {len(ai_text_cleaned)} chars)")
                ai_text = ai_text_cleaned
            
            # GENERAR FOREST PLOT SI SE SOLICITA
            forest_figure = None
            if 'graph' in options and not summary.empty:
                print(f"[COX PDF] Generando Forest Plot...")
                try:
                    from cox_regression import create_forest_plot
                    forest_figure = create_forest_plot(summary)
                    if forest_figure:
                        print(f"[COX PDF] ✓ Forest Plot generado")
                except Exception as e:
                    print(f"[COX PDF] ✗ Error generando Forest Plot: {e}")
                    forest_figure = None
            
            print(f"[COX PDF] Generando PDF: {filename}")
            print(f"[COX PDF] Variables: {cox_covariables} | Pacientes: {n_patients} | Eventos: {n_events}")
            print(f"[COX PDF] Tabla Cox será incluida: {'table' in options and cox_table is not None}")
            print(f"[COX PDF] Forest Plot será incluido: {'graph' in options and forest_figure is not None}")
            # GENERAR PDF CON DATOS REALES - SOLO LO QUE SE MARCÓ
            report_title = (
                f"COX REGRESSION REPORT: {', '.join(cox_covariables) if cox_covariables else 'GENERAL'}"
            )
            export_survival_analysis_to_pdf(
                filename=pdf_path,
                title=report_title,
                report_name=report_name,
                include_summary='summary' in options,  # Mostrar resumen SOLO si se marcó
                include_km=False,
                include_cox='table' in options or 'graph' in options,  # Mostrar Cox si marcó tabla O gráfica
                cox_table=cox_table if 'table' in options else None,
                forest_figure=forest_figure if 'graph' in options else None,  # Mostrar gráfica SOLO si se marcó
                include_logrank=False,
                include_ai_interpretation='ai_interpretation' in options,  # Mostrar IA SOLO si se marcó
                ai_text=ai_text,
                summary_stats=summary_stats if 'summary' in options else None,
                language=language
            )
            
            print(f"[COX PDF] ✓ PDF generado en: {pdf_path}")
            print(f"[COX PDF] ========== EXPORTACIÓN COMPLETADA ==========\n")
            return dcc.send_file(pdf_path), "", {'display': 'none'}
        
        except Exception as e:
            print(f"ERROR en download_cox_pdf: {str(e)}")
            print(traceback.format_exc())
            return None, (f"❌ Error al generar el PDF de Cox: {str(e)}" if language == 'es' else f"❌ Error generating the Cox PDF: {str(e)}"), {
                'display': 'block',
                'marginTop': '15px',
                'padding': '10px 12px',
                'borderRadius': '6px',
                'backgroundColor': '#fff1f0',
                'color': '#c0392b',
                'border': '1px solid #f5c6cb',
                'fontSize': '14px',
                'fontWeight': 'bold'
            }
    
    
    # ===== LOG-RANK TEST PDF CALLBACKS =====
    
    @app.callback(
        [Output('logrank-pdf-modal-overlay', 'style'),
         Output('logrank-pdf-modal-container', 'style')],
        [Input('export-logrank-btn', 'n_clicks'),
         Input('logrank-pdf-modal-close-btn', 'n_clicks'),
         Input('logrank-pdf-modal-cancel-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def toggle_logrank_pdf_modal(export_clicks, close_clicks, cancel_clicks):
        """Abre/cierra el modal de exportación Log-Rank"""
        if not callback_context.triggered:
            return {'display': 'none'}, {'display': 'none'}
        
        triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
        
        if triggered_id == 'export-logrank-btn' and export_clicks:
            return _get_modal_styles(True)
        
        return {'display': 'none'}, {'display': 'none'}
    
    
    @app.callback(
        [Output('logrank-pdf-modal-download', 'data'),
         Output('logrank-pdf-modal-error', 'children'),
         Output('logrank-pdf-modal-error', 'style')],
        Input('logrank-pdf-modal-download-btn', 'n_clicks'),
        [State('logrank-pdf-modal-filename', 'value'),
         State('logrank-pdf-modal-checklist-content', 'value'),
         State('logrank-current-variable', 'data'),
         State('logrank-selected-covariables', 'data'),
         State('openai-answer-logrank', 'value'),
         State('df-store', 'data'),
         State('language-store', 'data')],
        prevent_initial_call=True
    )
    def download_logrank_pdf(n_clicks, filename, options, logrank_variable, logrank_covariables, ai_text_from_page, df_json, language):
        """Genera PDF de Log-Rank Test con datos REALES del análisis actualizado"""
        try:
            # Generar nombre de archivo
            if not filename or filename.strip() == '':
                var_str = logrank_variable if logrank_variable else 'general'
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"logrank_{var_str}_{timestamp}.pdf"
            else:
                if not filename.endswith('.pdf'):
                    filename += '.pdf'
            
            os.makedirs("downloads", exist_ok=True)
            pdf_path = f"downloads/{filename}"
            report_name = _get_report_name_from_filename(filename)
            
            options = options or ['summary', 'table']
            ai_error = _validate_ai_explanation('ai_interpretation' in options, ai_text_from_page, 'Log-Rank', language)
            if ai_error:
                return None, ai_error, {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }
            
            print(f"\n[LOGRANK PDF] ========== INICIANDO EXPORTACIÓN ==========")
            print(f"[LOGRANK PDF] Opciones marcadas: {options}")
            print(f"[LOGRANK PDF] Variables seleccionadas: {logrank_covariables}")
            
            # CAPTURAR DATOS REALES DEL ANÁLISIS ACTUAL
            df = None
            logrank_results = None
            logrank_figure = None
            
            if df_json and logrank_covariables:
                df = pd.read_json(df_json, orient='split')
                
                # RECALCULAR LOG-RANK TEST CON LA VARIABLE SELECCIONADA
                from log_rank_test import perform_log_rank_test, create_logrank_figure
                
                # Procesar cada covariable y acumular resultados
                all_results = []
                for covariable in logrank_covariables:
                    res_df = perform_log_rank_test(df, covariable)
                    if not res_df.empty:
                        all_results.append(res_df)
                
                if all_results:
                    logrank_results = pd.concat(all_results, ignore_index=True)
                
                # GENERAR GRÁFICA SI SE SOLICITA
                if 'graph' in options and logrank_covariables:
                    print(f"[LOGRANK PDF] Generando gráfica...")
                    try:
                        logrank_figure = create_logrank_figure(df, logrank_covariables[0])
                        if logrank_figure:
                            print(f"[LOGRANK PDF] ✓ Gráfica Log-Rank generada")
                    except Exception as e:
                        print(f"[LOGRANK PDF] ✗ Error generando gráfica: {e}")
                        logrank_figure = None
            
            # CALCULAR ESTADÍSTICAS DE RESUMEN
            n_patients = len(df) if df is not None else 0
            n_events = int(df['final_result'].sum()) if df is not None and 'final_result' in df.columns else 0
            follow_up_mean = float(df['date'].mean()) if df is not None and 'date' in df.columns else 0
            follow_up_median = float(df['date'].median()) if df is not None and 'date' in df.columns else 0
            
            summary_stats = {
                'n_patients': n_patients,
                'n_events': n_events,
                'follow_up_mean': follow_up_mean,
                'follow_up_median': follow_up_median,
                'variable_name': ', '.join(logrank_covariables) if logrank_covariables else 'N/A'
            }
            
            print(f"[LOGRANK PDF] Tabla será incluida: {'table' in options and logrank_results is not None}")
            print(f"[LOGRANK PDF] Gráfica será incluida: {'graph' in options and logrank_figure is not None}")
            print(f"[LOGRANK PDF] Resumen será incluido: {'summary' in options}")

            ai_text = ""
            if 'ai_interpretation' in options:
                print(f"[LOGRANK PDF] Generando interpretación de IA...")
                ai_text = ai_text_from_page or generate_interpretation_for_pdf('log-rank', summary_stats, logrank_results, language=language)
                ai_text = clean_markdown_text(ai_text)
            
            # GENERAR PDF CON DATOS REALES - SOLO LO QUE SE MARCÓ
            report_title = (
                f"LOG-RANK TEST REPORT: {', '.join(logrank_covariables) if logrank_covariables else 'GENERAL'}"
            )
            export_survival_analysis_to_pdf(
                filename=pdf_path,
                title=report_title,
                report_name=report_name,
                include_summary='summary' in options,  # Mostrar resumen SOLO si se marcó
                include_km=False,
                include_cox=False,
                include_logrank='table' in options or 'graph' in options,  # Mostrar Log-Rank si marcó tabla O gráfica
                logrank_figure=logrank_figure if 'graph' in options else None,  # Mostrar gráfica SOLO si se marcó
                logrank_results=logrank_results if 'table' in options else None,
                include_ai_interpretation='ai_interpretation' in options,
                ai_text=ai_text,
                summary_stats=summary_stats if 'summary' in options else None,
                language=language,
                landscape_tables=['logrank']  # Hacer tabla Log-Rank en horizontal
            )
            
            print(f"[LOGRANK PDF] ✓ PDF generado en: {pdf_path}")
            print(f"[LOGRANK PDF] ========== EXPORTACIÓN COMPLETADA ==========\n")
            return dcc.send_file(pdf_path), "", {'display': 'none'}
        
        except Exception as e:
            print(f"ERROR en download_logrank_pdf: {str(e)}")
            print(traceback.format_exc())
            return None, (f"❌ Error al generar el PDF de Log-Rank: {str(e)}" if language == 'es' else f"❌ Error generating the Log-Rank PDF: {str(e)}"), {
                'display': 'block',
                'marginTop': '15px',
                'padding': '10px 12px',
                'borderRadius': '6px',
                'backgroundColor': '#fff1f0',
                'color': '#c0392b',
                'border': '1px solid #f5c6cb',
                'fontSize': '14px',
                'fontWeight': 'bold'
            }


    # ===== COMBINED WEIBULL + EXPONENTIAL PDF CALLBACKS =====

    @app.callback(
        [Output('weibexp-pdf-modal-overlay', 'style'),
         Output('weibexp-pdf-modal-container', 'style')],
        [Input('export-weibexp-btn', 'n_clicks'),
         Input('weibexp-pdf-modal-close-btn', 'n_clicks'),
         Input('weibexp-pdf-modal-cancel-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def toggle_weibexp_pdf_modal(export_clicks, close_clicks, cancel_clicks):
        """Abre/cierra el modal de exportación combinada Weibull + Exponencial."""
        if not callback_context.triggered:
            return {'display': 'none'}, {'display': 'none'}

        triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

        if triggered_id == 'export-weibexp-btn' and export_clicks:
            return _get_modal_styles(True)

        return {'display': 'none'}, {'display': 'none'}


    @app.callback(
        [Output('weibexp-pdf-modal-download', 'data'),
         Output('weibexp-pdf-modal-error', 'children'),
         Output('weibexp-pdf-modal-error', 'style')],
        Input('weibexp-pdf-modal-download-btn', 'n_clicks'),
        [State('weibexp-pdf-modal-filename', 'value'),
         State('weibexp-pdf-modal-techniques', 'value'),
         State('weibexp-pdf-modal-content', 'value'),
         State('weibull-ai-text-store', 'data'),
         State('exponential-ai-text-store', 'data'),
         State('weibull-ai-language-store', 'data'),
         State('exponential-ai-language-store', 'data'),
         State('df-store', 'data'),
         State('language-store', 'data')],
        prevent_initial_call=True
    )
    def download_weibexp_pdf(
        n_clicks,
        filename,
        techniques,
        options,
        weibull_ai_text,
        exponential_ai_text,
        weibull_ai_language,
        exponential_ai_language,
        df_json,
        language
    ):
        """Genera un PDF combinado de Weibull + Exponencial."""
        try:
            if not filename or filename.strip() == '':
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"weibull_exponential_combined_{timestamp}.pdf"
            else:
                if not filename.endswith('.pdf'):
                    filename += '.pdf'

            report_name = _get_report_name_from_filename(filename)
            os.makedirs("downloads", exist_ok=True)
            pdf_path = f"downloads/{filename}"

            techniques = techniques or ['weibull', 'exponential']
            options = options or ['summary', 'table', 'graph']

            if 'ai_interpretation' in options:
                if 'weibull' in techniques:
                    ai_error = _validate_ai_explanation(True, weibull_ai_text, 'Weibull', language)
                    if ai_error:
                        return None, ai_error, {
                            'display': 'block',
                            'marginTop': '15px',
                            'padding': '10px 12px',
                            'borderRadius': '6px',
                            'backgroundColor': '#fff1f0',
                            'color': '#c0392b',
                            'border': '1px solid #f5c6cb',
                            'fontSize': '14px',
                            'fontWeight': 'bold'
                        }

                    lang_error = _validate_ai_language(True, weibull_ai_language, language)
                    if lang_error:
                        return None, lang_error, {
                            'display': 'block',
                            'marginTop': '15px',
                            'padding': '10px 12px',
                            'borderRadius': '6px',
                            'backgroundColor': '#fff1f0',
                            'color': '#c0392b',
                            'border': '1px solid #f5c6cb',
                            'fontSize': '14px',
                            'fontWeight': 'bold'
                        }

                if 'exponential' in techniques:
                    ai_error = _validate_ai_explanation(True, exponential_ai_text, 'Exponencial' if language == 'es' else 'Exponential', language)
                    if ai_error:
                        return None, ai_error, {
                            'display': 'block',
                            'marginTop': '15px',
                            'padding': '10px 12px',
                            'borderRadius': '6px',
                            'backgroundColor': '#fff1f0',
                            'color': '#c0392b',
                            'border': '1px solid #f5c6cb',
                            'fontSize': '14px',
                            'fontWeight': 'bold'
                        }

                    lang_error = _validate_ai_language(True, exponential_ai_language, language)
                    if lang_error:
                        return None, lang_error, {
                            'display': 'block',
                            'marginTop': '15px',
                            'padding': '10px 12px',
                            'borderRadius': '6px',
                            'backgroundColor': '#fff1f0',
                            'color': '#c0392b',
                            'border': '1px solid #f5c6cb',
                            'fontSize': '14px',
                            'fontWeight': 'bold'
                        }

            if not techniques:
                return None, (
                    "⚠️ Selecciona al menos una técnica (Weibull o Exponencial)."
                    if language == 'es' else
                    "⚠️ Select at least one technique (Weibull or Exponential)."
                ), {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            if not df_json:
                return None, (
                    "❌ No hay datos cargados para exportar el informe combinado."
                    if language == 'es' else
                    "❌ No dataset loaded to export the combined report."
                ), {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            df = pd.read_json(df_json, orient='split')
            weibull_analysis = build_weibull_analysis(df, language=language) if 'weibull' in techniques else None
            exponential_analysis = build_exponential_analysis(df, language=language) if 'exponential' in techniques else None

            if ('weibull' in techniques and not weibull_analysis) and ('exponential' in techniques and not exponential_analysis):
                return None, (
                    "❌ No hay datos suficientes para construir el informe combinado."
                    if language == 'es' else
                    "❌ Not enough data to build the combined report."
                ), {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            summary_stats = {
                'n_patients': len(df),
                'n_events': int(df['final_result'].sum()) if 'final_result' in df.columns else 0,
                'follow_up_mean': float(df['date'].mean()) if 'date' in df.columns else 0,
                'follow_up_median': float(df['date'].median()) if 'date' in df.columns else 0,
            }

            combined_ai_text = ""
            if 'ai_interpretation' in options:
                ai_sections = []
                if weibull_analysis:
                    weibull_ai = weibull_ai_text
                    if weibull_ai:
                        ai_sections.append("Weibull:\n" + clean_markdown_text(weibull_ai))

                if exponential_analysis:
                    exponential_ai = exponential_ai_text
                    if exponential_ai:
                        ai_sections.append(("Exponencial" if language == 'es' else "Exponential") + ":\n" + clean_markdown_text(exponential_ai))

                combined_ai_text = "\n\n".join(ai_sections)

            export_weibull_exponential_combined_pdf(
                filename=pdf_path,
                title="WEIBULL + EXPONENTIAL REPORT: COMBINED",
                report_name=report_name,
                include_summary='summary' in options,
                include_table='table' in options,
                include_graph='graph' in options,
                include_ai_interpretation='ai_interpretation' in options,
                include_weibull='weibull' in techniques and weibull_analysis is not None,
                include_exponential='exponential' in techniques and exponential_analysis is not None,
                weibull_table=weibull_analysis['summary_df'] if weibull_analysis is not None else None,
                weibull_figure=weibull_analysis['figure'] if weibull_analysis is not None else None,
                exponential_table=exponential_analysis['summary_df'] if exponential_analysis is not None else None,
                exponential_figure=exponential_analysis['figure'] if exponential_analysis is not None else None,
                ai_text=combined_ai_text,
                summary_stats=summary_stats,
                language=language
            )

            return dcc.send_file(pdf_path), "", {'display': 'none'}

        except Exception as e:
            print(f"ERROR en download_weibexp_pdf: {str(e)}")
            print(traceback.format_exc())
            return None, (
                f"❌ Error al generar el PDF combinado: {str(e)}"
                if language == 'es' else
                f"❌ Error generating the combined PDF: {str(e)}"
            ), {
                'display': 'block',
                'marginTop': '15px',
                'padding': '10px 12px',
                'borderRadius': '6px',
                'backgroundColor': '#fff1f0',
                'color': '#c0392b',
                'border': '1px solid #f5c6cb',
                'fontSize': '14px',
                'fontWeight': 'bold'
            }


    # ===== WEIBULL PDF CALLBACKS =====

    @app.callback(
        [Output('weibull-pdf-modal-overlay', 'style'),
         Output('weibull-pdf-modal-container', 'style')],
        [Input('export-weibull-btn', 'n_clicks'),
         Input('weibull-pdf-modal-close-btn', 'n_clicks'),
         Input('weibull-pdf-modal-cancel-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def toggle_weibull_pdf_modal(export_clicks, close_clicks, cancel_clicks):
        """Abre/cierra el modal de exportación Weibull"""
        if not callback_context.triggered:
            return {'display': 'none'}, {'display': 'none'}

        triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

        if triggered_id == 'export-weibull-btn' and export_clicks:
            return _get_modal_styles(True)

        return {'display': 'none'}, {'display': 'none'}


    @app.callback(
        [Output('weibull-pdf-modal-download', 'data'),
         Output('weibull-pdf-modal-error', 'children'),
         Output('weibull-pdf-modal-error', 'style')],
        Input('weibull-pdf-modal-download-btn', 'n_clicks'),
        [State('weibull-pdf-modal-filename', 'value'),
         State('weibull-pdf-modal-checklist-content', 'value'),
         State('openai-answer-weibull', 'children'),
         State('weibull-ai-language-store', 'data'),
         State('df-store', 'data'),
         State('language-store', 'data')],
        prevent_initial_call=True
    )
    def download_weibull_pdf(n_clicks, filename, options, ai_text_from_page, ai_text_language, df_json, language):
        """Genera PDF de Weibull con datos REALES del dataset limpio"""
        try:
            if not filename or filename.strip() == '':
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"weibull_{timestamp}.pdf"
            else:
                if not filename.endswith('.pdf'):
                    filename += '.pdf'

            os.makedirs("downloads", exist_ok=True)
            pdf_path = f"downloads/{filename}"
            report_name = _get_report_name_from_filename(filename)

            options = options or ['summary', 'table']
            ai_error = _validate_ai_explanation('ai_interpretation' in options, ai_text_from_page, 'Weibull', language)
            if ai_error:
                return None, ai_error, {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            ai_lang_error = _validate_ai_language('ai_interpretation' in options, ai_text_language, language)
            if ai_lang_error:
                return None, ai_lang_error, {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            df = None
            if df_json:
                try:
                    df = pd.read_json(df_json, orient='split')
                except Exception:
                    df = None

            analysis = build_weibull_analysis(df, language=language) if df is not None else None
            if not analysis:
                return None, ("❌ No hay datos suficientes para generar el PDF de Weibull." if language == 'es' else "❌ Not enough data to generate the Weibull PDF."), {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            summary_stats = {
                'n_patients': analysis['n_observations'],
                'n_events': analysis['n_events'],
                'event_rate': analysis['event_rate'],
                'variable_name': 'Weibull'
            }

            ai_text = ""
            if 'ai_interpretation' in options:
                ai_text = ai_text_from_page or generate_interpretation_for_pdf('weibull', summary_stats, analysis['summary_df'], language=language)
                ai_text = clean_markdown_text(ai_text)

            report_title = "WEIBULL METHOD REPORT: GENERAL"
            export_survival_analysis_to_pdf(
                filename=pdf_path,
                title=report_title,
                report_name=report_name,
                include_summary='summary' in options,
                include_km=False,
                include_cox=False,
                include_logrank=False,
                include_weibull='table' in options or 'graph' in options,
                weibull_table=analysis['summary_df'] if 'table' in options else None,
                weibull_figure=analysis['figure'] if 'graph' in options else None,
                include_ai_interpretation='ai_interpretation' in options,
                ai_text=ai_text,
                summary_stats=summary_stats if 'summary' in options else None,
                language=language
            )

            return dcc.send_file(pdf_path), "", {'display': 'none'}

        except Exception as e:
            print(f"ERROR en download_weibull_pdf: {str(e)}")
            print(traceback.format_exc())
            return None, (f"❌ Error al generar el PDF de Weibull: {str(e)}" if language == 'es' else f"❌ Error generating the Weibull PDF: {str(e)}"), {
                'display': 'block',
                'marginTop': '15px',
                'padding': '10px 12px',
                'borderRadius': '6px',
                'backgroundColor': '#fff1f0',
                'color': '#c0392b',
                'border': '1px solid #f5c6cb',
                'fontSize': '14px',
                'fontWeight': 'bold'
            }


    # ===== EXPONENTIAL PDF CALLBACKS =====

    @app.callback(
        [Output('exponential-pdf-modal-overlay', 'style'),
         Output('exponential-pdf-modal-container', 'style')],
        [Input('export-exponential-btn', 'n_clicks'),
         Input('exponential-pdf-modal-close-btn', 'n_clicks'),
         Input('exponential-pdf-modal-cancel-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def toggle_exponential_pdf_modal(export_clicks, close_clicks, cancel_clicks):
        if not callback_context.triggered:
            return {'display': 'none'}, {'display': 'none'}

        triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

        if triggered_id == 'export-exponential-btn' and export_clicks:
            return _get_modal_styles(True)

        return {'display': 'none'}, {'display': 'none'}


    @app.callback(
        [Output('exponential-pdf-modal-download', 'data'),
         Output('exponential-pdf-modal-error', 'children'),
         Output('exponential-pdf-modal-error', 'style')],
        Input('exponential-pdf-modal-download-btn', 'n_clicks'),
        [State('exponential-pdf-modal-filename', 'value'),
         State('exponential-pdf-modal-checklist-content', 'value'),
         State('openai-answer-exponential', 'children'),
         State('exponential-ai-language-store', 'data'),
         State('df-store', 'data'),
         State('language-store', 'data')],
        prevent_initial_call=True
    )
    def download_exponential_pdf(n_clicks, filename, options, ai_text_from_page, ai_text_language, df_json, language):
        try:
            if not filename or filename.strip() == '':
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"exponential_{timestamp}.pdf"
            else:
                if not filename.endswith('.pdf'):
                    filename += '.pdf'

            os.makedirs("downloads", exist_ok=True)
            pdf_path = f"downloads/{filename}"
            report_name = _get_report_name_from_filename(filename)

            options = options or ['summary', 'table']
            ai_error = _validate_ai_explanation('ai_interpretation' in options, ai_text_from_page, 'Exponencial', language)
            if ai_error:
                return None, ai_error, {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            ai_lang_error = _validate_ai_language('ai_interpretation' in options, ai_text_language, language)
            if ai_lang_error:
                return None, ai_lang_error, {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            df = None
            if df_json:
                try:
                    df = pd.read_json(df_json, orient='split')
                except Exception:
                    df = None

            analysis = build_exponential_analysis(df, language=language) if df is not None else None
            if not analysis:
                return None, ("❌ No hay datos suficientes para generar el PDF de Exponencial." if language == 'es' else "❌ Not enough data to generate the Exponential PDF."), {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            summary_stats = {
                'n_patients': analysis['n_observations'],
                'n_events': analysis['n_events'],
                'follow_up_mean': float(df['date'].mean()) if df is not None and 'date' in df.columns else 0,
                'follow_up_median': float(df['date'].median()) if df is not None and 'date' in df.columns else 0,
                'variable_name': 'Exponencial'
            }

            ai_text = ""
            if 'ai_interpretation' in options:
                ai_text = ai_text_from_page or analysis['interpretation']
                ai_text = clean_markdown_text(ai_text)

            report_title = "EXPONENTIAL METHOD REPORT: GENERAL"
            export_survival_analysis_to_pdf(
                filename=pdf_path,
                title=report_title,
                report_name=report_name,
                include_summary='summary' in options,
                include_km=False,
                include_cox=False,
                include_logrank=False,
                include_weibull=False,
                include_exponential=True,
                exponential_table=analysis['summary_df'],
                exponential_figure=analysis['figure'] if 'graph' in options else None,
                include_ai_interpretation='ai_interpretation' in options,
                ai_text=ai_text,
                summary_stats=summary_stats if 'summary' in options else None,
                language=language
            )

            return dcc.send_file(pdf_path), "", {'display': 'none'}

        except Exception as e:
            print(f"ERROR en download_exponential_pdf: {str(e)}")
            print(traceback.format_exc())
            return None, (f"❌ Error al generar el PDF de Exponencial: {str(e)}" if language == 'es' else f"❌ Error generating the Exponential PDF: {str(e)}"), {
                'display': 'block',
                'marginTop': '15px',
                'padding': '10px 12px',
                'borderRadius': '6px',
                'backgroundColor': '#fff1f0',
                'color': '#c0392b',
                'border': '1px solid #f5c6cb',
                'fontSize': '14px',
                'fontWeight': 'bold'
            }


    # ===== RSF PDF CALLBACKS =====

    @app.callback(
        [Output('rsf-pdf-modal-overlay', 'style'),
         Output('rsf-pdf-modal-container', 'style')],
        [Input('export-rsf-btn', 'n_clicks'),
         Input('rsf-pdf-modal-close-btn', 'n_clicks'),
         Input('rsf-pdf-modal-cancel-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def toggle_rsf_pdf_modal(export_clicks, close_clicks, cancel_clicks):
        """Abre/cierra el modal de exportación RSF"""
        if not callback_context.triggered:
            return {'display': 'none'}, {'display': 'none'}

        triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

        if triggered_id == 'export-rsf-btn' and export_clicks:
            return _get_modal_styles(True)

        return {'display': 'none'}, {'display': 'none'}


    @app.callback(
        [Output('rsf-pdf-modal-download', 'data'),
         Output('rsf-pdf-modal-error', 'children'),
         Output('rsf-pdf-modal-error', 'style')],
        Input('rsf-pdf-modal-download-btn', 'n_clicks'),
        [State('rsf-pdf-modal-filename', 'value'),
         State('rsf-pdf-modal-checklist-content', 'value'),
         State('rsf-analysis-data', 'data'),
         State('openai-answer-rsf', 'value'),
         State('rsf-profile-gender', 'value'),
         State('rsf-profile-disability', 'value'),
         State('rsf-profile-age-band', 'value'),
         State('rsf-profile-education', 'value'),
         State('rsf-profile-credits', 'value'),
         State('df-store', 'data'),
         State('language-store', 'data')],
        prevent_initial_call=True
    )
    def download_rsf_pdf(
        n_clicks,
        filename,
        options,
        rsf_store_data,
        ai_text_from_page,
        profile_gender,
        profile_disability,
        profile_age_band,
        profile_education,
        profile_credits_level,
        df_json,
        language,
    ):
        """Genera PDF de Random Survival Forest con datos reales del análisis actual"""
        try:
            if not filename or filename.strip() == '':
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"rsf_{timestamp}.pdf"
            else:
                if not filename.endswith('.pdf'):
                    filename += '.pdf'

            os.makedirs("downloads", exist_ok=True)
            pdf_path = f"downloads/{filename}"
            report_name = _get_report_name_from_filename(filename)

            options = options or ['general_summary', 'model_summary']
            ai_error = _validate_ai_explanation('ai_interpretation' in options, ai_text_from_page, 'Random Survival Forest', language)
            if ai_error:
                return None, ai_error, {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            df = None
            if df_json:
                try:
                    df = pd.read_json(df_json, orient='split')
                except Exception:
                    df = None

            analysis = build_rsf_analysis(df, language=language) if df is not None else None
            if not analysis:
                return None, ("❌ No hay datos suficientes para generar el PDF de RSF." if language == 'es' else "❌ Not enough data to generate the RSF PDF."), {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            # Usar la misma tabla que se pinta en el dashboard para que el PDF refleje exactamente
            # el bloque "Resumen del modelo RSF" mostrado arriba.
            rsf_table_for_pdf = analysis['summary_df']
            if rsf_store_data and isinstance(rsf_store_data, dict) and rsf_store_data.get('summary_json'):
                try:
                    rsf_table_for_pdf = pd.read_json(rsf_store_data['summary_json'], orient='split')
                except Exception:
                    rsf_table_for_pdf = analysis['summary_df']

            summary_stats = {
                'n_patients': analysis['n_observations'],
                'n_events': analysis['n_events'],
                'follow_up_mean': float(df['date'].mean()) if df is not None and 'date' in df.columns else 0,
                'follow_up_median': float(df['date'].median()) if df is not None and 'date' in df.columns else 0,
                'variable_name': analysis['top_feature']
            }

            rsf_profile_analysis = None
            if 'profile' in options:
                credits_map = {'few': 30, 'medium': 60, 'many': 120}
                profile = {
                    'gender_F': profile_gender if profile_gender is not None else 1,
                    'disability_N': profile_disability if profile_disability is not None else 1,
                    'age_band': profile_age_band if profile_age_band else 'age_band_0-35',
                    'highest_education': profile_education if profile_education else 'highest_education_A Level or Equivalent',
                    'studied_credits': credits_map.get(profile_credits_level, 30),
                }
                rsf_profile_analysis = build_rsf_profile_analysis(df, profile, language=language)

            ai_text = ""
            if 'ai_interpretation' in options:
                ai_text = ai_text_from_page or analysis['interpretation']
                ai_text = clean_markdown_text(ai_text)

            report_title = "RANDOM SURVIVAL FOREST REPORT: MODEL"
            export_kwargs = dict(
                filename=pdf_path,
                title=report_title,
                report_name=report_name,
                include_summary='general_summary' in options,
                include_km=False,
                include_cox=False,
                include_logrank=False,
                include_weibull=False,
                include_rsf='model_summary' in options or 'graph' in options or 'importance' in options,
                rsf_table=rsf_table_for_pdf if 'model_summary' in options else None,
                rsf_figure=analysis['figure'] if 'graph' in options else None,
                rsf_importance_figure=analysis['importance_figure'] if 'importance' in options else None,
                include_rsf_profile='profile' in options and rsf_profile_analysis is not None,
                rsf_profile_figure=rsf_profile_analysis['figure'] if rsf_profile_analysis is not None else None,
                rsf_profile_text=clean_markdown_text(rsf_profile_analysis['interpretation']) if rsf_profile_analysis is not None else "",
                include_ai_interpretation='ai_interpretation' in options,
                ai_text=ai_text,
                summary_stats=summary_stats if 'general_summary' in options else None,
                language=language,
            )

            try:
                export_survival_analysis_to_pdf(**export_kwargs)
            except TypeError as type_error:
                # Compatibilidad con recargas parciales en caliente (firma antigua de exportador en memoria).
                if "include_rsf_profile" not in str(type_error):
                    raise

                fallback_kwargs = dict(export_kwargs)
                fallback_kwargs.pop('include_rsf_profile', None)
                fallback_kwargs.pop('rsf_profile_figure', None)
                fallback_kwargs.pop('rsf_profile_text', None)
                export_survival_analysis_to_pdf(**fallback_kwargs)

            return dcc.send_file(pdf_path), "", {'display': 'none'}

        except Exception as e:
            print(f"ERROR en download_rsf_pdf: {str(e)}")
            print(traceback.format_exc())
            return None, (f"❌ Error al generar el PDF de RSF: {str(e)}" if language == 'es' else f"❌ Error generating the RSF PDF: {str(e)}"), {
                'display': 'block',
                'marginTop': '15px',
                'padding': '10px 12px',
                'borderRadius': '6px',
                'backgroundColor': '#fff1f0',
                'color': '#c0392b',
                'border': '1px solid #f5c6cb',
                'fontSize': '14px',
                'fontWeight': 'bold'
            }


def _get_modal_styles(show=True):
    """Helper para obtener estilos del modal"""
    if not show:
        return {'display': 'none'}, {'display': 'none'}
    
    overlay_style = {
        'display': 'block',
        'position': 'fixed',
        'top': 0,
        'left': 0,
        'width': '100%',
        'height': '100%',
        'backgroundColor': 'rgba(0,0,0,0.5)',
        'zIndex': 999
    }
    container_style = {
        'display': 'block',
        'position': 'fixed',
        'top': '50%',
        'left': '50%',
        'transform': 'translate(-50%, -50%)',
        'backgroundColor': 'white',
        'padding': '30px',
        'borderRadius': '12px',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.3)',
        'zIndex': 1000,
        'width': '90%',
        'maxWidth': '500px',
        'maxHeight': '600px',
        'overflowY': 'auto'
    }
    return overlay_style, container_style
