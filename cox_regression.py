from lifelines import CoxPHFitter
import numpy as np
from dash import dash_table

def run_cox_regression(df_limpio, covariables):
    # Preparamos el DataFrame para la regresión de Cox
    df_cox = df_limpio[['date', 'final_result'] + covariables]

    # Creamos el objeto CoxPHFitter y ajustamos el modelo
    cph = CoxPHFitter()
    cph.fit(df_cox, duration_col='date', event_col='final_result')

    # resumen del modelo de Cox
    summary = cph.summary
    summary['-log2(p)'] = -np.log2(summary['p'])

    #  tabla HTML a partir del resumen del modelo de Cox
    summary = summary[['coef', 'exp(coef)', 'se(coef)', 'coef lower 95%', 'coef upper 95%',
                       'exp(coef) lower 95%', 'exp(coef) upper 95%', 'cmp to', 'z', 'p', '-log2(p)']]
    
    summary.reset_index(inplace=True)
    summary.columns = ['Covariable', 'Coef.', 'exp(Coef.)', 'SE(Coef.)', 'Coef. lower 95%', 'Coef. upper 95%',
                       'exp(Coef.) lower 95%', 'exp(Coef.) upper 95%', 'cmp to', 'z', 'p', '-log2(p)']

    # Crear el DataTable
    cox_table_html = dash_table.DataTable(
        id='cox-summary-table',
        columns=[{"name": col, "id": col} for col in summary.columns],
        data=summary.to_dict('records'),
        style_table={
            'height': '200px',
            'overflowY': 'auto',  
            'overflowX': 'auto',  
            'width': '100%'
        },
        style_cell={
            'textAlign': 'center',
            'whiteSpace': 'normal',
            'height': 'auto',
            'lineHeight': '15px',
            'backgroundColor': 'transparent'
        },
        style_header={'fontWeight': 'bold', 'backgroundColor': '#f4f7f6'},
        style_cell_conditional=[
            {'if': {'column_id': 'Covariable'}, 'textAlign': 'center'}
        ]
    )

    # Retornar tanto el resumen como la tabla HTML
    return summary, cox_table_html