# Diccionario de traducciones español-inglés
translations = {
    'es': {
        # Navbar
        'inicio': 'INICIO',
        'ver_dataset': 'VER DATASET',
        'analisis_covariables': 'ANÁLISIS DE COVARIABLES',
        'analisis_supervivencia': 'ANÁLISIS DE SUPERVIVENCIA',
        
        # Home page
        'dashboard_title': "DASHBOARD 'SURVIVAL ANALYSIS'",
        'cargar_dataset': 'Cargar Dataset para preprocesarlo y analizarlo',
        'sube_csv': 'Sube tu CSV',
        'preprocesa_csv': 'Preprocesa CSV',
        'no_archivo_cargado': 'No se ha cargado ningún archivo aún.',
        'error_archivo': 'ERROR: El archivo cargado no es el adecuado.',
        'archivo_correcto': 'Este sistema es compatible únicamente con el dataset temp_data.csv',
        'archivo_bruto': 'Archivo Bruto',
        'archivo_preprocesado': 'Archivo Preprocesado',
        
        # Diálogo de confirmación
        'confirmar_inicio': '¿Estás seguro de que deseas volver a la página inicial? Perderás el dataset cargado y todo el análisis realizado.',
        
        # Survival Analysis
        'survival_analysis': 'Análisis de Supervivencia',
        'kaplan_meier': 'Kaplan-Meier',
        'cox_regression': 'Regresión de Cox',
        'log_rank_test': 'Test de Log-Rank',
        
        # Dataset Limpio
        'dataset_limpio': 'Dataset Limpio',
        
        # Análisis de Covariables
        'analisis_covariables_title': 'Análisis de Covariables',
        'abandono_total': 'Abandono Total',
        'abandono_genero': 'Abandono por Género',
        'abandono_discapacidad': 'Abandono por Discapacidad',
        'abandono_age_band': 'Abandono por Banda de Edad',
        'abandono_highest_education': 'Abandono por Educación Más Alta',
        'femenino': 'Femenino',
        'masculino': 'Masculino',
        'con_discapacidad': 'Con Discapacidad',
        'sin_discapacidad': 'Sin Discapacidad',
        'abandono': 'Abandono',
        'no_abandono': 'No abandono',
        'withdrawn': 'Withdrawn',
        'edad_banda': 'Banda de Edad',
        'educacion_mas_alta': 'Educación Más Alta',
        'creditos_estudiados': 'Créditos Estudiados',
        
        # Kaplan-Meier
        'curva_supervivencia': 'Curva de Supervivencia Kaplan-Meier',
        'tiempo': 'Tiempo',
        'probabilidad_supervivencia': 'Probabilidad de Supervivencia',
        'explicar_kaplan': 'Explicar',
        'explicar_cox': 'Explicar',
        'explicar_logrank': 'Explicar',
        'selecciona_covariable': 'Selecciona 1 covariable para ver su curva de Kaplan:',
        'selecciona_covariable_kaplan': 'Selecciona 1 covariable para ver su curva de Kaplan:',
        'genero': 'Género',
        'discapacidad': 'Discapacidad',
        'ninguna': 'Ninguna',
        
        # Cox Regression page
        'elige_covariable_cox': 'Elige 1 o más covariable para la regresión de Cox',
        'selecciona_covariable_logrank': 'Selecciona 1 o más covariables para el Test de Log-Rank:',
        'selecciona_covariable_minimo': 'Selecciona al menos una covariable.',
        'selecciona_covariable_comparar': 'Selecciona al menos una covariable para comparar.',
        
        # Gráficos - Titles
        'abandono_vs_no_abandono': 'Abandono vs No Abandono',
        'resultado_final': 'Resultado Final',
        'num_estudiantes': 'Número de Estudiantes',
        'abandono_genero_title': 'Abandono según Género',
        'abandono_discapacidad_title': 'Abandono según Discapacidad',
        'abandono_age_band_title': 'Abandono según Banda de Edad',
        'abandono_highest_education_title': 'Abandono según Educación Más Alta',
        'evento': 'Evento',
        
        # Explicaciones
        'exp_abandono': '''Este gráfico muestra la distribución de los estudiantes en función de si han abandonado o no el curso. 
            La columna verde representa a los estudiantes que no han abandonado, mientras que la morada a los que sí.
            Como se puede observar, la mayoría de los estudiantes no han llevado a cabo el evento de abandono 
            lo que sugiere que la tasa de abandono es relativamente baja en este conjunto de datos.''',
        'exp_genero': '''Este gráfico muestra la distribución del abandono influido por el género. Podemos concluir en que la mayoría de los estudiantes
            no han abandonado el curso y se observa una diferencia destacable en el abandono por género. Las estudiantes femeninas
            presentan una mayor proporción de abandono en comparación con los estudiantes masculinos, por ello, en este conjunto las mujeres tienen una
            tasa de abandono superior. A pesar de ello, en ambos sexos, el número de estudiantes 
            que no abandonan el curso es significativamente mayor que el número de los que si abandonan.''',
        'exp_discapacidad': '''Este gráfico muestra la distribución del abandono influido por la discapacidad. A pesar de que el número de estudiantes sin discapacidad
            es considerablemente menor, la tasa de abandono en este conjunto es más alta en comparación con los estudiantes que presentan discapacidad.
            Aún así, la mayoría de estudiantes, tanto con como sin, no abandonan el curso, y es la conclusión que podemos destacar del grafo presente. Esto
            nos lleva a entender que aunque la discapacidad puede estar asociada a un riesgo superior de que suceda este evento, la diferencia no es 
            considerable para este conjunto de datos.''',
        'exp_age_band': '''Este gráfico muestra la distribución del abandono según la banda de edad de los estudiantes. Podemos observar cómo la tasa de abandono
            varía según los diferentes grupos de edad. Este análisis es importante para comprender si existe una relación entre la edad del estudiante y la probabilidad
            de abandonar el curso. La mayoría de estudiantes no abandonan, pero es notable ver si hay diferencias significativas entre grupos de edad.''',
        'exp_highest_education': '''Este gráfico muestra la distribución del abandono según el nivel de educación más alto de los estudiantes. Podemos analizar si existe
            una correlación entre el nivel educativo previo y el abandono del curso. Este análisis ayuda a entender cómo el trasfondo educativo influye en la persistencia
            del estudiante. Es importante observar si ciertos niveles educativos tienen tasas de abandono más altas o más bajas.''',
        'abandono_studied_credits': 'Abandono por Créditos Estudiados',
        'abandono_studied_credits_title': 'Abandono según Créditos Estudiados',
        'exp_studied_credits': '''Este gráfico muestra la distribución del abandono según los créditos estudiados por los estudiantes. Podemos observar si existe una relación
            entre la cantidad de créditos cursados y la probabilidad de abandonar. Este análisis es importante para entender cómo el progreso académico
            influye en la retención del estudiante.''',
        
        # General
        'cargando': 'Cargando...',
        'respuesta': 'La respuesta es...',
    },
    'en': {
        # Navbar
        'inicio': 'HOME',
        'ver_dataset': 'VIEW DATASET',
        'analisis_covariables': 'COVARIATE ANALYSIS',
        'analisis_supervivencia': 'SURVIVAL ANALYSIS',
        
        # Home page
        'dashboard_title': "SURVIVAL ANALYSIS DASHBOARD",
        'cargar_dataset': 'Load Dataset to preprocess and analyze',
        'sube_csv': 'Upload your CSV',
        'preprocesa_csv': 'Preprocess CSV',
        'no_archivo_cargado': 'No file has been loaded yet.',
        'error_archivo': 'ERROR: The uploaded file is not suitable.',
        'archivo_correcto': 'This system is only compatible with the temp_data.csv dataset',
        'archivo_bruto': 'Raw File',
        'archivo_preprocesado': 'Preprocessed File',
        
        # Diálogo de confirmación
        'confirmar_inicio': 'Are you sure you want to return to the home page? You will lose the loaded dataset and all analysis performed.',
        
        # Survival Analysis
        'survival_analysis': 'Survival Analysis',
        'kaplan_meier': 'Kaplan-Meier',
        'cox_regression': 'Cox Regression',
        'log_rank_test': 'Log-Rank Test',
        
        #abandono_age_band': 'Dropout by Age Band',
        'abandono_highest_education': 'Dropout by Highest Education',
        'femenino': 'Female',
        'masculino': 'Male',
        'con_discapacidad': 'With Disability',
        'sin_discapacidad': 'Without Disability',
        'abandono': 'Dropout',
        'no_abandono': 'No Dropout',
        'withdrawn': 'Withdrawn',
        'edad_banda': 'Age Band',
        'educacion_mas_alta': 'Highest Education',
        'creditos_estudiados': 'Studied Creditspout by Gender',
        'abandono_discapacidad': 'Dropout by Disability',
        'femenino': 'Female',
        'masculino': 'Male',
        'con_discapacidad': 'With Disability',
        'sin_discapacidad': 'Without Disability',
        'abandono': 'Dropout',
        'no_abandono': 'No Dropout',
        'withdrawn': 'Withdrawn',
        
        # Kaplan-Meier
        'curva_supervivencia': 'Kaplan-Meier Survival Curve',
        'tiempo': 'Time',
        'probabilidad_supervivencia': 'Survival Probability',
        'explicar_kaplan': 'Explain',
        'explicar_cox': 'Explain',
        'explicar_logrank': 'Explain',
        'selecciona_covariable': 'Select 1 covariate to see its Kaplan curve:',
        'selecciona_covariable_kaplan': 'Select 1 covariate to see its Kaplan curve:',
        'genero': 'Gender',
        'discapacidad': 'Disability',
        'ninguna': 'None',
        
        # Cox Regression page
        'abandono_age_band_title': 'Dropout by Age Band',
        'abandono_highest_education_title': 'Dropout by Highest Education',
        'elige_covariable_cox': 'Choose 1 or more covariate for Cox regression',
        'selecciona_covariable_logrank': 'Select 1 or more covariates for the Log-Rank Test:',
        'selecciona_covariable_minimo': 'Please select at least one covariate.',
        'selecciona_covariable_comparar': 'Please select at least one covariate to compare.',
        
        # Gráficos - Titles
        'abandono_vs_no_abandono': 'Dropout vs No Dropout',
        'resultado_final': 'Final Result',
        'num_estudiantes': 'Number of Students',
        'abandono_genero_title': 'Dropout by Gender',
        'abandono_discapacidad_title': 'Dropout by Disability',
        'evento': 'Event',
        
        # Explicaciones
        'exp_age_band': '''This chart shows the distribution of dropout by the age band of students. We can observe how the dropout rate
            varies among different age groups. This analysis is important to understand if there is a relationship between student age and the probability
            of dropping out of the course. Most students do not drop out, but it is notable to see if there are significant differences between age groups.''',
        'exp_highest_education': '''This chart shows the distribution of dropout by the highest education level of students. We can analyze if there is
            a correlation between prior educational level and course dropout. This analysis helps understand how educational background influences student
            persistence. It is important to observe if certain education levels have higher or lower dropout rates.''',
        'abandono_studied_credits': 'Dropout by Studied Credits',
        'abandono_studied_credits_title': 'Dropout by Studied Credits',
        'exp_studied_credits': '''This chart shows the distribution of dropout by the studied credits of students. We can observe if there is a relationship
            between the number of credits taken and the probability of dropping out. This analysis is important to understand how academic progress
            influences student retention.''',
        'exp_abandono': '''This chart shows the distribution of students based on whether they have dropped out or not. 
            The green column represents students who have not dropped out, while the purple represents those who have.
            As can be seen, most students have not experienced the dropout event, suggesting that the dropout rate is relatively low in this dataset.''',
        'exp_genero': '''This chart shows the distribution of dropout influenced by gender. We can conclude that most students
            have not dropped out of the course and there is a notable difference in dropout by gender. Female students
            have a higher proportion of dropout compared to male students, therefore, in this dataset women have a
            higher dropout rate. However, in both sexes, the number of students 
            who do not drop out is significantly greater than the number who do.''',
        'exp_discapacidad': '''This chart shows the distribution of dropout influenced by disability. Although the number of students without disability
            is considerably lower, the dropout rate in this dataset is higher compared to students with disability.
            Still, most students, both with and without disability, do not drop out, and this is the conclusion we can highlight from this chart. This
            leads us to understand that although disability can be associated with a higher risk of this event occurring, the difference is not 
            significant for this dataset.''',
        
        # General
        'cargando': 'Loading...',
        'respuesta': 'The answer is...',
    }
}

def get_translation(lang, key):
    """Obtiene una traducción basada en el idioma y la clave"""
    return translations.get(lang, {}).get(key, key)
