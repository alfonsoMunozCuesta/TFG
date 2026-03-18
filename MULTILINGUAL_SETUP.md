# Soporte Multilingüe - Survival Analysis Dashboard

## ✅ Implementación Completada

Se ha agregado **soporte multilingüe (Español e Inglés)** con un botón selector en la esquina superior derecha de la navbar.

### 📁 Archivos Modificados

1. **translations.py** (NUEVO)
   - Diccionario completo de traducciones español/inglés
   - Función `get_translation(lang, key)` para obtener traducciones

2. **cargaDataset.py** (MODIFICADO)
   - Importación de funciones dinámicas de layout
   - Selector de idiomas en la navbar (botón radio)
   - Store para almacenar el idioma actual
   - Callback para actualizar idioma globalmente

3. **layout.py** (MODIFICADO)
   - Convertir páginas estáticas a funciones dinámicas:
     - `create_survival_analysis_page(language)`
     - `create_covariate_analysis_page(language)`
     - `create_kaplan_meier_page(language)`
     - `create_cox_regression_page(language)`
     - `create_log_rank_page(language)`
     - `create_ver_dataset_page(language)`
   - Todas las páginas ahora usan `get_translation()` para textos dinámicos

4. **assets/style.css** (MODIFICADO)
   - Estilos mejorados para el selector de idiomas
   - Estilos para la navbar con mejor layout
   - Responsividad del selector

### 🎯 Características

- ✅ Selector de idiomas en esquina superior derecha
- ✅ Cambio dinámico de idioma sin recargar página
- ✅ Mensajes de confirmación traducidos
- ✅ Títulos, botones y textos en ambos idiomas
- ✅ Persistencia del idioma durante la sesión

### 📝 Cómo Usar

**Selector de Idiomas:**
- Ubicado en la esquina superior derecha de la barra de navegación
- Opciones: ES (Español) | EN (Inglés)
- Cambio instantáneo al seleccionar

**Para agregar nuevas traducciones:**

1. Abrir `translations.py`
2. Agregar la clave en ambos idiomas:
```python
'mi_nueva_clave': 'Texto en español',
...
'en': {
    'mi_nueva_clave': 'Text in English',
}
```

3. En el código Dash, usar:
```python
get_translation(language, 'mi_nueva_clave')
```

### 🔄 Flujo de Traducción

1. Usuario selecciona idioma en el selector
2. `language-store` se actualiza con el idioma seleccionado
3. Callbacks detectan cambio y pasan el idioma a las funciones
4. Las funciones dinámicas crean el layout con traducciones

### 💾 Archivos Clave

```
Survival-Analysis/
├── translations.py              # Diccionario de traducciones
├── cargaDataset.py             # App principal + callbacks
├── layout.py                   # Funciones dinámicas de páginas
├── assets/
│   └── style.css              # Estilos mejorados
```

### 🚀 Para Ejecutar

```bash
cd c:\Users\LENOVO\Desktop\CODE_LUCI\Survival-Analysis
python cargaDataset.py
```

Luego accede a `http://localhost:8050` en tu navegador.

### 📚 Traducciones Disponibles

**Español:**
- Todos los títulos, botones y mensajes en español
- Mensajes de error personalizados
- Confirmaciones de navegación

**English:**
- All titles, buttons, and messages in English
- Customized error messages
- Navigation confirmations

---

**Nota:** El selector de idiomas aparece arriba a la derecha con dos opciones: **ES** e **EN**
