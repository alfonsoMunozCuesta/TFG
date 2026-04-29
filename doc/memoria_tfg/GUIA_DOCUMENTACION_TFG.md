# Guia de documentacion del TFG

Esta carpeta contiene la plantilla LaTeX extraida de `ALFONSO.zip` y preparada para documentar el TFG:

- Archivo principal: `__memoria.tex`
- Datos de portada: `Configuracion/_datos_proyecto.tex`
- Capitulos: `Unidades/*/info.tex`
- Bibliografia: `referencias.bib`
- Manuales anexos: `Unidades/Apéndice/manual_de_usuario.tex` y `Unidades/Apéndice/manual_de_codigo.tex`

## Datos del anteproyecto que deben mantenerse

- Titulo: Ampliacion y mejora de un Dashboard interactivo para el analisis de supervivencia aplicado al abandono escolar universitario.
- Autor: Alfonso Munoz Cuesta.
- Director: Dr. Juan Alfonso Lara Torralbo.
- Universidad: Universidad de Cordoba, Escuela Politecnica Superior de Cordoba.
- Grado: Ingenieria Informatica.
- Tipo de TFG: analisis y resolucion de casos practicos reales en el ambito de la ingenieria.
- Linea: analisis, diseno y desarrollo de aplicaciones web.

## Objetivos obligatorios

La memoria debe demostrar que se han cubierto estos puntos:

- Mejora del modulo de inteligencia artificial para generar interpretaciones con mayor rapidez.
- Soporte multilingue, al menos castellano e ingles.
- Incorporacion o tratamiento de nuevas variables, incluyendo variables no binarias cuando proceda.
- Graficos y analisis comparativos basados en Log-Rank y Cox.
- Integracion de nuevas tecnicas de analisis de supervivencia: Weibull, Exponencial y Random Survival Forest, o equivalentes justificados.
- Exportacion automatica de informes, graficos o resultados.

## Estructura esperada de la memoria

El anteproyecto pide cubrir, como minimo:

- Introduccion y motivacion.
- Objetivos generales y especificos.
- Antecedentes y relacion con el TFG previo.
- Marco teorico sobre analisis de supervivencia y abandono universitario.
- Metodologia: datos, preprocesamiento, modelos, IA, multilingue y exportacion.
- Resultados y mejoras implementadas.
- Desarrollo tecnico de la aplicacion.
- Discusion, limitaciones y conclusiones.
- Bibliografia.
- Anexos con manual de usuario, codigo, capturas y material complementario.

## Planificacion del anteproyecto

La estimacion total es de 300 horas:

| Fase | Horas |
| --- | ---: |
| Preparacion | 30 |
| Analisis y diseno | 60 |
| Desarrollo e implementacion | 120 |
| Pruebas y validacion | 50 |
| Documentacion y manuales | 25 |
| Evaluacion final y conclusiones | 15 |

## Estado inicial detectado

- La plantilla ya contiene texto avanzado en introduccion, objetivos, antecedentes, recursos, requisitos y analisis.
- Los capitulos de marco teorico, restricciones, diseno, implementacion, pruebas y conclusiones necesitan desarrollo.
- `__memoria.tex` se ha corregido para incluir `Unidades/11-Implementación/info.tex`, que estaba fuera por un `\input` incompleto.
- `anteproyecto_extract.txt` contiene el texto extraido del PDF para consultar requisitos y bibliografia de partida.

## Siguiente trabajo recomendado

1. Revisar y completar el marco teorico con las tecnicas reales del proyecto: Kaplan-Meier, Cox, Log-Rank, Weibull, Exponencial y Random Survival Forest.
2. Documentar la arquitectura real del codigo usando los modulos del repositorio (`cargaDataset.py`, `layout.py`, `analysis_callbacks.py`, `pdf_exporter.py`, etc.).
3. Pasar el manual de usuario existente de `doc/MANUAL DE USUARIO.pdf` al anexo de la memoria o resumirlo en LaTeX.
4. Depurar `referencias.bib`, porque contiene referencias heredadas no relacionadas con este TFG.
5. Anadir capturas reales de la aplicacion y resultados exportados cuando la parte funcional este cerrada.
