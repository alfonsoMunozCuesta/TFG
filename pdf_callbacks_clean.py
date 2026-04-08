"""
Compatibilidad para exportación de análisis a PDF.

Esta variante delega en pdf_callbacks.py para evitar divergencias entre dos
implementaciones distintas de los mismos callbacks.
"""

from pdf_callbacks import register_pdf_export_callbacks


__all__ = ["register_pdf_export_callbacks"]
