"""Paquete principal de módulos para el procesador de OCR y metadatos"""

from .config import config
from .ocr import OCRProcessor
from .metadata import MetadataExtractor
from .rag import RAGSystem
from .utils import setup_logging

__all__ = ['config', 'OCRProcessor', 'MetadataExtractor', 'RAGSystem', 'setup_logging']