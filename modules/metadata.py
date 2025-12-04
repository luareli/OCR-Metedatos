import json
import re
import google.generativeai as genai
from typing import Dict, List, Any, Optional
from modules.config import config
from modules.utils import get_logger

logger = get_logger(__name__)


class MetadataExtractor:
    """Handles metadata extraction using various methods

    This class provides functionality to extract metadata from text using
    AI (Gemini), regex rules, or manual input. It also provides methods
    to merge metadata from different sources with proper priority.
    """

    def __init__(self):
        """Initialize the metadata extractor with AI model if API key is configured."""
        if config.GEMINI_API_KEY:
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel(config.GEMINI_MODEL)
        else:
            self.gemini_model = None

    def extract_with_gemini(self, text_content: str) -> Dict[str, Any]:
        """Extract metadata using Gemini AI"""
        try:
            # Validate inputs
            if not isinstance(text_content, str):
                raise TypeError("Text content must be a string")

            if not self.gemini_model:
                error_msg = "Gemini API key not configured"
                logger.error(error_msg)
                return {"error": error_msg}

            if not text_content or not text_content.strip():
                error_msg = "No text to process with Gemini"
                logger.error(error_msg)
                return {"error": error_msg}

            # Limit text length to prevent API issues
            max_length = 100000  # Adjust based on Gemini's limits
            if len(text_content) > max_length:
                logger.warning(f"Text content is too long ({len(text_content)} > {max_length}), truncating...")
                text_content = text_content[:max_length]

            prompt = f"""
            Eres un asistente experto en extracción de metadatos de documentos. Analiza el siguiente texto y extrae la siguiente información en formato JSON. Si no encuentras un dato, déjalo como null o cadena vacía. Trata de ser lo más preciso posible.

            Formato JSON requerido:
            {{
                "titulo": "Título principal del documento",
                "autor": "Autor o fuente del documento",
                "fecha_documento": "Fecha más relevante mencionada en el documento (formato YYYY-MM-DD si es posible)",
                "tipo_documento": "Clasificación general (ej. 'Factura', 'Contrato', 'Reporte', 'Carta', 'Artículo', 'Acta', 'Memorándum')",
                "palabras_clave": ["palabra1", "palabra2", "palabra3", ...],
                "resumen_corto": "Un resumen conciso de 2-3 oraciones"
            }}

            Texto del documento:
            ---
            {text_content}
            ---
            """

            response = self.gemini_model.generate_content(prompt)
            json_str = response.text.replace("```json", "").replace("```", "").strip()
            metadata = json.loads(json_str)
            return metadata
        except json.JSONDecodeError as e:
            error_msg = f"Error decoding JSON response from Gemini: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"Error with Gemini: {e}"
            logger.error(error_msg)
            return {"error": error_msg}

    def extract_with_rules(self, text_content: str, rules: Dict[str, str]) -> Dict[str, Any]:
        """Extract metadata using regex rules"""
        extracted_data = {}

        for key, pattern in rules.items():
            try:
                match = re.search(pattern, text_content, re.IGNORECASE | re.MULTILINE)
                if match:
                    # If pattern has named capture groups, use them
                    if match.groupdict():
                        for group_name, value in match.groupdict().items():
                            extracted_data[f"{key}_{group_name}"] = value.strip() if value else ""
                    # If pattern has unnamed capture groups, take the first one
                    elif match.groups():
                        extracted_data[key] = match.group(1).strip() if match.group(1) else ""
                    # If no groups, take the complete match
                    else:
                        extracted_data[key] = match.group(0).strip()
            except re.error as e:
                error_msg = f"Error in regex pattern for '{key}': '{pattern}'. Error: {e}"
                logger.warning(error_msg)  # Use warning since it's likely a user issue with the pattern
            except Exception as e:
                error_msg = f"Unexpected error applying rule for '{key}': {e}"
                logger.error(error_msg)

        return extracted_data

    def merge_metadata(self, manual_metadata: Dict[str, Any], 
                      gemini_metadata: Optional[Dict[str, Any]] = None,
                      rule_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Merge metadata from different sources with priority: manual > rules > gemini"""
        merged = manual_metadata.copy()
        
        # Add rule metadata with priority over Gemini but lower than manual
        if rule_metadata:
            for key, value in rule_metadata.items():
                if key not in merged or not merged[key]:  # Only add if not already set manually
                    merged[key] = value
        
        # Add Gemini metadata with lowest priority
        if gemini_metadata and "error" not in gemini_metadata:
            for key, value in gemini_metadata.items():
                # Map Gemini keys to more user-friendly keys
                mapped_key = self._map_gemini_key(key)
                if mapped_key not in merged or not merged[mapped_key]:  # Only add if not already set
                    merged[mapped_key] = value
        
        return merged

    def _map_gemini_key(self, key: str) -> str:
        """Map Gemini response keys to user-friendly keys"""
        mapping = {
            "titulo": "Título",
            "autor": "Autor",
            "fecha_documento": "Fecha del Documento",
            "tipo_documento": "Tipo de Documento", 
            "palabras_clave": "Palabras Clave",
            "resumen_corto": "Resumen"
        }
        return mapping.get(key, key)