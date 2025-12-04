import streamlit as st
from PIL import Image
import json

# Importar módulos personalizados
from modules.config import config
from modules.ocr import OCRProcessor
from modules.metadata import MetadataExtractor
from modules.rag import RAGSystem
from modules.utils import setup_logging, validate_file_type, format_metadata_for_download, validate_file_size, calculate_file_hash, load_cached_result, save_processed_result

# Setup logging
logger = setup_logging()

# Validar configuración
config_errors, config_warnings = config.validate()
if config_errors:
    st.error("Errores críticos de configuración:")
    for error in config_errors:
        st.error(error)
    st.stop()

if config_warnings:
    st.warning("Advertencias de configuración:")
    for warning in config_warnings:
        st.warning(warning)

# Inicializar procesadores
# Inicializar procesadores con caché
@st.cache_resource
def get_processors():
    try:
        ocr = OCRProcessor()
        metadata = MetadataExtractor()
        rag = RAGSystem(persist_directory="./chroma_db")  # Usar directorio persistente
        return ocr, metadata, rag
    except Exception as e:
        logger.error(f"Error al inicializar los procesadores: {e}")
        return None, None, None

ocr_processor, metadata_extractor, rag_system = get_processors()

if not ocr_processor or not metadata_extractor or not rag_system:
    st.error("Error crítico al inicializar los componentes del sistema. Revisa los logs.")
    st.stop()

# Inicializar variables de session_state
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'document_metadata' not in st.session_state:
    st.session_state.document_metadata = {}
if 'document_name' not in st.session_state:
    st.session_state.document_name = ""
if 'current_rag_system' not in st.session_state:
    st.session_state.current_rag_system = None
if 'document_indexed' not in st.session_state:
    st.session_state.document_indexed = False
if 'document_id' not in st.session_state:
    st.session_state.document_id = None

# Page configuration
st.set_page_config(
    page_title="Procesador de Documentos con OCR",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("📄 Procesador de Documentos Avanzado con OCR y RAG")
st.markdown("""
Esta aplicación te permite subir documentos (PDF, JPG, PNG, DOCX, XLSX, XLS), extraer texto mediante OCR,
generar metadatos y consultar el contenido mediante un sistema de
preguntas y respuestas basado en RAG (Retrieval Augmented Generation).
""")

# Add a brief description in the sidebar
with st.sidebar:
    st.info("### Instrucciones de uso:\n"
            "1. Sube un documento (PDF, JPG, PNG, DOCX, XLSX, XLS)\n"
            "2. Accede a la pestaña 'Metadatos' para ver el texto extraído\n"
            "3. Genera metadatos manualmente o con IA\n"
            "4. Ve a la pestaña 'RAG' para hacer preguntas al documento")

# --- Configuración de pestañas ---
tab_metadatos, tab_rag = st.tabs(["🏷️ Metadatos y Extracción", "🧠 RAG & Preguntas"])


# --- Sidebar para configuración o carga ---
st.sidebar.header("📁 Cargar Documento")
uploaded_file = st.sidebar.file_uploader(
    "Selecciona un documento para procesar",
    type=["pdf", "jpg", "jpeg", "png", "docx", "xlsx", "xls"],
    accept_multiple_files=False,
    help="Soporta archivos PDF, JPG, PNG, DOCX, XLSX, XLS de hasta 50MB"
)

# Show file info if uploaded
if uploaded_file:
    file_details = {
        "Nombre": uploaded_file.name,
        "Tamaño": f"{len(uploaded_file.getvalue()) / (1024 * 1024):.2f} MB",
        "Tipo": uploaded_file.type
    }
    st.sidebar.json(file_details)

st.sidebar.header("⚙️ Configuración de Reglas (Regex)")
st.sidebar.markdown("Define tus propias reglas de extracción de metadatos usando expresiones regulares.")

# Inicializar reglas en session_state si no existen
if 'regex_rules' not in st.session_state:
    st.session_state.regex_rules = {
        "Número de Factura": r"Factura Nº\s*(\S+)",
        "Fecha de Emisión": r"(?:Fecha|Fecha de Emisión):\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
        "Total a Pagar": r"(?:Total|Total a Pagar|Importe Total):\s*([€$]?\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)",
        "Nombre Cliente": r"(?:Cliente|Nombre Cliente):\s*([^\n]+)",
        "CIF/NIF": r"(?:CIF|NIF|VAT|TAX ID):\s*([A-Z0-9\-]{9,})"
    }

# Mostrar las reglas actuales y permitir edición/adición
col_key, col_pattern = st.sidebar.columns([0.4, 0.6])
col_key.write("**Clave**")
col_pattern.write("**Patrón Regex**")

rule_index_to_delete = None
for i, (key, pattern) in enumerate(st.session_state.regex_rules.items()):
    cols_edit = st.sidebar.columns([0.35, 0.55, 0.1])
    new_key = cols_edit[0].text_input("Clave", value=key, key=f"key_{i}", label_visibility="collapsed")
    new_pattern = cols_edit[1].text_input("Patrón", value=pattern, key=f"pattern_{i}", label_visibility="collapsed")
    
    if cols_edit[2].button("🗑️", key=f"delete_rule_{i}"):
        rule_index_to_delete = i

    # Actualizar la regla si ha cambiado (sin cambiar la clave directamente en el diccionario mientras iteramos)
    if new_key != key or new_pattern != pattern:
        st.session_state.regex_rules[new_key] = new_pattern
        if new_key != key: # Si la clave cambió, eliminar la antigua
             del st.session_state.regex_rules[key]

# Eliminar la regla después de la iteración
if rule_index_to_delete is not None:
    del st.session_state.regex_rules[list(st.session_state.regex_rules.keys())[rule_index_to_delete]]
    st.session_state.regex_rules = st.session_state.regex_rules # Trigger rerun with updated state
    st.experimental_rerun() # For immediate refresh


if st.sidebar.button("➕ Añadir Nueva Regla"):
    # Añadir una nueva regla con una clave única temporal
    new_temp_key = f"Nueva Regla {len(st.session_state.regex_rules) + 1}"
    st.session_state.regex_rules[new_temp_key] = r"Ejemplo de Patrón"
    st.experimental_rerun() # Para que aparezca inmediatamente

if uploaded_file is not None:
    # Validate file type and size
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in config.ALLOWED_EXTENSIONS:
        st.error(f"Tipo de archivo no permitido. Extensiones permitidas: {config.ALLOWED_EXTENSIONS}")
        uploaded_file = None
    elif not validate_file_size(uploaded_file, config.MAX_FILE_SIZE_MB):
        st.error(f"El archivo es demasiado grande. Tamaño máximo: {config.MAX_FILE_SIZE_MB} MB")
        uploaded_file = None
    elif uploaded_file.name != st.session_state.document_name:
        st.session_state.document_name = uploaded_file.name
        st.session_state.extracted_text = ""
        st.session_state.document_metadata = {} # Reset metadatos al subir nuevo archivo
        st.session_state.current_rag_system = None
        st.session_state.document_indexed = False

    document_name = st.session_state.document_name
    st.sidebar.write(f"Archivo cargado: {document_name}")

    file_extension = document_name.split('.')[-1].lower()

    if not st.session_state.extracted_text:
        # Calculate file hash for caching
        uploaded_file.seek(0)  # Reset to beginning
        file_hash = calculate_file_hash(uploaded_file)

        # Check if result is already cached
        cached_result = load_cached_result(file_hash)
        if cached_result:
            logger.info(f"Using cached result for file {document_name}")
            st.session_state.extracted_text = cached_result.get("extracted_text", "")
            st.session_state.document_metadata = cached_result.get("metadata", {})
            st.success("Documento procesado (usando caché)")
        else:
            # Determine if the file is large and show appropriate message
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            processing_msg = f"Procesando {document_name} ({file_size_mb:.1f} MB)... Esto puede tardar un momento."

            with st.spinner(processing_msg):
                try:
                    # Reset the file pointer to the beginning
                    uploaded_file.seek(0)

                    # Process the file using the OCR module
                    # Pass the file object directly to handle both file-like objects and bytes appropriately
                    extracted_text = ocr_processor.process_file(uploaded_file, file_extension)
                    st.session_state.extracted_text = extracted_text

                    # Save to cache
                    save_processed_result(file_hash, extracted_text, st.session_state.document_metadata)

                    st.success("Documento procesado y texto extraído.")

                except Exception as e:
                    logger.error(f"Error procesando el documento: {e}")
                    st.error(f"Error al procesar el documento: {e}")
    
    extracted_text = st.session_state.extracted_text

    with tab_metadatos:
        # --- Sección de Resultados y Metadatos ---
        if extracted_text:
            st.subheader("📝 Texto Extraído por OCR")

            # Add a progress indicator for text display
            with st.expander("Ver texto extraído", expanded=True):
                st.text_area("Texto del documento:", extracted_text, height=300, key="ocr_text_area")

            st.subheader("🏷️ Añadir Metadatos")
            st.write("Introduce los metadatos clave-valor para este documento, genéralos automáticamente o usa reglas.")

            # Use containers for better organization
            col_buttons_meta = st.columns([1, 1])

            with col_buttons_meta[0]:
                if st.button("🔍 Extraer Metadatos por Reglas", type="secondary"):
                    if st.session_state.regex_rules:
                        with st.spinner("Aplicando reglas de extracción..."):
                            try:
                                rule_suggestions = metadata_extractor.extract_with_rules(extracted_text, st.session_state.regex_rules)
                                if rule_suggestions:
                                    # Actualizar los metadatos del documento con las reglas
                                    for key, value in rule_suggestions.items():
                                        st.session_state.document_metadata[key] = value
                                    st.success(f"✅ Metadatos extraídos por reglas: {len(rule_suggestions)} campos encontrados")
                                else:
                                    st.info("ℹ️ No se encontraron metadatos con las reglas actuales.")
                            except Exception as e:
                                logger.error(f"Error aplicando reglas de metadatos: {e}")
                                st.error(f"❌ Error aplicando reglas de metadatos: {e}")
                    else:
                        st.warning("⚠️ No hay reglas de expresión regular definidas.")

            with col_buttons_meta[1]:
                if st.button("✨ Generar Metadatos con IA", type="primary"):
                    with st.spinner("Generando metadatos con IA (Gemini)..."):
                        try:
                            gemini_suggestions = metadata_extractor.extract_with_gemini(extracted_text)
                            if gemini_suggestions and "error" not in gemini_suggestions:
                                # Pre-rellenar los metadatos actuales con las sugerencias de Gemini
                                # Las reglas tienen prioridad si ya se han aplicado
                                st.session_state.document_metadata.update({
                                    "Título": st.session_state.document_metadata.get("Título") or gemini_suggestions.get("titulo", ""),
                                    "Autor": st.session_state.document_metadata.get("Autor") or gemini_suggestions.get("autor", ""),
                                    "Fecha del Documento": st.session_state.document_metadata.get("Fecha del Documento") or gemini_suggestions.get("fecha_documento", ""),
                                    "Tipo de Documento": st.session_state.document_metadata.get("Tipo de Documento") or gemini_suggestions.get("tipo_documento", ""),
                                    "Palabras Clave": st.session_state.document_metadata.get("Palabras Clave") or ", ".join(gemini_suggestions.get("palabras_clave", [])),
                                    "Resumen": st.session_state.document_metadata.get("Resumen") or gemini_suggestions.get("resumen_corto", ""),
                                })
                                st.success("✅ Metadatos generados por IA. ¡Revísalos y edítalos si es necesario!")
                            elif gemini_suggestions and "error" in gemini_suggestions:
                                st.error(f"❌ No se pudieron generar metadatos con IA: {gemini_suggestions['error']}")
                            else:
                                st.warning("ℹ️ La IA no pudo generar metadatos. Por favor, introduce manualmente.")
                        except Exception as e:
                            logger.error(f"Error generando metadatos con Gemini: {e}")
                            st.error(f"❌ Error generando metadatos con IA: {e}")

            # Campos de metadatos (pre-rellenados por Gemini/Reglas o entrada manual)
            st.session_state.document_metadata['Nombre del Archivo'] = st.session_state.get('document_name', document_name)
            st.session_state.document_metadata['Fecha de Procesamiento'] = st.date_input(
                "Fecha de Procesamiento",
                value=st.session_state.document_metadata.get('Fecha de Procesamiento', None) or st.date_input("default_fecha_proc_val").today(),
                key="fecha_proc_input"
            )
            st.session_state.document_metadata['Título'] = st.text_input(
                "Título del Documento",
                value=st.session_state.document_metadata.get('Título', ""),
                key="titulo_input"
            )
            st.session_state.document_metadata['Autor'] = st.text_input(
                "Autor/Fuente",
                value=st.session_state.document_metadata.get('Autor', ""),
                key="autor_input"
            )
            st.session_state.document_metadata['Fecha del Documento'] = st.text_input(
                "Fecha del Documento (ej. YYYY-MM-DD)",
                value=st.session_state.document_metadata.get('Fecha del Documento', ""),
                key="fecha_doc_input"
            )
            st.session_state.document_metadata['Tipo de Documento'] = st.text_input(
                "Tipo de Documento (Ej: Factura, Contrato, Reporte)",
                value=st.session_state.document_metadata.get('Tipo de Documento', ""),
                key="tipo_doc_input"
            )
            st.session_state.document_metadata['Palabras Clave'] = st.text_input(
                "Palabras Clave (separadas por comas)",
                value=st.session_state.document_metadata.get('Palabras Clave', ""),
                key="keywords_input"
            )
            st.session_state.document_metadata['Resumen'] = st.text_area(
                "Resumen del Documento",
                value=st.session_state.document_metadata.get('Resumen', ""),
                height=100,
                key="resumen_input"
            )
            st.session_state.document_metadata['Notas'] = st.text_area(
                "Notas Adicionales",
                value=st.session_state.document_metadata.get('Notas', ""),
                key="notas_input"
            )

            st.subheader("Metadatos Actuales")
            st.json(st.session_state.document_metadata)

            # --- Descarga de Resultados ---
            st.subheader("📥 Descargar Resultados")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📄 Descargar Texto Extraído", key="download_text_btn", type="secondary"):
                    st.download_button(
                        label="📥 Descargar Archivo de Texto",
                        data=extracted_text,
                        file_name=f"{document_name}_texto.txt",
                        mime="text/plain"
                    )
            with col2:
                if st.button("🏷️ Descargar Metadatos", key="download_metadata_btn", type="secondary"):
                    try:
                        metadata_for_download = format_metadata_for_download(st.session_state.document_metadata)
                        # Handle date objects for JSON serialization
                        import json
                        from datetime import datetime

                        def json_serializer(obj):
                            if isinstance(obj, (datetime, datetime.date)):
                                return obj.isoformat()
                            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

                        metadata_json = json.dumps(metadata_for_download, indent=4, ensure_ascii=False, default=json_serializer)

                        # Create download button
                        st.download_button(
                            label="📥 Descargar Archivo de Metadatos",
                            data=metadata_json,
                            file_name=f"{document_name}_metadatos.json",
                            mime="application/json"
                        )
                    except Exception as e:
                        logger.error(f"Error al preparar metadatos para descarga: {e}")
                        st.error(f"❌ Error al preparar metadatos para descarga: {e}")
    
    # RAG Tab
    with tab_rag:
        st.subheader("🧠 Sistema RAG (Recuperación Aumentada por Generación)")
        
        if not st.session_state.extracted_text:
            st.info("Por favor, sube un documento en la pestaña de Metadatos para habilitar el sistema RAG.")
        else:
            # Initialize RAG system for this document if needed
            if not st.session_state.current_rag_system:
                st.session_state.current_rag_system = rag_system

            # Generate a unique document ID if not already generated
            if not st.session_state.document_id:
                import uuid
                st.session_state.document_id = str(uuid.uuid4())

            # Index the document if not already indexed
            if not st.session_state.document_indexed:
                if st.button(f"🚀 Indexar Documento para Búsqueda RAG", type="primary"):
                    with st.spinner("Indexando documento para RAG (esto puede tardar unos segundos)..."):
                        try:
                            st.session_state.current_rag_system.index_document(
                                st.session_state.extracted_text,
                                doc_id=st.session_state.document_id
                            )
                            st.session_state.document_indexed = True
                            st.success("✅ Documento indexado exitosamente para RAG!")
                            # Forzar actualización de la interfaz para mostrar el campo de preguntas
                            st.rerun()
                        except Exception as e:
                            logger.error(f"Error indexando documento para RAG: {e}")
                            st.error(f"❌ Error indexando documento para RAG: {e}")
                            # Actualizar el estado de indexado si falló
                            st.session_state.document_indexed = False
            else:
                st.success(f"✅ Documento '{document_name}' ya está indexado para RAG")

                # Query interface
                st.subheader("💬 Pregunta al Documento")
                query = st.text_input("Introduce tu pregunta sobre el contenido del documento:",
                                    placeholder="Ej: ¿Cuál es el monto total del documento?")

                if st.button("🔍 Obtener Respuesta del Documento", type="primary") and query:
                    with st.spinner("🔍 Buscando información relevante..."):
                        try:
                            # Make sure to use the correct RAG system instance and document ID
                            rag_response = st.session_state.current_rag_system.query_document(
                                query,
                                doc_id=st.session_state.document_id
                            )

                            if "error" in rag_response:
                                st.error(f"❌ Error en el sistema RAG: {rag_response['error']}")
                            else:
                                st.subheader("🤖 Respuesta del Documento")
                                st.success("✅ Respuesta generada exitosamente")
                                st.write(rag_response["response"])

                                # Show sources
                                with st.expander("🔍 Fragmentos Relevantes Usados"):
                                    for i, source in enumerate(rag_response["sources"]):
                                        st.write(f"**Fragmento {i+1}** (similitud: {source['similarity']:.3f}):")
                                        st.text_area(f"Contenido del fragmento {i+1}:",
                                                   value=source["chunk"], height=100, key=f"source_{i}")
                        except Exception as e:
                            logger.error(f"Error consultando RAG: {e}")
                            st.error(f"❌ Error consultando RAG: {e}")

                # Chunk retrieval test
                st.subheader("🔍 Test de Recuperación de Chunks")
                test_query = st.text_input("Prueba una consulta para ver los chunks recuperados:",
                                         key="test_query_input")

                if st.button("🔍 Probar Recuperación", key="test_retrieval_btn") and test_query:
                    try:
                        retrieved_chunks = st.session_state.current_rag_system.retrieve_chunks(
                            test_query,
                            doc_id=st.session_state.document_id
                        )

                        if retrieved_chunks:
                            st.write(f"Encontrados {len(retrieved_chunks)} chunks relevantes:")
                            for i, (chunk, similarity) in enumerate(retrieved_chunks):
                                with st.expander(f"Fragmento {i+1} (similitud: {similarity:.3f})"):
                                    st.text_area(f"", value=chunk.strip(), height=150, key=f"test_chunk_{i}")
                        else:
                            st.info("No se encontraron chunks relevantes para la consulta.")
                    except Exception as e:
                        logger.error(f"Error probando recuperación de chunks: {e}")
                        st.error(f"Error probando recuperación de chunks: {e}")

else:
    st.info("Sube un documento para empezar el procesamiento.")
    # Limpiar el estado de la sesión cuando no hay archivo cargado
    if st.session_state.extracted_text or st.session_state.document_metadata or st.session_state.document_name:
        st.session_state.extracted_text = ""
        st.session_state.document_metadata = {}
        st.session_state.document_name = ""
        st.session_state.current_rag_system = None
        st.session_state.document_indexed = False
        st.session_state.document_id = None


st.sidebar.markdown("---")
st.sidebar.markdown("Hecho con ❤️ usando Streamlit, Tesseract OCR y Gemini AI")
