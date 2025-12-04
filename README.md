# 📄 Procesador de Documentos Avanzado con OCR, Metadatos y RAG

Esta es una aplicación Streamlit avanzada y modular que permite a los usuarios subir documentos (PDF, JPG, PNG), extraer texto mediante OCR, aplicar preprocesamiento avanzado a imágenes, gestionar metadatos de diversas formas y, lo más importante, interactuar con el contenido del documento a través de un sistema de Preguntas y Respuestas (RAG - Retrieval Augmented Generation).

## ✨ Características Principales

*   **Carga de Documentos:** Soporte para archivos PDF, JPG y PNG con una interfaz sencilla.
*   **Extracción de Texto Robusta:**
    *   **PDFs Nativos:** Extrae texto directamente de PDFs con capa de texto, garantizando máxima precisión.
    *   **OCR con Tesseract:** Utiliza Tesseract OCR para extraer texto de imágenes (JPG/PNG) y PDFs escaneados.
    *   **Preprocesamiento Avanzado de Imágenes:** Incluye técnicas avanzadas para mejorar la precisión del OCR, como umbralización, corrección de sesgo (deskewing), y mejora de contraste.
*   **Gestión de Metadatos Flexible:**
    *   **Manual:** Interfaz intuitiva para que el usuario introduzca metadatos clave-valor.
    *   **Extracción por Reglas (Regex):** Permite definir y aplicar patrones de expresiones regulares personalizadas para extraar campos específicos de documentos estructurados (ej., números de factura, fechas, totales).
    *   **Generación con Inteligencia Artificial (Gemini AI):** Integra la API de Google Gemini para analizar el texto extraído y sugerir automáticamente metadatos como título, autor, tipo de documento, palabras clave y un resumen conciso.
*   **Sistema de Preguntas y Respuestas (RAG):**
    *   **Indexación de Documentos:** Divide el texto extraído en \"chunks\" (fragmentos), los vectoriza utilizando embeddings de Gemini y los almacena en un índice vectorial **ChromaDB** con almacenamiento persistente.
    *   **Generación Aumentada:** Permite a los usuarios hacer preguntas sobre el contenido del documento. El sistema recupera los chunks más relevantes del índice ChromaDB y los utiliza como contexto para que Gemini AI genere respuestas precisas y basadas en el documento.
    *   **Visualización de Fuentes:** Muestra los fragmentos específicos del documento que se utilizaron para responder a la pregunta.
*   **Test de Recuperación de Chunks:** Una herramienta integrada que permite probar consultas directamente sobre el índice ChromaDB, mostrando los chunks recuperados y su \"similitud\" a la consulta, ayudando a entender cómo funciona el RAG.
*   **Visualización y Descarga:** Muestra el texto extraído y los metadatos. Permite descargar ambos en formatos `.txt` y `.json`.
*   **Arquitectura Modular:** El código está organizado en módulos para una mejor mantenibilidad, escalabilidad y claridad.
*   **Gestión de Errores y Logging:** Sistema robusto de manejo de errores y registro de actividades.
*   **Mejora de Procesamiento de Imágenes:** Técnicas avanzadas para mejorar la calidad del OCR en imágenes de baja calidad.

## 🛠️ Tecnologías Utilizadas

*   **Python:** Lenguaje de programación principal.
*   **Streamlit:** Para la creación rápida de la interfaz de usuario interactiva y el backend ligero.
*   **Tesseract OCR (`pytesseract`):** Motor de reconocimiento óptico de caracteres.
*   **Pillow (PIL Fork):** Para manipulación y preprocesamiento de imágenes.
*   **OpenCV (`cv2`):** Para técnicas avanzadas de preprocesamiento de imágenes.
*   **PyMuPDF (`fitz`):** Para el manejo eficiente de documentos PDF, incluyendo extracción directa de texto y renderizado.
*   **Google Generative AI (`google-generativeai`):** Para la integración con la API de Gemini (modelos `gemini-pro` para generación de texto y `embedding-001` para embeddings).
*   **`re` (módulo de Python):** Para la aplicación de expresiones regulares en la extracción de metadatos.
*   **NumPy:** Para operaciones numéricas, especialmente con embeddings.
*   **ChromaDB:** Base de datos vectorial para el almacenamiento persistente de embeddings y chunks del sistema RAG.
*   **python-dotenv:** Para la gestión de variables de entorno.

## ✨ Características Mejoradas

*   **Persistencia de datos RAG:** Los documentos indexados se almacenan persistentemente en ChromaDB, lo que permite consultas incluso después de reiniciar la aplicación.
*   **Seguridad mejorada:** Archivos de configuración sensible están excluidos del control de versiones y se incluye un archivo .env.dist para guiar la configuración.
*   **Validación robusta:** Validación mejorada de entradas del usuario y manejo de errores en todos los módulos.
*   **Soporte Docker:** Incluye Dockerfile y .dockerignore para despliegue contenedorizado.
*   **Documentación completa:** Mejora significativa de los docstrings y comentarios en el código.
*   **Pruebas unitarias extendidas:** Pruebas mejoradas para cubrir casos de error y funcionalidades críticas.

## 🚀 Cómo Empezar

Sigue estos pasos para configurar y ejecutar la aplicación en tu entorno local.

### 1. Requisitos Previos

*   Python 3.8+ (para instalación local)
*   Tesseract OCR instalado en tu sistema operativo (ver sección 2) - No es necesario si usas Docker
*   Acceso a la API de Google Gemini y una clave API (ver sección 5).

### 2. Instalación de Tesseract OCR (solo para instalación local)

Tesseract es una dependencia externa esencial para instalación local.

*   **En sistemas basados en Debian/Ubuntu:**
    ```bash
    sudo apt update
    sudo apt install tesseract-ocr
    sudo apt install tesseract-ocr-spa # Para soporte del idioma español
    ```
*   **En macOS (usando Homebrew):**
    ```bash
    brew install tesseract
    brew install tesseract-lang # Para instalar idiomas adicionales como el español
    ```
*   **En Windows:**
    Descarga el instalador desde la página oficial de Tesseract OCR (recomendado el instalador de UB Mannheim): [https://tesseract-ocr.github.io/tessdoc/Installation.html](https://tesseract-ocr.github.io/tessdoc/Installation.html). Durante la instalación, asegúrate de seleccionar el idioma español y, si no lo añades al PATH del sistema, deberás especificar la ruta completa al ejecutable `tesseract.exe` en el archivo `modules/config.py`.

### 3. Opción A: Ejecución con Docker (recomendada)

Docker simplifica el despliegue ya que incluye todas las dependencias necesarias:

1. Asegúrate de tener Docker instalado en tu sistema
2. Crea un archivo `.env` como se describe en la sección 5
3. Desde el directorio raíz del proyecto, construye la imagen:
   ```bash
   docker build -t metadatos-ocr .
   ```
4. Ejecuta el contenedor con tu clave API:
   ```bash
   docker run -p 8501:8501 -e GOOGLE_API_KEY=tu_clave_api_de_gemini_aqui metadatos-ocr
   ```
5. Accede a la aplicación en `http://localhost:8501`

### 4. Opción B: Instalación Local

Sigue estos pasos si prefieres instalar localmente:

1.  **Clona este repositorio (o descarga los archivos):**
    ```bash
    git clone https://github.com/tu-usuario/nombre-del-repo.git # Reemplaza con tu repo
    cd nombre-del-repo
    ```
2.  **Crea un entorno virtual (muy recomendado):**
    ```bash
    python -m venv venv
    ```
3.  **Activa el entorno virtual:**
    *   **Windows:** `.\\venv\\Scripts\\activate`
    *   **macOS/Linux:** `source venv/bin/activate`

4.  **Instala las dependencias de Python:**
    ```bash
    pip install -r requirements.txt
    ```

### 5. Configuración de la API de Gemini

Para utilizar la extracción de metadatos con Gemini y el sistema RAG, necesitas una clave API:

1.  Obtén tu clave API desde [Google AI Studio](https://aistudio.google.com/app/apikey).
2.  Crea un archivo `.env` en la raíz del proyecto con el siguiente contenido:
    ```
    GOOGLE_API_KEY=tu_clave_api_de_gemini_aqui
    GEMINI_MODEL=gemini-1.5-flash
    GEMINI_EMBEDDING_MODEL=models/text-embedding-004
    ```
    **Importante:** No subas el archivo `.env` al repositorio. El archivo `.gitignore` ya lo excluye por seguridad.
3.  Opcionalmente, puedes crear un directorio `.streamlit` en la raíz de tu proyecto (al mismo nivel que `app.py`).
4.  Dentro del directorio `.streamlit`, crea un archivo llamado `secrets.toml`.
5.  Añade tu clave API a este archivo de la siguiente manera:
    ```toml
    # .streamlit/secrets.toml
    GEMINI_API_KEY = "tu_clave_api_de_gemini_aqui"
    ```
    Este archivo es útil para despliegue en Streamlit Cloud, pero también debe mantenerse seguro y no ser compartido públicamente.

### 6. Ajustes de Configuración (Opcional)

Puedes revisar y ajustar las configuraciones en `modules/config.py`:
*   `TESSERACT_PATH`: La ruta al ejecutable de Tesseract si no está en el PATH del sistema.
*   `TESSDATA_PREFIX`: La ruta a la carpeta `tessdata` de Tesseract.
*   `CHUNK_SIZE`, `CHUNK_OVERLAP`: Parámetros para la división de texto en el sistema RAG.
*   `RAG_NUM_RESULTS`: Número de fragmentos que se recuperan para contextualizar la respuesta RAG.
*   `DEFAULT_DPI`: Resolución para preprocesamiento de imágenes.
*   `CONTRAST_ENHANCEMENT`: Habilitar/deshabilitar mejora de contraste.

### 7. Ejecutar la Aplicación

**Opción A: Con Docker (recomendada):**
1. Asegúrate de tener Docker instalado en tu sistema
2. Desde el directorio raíz del proyecto, construye la imagen:
   ```bash
   docker build -t metadatos-ocr .
   ```
3. Ejecuta el contenedor con tu clave API:
   ```bash
   docker run -p 8501:8501 -e GOOGLE_API_KEY=tu_clave_api_de_gemini_aqui metadatos-ocr
   ```
4. Accede a la aplicación en `http://localhost:8501`

**Opción B: Instalación local:**
Una vez que todas las dependencias y configuraciones estén listas, ejecuta la aplicación Streamlit desde tu terminal (asegurándote de que tu entorno virtual esté activado):

```bash
streamlit run app.py
```

Esto abrirá la aplicación en tu navegador web predeterminado (generalmente en http://localhost:8501).

## 💡 Uso de la Aplicación

1. **Carga un Documento:** En la barra lateral izquierda, usa el cargador de archivos para subir un PDF, JPG o PNG.
2. **Pestaña \"🏷️ Metadatos y Extracción\":**
    * Texto Extraído: Visualiza el texto obtenido mediante OCR o extracción directa.
    * Metadatos:
        * Manual: Edita los campos de metadatos directamente.
        * Por Reglas (Regex): Configura y aplica tus propias expresiones regulares desde la barra lateral para extraer datos estructurados.
        * Con Gemini AI: Haz clic en \"✨ Generar Metadatos con Gemini\" para que la IA sugiera metadatos automáticamente.
    * Descarga: Guarda el texto y los metadatos en archivos .txt y .json respectivamente.
3. **Pestaña \"🧠 RAG & Preguntas\":**
    * Indexar Documento: Si el documento no está indexado para RAG, aparecerá un botón \"🚀 Indexar '[nombre del documento]' para RAG\". Haz clic en él para procesar el documento para el sistema de preguntas y respuestas.
    * Pregunta al Documento: Una vez indexado, escribe tu pregunta en el campo de texto y haz clic en \"Obtener Respuesta RAG\". La IA generará una respuesta basada exclusivamente en el contenido de tu documento.
    * Fuentes: Expande las secciones de \"Fuentes\" para ver los fragmentos de texto del documento que se utilizaron para construir la respuesta.
    * 🔍 Test de Recuperación de Chunks: En esta misma pestaña, puedes usar la sección de test para introducir una consulta y ver directamente los N chunks más relevantes que el sistema recuperaría de tu documento, junto con sus distancias de similitud. Esto es útil para depurar y entender el comportamiento del RAG.

## 🧪 Pruebas

Para ejecutar las pruebas unitarias:

```bash
python -m unittest discover tests
```

## 📚 Estructura del Proyecto

```
Metadatos OCR/
│
├── app.py                 # Aplicación Streamlit principal
├── requirements.txt       # Dependencias del proyecto
├── .env                  # Variables de entorno (no incluido en el repo)
├── .streamlit/
│   └── secrets.toml      # Claves API para Streamlit Cloud
├── modules/              # Módulos personalizados
│   ├── __init__.py
│   ├── config.py         # Configuración del proyecto
│   ├── ocr.py           # Procesamiento OCR
│   ├── metadata.py      # Extracción de metadatos
│   ├── rag.py           # Sistema RAG
│   └── utils.py         # Funciones utilitarias
├── tests/                # Pruebas unitarias
│   └── test_modules.py
└── README.md
```

## 🔄 Próximos Pasos / Mejoras Futuras

*   Gestión de Múltiples Documentos: Extender el RAG para buscar en un corpus de varios documentos subidos previamente.
*   Extracción de Tablas: Integrar librerías como Camelot o Tabula-py para extraer datos tabulares de PDFs.
*   Interfaz de Usuario Mejorada: Un editor de reglas Regex más interactivo y validación en tiempo real.
*   Gestión de Usuarios: Implementar autenticación y autorización para entornos multiusuario.
*   Soporte para más formatos: Añadir soporte para formatos como TXT, RTF y otros tipos de documentos.

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Si tienes ideas para mejoras, encuentras errores o quieres añadir nuevas funcionalidades, no dudes en abrir un issue o enviar un pull request.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.