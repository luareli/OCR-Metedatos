import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the OCR metadata processor"""
    
    # Tesseract configuration
    TESSERACT_PATH = os.getenv('TESSERACT_PATH', '/usr/bin/tesseract')
    TESSDATA_PREFIX = os.getenv('TESSDATA_PREFIX', '/usr/share/tesseract-ocr/tessdata')
    
    # Gemini API configuration
    GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
    GEMINI_EMBEDDING_MODEL = os.getenv('GEMINI_EMBEDDING_MODEL', 'models/text-embedding-004')
    
    # RAG configuration
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '200'))
    RAG_NUM_RESULTS = int(os.getenv('RAG_NUM_RESULTS', '5'))
    
    # Preprocessing configuration
    DEFAULT_DPI = int(os.getenv('DEFAULT_DPI', '300'))
    CONTRAST_ENHANCEMENT = os.getenv('CONTRAST_ENHANCEMENT', 'True').lower() == 'true'
    
    # File upload limits
    MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '50'))
    ALLOWED_EXTENSIONS = set(os.getenv('ALLOWED_EXTENSIONS', 'pdf,jpg,jpeg,png,docx,xlsx,xls').split(','))
    
    @classmethod
    def validate(cls):
        """Validate that the required configuration is present"""
        errors = []
        warnings = []

        if not cls.GEMINI_API_KEY:
            errors.append("❌ GEMINI_API_KEY is not set in environment variables or .env file")
        else:
            # Basic check for API key format (should be a long string)
            if len(cls.GEMINI_API_KEY) < 20:
                warnings.append("⚠️ GEMINI_API_KEY seems too short, please verify it's correct")

        if not cls.TESSERACT_PATH or not os.path.exists(cls.TESSERACT_PATH):
            warnings.append(f"⚠️ TESSERACT_PATH '{cls.TESSERACT_PATH}' does not exist, OCR may fail. Please install Tesseract OCR or update TESSERACT_PATH in .env")

        if cls.CHUNK_SIZE <= 0:
            errors.append("❌ CHUNK_SIZE must be a positive integer")

        if cls.CHUNK_OVERLAP < 0:
            errors.append("❌ CHUNK_OVERLAP must be a non-negative integer")

        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            errors.append("❌ CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

        if cls.MAX_FILE_SIZE_MB <= 0:
            errors.append("❌ MAX_FILE_SIZE_MB must be a positive number")

        if not cls.ALLOWED_EXTENSIONS:
            errors.append("❌ ALLOWED_EXTENSIONS cannot be empty")

        # Check if Gemini models exist (basic validation)
        valid_models = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.0-flash-exp']
        if cls.GEMINI_MODEL not in valid_models:
            warnings.append(f"⚠️ GEMINI_MODEL '{cls.GEMINI_MODEL}' may not be valid. Recommended: {valid_models[0]}")

        return errors, warnings

# Initialize the config
config = Config()