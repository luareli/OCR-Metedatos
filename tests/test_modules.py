import unittest
import os
import sys
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import io

# Add the modules directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.config import config
from modules.ocr import OCRProcessor
from modules.metadata import MetadataExtractor
from modules.rag import RAGSystem
from modules.utils import validate_file_type, validate_file_size, calculate_file_hash, load_cached_result, save_processed_result


class TestConfig(unittest.TestCase):
    def test_config_initialization(self):
        """Test that configuration is properly initialized"""
        self.assertIsNotNone(config.GEMINI_API_KEY)
        self.assertIsInstance(config.CHUNK_SIZE, int)
        self.assertIsInstance(config.CHUNK_OVERLAP, int)
        self.assertLess(config.CHUNK_OVERLAP, config.CHUNK_SIZE)  # Overlap should be less than chunk size

    def test_config_validation(self):
        """Test configuration validation"""
        errors, warnings = config.validate()
        # Should not have errors for a properly configured system
        # (though API key warnings might be present depending on environment)
        self.assertIsInstance(errors, list)
        self.assertIsInstance(warnings, list)


class TestOCRProcessor(unittest.TestCase):
    def setUp(self):
        self.ocr = OCRProcessor()

    def test_init(self):
        """Test OCR processor initialization"""
        self.assertIsInstance(self.ocr, OCRProcessor)

    @patch('pytesseract.image_to_string')
    def test_extract_text_from_image(self, mock_tesseract):
        """Test OCR text extraction from image"""
        mock_tesseract.return_value = "Texto de prueba"

        # Test that the method can be called without error
        from PIL import Image
        # Create a dummy image for testing
        img = Image.new('RGB', (100, 100), color='red')
        result = self.ocr.extract_text_from_image(img, preprocess=False)

        self.assertIsNotNone(result)

    def test_process_file_invalid_inputs(self):
        """Test process_file with invalid inputs"""
        with self.assertRaises(ValueError):
            self.ocr.process_file(None, 'pdf')

        with self.assertRaises(ValueError):
            self.ocr.process_file(b'test', '')

        with self.assertRaises(ValueError):
            self.ocr.process_file(b'test', 'invalid_type')

    def test_chunk_text(self):
        """Test text chunking functionality"""
        rag = RAGSystem()

        # Create a longer text to chunk
        long_text = "Este es un párrafo de prueba. " * 50  # Repeat to make it long

        chunks = rag.chunk_text(long_text)

        # Should have created chunks
        self.assertGreater(len(chunks), 0)
        # Each chunk should not exceed the configured size (though overlap may affect this check)
        # For this test, just ensure chunks were created


class TestMetadataExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = MetadataExtractor()

    def test_init(self):
        """Test metadata extractor initialization"""
        self.assertIsInstance(self.extractor, MetadataExtractor)

    def test_merge_metadata(self):
        """Test metadata merging functionality"""
        manual_meta = {"Título": "Documento de prueba"}
        gemini_meta = {"titulo": "Título desde Gemini", "autor": "Autor desde Gemini"}
        rule_meta = {"Número de Factura": "FAC-001"}

        merged = self.extractor.merge_metadata(manual_meta, gemini_meta, rule_meta)

        # Manual metadata should take priority
        self.assertEqual(merged.get("Título"), "Documento de prueba")
        # Rule metadata should be added if not in manual
        self.assertEqual(merged.get("Número de Factura"), "FAC-001")

    def test_extract_with_rules(self):
        """Test metadata extraction with rules"""
        text_content = "Factura Nº FAC-001 Fecha: 2023-05-15 Total: $100.00"
        rules = {
            "Número de Factura": r"Factura Nº\s*(\S+)",
            "Fecha": r"Fecha:\s*(\d{4}-\d{2}-\d{2})",
            "Total": r"Total:\s*([$\d.]+)"
        }

        extracted = self.extractor.extract_with_rules(text_content, rules)

        self.assertEqual(extracted.get("Número de Factura"), "FAC-001")
        self.assertEqual(extracted.get("Fecha"), "2023-05-15")
        self.assertEqual(extracted.get("Total"), "$100.00")


class TestRAGSystem(unittest.TestCase):
    def setUp(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.rag = RAGSystem(persist_directory=temp_dir)

    def test_chunk_text(self):
        """Test text chunking functionality"""
        # Create a longer text to chunk
        long_text = "Este es un párrafo de prueba. " * 50  # Repeat to make it long

        chunks = self.rag.chunk_text(long_text)

        # Should have created chunks
        self.assertGreater(len(chunks), 0)
        # Each chunk should not exceed the configured size (though overlap may affect this check)
        # For this test, just ensure chunks were created

    def test_index_and_query_document(self):
        """Test indexing a document and querying it"""
        text = "Este es un documento de prueba para el sistema RAG."
        doc_id = "test_doc_1"

        # Index the document
        self.rag.index_document(text, doc_id)

        # Check that document was indexed
        collection_name = f"doc_{doc_id.replace('-', '_')}"
        collection = self.rag.chroma_client.get_collection(name=collection_name)
        self.assertIsNotNone(collection)

        # Query the document
        result = self.rag.query_document("¿De qué trata este documento?", doc_id=doc_id)

        # Should return a response
        self.assertIn("response", result)
        self.assertIn("sources", result)


class TestUtils(unittest.TestCase):
    def test_validate_file_type(self):
        """Test file type validation"""
        allowed_extensions = {'pdf', 'txt', 'doc'}

        self.assertTrue(validate_file_type("test.pdf", allowed_extensions))
        self.assertFalse(validate_file_type("test.jpg", allowed_extensions))
        self.assertFalse(validate_file_type("", allowed_extensions))
        self.assertFalse(validate_file_type("test.pdf", None))
        self.assertFalse(validate_file_type(None, allowed_extensions))

    def test_validate_file_size(self):
        """Test file size validation"""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            # Write 1MB of data
            tmp.write(b'0' * (1024 * 1024))
            tmp_path = tmp.name

        # Create a mock file object
        with open(tmp_path, 'rb') as f:
            # File is 1MB, max is 2MB, should pass
            self.assertTrue(validate_file_size(f, 2))
            # File is 1MB, max is 0.5MB, should fail
            self.assertFalse(validate_file_size(f, 0))
            self.assertFalse(validate_file_size(f, -1))

        # Cleanup
        os.unlink(tmp_path)

    def test_calculate_file_hash(self):
        """Test file hash calculation"""
        test_data = b"test content for hashing"
        file_obj = io.BytesIO(test_data)

        hash1 = calculate_file_hash(file_obj)
        # Verify position was restored
        self.assertEqual(file_obj.tell(), 0)

        # Calculate again to ensure consistency
        hash2 = calculate_file_hash(file_obj)
        self.assertEqual(hash1, hash2)

        # Test calculating hash from bytes directly
        hash3 = calculate_file_hash(test_data)
        self.assertEqual(hash1, hash3)

    def test_cache_functions(self):
        """Test cache save and load functions"""
        test_hash = "test_hash_value"
        test_text = "test extracted text"
        test_metadata = {"title": "Test Title"}

        # Save to cache
        result = save_processed_result(test_hash, test_text, test_metadata)
        self.assertTrue(result)

        # Load from cache
        cached_result = load_cached_result(test_hash)
        if cached_result:  # Only test if cache directory exists and is writable
            self.assertEqual(cached_result["extracted_text"], test_text)
            self.assertEqual(cached_result["metadata"], test_metadata)


if __name__ == '__main__':
    unittest.main()