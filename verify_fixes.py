
import os
import sys
import logging
from datetime import date, datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rag_system():
    print("\n=== TESTING RAG SYSTEM ===")
    try:
        from modules.rag import RAGSystem
        rag = RAGSystem()
        
        # Test text
        text = """
        Python es un lenguaje de programación de alto nivel, interpretado y de propósito general.
        Fue creado por Guido van Rossum y lanzado por primera vez en 1991.
        La filosofía de diseño de Python enfatiza la legibilidad del código.
        """
        
        print("1. Testing Indexing (Embedding Generation)...")
        rag.index_document(text, doc_id="test_doc_001")
        print("✅ Indexing successful")
        
        print("2. Testing Query (Text Generation)...")
        response = rag.query_document("¿Quién creó Python?")
        
        if "error" in response:
            print(f"❌ Query failed: {response['error']}")
            return False
            
        print(f"✅ Query successful. Response: {response['response'][:100]}...")
        return True
    except Exception as e:
        print(f"❌ RAG System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metadata_extraction():
    print("\n=== TESTING METADATA EXTRACTION ===")
    try:
        from modules.metadata import MetadataExtractor
        from modules.config import config
        
        if not config.GEMINI_API_KEY:
            print("❌ Skipped: No API Key found")
            return False
            
        extractor = MetadataExtractor()
        
        text = "Informe anual de ventas 2023. Autor: Juan Pérez. Fecha: 12/01/2024."
        
        print("1. Testing Metadata Generation with Gemini...")
        metadata = extractor.extract_with_gemini(text)
        
        if not metadata:
            print("❌ Metadata generation returned empty")
            return False
            
        print(f"✅ Metadata generated: {json.dumps(metadata, indent=2, ensure_ascii=False)}")
        return True
    except Exception as e:
        print(f"❌ Metadata test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_json_serialization():
    print("\n=== TESTING JSON SERIALIZATION ===")
    try:
        from modules.utils import format_metadata_for_download
        
        data = {
            "title": "Test Document",
            "created_date": date(2024, 3, 15),
            "processed_at": datetime.now(),
            "tags": ["test", "python"]
        }
        
        print("1. Testing serialization of date objects...")
        formatted = format_metadata_for_download(data)
        
        # Try to dump to JSON string to verify it works
        json_str = json.dumps(formatted)
        print(f"✅ Serialization successful: {json_str}")
        return True
    except Exception as e:
        print(f"❌ Serialization test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting Verification of Fixes...")
    
    rag_success = test_rag_system()
    meta_success = test_metadata_extraction()
    json_success = test_json_serialization()
    
    if rag_success and meta_success and json_success:
        print("\n🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n⚠️ SOME TESTS FAILED")
        sys.exit(1)
