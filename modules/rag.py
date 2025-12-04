import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Tuple
from modules.config import config
import os
from modules.utils import get_logger
from pathlib import Path

logger = get_logger(__name__)


class RAGSystem:
    """Implementation of Retrieval Augmented Generation system using ChromaDB

    This class provides methods to index documents, chunk text, retrieve relevant
    information, and generate responses based on the indexed content using AI.
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize the RAG system with persistent storage

        Args:
            persist_directory (str): Directory to store the persistent ChromaDB
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)  # Create directory if it doesn't exist

        if config.GEMINI_API_KEY:
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.embedding_model_name = config.GEMINI_EMBEDDING_MODEL
            # For generating text responses
            self.text_model = genai.GenerativeModel(config.GEMINI_MODEL)
        else:
            self.embedding_model_name = None
            self.text_model = None

        # Initialize ChromaDB client with persistent storage
        try:
            # Use persistent client instead of ephemeral
            self.chroma_client = chromadb.PersistentClient(path=str(self.persist_directory))
            logger.info(f"ChromaDB initialized successfully with persistent storage at {self.persist_directory}")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise RuntimeError(f"Failed to initialize ChromaDB: {e}")

        self.collection = None
        self.document_id = None

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap, respecting sentence boundaries"""
        try:
            # Split by paragraphs first
            paragraphs = text.split('\n\n')
            chunks = []
            current_chunk = ""

            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue

                # If the paragraph itself is too large, split it into sentences
                if len(paragraph) > config.CHUNK_SIZE:
                    # Split paragraph into sentences
                    import re
                    sentences = re.split(r'[.!?]+\s+', paragraph)

                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue

                        # Check if adding this sentence would exceed chunk size
                        if len(current_chunk) + len(sentence) <= config.CHUNK_SIZE:
                            current_chunk += " " + sentence if current_chunk else sentence
                        else:
                            # If current chunk is not empty, save it and start a new one
                            if current_chunk:
                                chunks.append(current_chunk)

                            # If the sentence is still too large, split by length
                            if len(sentence) > config.CHUNK_SIZE:
                                # Split large sentence into smaller parts
                                for i in range(0, len(sentence), config.CHUNK_SIZE):
                                    chunks.append(sentence[i:i+config.CHUNK_SIZE])
                                current_chunk = ""  # Reset as we've added all parts
                            else:
                                current_chunk = sentence  # Start new chunk with this sentence
                else:
                    # The paragraph fits, but check if it fits with current chunk
                    if len(current_chunk) + len(paragraph) <= config.CHUNK_SIZE:
                        current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                    else:
                        # Current chunk is full, save it
                        if current_chunk:
                            chunks.append(current_chunk)

                        # Start new chunk
                        current_chunk = paragraph

            # Add the last chunk if it exists
            if current_chunk:
                chunks.append(current_chunk)

            # Add overlap between chunks
            if config.CHUNK_OVERLAP > 0 and len(chunks) > 1:
                overlapped_chunks = []
                for i, chunk in enumerate(chunks):
                    if i > 0:
                        # Add overlap from previous chunk (without exceeding overlap size)
                        prev_chunk = chunks[i-1]
                        overlap_size = min(config.CHUNK_OVERLAP, len(prev_chunk))
                        overlap_text = prev_chunk[-overlap_size:]

                        # Ensure we don't duplicate content
                        if not chunk.startswith(overlap_text):
                            chunk = overlap_text + " " + chunk
                    overlapped_chunks.append(chunk)
                return overlapped_chunks

            return chunks
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            # Return at least the original text as a single chunk in case of failure
            return [text] if text else []

    def index_document(self, text: str, doc_id: str = None) -> None:
        """Index a document by chunking and storing in ChromaDB"""
        try:
            logger.info("Iniciando indexación del documento")

            if not doc_id:
                import uuid
                doc_id = str(uuid.uuid4())
            self.document_id = doc_id

            chunks = self.chunk_text(text)
            logger.info(f"Creados {len(chunks)} chunks desde el documento")

            # Create or get collection for this document
            collection_name = f"doc_{doc_id.replace('-', '_')}"
            try:
                # Try to get existing collection
                self.collection = self.chroma_client.get_collection(name=collection_name)
                # If collection exists, clear all existing documents
                self.collection.delete(where={})
            except:
                # Create new collection if it doesn't exist
                self.collection = self.chroma_client.create_collection(name=collection_name)

            # Add chunks to the collection
            documents = []
            metadatas = []
            ids = []

            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only add non-empty chunks
                    doc_id_chunk = f"{doc_id}_{i}"
                    documents.append(chunk)
                    metadatas.append({"doc_id": doc_id, "chunk_id": i, "chunk_text": chunk})
                    ids.append(doc_id_chunk)

            if documents:  # Only add if there are documents to add
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )

            logger.info(f"Indexado {len(documents)} chunks en la colección ChromaDB")
        except Exception as e:
            logger.error(f"Error indexing document in ChromaDB: {e}")
            raise

    def retrieve_chunks(self, query: str, k: int = None, doc_id: str = None) -> List[Tuple[str, float]]:
        """Retrieve the most relevant chunks for a query from ChromaDB"""
        if k is None:
            k = config.RAG_NUM_RESULTS

        # If no specific doc_id provided, use the current document_id
        if doc_id is None:
            doc_id = self.document_id

        if doc_id is None:
            logger.warning("No document ID available for retrieval")
            return []

        # Get the correct collection for this document
        collection_name = f"doc_{doc_id.replace('-', '_')}"
        try:
            collection = self.chroma_client.get_collection(name=collection_name)
        except:
            logger.warning(f"Collection {collection_name} not found for retrieval")
            return []

        try:
            # Query the collection
            results = collection.query(
                query_texts=[query],
                n_results=k
            )

            # Format results as (chunk, similarity) tuples
            chunks = results['documents'][0] if results['documents'] else []
            distances = results['distances'][0] if results['distances'] else []

            # Convert distances to similarity (1 - distance) for consistency with FAISS
            similarities = [1 - dist for dist in distances]

            return list(zip(chunks, similarities))
        except Exception as e:
            logger.error(f"Error retrieving chunks from ChromaDB: {e}")
            return []

    def query_document(self, query: str, doc_id: str = None) -> Dict[str, Any]:
        """Query the indexed document and generate a response using RAG"""
        try:
            if not self.text_model:
                return {"error": "Gemini API key not configured"}

            retrieved_chunks = self.retrieve_chunks(query, doc_id=doc_id)

            if not retrieved_chunks:
                return {"response": "No relevant content found in the document.", "sources": []}

            # Prepare context from retrieved chunks
            context_parts = [f"Fragmento {i+1}: {chunk}" for i, (chunk, score) in enumerate(retrieved_chunks)]
            context = "\n\n".join(context_parts)

            # Generate response using Gemini
            prompt = f"""
            Responde la siguiente pregunta basándote únicamente en el contexto proporcionado.

            Contexto:
            {context}

            Pregunta: {query}

            Responde de forma clara y concisa, y menciona qué fragmentos del documento usaste para formular la respuesta.
            """

            response = self.text_model.generate_content(prompt)

            return {
                "response": response.text,
                "sources": [{"chunk": chunk, "similarity": score} for chunk, score in retrieved_chunks]
            }
        except Exception as e:
            logger.error(f"Error in query_document: {e}")
            return {"error": f"Error processing query: {e}", "sources": []}

    def delete_document_index(self, doc_id: str):
        """Delete the index for a specific document"""
        try:
            collection_name = f"doc_{doc_id.replace('-', '_')}"
            self.chroma_client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection for document: {doc_id}")
        except Exception as e:
            logger.error(f"Error deleting document index: {e}")

    def list_collections(self):
        """List all available collections"""
        try:
            collections = self.chroma_client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []