"""
RAG (Retrieval-Augmented Generation) Service for Compliance Documents
This module handles ingestion, storage, and retrieval of compliance documents.
"""
import os
import logging
import hashlib
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available. Install with: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

logger = logging.getLogger(__name__)

# Default embedding model (lightweight and effective)
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks


class RAGService:
    """RAG service for storing and retrieving compliance documents."""
    
    def __init__(self, collection_name: str = "compliance_docs", persist_directory: str = "./rag_db"):
        """
        Initialize the RAG service.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector database
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is required. Install with: pip install chromadb")
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.embedding_model = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB client, collection, and embedding model."""
        try:
            # Initialize ChromaDB client with persistence
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(name=self.collection_name)
                logger.info(f"Created new collection: {self.collection_name}")
            
            # Initialize embedding model
            logger.info(f"Loading embedding model: {DEFAULT_EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
            logger.info("RAG service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            raise
    
    def _chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence or paragraph boundary
            if end < len(text):
                # Look for sentence endings
                for sep in ['\n\n', '\n', '. ', '! ', '? ']:
                    last_sep = chunk.rfind(sep)
                    if last_sep > chunk_size // 2:  # Only break if we're past halfway
                        chunk = chunk[:last_sep + len(sep)].strip()
                        end = start + len(chunk)
                        break
            
            chunks.append(chunk.strip())
            start = end - chunk_overlap  # Overlap with previous chunk
            
            if start >= len(text):
                break
        
        return chunks
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text string."""
        if not self.embedding_model:
            raise RuntimeError("Embedding model not initialized")
        return self.embedding_model.encode(text).tolist()
    
    def _hash_text(self, text: str) -> str:
        """Generate a hash for text to check for duplicates."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def ingest_document(self, 
                       document_text: str, 
                       document_name: str,
                       metadata: Optional[Dict] = None) -> int:
        """
        Ingest a document into the RAG system.
        
        Args:
            document_text: Full text of the document
            document_name: Name/identifier of the document
            metadata: Optional metadata dictionary
            
        Returns:
            Number of chunks created
        """
        try:
            # Check if document already exists
            existing = self.collection.get(
                where={"document_name": document_name}
            )
            
            if existing['ids'] and len(existing['ids']) > 0:
                logger.info(f"Document '{document_name}' already exists. Updating...")
                # Delete existing chunks
                self.collection.delete(where={"document_name": document_name})
            
            # Chunk the document
            chunks = self._chunk_text(document_text)
            logger.info(f"Created {len(chunks)} chunks from document '{document_name}'")
            
            if not chunks:
                logger.warning(f"No chunks created from document '{document_name}'")
                return 0
            
            # Generate embeddings and prepare data
            ids = []
            metadatas = []
            documents = []
            embeddings = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_name}_chunk_{i}"
                ids.append(chunk_id)
                
                chunk_metadata = {
                    "document_name": document_name,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                if metadata:
                    chunk_metadata.update(metadata)
                
                metadatas.append(chunk_metadata)
                documents.append(chunk)
                
                # Generate embedding for each chunk
                chunk_embedding = self._generate_embedding(chunk)
                embeddings.append(chunk_embedding)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            logger.info(f"Successfully ingested document '{document_name}' with {len(chunks)} chunks")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error ingesting document '{document_name}': {e}")
            raise
    
    def retrieve_relevant_chunks(self, 
                                 query: str, 
                                 n_results: int = 5,
                                 filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Retrieve relevant chunks based on a query.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of dictionaries with 'text', 'metadata', and 'distance' keys
        """
        try:
            if not query or not query.strip():
                return []
            
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata if filter_metadata else None
            )
            
            # Format results
            retrieved_chunks = []
            if results['ids'] and len(results['ids']) > 0:
                ids = results['ids'][0]
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0] if 'distances' in results else [0] * len(ids)
                
                for i, doc_id in enumerate(ids):
                    retrieved_chunks.append({
                        'id': doc_id,
                        'text': documents[i],
                        'metadata': metadatas[i] if i < len(metadatas) else {},
                        'distance': distances[i] if i < len(distances) else 0.0,
                        'document_name': metadatas[i].get('document_name', 'Unknown') if i < len(metadatas) else 'Unknown'
                    })
            
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query: {query[:50]}...")
            return retrieved_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    def get_all_documents(self) -> List[str]:
        """Get list of all document names in the collection."""
        try:
            results = self.collection.get()
            document_names = set()
            
            if results['metadatas']:
                for metadata in results['metadatas']:
                    if 'document_name' in metadata:
                        document_names.add(metadata['document_name'])
            
            return sorted(list(document_names))
        except Exception as e:
            logger.error(f"Error getting document list: {e}")
            return []
    
    def delete_document(self, document_name: str) -> bool:
        """Delete a document and all its chunks from the collection."""
        try:
            self.collection.delete(where={"document_name": document_name})
            logger.info(f"Deleted document '{document_name}' from RAG collection")
            return True
        except Exception as e:
            logger.error(f"Error deleting document '{document_name}': {e}")
            return False
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the RAG collection."""
        try:
            results = self.collection.get()
            total_chunks = len(results['ids']) if results['ids'] else 0
            document_names = self.get_all_documents()
            
            return {
                'total_chunks': total_chunks,
                'total_documents': len(document_names),
                'documents': document_names
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'total_chunks': 0, 'total_documents': 0, 'documents': []}


# Global RAG service instance
_rag_service_instance = None


def get_rag_service() -> Optional[RAGService]:
    """Get or create the global RAG service instance."""
    global _rag_service_instance
    
    if _rag_service_instance is None:
        try:
            _rag_service_instance = RAGService()
            return _rag_service_instance
        except Exception as e:
            logger.warning(f"Failed to initialize RAG service: {e}")
            return None
    
    return _rag_service_instance

