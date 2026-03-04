import os
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import settings

class VectorStoreManager:
    """
    Manages embedding model initialization and FAISS vector index logic.
    Provides methods for index creation, persistence, and semantic search retrieval.
    """
    def __init__(self):
        # Using API-based Google Generative AI embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GEMINI_API_KEY
        )
        self.vector_store_dir = settings.VECTOR_STORE_DIR
        self.vector_store = None
        
    def create_and_save_index(self, documents: List[Document], save_path: str = None) -> FAISS:
        """
        Creates a new FAISS vector index from chunked text documents.
        Saves the resulting index to disk for fast continuous retrieval later.
        
        Args:
            documents: List of chunks (LangChain Document objects).
            save_path: Location to save the FAISS binary file and index map.
            
        Returns:
            The initialized FAISS index.
        """
        if not save_path:
            save_path = self.vector_store_dir
            
        print(f"Creating vector index with {len(documents)} chunks...")
        print(f"Embedding model: {settings.EMBEDDING_MODEL}")
        
        # Batch size for OpenAI API is handled natively by langchain
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Ensure directory exists before saving
        os.makedirs(save_path, exist_ok=True)
        self.vector_store.save_local(save_path)
        print(f"Index successfully created and saved locally to: {save_path}")
        
        return self.vector_store
        
    def load_index(self, load_path: str = None) -> FAISS:
        """
        Loads a previously saved FAISS index from disk.
        """
        if not load_path:
            load_path = self.vector_store_dir
            
        if not os.path.exists(os.path.join(load_path, "index.faiss")):
            raise ValueError(f"No valid FAISS index found at {load_path}. Please ingest documents first.")
            
        print(f"Loading existing vector index from: {load_path}")
        # Allow loading local file - a common FAISS setting when deserializing
        self.vector_store = FAISS.load_local(
            load_path, 
            self.embeddings,
            allow_dangerous_deserialization=True  # Required by recent FAISS updates for trusted local files
        )
        return self.vector_store
        
    def get_retriever(self, k: int = None):
        """
        Returns a LangChain retriever interface for the underlying FAISS index.
        Limits results to the top k nearest neighbors.
        """
        if not self.vector_store:
            try:
                self.load_index()
            except Exception as e:
                raise RuntimeError("Vector store not initialized. Load or create an index first.") from e
                
        if not k:
            k = settings.RETRIEVER_K
            
        # Standard semantic similarity search
        return self.vector_store.as_retriever(search_kwargs={"k": k})
        
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Utility method to run a raw vector search for debugging or manual testing.
        """
        if not self.vector_store:
            self.load_index()
            
        return self.vector_store.similarity_search(query, k=k)

if __name__ == "__main__":
    # Test script if executed directly
    from dotenv import load_dotenv
    from src.document_processor import DocumentProcessor
    import time
    
    load_dotenv()
    
    # Simple flow to test ingestion pipeline from start to finish
    doc_paths = settings.DEFAULT_PDF_PATH
    if os.path.exists(doc_paths):
        processor = DocumentProcessor()
        manager = VectorStoreManager()
        
        print("Starting ingestion test...")
        # Reduce size for rapid dev testing
        start_time = time.time()
        docs = processor.load_and_process(doc_paths)
        manager.create_and_save_index(docs)
        print(f"Finished. Took {time.time() - start_time:.2f} seconds.")
    else:
        print("PDF test missing. Create a dummy test if needed.")
