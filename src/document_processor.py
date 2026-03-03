import os
import re
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings

class DocumentProcessor:
    """
    Handles extracting, cleaning, and chunking text from PDF documents.
    """
    def __init__(self):
        # We use standard recursive chunking to keep sentences and paragraphs grouped
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def load_and_process(self, file_path: str) -> List[Document]:
        """
        Loads the PDF, cleans its text, chunks it, and returns the list of Documents.
        
        Args:
            file_path: The absolute or relative path to the PDF file.
            
        Returns:
            A list of chunked Document objects containing text and metadata.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Source document not found at: {file_path}")
            
        print(f"Loading document: {file_path}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        print(f"Loaded {len(documents)} pages. Beginning cleaning and chunking...")
        clean_docs = self._clean_documents(documents)
        
        # Split documents into chunks while preserving page number metadata
        chunks = self.text_splitter.split_documents(clean_docs)
        
        print(f"Produced {len(chunks)} chunks from the document.")
        return chunks

    def _clean_documents(self, documents: List[Document]) -> List[Document]:
        """
        Applies basic text cleaning to remove excessive whitespace and artifacts.
        """
        cleaned_docs = []
        for doc in documents:
            # 1. Replace multiple newlines with a single newline
            text = re.sub(r'\n{3,}', '\n\n', doc.page_content)
            
            # 2. Remove excessively long sequences of spaces
            text = re.sub(r' {3,}', '  ', text)
            
            # 3. Strip leading/trailing whitespace
            text = text.strip()
            
            # Update the document content
            doc.page_content = text
            
            # Ensure basic metadata is retained
            if "page" not in doc.metadata:
                doc.metadata["page"] = 0
            # Langchain PyPDFLoader 0-indexes pages; adjust to 1-index for human readability
            doc.metadata["page"] = int(doc.metadata.get("page", 0)) + 1
            
            cleaned_docs.append(doc)
            
        return cleaned_docs

if __name__ == "__main__":
    # Simple test execution pattern when run directly
    from dotenv import load_dotenv
    load_dotenv()
    
    processor = DocumentProcessor()
    test_path = settings.DEFAULT_PDF_PATH
    if os.path.exists(test_path):
        docs = processor.load_and_process(test_path)
        print(f"First chunk: {docs[0].page_content[:100]}...")
    else:
        print(f"Test file not found at {test_path}. Skipping direct test.")
