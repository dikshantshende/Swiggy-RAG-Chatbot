import argparse
import sys
import os
from dotenv import load_dotenv

# Load env variables before importing local modules
load_dotenv()

from src.rag_pipeline import RAGPipeline
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.config import settings

def ingest_mode(file_path: str):
    """Handles the document ingestion flow."""
    print(f"--- INGESTION MODE ---")
    print(f"File to process: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found exactly at '{file_path}'.")
        print("Please place the target PDF in that location and run again.")
        sys.exit(1)
        
    try:
        processor = DocumentProcessor()
        docs = processor.load_and_process(file_path)
        
        manager = VectorStoreManager()
        manager.create_and_save_index(docs)
        print("--- INGESTION COMPLETE ---")
    except Exception as e:
        print(f"Ingestion failed: {e}")
        sys.exit(1)


def interactive_mode():
    """Runs a simple CLI loop for submitting questions."""
    print("--- SWIGGY ANNUAL REPORT CLI ---")
    print("Type 'exit' or 'quit' to close.")
    
    # Check if index exists before starting
    index_path = os.path.join(settings.VECTOR_STORE_DIR, "index.faiss")
    if not os.path.exists(index_path):
        print("\n[WARNING] FAISS index not found. You must ingest the document first.")
        print(f"Run: python main.py --ingest {settings.DEFAULT_PDF_PATH}\n")
        sys.exit(1)
        
    try:
        pipeline = RAGPipeline()
    except Exception as e:
         print(f"Failed to initialize RAG Pipeline. Ensure API keys are set and index exists. Error: {e}")
         sys.exit(1)
    
    while True:
        try:
            query = input("\nQuery: ").strip()
            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            if not query:
                continue
                
            print("Generating answer (this might take a few seconds)...")
            response = pipeline.generate_answer(query)
            
            # Print the formatted output cleanly
            print("\n---------------------------------------------------------")
            print(f"ANSWER:\n{response['answer']}")
            print(f"\nCONFIDENCE: {response['confidence']}")
            
            if response.get('sources'):
                print("\nSOURCES:")
                for i, src in enumerate(response['sources']):
                    page = src.get('page', 'Unknown')
                    quote = src.get('text', '')
                    print(f"  [{i+1}] Page {page}: \"{quote}\"")
            print("---------------------------------------------------------")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred resolving your query: {e}")


def main():
    parser = argparse.ArgumentParser(description="Swiggy RAG System CLI")
    parser.add_argument(
        "--ingest", 
        type=str, 
        nargs="?",
        const=settings.DEFAULT_PDF_PATH,
        help="Path to the PDF file to ingest. Default is 'data/swiggy_annual_report.pdf'"
    )
    
    args = parser.parse_args()
    
    # Check environment variable presence gracefully
    if not os.environ.get("GEMINI_API_KEY"):
         print("[WARNING] GEMINI_API_KEY is not set in environment or .env file.")
         print("You need a valid Gemini API key to run this pipeline with the chosen models.")
    
    if args.ingest:
        ingest_mode(args.ingest)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
