import sys
import os

# Ensure the root directory is in sys.path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.pipeline import RAGPipeline
from src.rag.document_processor import DocumentProcessor
from src.rag.retriever import MedicalRetriever
from src.utils.helpers import get_logger

logger = get_logger(__name__)

def initialize_system():
    """Builds the vector database from sample documents."""
    processor = DocumentProcessor()
    retriever = MedicalRetriever()
    
    data_dir = os.path.join(os.getcwd(), "data")
    if os.path.exists(data_dir) and os.listdir(data_dir):
        chunks = processor.load_and_split(data_dir)
        retriever.create_vector_store(chunks)
        print("✅ System initialized: Documents processed and indexed.")
    else:
        print("⚠️ No documents found in 'data/' directory. System started with empty database.")

def main():
    """Keyboard simulation for medical queries."""
    if not os.path.exists("vector_db"):
        initialize_system()

    pipeline = RAGPipeline()
    print("\n--- Medical Information Assistant ---")
    print("Type 'exit' or 'quit' to end.")

    while True:
        query = input("\nYour Health Query: ")
        if query.lower() in ["exit", "quit", "q"]:
            break
            
        answer, sources = pipeline.process_query(query)
        print(f"\nResponse:\n{answer}")
        if sources:
            print(f"\nSources: {', '.join(sources)}")

if __name__ == "__main__":
    main()
