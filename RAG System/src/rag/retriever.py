from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from src.config.settings import settings
from src.utils.helpers import get_logger

logger = get_logger(__name__)

class MedicalRetriever:
    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory or settings.VECTOR_DB_DIR
        self.embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
        self.vector_store = None

    def create_vector_store(self, chunks):
        """Creates a vector store from document chunks."""
        logger.info("Creating new vector store.")
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        return self.vector_store

    def load_vector_store(self):
        """Loads an existing vector store."""
        logger.info(f"Loading vector store from {self.persist_directory}")
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        return self.vector_store

    def get_relevant_documents(self, query: str, k: int = 3):
        """Retrieves top-k relevant documents."""
        if not self.vector_store:
            self.load_vector_store()
        
        # Use simple similarity search
        return self.vector_store.similarity_search(query, k=k)
