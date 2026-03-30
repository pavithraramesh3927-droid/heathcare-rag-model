from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_spliters import RecursiveCharacterTextSplitter
from src.config.settings import settings
from src.utils.helpers import get_logger

logger = get_logger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

    def load_and_split(self, directory_path: str):
        """Loads markdown files from a directory and splits them into chunks."""
        logger.info(f"Loading documents from {directory_path}")
        loader = DirectoryLoader(directory_path, glob="**/*.md", loader_cls=TextLoader)
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks
