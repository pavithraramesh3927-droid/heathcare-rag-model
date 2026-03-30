import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    VECTOR_DB_DIR = "vector_db"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    LLM_MODEL = "gpt-4-turbo-preview"
    EMBEDDING_MODEL = "text-embedding-3-small"

settings = Settings()
