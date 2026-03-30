from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from src.config.settings import settings
from src.utils.helpers import get_logger

logger = get_logger(__name__)

class MedicalGenerator:
    def __init__(self, model_name: str = None):
        self.llm = ChatOpenAI(
            model=model_name or settings.LLM_MODEL,
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
        )

    def generate_response(self, query: str, context: str):
        """Generates a response based on retrieved medical context."""
        logger.info(f"Generating response for query: {query}")
        
        prompt = ChatPromptTemplate.from_template("""
            You are a highly accurate Medical Assistant. 
            Answer the user's healthcare query based ONLY on the provided context below.
            If the context does not contain the answer, say "I don't have enough verified information to answer this based on the available sources."
            
            Maintain a helpful, clear, and professional tone.
            Use simple language suitable for patients.
            Do not make any unsupported medical claims.
            Always cite the source document name if available.

            Context:
            {context}

            User Query:
            {query}

            Assistant Answer:
        """)
        
        chain = prompt | self.llm
        response = chain.invoke({"context": context, "query": query})
        return response.content
