from src.rag.retriever import MedicalRetriever
from src.rag.generator import MedicalGenerator
from src.utils.guardrails import HealthcareGuardrails
from src.utils.helpers import get_logger

logger = get_logger(__name__)

class RAGPipeline:
    def __init__(self, persist_dir: str = None):
        self.retriever = MedicalRetriever(persist_directory=persist_dir)
        self.generator = MedicalGenerator()
        self.guardrails = HealthcareGuardrails()

    def process_query(self, query: str):
        """Main RAG flow: Detect Emergency -> Retrieve -> Generate."""
        # 1. Check for Emergency
        if self.guardrails.detect_emergency(query):
            logger.info("Emergency query detected!")
            return self.guardrails.get_emergency_response(), []

        # 2. Retrieve Documents
        docs = self.retriever.get_relevant_documents(query)
        context = "\n\n".join([d.page_content for d in docs])
        sources = list(set([d.metadata.get("source", "Unknown Source") for d in docs]))

        # 3. Generate Answer
        raw_answer = self.generator.generate_response(query, context)

        # 4. Post-process (Add disclaimer)
        final_answer = f"{raw_answer}\n{self.guardrails.DISCLAIMER}"
        
        return final_answer, sources
