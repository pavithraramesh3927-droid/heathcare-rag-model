import pytest
from unittest.mock import MagicMock, patch
from src.rag.retriever import MedicalRetriever

@pytest.fixture
def mock_chunks():
    doc = MagicMock()
    doc.page_content = "Hypertension causes headaches."
    doc.metadata = {"source": "test.md"}
    return [doc]

@patch("src.rag.retriever.Chroma")
@patch("src.rag.retriever.OpenAIEmbeddings")
def test_retriever_creation(mock_embeddings, mock_chroma, mock_chunks):
    retriever = MedicalRetriever(persist_directory=":memory:")
    retriever.create_vector_store(mock_chunks)
    
    mock_chroma.from_documents.assert_called_once()

@patch("src.rag.retriever.Chroma")
@patch("src.rag.retriever.OpenAIEmbeddings")
def test_retriever_query(mock_embeddings, mock_chroma, mock_chunks):
    mock_instance = mock_chroma.return_value
    mock_instance.similarity_search.return_value = mock_chunks
    
    retriever = MedicalRetriever(persist_directory=":memory:")
    retriever.vector_store = mock_instance
    
    results = retriever.get_relevant_documents("What causes headaches?")
    assert len(results) == 1
    assert results[0].page_content == "Hypertension causes headaches."
