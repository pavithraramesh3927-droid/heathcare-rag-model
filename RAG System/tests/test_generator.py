import pytest
from unittest.mock import MagicMock, patch
from src.rag.generator import MedicalGenerator

@patch("src.rag.generator.ChatOpenAI")
def test_generator_prompt_logic(mock_chat_openai):
    # Mock LLM response
    mock_llm = mock_chat_openai.return_value
    mock_response = MagicMock()
    mock_response.content = "Filtered answer."
    mock_llm.invoke.return_value = mock_response
    
    generator = MedicalGenerator()
    answer = generator.generate_response("query", "context")
    
    assert answer == "Filtered answer."
    mock_llm.invoke.assert_called_once()
