import pytest
from src.utils.guardrails import HealthcareGuardrails

def test_emergency_detection_positive():
    guardrails = HealthcareGuardrails()
    query = "I think I am having a heart attack help me"
    assert guardrails.detect_emergency(query) is True

def test_emergency_detection_negative():
    guardrails = HealthcareGuardrails()
    query = "How do I naturally lower my blood pressure?"
    assert guardrails.detect_emergency(query) is False

def test_disclaimer_content():
    guardrails = HealthcareGuardrails()
    assert "not constitute medical advice" in guardrails.DISCLAIMER
