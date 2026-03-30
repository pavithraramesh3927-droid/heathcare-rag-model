import re

class HealthcareGuardrails:
    EMERGENCY_KEYWORDS = [
        "chest pain", "heart attack", "can't breathe", "shortness of breath",
        "stroke", "unconscious", "heavy bleeding", "suicide", "kill myself"
    ]

    DISCLAIMER = (
        "\n\n**Disclaimer:** This information is for educational purposes only and "
        "does NOT constitute medical advice, diagnosis, or treatment. "
        "Always seek the advice of your physician or other qualified health provider "
        "with any questions you may have regarding a medical condition. "
        "If you are experiencing a medical emergency, call your local emergency services immediately."
    )

    @staticmethod
    def detect_emergency(query: str) -> bool:
        """Detects if the query contains emergency-related keywords."""
        query_lower = query.lower()
        for keyword in HealthcareGuardrails.EMERGENCY_KEYWORDS:
            if re.search(rf"\b{keyword}\b", query_lower):
                return True
        return False

    @staticmethod
    def get_emergency_response() -> str:
        """Returns an immediate emergency guidance message."""
        return (
            "🚨 **URGENT:** Based on your symptoms, you may be experiencing a medical emergency. "
            "Please stop using this chat and **call emergency services (e.g., 911) immediately** "
            "or go to the nearest Emergency Room."
        )
