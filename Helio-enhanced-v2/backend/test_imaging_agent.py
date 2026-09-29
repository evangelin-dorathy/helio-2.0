from app.agents.imaging_agent import ImagingAgent, detect_mime


class MockPatient:
    age = 61
    gender = "Male"
    symptoms = "Cough and fever"
    heart_rate = 110
    spo2 = 91
    temperature = 38.6


def test_detect_mime_uses_magic_bytes():
    assert detect_mime(b"\x89PNG\r\n\x1a\n" + b"0" * 10) == "image/png"
    assert detect_mime(b"\xff\xd8\xff\xe0" + b"0" * 10) == "image/jpeg"
    assert detect_mime(b"%PDF-1.4") is None


def test_no_api_key_returns_unavailable_not_a_made_up_result():
    agent = ImagingAgent()
    agent.client = None  # simulate missing GEMINI_API_KEY
    result = agent.analyse(b"\x89PNG\r\n\x1a\n", "image/png", "Chest X-Ray", MockPatient())
    assert result.available is False
    assert result.confidence == 0.0
    assert "manual" in result.recommended_action.lower()
