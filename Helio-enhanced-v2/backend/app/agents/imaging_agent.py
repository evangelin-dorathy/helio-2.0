"""Helio Imaging Agent - computer-vision analysis of diagnostic images.

A nurse uploads an X-ray / CT / MRI / ultrasound image; this agent sends it to
a multimodal vision model (Gemini) together with the patient context and returns
a structured report for the doctor: the suspected problem, the reasoning behind
it and a confidence score.

Important safety behaviour
--------------------------
* The agent NEVER invents a result. If the vision model is not configured or
  the call fails, it returns ``available=False`` with a clear message so the
  image goes to a doctor / radiologist for a manual read.
* The output is decision support only. Every report carries a disclaimer and
  the doctor must review it before the diagnostic is completed.
* The confidence value is the model's own self-reported estimate. It is not a
  calibrated probability, and the UI says so.
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None
    genai_types = None

from app.core.config import settings

DISCLAIMER = (
    "AI-assisted preliminary reading for clinical decision support only. "
    "It is not a diagnosis and must be reviewed by a qualified doctor or radiologist."
)

ALLOWED_MIME = {"image/png", "image/jpeg", "image/webp"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8 MB


def detect_mime(data: bytes) -> str | None:
    """Identify the real image type from magic bytes (never trust the filename)."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


class ImagingResult(BaseModel):
    """Structured report handed to the doctor."""

    available: bool = Field(
        default=True,
        description="False when no AI reading could be produced.",
    )
    problem: str = Field(
        description="Most likely finding / problem seen in the image, in a short phrase."
    )
    reason: str = Field(
        description="Visible evidence in the image that supports the problem."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Self-reported confidence between 0.0 and 1.0.",
    )
    findings: list[str] = Field(
        default_factory=list,
        description="Individual observations from the image.",
    )
    urgency: Literal["ROUTINE", "URGENT", "CRITICAL"] = Field(
        default="ROUTINE",
        description="How quickly a doctor should look at this image.",
    )
    image_quality: Literal["GOOD", "LIMITED", "POOR", "NOT_A_MEDICAL_IMAGE"] = Field(
        default="GOOD",
        description="Whether the image is adequate for reading.",
    )
    recommended_action: str = Field(
        default="",
        description="Suggested next clinical step for the doctor (no medication).",
    )
    model: str = ""
    disclaimer: str = DISCLAIMER


class ImagingAgent:
    """Vision-model based reader for diagnostic images."""

    def __init__(self):
        self.client = None
        if settings.GEMINI_API_KEY and genai is not None:
            try:
                self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
            except Exception as error:  # pragma: no cover
                print(f"[Imaging Agent] Gemini disabled: {error}")
        self.model = settings.GEMINI_MODEL

    # ------------------------------------------------------------------
    def analyse(self, image_bytes: bytes, mime_type: str, test_type: str, patient) -> ImagingResult:
        if self.client is None:
            if not settings.GEMINI_API_KEY:
                why = "GEMINI_API_KEY is not set on the server"
            elif genai is None:
                why = "the google-genai package is not installed on the server"
            else:
                why = "the Gemini client could not start (check the key is valid)"
            return self._unavailable(
                f"AI image analysis is not configured: {why}. "
                "The image is saved; please review it manually."
            )
        try:
            return self._gemini_read(image_bytes, mime_type, test_type, patient)
        except Exception as error:
            print(f"[Imaging Agent] analysis failed: {error}")
            return self._unavailable(
                f"AI image analysis failed ({str(error)[:160]}). "
                "The image is saved; please review it manually."
            )

    # ------------------------------------------------------------------
    def _gemini_read(self, image_bytes, mime_type, test_type, patient) -> ImagingResult:
        prompt = f"""
You are the Imaging Agent in Helio, a hospital operations system. You assist a
doctor by giving a preliminary reading of ONE diagnostic image.

Study type requested: {test_type}

PATIENT CONTEXT (may be incomplete)
Age: {patient.age or "unknown"}
Gender: {patient.gender or "unknown"}
Symptoms: {patient.symptoms or "not recorded"}
Heart rate: {patient.heart_rate if patient.heart_rate is not None else "not recorded"} bpm
SpO2: {patient.spo2 if patient.spo2 is not None else "not recorded"} %
Temperature: {patient.temperature if patient.temperature is not None else "not recorded"} C

INSTRUCTIONS
- Look at the image and report the single most likely problem. If nothing
  abnormal is visible, say "No acute abnormality detected".
- "reason" must cite what is VISIBLE in the image (location, appearance).
- "confidence" is your honest 0.0-1.0 certainty. Use lower values when the image
  is poor, cropped, or the finding is subtle. Never claim certainty.
- If the image is not a medical scan, set image_quality to NOT_A_MEDICAL_IMAGE,
  confidence to 0.0 and explain in "reason".
- "urgency": CRITICAL only for findings that need immediate attention
  (e.g. pneumothorax, large bleed, displaced fracture with compromise).
- Focus on simple, common findings a first-line reader would name, e.g. bone
  fracture, pneumonia / lung opacity, pleural effusion, pneumothorax,
  cardiomegaly, joint dislocation, sinus opacification, obvious mass or
  hemorrhage on MRI/CT. Name the body part and side in "problem".
- Write "problem", "reason" and "findings" in plain language a nurse can follow.
- If the scan is complex or the finding is unclear, say so, lower the
  confidence and recommend specialist review instead of guessing.
- Do not prescribe medication. Do not invent patient information.
"""
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                prompt,
            ],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ImagingResult,
                temperature=0.2,
            ),
        )

        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Vision model returned an empty response.")

        result = ImagingResult.model_validate(json.loads(text))
        result.available = True
        result.model = self.model
        result.disclaimer = DISCLAIMER
        if result.image_quality == "NOT_A_MEDICAL_IMAGE":
            result.confidence = 0.0
        return result

    # ------------------------------------------------------------------
    def _unavailable(self, message: str) -> ImagingResult:
        return ImagingResult(
            available=False,
            problem="AI reading unavailable",
            reason=message,
            confidence=0.0,
            findings=[],
            urgency="ROUTINE",
            image_quality="LIMITED",
            recommended_action="Manual review by a doctor or radiologist.",
            model="",
        )
