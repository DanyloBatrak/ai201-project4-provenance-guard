import json
import os
import re
import statistics
from typing import List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Signal 1 - Predictability via Groq attribution
# ---------------------------------------------------------------------------

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"

ATTRIBUTION_SYSTEM_PROMPT = """You are a text-provenance classifier.

Assess whether the given text reads as AI-generated or human-written.
Weigh these concrete signals rather than a general impression of
"quality" -- fluent, well-structured writing is NOT on its own evidence
of human authorship; modern LLMs write fluently by default.

Signals that lean AI-generated:
- Generic hedging/transition phrases ("It is important to note that",
  "Furthermore", "That said", "In conclusion")
- Formulaic structure (e.g. intro claim, three balanced supporting
  points, wrap-up restatement) with even paragraph/sentence lengths
- Abstract, general claims with no concrete specific details (names,
  numbers, dates, sensory detail, personal anecdotes)
- Studied even-handedness ("there are pros and cons on both sides")
  without a genuine personal stance
- Absence of any idiosyncrasy: no typos, no informal punctuation, no
  tangents, no self-interruptions

Signals that lean human-written:
- Concrete, specific, verifiable details (a named place, a specific
  number, an offhand personal reference)
- Informal or inconsistent punctuation/capitalization, contractions,
  filler words, run-on or fragment sentences
- A genuine opinion or complaint, not just balanced pros/cons
- Irregular structure -- sentences and paragraphs that vary a lot in
  length rather than following a template
- Domain jargon used naturally and specifically, rather than a
  textbook-style general overview of a topic

Fluent, grammatically correct, or sophisticated-vocabulary text is not
by itself evidence of human authorship -- judge structure and
specificity, not polish.

Respond ONLY with a JSON object matching this exact schema, with no
extra commentary:

{
  "verdict": "ai" | "human" | "uncertain",
  "confidence": <float between 0 and 1>,
  "reasoning": "<one or two sentence justification citing specific signals from the text>"
}
"""

VALID_VERDICTS = {"ai", "human", "uncertain"}


class AttributionSignalError(Exception):
    """Raised when the Groq call fails or returns something unusable."""


def get_groq_attribution(
    text: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    timeout: float = 15.0,
) -> dict:
    """Call Groq for a structured AI-vs-human attribution assessment.

    Returns:
        dict with keys: verdict ("ai" | "human" | "uncertain"),
        confidence (float in [0, 1] or None), reasoning (str).

    Raises:
        AttributionSignalError if the API key is missing, the request
        fails, or the model's response can't be parsed into the schema.
    """
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise AttributionSignalError("GROQ_API_KEY is not set")

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": ATTRIBUTION_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise AttributionSignalError(f"Groq request failed: {exc}") from exc

    try:
        raw_content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(raw_content)
    except (KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
        raise AttributionSignalError(f"Could not parse Groq response: {exc}") from exc

    return _normalize(parsed)


def _normalize(parsed: dict) -> dict:
    """Coerce a raw parsed JSON dict into the fixed output schema."""
    verdict = parsed.get("verdict")
    if verdict not in VALID_VERDICTS:
        verdict = "uncertain"

    confidence = parsed.get("confidence")
    try:
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = None

    reasoning = parsed.get("reasoning")
    if not isinstance(reasoning, str):
        reasoning = ""

    return {"verdict": verdict, "confidence": confidence, "reasoning": reasoning}


# ---------------------------------------------------------------------------
# Signal 2 - Burstiness (clustering_score)
# ---------------------------------------------------------------------------

MIN_SENTENCES_FOR_RELIABILITY = 5
CV_CLIP_MIN = 0.0
CV_CLIP_MAX = 1.0


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


def _sentence_lengths(sentences: List[str]) -> List[int]:
    return [len(re.findall(r"\w+", s)) for s in sentences]


def _coefficient_of_variation(lengths: List[int]) -> float:
    if len(lengths) < 2:
        return 0.0
    mean = statistics.mean(lengths)
    if mean == 0:
        return 0.0
    stdev = statistics.pstdev(lengths)
    return stdev / mean


def _normalize_cv(cv: float) -> float:
    clipped = max(CV_CLIP_MIN, min(CV_CLIP_MAX, cv))
    span = CV_CLIP_MAX - CV_CLIP_MIN
    normalized = clipped / span if span else 0.0
    return 1.0 - normalized


def clustering_score(text: str) -> Tuple[float, bool]:
    """Signal 2 - Burstiness. Returns (score, reliable).

    score: float in [0,1], 1.0 = low variation/AI-like, 0.0 = high
        variation/human-like.
    reliable: False if fewer than MIN_SENTENCES_FOR_RELIABILITY sentences.
    """
    sentences = _split_sentences(text)
    reliable = len(sentences) >= MIN_SENTENCES_FOR_RELIABILITY

    if not sentences:
        return 0.0, False

    lengths = _sentence_lengths(sentences)
    cv = _coefficient_of_variation(lengths)
    score = _normalize_cv(cv)
    return score, reliable


# ---------------------------------------------------------------------------
# Standalone check (run: python signals.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Requires GROQ_API_KEY to be set and network access.
    samples = [
        "As an AI language model, I don't have personal opinions, but I can provide a balanced overview of the topic.",
        "honestly? my cat knocked the vase off the shelf AGAIN. third time this week. i give up, he wins.",
        "The report outlines quarterly revenue growth of 4.2%, driven primarily by increased demand in the enterprise segment.",
    ]
    for s in samples:
        try:
            result = get_groq_attribution(s)
        except AttributionSignalError as e:
            result = {"error": str(e)}
        print(f"{s[:60]!r} -> attribution={result}")
        print(f"{'':60s}    clustering={clustering_score(s)}")