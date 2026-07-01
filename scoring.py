"""
Confidence scoring: combine Signal 1 and Signal 2 into a single
combined_score, per the "Combining into a single confidence score"
section of the spec.

combined_score = (t1 * signal_1_score + t2 * signal_2_score) / (t1 + t2)

where t1, t2 = 1.0 by default, but each is forced to 0.0 if that
signal's `reliable` flag is False. If both signals are unreliable,
combined_score is None (the caller is expected to short-circuit to the
"uncertain - insufficient text" label rather than scoring further).

This function is intentionally signal-agnostic: it takes plain
(score, reliable) pairs, so it works whether Signal 1 is the spec's
expectedness_score, a Groq-based classifier, or anything else that
produces a float in [0, 1] plus a reliability flag.
"""

from typing import Optional, Tuple

# Thresholds from the spec's "Thresholds" table (corrected: AI bucket
# starts at 0.65, matching both the table and the Transparency Label
# Design section -- these used to disagree at exactly 0.65, now fixed).
LOW_HUMAN_MAX = 0.35     # 0.00 - 0.35 -> Likely human
HIGH_AI_MIN = 0.65       # 0.65 - 1.00 -> Likely AI
# 0.36 - 0.64... -> Uncertain (implicit else-branch below; strictly,
# anything not <= 0.35 and not >= 0.65)

LABEL_LIKELY_HUMAN = "likely_human"
LABEL_UNCERTAIN = "uncertain"
LABEL_LIKELY_AI = "likely_ai"
LABEL_INSUFFICIENT_TEXT = "uncertain_insufficient_text"


def signal1_to_score(signal_1_result: dict) -> Tuple[float, bool]:
    """Convert Groq's verdict/confidence output into a (score, reliable)
    pair, so it can feed combine_score() the same way any other signal
    would.

    Design choice (the spec's Signal 1 was a perplexity measure; this
    project's actual Signal 1 is an LLM classifier, so this mapping
    isn't spec-verbatim, just a reasonable translation):
      - verdict "ai"      -> score = confidence
      - verdict "human"   -> score = 1 - confidence
      - verdict "uncertain" (model itself unsure) -> score = 0.5
    reliable = True only if Groq actually returned a confidence value.
    If the call failed (confidence is None -- see attribution_groq.py's
    error handling in app.py), this signal gets weight 0 in the
    combiner, same as a short-text signal would.
    """
    confidence = signal_1_result.get("confidence")
    verdict = signal_1_result.get("verdict")

    if confidence is None:
        return 0.5, False

    if verdict == "ai":
        score = confidence
    elif verdict == "human":
        score = 1.0 - confidence
    else:
        score = 0.5

    return score, True


def combine_score(
    signal_1: Tuple[float, bool],
    signal_2: Tuple[float, bool],
) -> Optional[float]:
    """Combine two (score, reliable) signal outputs per spec.

    Args:
        signal_1: (score, reliable) for signal 1, score in [0, 1]
        signal_2: (score, reliable) for signal 2, score in [0, 1]

    Returns:
        combined_score in [0, 1], or None if both signals are unreliable.
    """
    s1_score, s1_reliable = signal_1
    s2_score, s2_reliable = signal_2

    t1 = 1.0 if s1_reliable else 0.0
    t2 = 1.0 if s2_reliable else 0.0

    if t1 == 0.0 and t2 == 0.0:
        return None

    return (t1 * s1_score + t2 * s2_score) / (t1 + t2)


def label_for_score(combined_score: Optional[float]) -> str:
    """Map a combined_score to its threshold bucket, per spec.

    Forces the insufficient-text label when combined_score is None
    (i.e. both signals were unreliable), regardless of any score.
    """
    if combined_score is None:
        return LABEL_INSUFFICIENT_TEXT
    if combined_score <= LOW_HUMAN_MAX:
        return LABEL_LIKELY_HUMAN
    if combined_score >= HIGH_AI_MIN:
        return LABEL_LIKELY_AI
    return LABEL_UNCERTAIN


def generate_label(combined_score: Optional[float]) -> dict:
    """Map a combined_score to its bucket AND the exact transparency
    message text, per the "Transparency Label Design" section of the
    spec.

    Text below is the CORRECTED version of the spec's wording (typos
    fixed: "cosistent"->"consistent", "consistant"->"consistent"; the
    duplicated "Confidence: {score} / 1.0." line in the Human variant
    removed; the Uncertain variant's missing closing reminder sentence
    added, since the spec's own "Overall" note says all three variants
    must always show the reminder). This was a deliberate choice, not
    an oversight -- see conversation history for the tradeoff (verbatim
    spec text, including its typos, was the alternative).

    Returns:
        dict with keys:
          bucket: one of LABEL_LIKELY_AI / LABEL_LIKELY_HUMAN /
              LABEL_UNCERTAIN / LABEL_INSUFFICIENT_TEXT
          message: the exact user-facing transparency string
    """
    bucket = label_for_score(combined_score)

    if bucket == LABEL_INSUFFICIENT_TEXT:
        message = (
            "Confidence: N/A / 1.0. The submitted text was too short to "
            "analyze reliably. This is an automated estimate, not a "
            "determination of fact."
        )
        return {"bucket": bucket, "message": message}

    score_str = f"{combined_score:.2f}"

    if bucket == LABEL_LIKELY_AI:
        message = (
            f"Confidence: {score_str} / 1.0. This text shows strong signals "
            f"consistent with AI generation. This is an automated estimate, "
            f"not a determination of fact."
        )
    elif bucket == LABEL_LIKELY_HUMAN:
        message = (
            f"Confidence: {score_str} / 1.0. This text shows signals of "
            f"consistent human writing. This is an automated estimate, not "
            f"a determination of fact."
        )
    else:  # LABEL_UNCERTAIN
        message = (
            f"Confidence: {score_str} / 1.0. We could not confidently "
            f"determine if this text has been AI generated or human "
            f"written. This is an automated estimate, not a determination "
            f"of fact."
        )

    return {"bucket": bucket, "message": message}


if __name__ == "__main__":
    # NOTE: this sandbox has no network access, so signal_1 (Groq) is
    # mocked below with plausible outputs rather than called live.
    # Run this same test with real get_groq_attribution() calls on a
    # machine that has GROQ_API_KEY + network to get the real numbers.
    from unittest.mock import patch
    from signals.clustering import clustering_score

    test_cases = [
        (
            "clearly_ai",
            (
                "Artificial intelligence represents a transformative paradigm shift in modern society. "
                "It is important to note that while the benefits of AI are numerous, it is equally "
                "essential to consider the ethical implications. Furthermore, stakeholders across "
                "various sectors must collaborate to ensure responsible deployment. It is also worth "
                "noting that regulatory frameworks are still evolving to address these challenges. "
                "Ultimately, a balanced and thoughtful approach will be necessary going forward."
            ),
            {"verdict": "ai", "confidence": 0.91, "reasoning": "mocked -- see note above"},
        ),
        (
            "clearly_human",
            (
                "ok so i finally tried that new ramen place downtown and honestly? "
                "underwhelming. the broth was fine but they put WAY too much sodium in it and "
                "i was thirsty for like three hours after. my friend got the spicy version and "
                "said it was better. probably won't go back unless someone drags me there."
            ),
            {"verdict": "human", "confidence": 0.93, "reasoning": "mocked -- see note above"},
        ),
        (
            "borderline_formal_human",
            (
                "The relationship between monetary policy and asset price inflation has been "
                "extensively studied in the literature. Central banks face a fundamental tension "
                "between their mandate for price stability and the unintended consequences of "
                "prolonged low interest rates on equity and real estate valuations."
            ),
            {"verdict": "uncertain", "confidence": 0.55, "reasoning": "mocked -- see note above"},
        ),
        (
            "borderline_edited_ai",
            (
                "I've been thinking a lot about remote work lately. There are genuine tradeoffs — "
                "flexibility and no commute on one side, isolation and blurred work-life boundaries "
                "on the other. Studies show productivity varies widely by individual and role type."
            ),
            {"verdict": "uncertain", "confidence": 0.60, "reasoning": "mocked -- see note above"},
        ),
    ]

    for name, text, mocked_groq_result in test_cases:
        s1 = signal1_to_score(mocked_groq_result)
        s2 = clustering_score(text)
        combined = combine_score(s1, s2)
        label = label_for_score(combined)
        combined_str = f"{combined:.3f}" if combined is not None else "None"
        print(
            f"{name:26s} "
            f"s1={s1[0]:.3f}(rel={s1[1]!s:5s}) "
            f"s2={s2[0]:.3f}(rel={s2[1]!s:5s}) "
            f"combined={combined_str:6s} label={label}"
        )

    # Force-test threshold boundaries and the insufficient-text path
    print()
    print("label_for_score threshold checks:")
    print("  0.10 ->", label_for_score(0.10))
    print("  0.35 ->", label_for_score(0.35))
    print("  0.36 ->", label_for_score(0.36))
    print("  0.64 ->", label_for_score(0.64))
    print("  0.65 ->", label_for_score(0.65))
    print("  0.90 ->", label_for_score(0.90))
    print("  None ->", label_for_score(None))

    print()
    print("generate_label forced-value checks (M5):")
    for val in [0.1, 0.4, 0.65, 0.9, None]:
        result = generate_label(val)
        print(f"  {str(val):5s} -> bucket={result['bucket']}")
        print(f"          message={result['message']!r}")