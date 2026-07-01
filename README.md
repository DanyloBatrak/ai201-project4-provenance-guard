# Provenance Guard

A text-provenance classifier: the user submit a piece of text, get back an estimate of whether it reads as AI-generated or human-written, with a transparent confidence score, an audit trail, and an appeals process for contesting the result.

## Architecture

### Submission Flow

+----------+ text +------------+ text +----------------+
| Client | ---------> | POST /submit| -----> | Signal 1: |
+----------+ | (API layer) | | Predictability |
+------------+ +-------+--------+
| | score(0-1)+reliable
| |
| text v
+------------->+-----------------+
| Signal 2 |
| Burstiness |
+-----------------+
|
| score(0-1)+reliable
|
V
+------------------+
| Confidense Score |
| (combine) |
+------------------+
|
| label + score
|
V
+------------------+
| Audit Log |
+------------------+
|
|
V
+------------------+
| Response to User |
+------------------+

### Appeal Flow

+-----------+ submission_id +---------------+ status_update +-------------+
| Client | --------------->| POST / appeal | --------------->| Audit |
+-----------+ +---------------+ +-------------+
|
| appeal_id + status
V
+---------------+  
 | Response |
+---------------+

### Stack:

Flask, Groq API (llama-3.3-70b-versatile), Flask-Limiter, a JSONL audit log.

## Detection Signals

### Signal 1 - Predictability (expectedness_score)

This signal would measures how statistically each word/token is given its preceding context, using a small reference language model

Output: normalized float in [0, 1], where 1.0 is maximally predictable (low perplexity, more AI-like) while 0.0 is maximally unpredictable (high perplexity, more human-like)

Normallization: clip my raw perpexity to an chosen range [10, 100] and rescale it linearly, inverted, into [0, 1]

Reliable(?): bool flag would returns False if text would be under 20 tokens since perplexity is noisy on very short output

### Signal 2 - Burstiness (clustering_score)

This signal would measure variation in sentence length and structure across the document

Output: normalized float in [0, 1], where 1.0 is low variation (more AI-like) while 0.0 is highly varied (more human-like). Computed from coefficient of variation, inverted and rescaled.

Reliable(?): bool flag would returns False if text would have fewer than 5 sentences because variation isn't very measurable on short input.

### Combining into a single confidence score

Our combined score would be like that:
combine*score = (t1 * expectedness*score + t2 * clustering_score) / (t1 + t2)

where t1 and t2 are equal to 1.0 by default, but each weight is set to 0.0 if that signal's reliable flag is False. If both of them are unreliable then combined_score = None and the system returns the "uncertain — insufficient text" label directly without further scoring. combined_score is a float in [0, 1], where higher = more AI-like.

#### Threshold buckets:

Range | Label bucket |
0.00 - 0.35 | Likely human
0.36 - 0.65 | Uncertain
0.65 - 1.00 | Likely AI

#### Example of high-confidense case:

Input: Artificial intelligence represents a transformative paradigm shift in modern society. It is important to note that while the benefits of AI are numerous, it is equally essential to consider the ethical implications. Furthermore, stakeholders across various sectors must collaborate to ensure responsible deployment.

Signal 1: 0.9 (reliable) — Groq verdict human, confidence 0.9, citing generic hedging phrases and formulaic balance
Signal 2: 0.62 (unreliable — under 5 sentences, gets zero weight)
Combined: 0.9 -> likely ai

#### Example of lower-confidense case:

Input: ok so i finally tried that new ramen place downtown and honestly? underwhelming. the broth was fine but they put WAY too much sodium in it and i was thirsty for like three hours after. my friend got the spicy version and said it was better. probably wont go back unless someone drags me there...

Signal 1: 0.2 (reliable) — Groq verdict ai, confidence 0.9, citing generic hedging phrases and formulaic balance
Signal 2: 0.38 (reliable) - Clearly human-written
Combined: 0.9 -> likely human

## Transparancy Labels

### High Confidense AI

It would happen when: {score} >= 0.65

Response: "Confidence: {score} / 1.0. This text shows strong signals cosistent usage of AI generation. This is an automated estimate, not a determination of fact."

### High Confidense Human

It would happen when: {score} <= 0.35

Response: "Confidence: {score} / 1.0. This text shows signals of consistant human writing. Confidence: {score} / 1.0. This is an automated estimate, not a determination of fact."

### Uncertain

It would happen when: 0.35 < score < 0.65

Response: " Confidence: {score} / 1.0. We could not confidently determine if this text been AI generated or been human written.

### Insuficient text

It would happen when both signals are not reliable and combined_score = None

Response: Confidence: N/A / 1.0. The submitted text was too short to analyze reliably. This is an automated estimate, not a determination of fact.

## Appeals Workflow

Post /appeal accepts content_id and creator_reasoning

## Rate Limiting

POST /submit is limited to 10 requests/minute and 100 requests/day per client IP, via Flask-Limiter with in-memory storage.

Reasoning: 0/minute comfortably covers a real writer submitting multiple drafts in one sitting (no legitimate use case submits faster than one every ~6 seconds) while blocking a script flooding the endpoint. 100/day covers a very prolific user across a full workday with headroom, while still bounding worst-case load from a single client.

Evidence: 200
200
200
200
200
200
200
200
200
200
429
429

## Audit Log

Every submission and appeal writes a structured JSON entry (audit_log.jsonl) including: timestamp, content_id, the raw attribution result, confidence, both individual signal scores + their reliable flags, and an appealed boolean (computed live against current submission state, so it's visible directly on each entry rather than requiring cross-referencing a separate appeal event).

Example :

appeal_id status

---

814e056f-c8ce-4499-8e46-0a6e6fc09f2a under_review

    {
      "appealed": false,
      "attribution": {
        "confidence": 0.8,
        "reasoning": "The text contains informal language and a casual tone, with a specific statement of purpose that suggests a human-written text, and lacks the formulaic structure often seen in AI-generated content.",
        "verdict": "human"
      },
      "confidence": 0.19999999999999996,
      "content_id": "a7e3caef-c347-4678-a695-2e12db30e6ac",
      "creator_id": "ratelimit-test",
      "label": "likely_human",
      "llm_score": 0.8,
      "signal_1_reliable": true,
      "signal_1_score": 0.19999999999999996,
      "signal_2_reliable": false,
      "signal_2_score": 1.0,
      "status": "classified",
      "timestamp": "2026-07-01T04:36:49.795Z"
    }

## Known Limitations

### Sometimes formal/academic writing get misclassified as AI-generated

Both signals independently flag showed that dense academic or professional prose genuinely has: Signal 1's LLM classifier reads generic hedging language, balanced/non-committal claims, and absence of personal specifics as AI-like. Signal 2 reads the typically uniform sentence structure of edited professional writing as low-variation, also AI-like. Through testing, a genuine economics-paper-style paragraph (about monetary policy and asset prices) scored 0.748 — confidently likely_ai — despite the fact that it should be plausible piece of human academic writing. This isn't a random miscalibration; it's shows a direct consequence of what both signals are built to measure. The planning doc's own Anticipated Edge Cases section predicted a version of this (careful non-native writers triggering false positives for structural reasons) — this generalizes that same failure mode to any writer using a formal register, not just non-native speakers. The appeals workflow exists partly to catch exactly this category.

## Spec Reflection

### Where specs helped

The explicit reliability-weighted combiner formula (where t=0 for unreliable signals, forced None/insufficient-text if both are unreliable) gave a clean, unambiguous mechanism for handling short or low-confidence input without ad hoc guessing. This made the whole scoring layer easy to test rigorously — forcing combined_score to 0.1, 0.4, 0.65, 0.9, and None all traced to exactly the expected bucket and exact expected text, with no ambiguous cases.

### Where the implementation diverged

## AI Usage

## Portfolio Walkthrough
