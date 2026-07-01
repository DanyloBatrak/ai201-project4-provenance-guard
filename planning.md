# Provenance Guard

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

## Uncertain Representation

A combined_score is not "probability this is AI" in strict statistical sense but it is calibrated index of how strongly the two signals agree that the text looks AI-like. For example, when score is 0.6 then signals leans toward AI-like patterns but not strongly enough to rule out human authorship

###

### Thresholds

Range | Label bucket |
0.00 - 0.35 | Likely human
0.36 - 0.65 | Uncertain
0.65 - 1.00 | Likely AI

Although, regadless of the score, if reliable is False for both signals then bucket is forced to Uncertain with message that text was too short to analyze.

## Transparency Label Design

### High Confidense AI

It would happen when: {score} >= 0.65

Response: "Confidence: {score} / 1.0. This text shows strong signals cosistent usage of AI generation. This is an automated estimate, not a determination of fact."

### High Confidense Human

It would happen when: {score} <= 0.35

Response: "Confidence: {score} / 1.0. This text shows signals of consistant human writing. Confidence: {score} / 1.0. This is an automated estimate, not a determination of fact."

### Uncertain

It would happen when: 0.35 < score < 0.65

Response: " Confidence: {score} / 1.0. We could not confidently determine if this text been AI generated or been human written.

### Overall

All three variants always show the numeric score and a reminder that the result is an estimate — no label is ever presented as a bare verdict.

## Appeals Workflow

### Who can submit appeal?

The original submitter that identified by submission_id

### What information do they provide?

required submission_id, reason (optional free text), contact (optional)

### What does the system do when an appeal is received — what status changes, what gets logged?

System would do these following steps:

1. Look up submission by ID. If submission is not found then return 404
2. Set submission status from completed to under_review
3. Create an appeal record linked to submission_id storing the reason / contact / timestamp
4. Write a audit log entry: {event: "appeal_submitted", submission_id, appeal_id, timestamp }
5. Return {appeal_id, status: "under_review"} to the user.

### What would a human reviewer see when they open the appeal queue?

For each pending appeal - the original text, both raw signal scores and their reliable flags, the combined score, the label that was shown, the submitter's stated reason, and a side-by-side decision control (overturn to AI / overturn to human / overturn to uncertain) that, once submitted, writes a resolution audit entry and updates submission status to resolved.

## Anticipated Edge Cases

1. Repetitive, simple-vocabulary creative writing: a poem or children's-style text using heavy repetition and short. Simple sentences will likely score low on burstiness variation (looks uniform) and high on predictability (simple words are statistically likely) which push it toward "Likely AI" categoty despite the fact that it's clearly human and intentionally stylized.
2. Non-native English writing: a careful non-native speaker writing grammatically simple, correct sentences will often score low-perplexity and low-burstiness for the same structural reasons, producing a false "Likely AI" result for a human who is simply writing cautiously.
3. Very short submissions: both of signals are not reliable; the system must force "Uncertain — insufficient text" rather than send a misleadingly confident score from noisy statistics.
4. Heavily edited AI-assisted text: A human draft with AI assistance then substantially rewrites. This can land anywhere in the middle range and this case for "Uncertain" bucket, however reviewers should expect a high appeal volume from this category because the underlying authorship is genuinely mixed.

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

## AI Tool Plan

### M3 (Submission endpoint + first signal)

#### Spec sections to provide

Detection Signal (1 signal only) + Architecture Diagram

#### What to ask for?

A minimal Flask app skeleton with POST /submit accepting {text} and returning a stab response and also a independent expectedness_score(text) function that implement the perplexity-based signal with reliable flag logic.

#### How to verify?

It would call expectedness_score(text) directly on 3-4 hand-picked inputs (a clearly AI-sounding paragraph, a clearly human paragraph, a very short string) and confirm scores land in [0,1] and the reliable flag is False for the short input, before wiring it into the endpoint.

### M4 (Second signal + confidence scoring)

#### Spec sections to provide

Detection Signal (Second signal + confidence scoring) + Uncertainty Representation + Architecture Diagram

#### What to ask for?

A clustering_score function and a combine_score(sig1, sig2) function that implement the weighting/reliability logic from Detection Signals section

#### How to verify?

It would run both signals and the combiner on a known-AI sample and a known-human sample, and confirm the combined scores differ meaningfully (ideally landing in opposite threshold buckets) rather than clustering near the same value.

### M5 (Production layer)

#### Spec sections to provide

Transparency Label Design (all three variants) + Appeals Workflow + Architecture diagram (both flows).

#### What to ask for?

A generate_label(combined_score) function returning the exact label text and bucket from Transparency Label Design and POST / appeal endpoint implementing the status/audit logic from section Appeals Workflow.

#### How to verify?

Force comnined_score to test values (like 0.1, 0.4, 0.9 and None) to confirm all three label variants and the insufficient-text case are reachable and word-for-word correct. Then it would submit a test appeal against a known submission_id and confirm status flips to under_review and a corresponding audit log entry appears.
