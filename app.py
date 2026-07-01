"""
Provenance Guard - Flask app

NOTE on SUBMISSIONS: the audit log (audit.py) is append-only, so it
can't be queried/updated by content_id directly. This adds a simple
in-memory dict as a lookup store for "does this content_id exist, what
was its data, what's its current status" -- needed because /appeal has
to look up and update a specific submission (per spec: "look up
submission by ID, if not found return 404"). This resets on server
restart, which is fine for this milestone's scope (append-only audit
log remains the permanent record; SUBMISSIONS is just a live index).
"""

import uuid

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from audit import get_log, log_event
from signals import get_groq_attribution, clustering_score
from scoring import combine_score, generate_label, signal1_to_score

app = Flask(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

# In-memory index: content_id -> submission record. See module docstring.
SUBMISSIONS = {}


@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute;100 per day")
def submit():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")
    creator_id = payload.get("creator_id")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "field 'text' (non-empty string) is required"}), 400
    if not creator_id:
        return jsonify({"error": "field 'creator_id' is required"}), 400

    content_id = str(uuid.uuid4())

    try:
        signal_1_result = get_groq_attribution(text)
    except Exception as exc:
        signal_1_result = {
            "verdict": "uncertain",
            "confidence": None,
            "reasoning": f"signal_1 unavailable: {exc}",
        }

    s1_score, s1_reliable = signal1_to_score(signal_1_result)
    s2_score, s2_reliable = clustering_score(text)

    combined = combine_score((s1_score, s1_reliable), (s2_score, s2_reliable))
    label_result = generate_label(combined)  # {"bucket": ..., "message": ...}

    response = {
        "content_id": content_id,
        "creator_id": creator_id,
        "attribution": signal_1_result,
        "signal_scores": {
            "signal_1_score": s1_score,
            "signal_1_reliable": s1_reliable,
            "signal_2_score": s2_score,
            "signal_2_reliable": s2_reliable,
        },
        "confidence": combined,
        "label": label_result,
    }

    SUBMISSIONS[content_id] = {
        "text": text,
        "creator_id": creator_id,
        "attribution": signal_1_result,
        "signal_1_score": s1_score,
        "signal_1_reliable": s1_reliable,
        "signal_2_score": s2_score,
        "signal_2_reliable": s2_reliable,
        "confidence": combined,
        "label_bucket": label_result["bucket"],
        "label_message": label_result["message"],
        "status": "classified",
        "appeal_reasoning": None,
    }

    log_event(
        {
            "content_id": content_id,
            "creator_id": creator_id,
            "attribution": signal_1_result,
            "signal_1_score": s1_score,
            "signal_1_reliable": s1_reliable,
            "signal_2_score": s2_score,
            "signal_2_reliable": s2_reliable,
            "llm_score": signal_1_result.get("confidence"),
            "confidence": combined,
            "label": label_result["bucket"],
            "status": "classified",
        }
    )

    return jsonify(response), 200


@app.route("/appeal", methods=["POST"])
def appeal():
    payload = request.get_json(silent=True) or {}
    content_id = payload.get("content_id")
    creator_reasoning = payload.get("creator_reasoning")

    if not content_id:
        return jsonify({"error": "field 'content_id' is required"}), 400

    submission = SUBMISSIONS.get(content_id)
    if submission is None:
        return jsonify({"error": f"no submission found for content_id '{content_id}'"}), 404

    submission["status"] = "under_review"
    submission["appeal_reasoning"] = creator_reasoning

    appeal_id = str(uuid.uuid4())

    log_event(
        {
            "event": "appeal_submitted",
            "content_id": content_id,
            "appeal_id": appeal_id,
            "appeal_reasoning": creator_reasoning,
            "status": "under_review",
        }
    )

    return jsonify({"appeal_id": appeal_id, "status": "under_review"}), 200


@app.route("/log", methods=["GET"])
def get_log_route():
    entries = get_log()
    for entry in entries:
        content_id = entry.get("content_id")
        submission = SUBMISSIONS.get(content_id) if content_id else None
        entry["appealed"] = bool(submission and submission.get("appeal_reasoning") is not None)
    return jsonify({"entries": entries})


if __name__ == "__main__":
    app.run(debug=True)