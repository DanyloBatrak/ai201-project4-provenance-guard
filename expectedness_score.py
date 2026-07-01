"""
Signal 1 - Predictability (expectedness_score)

Measures how statistically predictable each token is given its preceding
context, using a small reference language model. Lower perplexity means
the text is more "expected" (AI-like); higher perplexity means it is more
surprising (human-like).

Per spec:
- Output: float in [0, 1]. 1.0 = maximally predictable (low perplexity,
  more AI-like). 0.0 = maximally unpredictable (high perplexity, more
  human-like).
- Normalization: clip raw perplexity to [10, 100], rescale linearly,
  then invert into [0, 1].
- reliable: False if the text is under 20 tokens (perplexity is noisy on
  very short output).
"""

import math
import re
from typing import List, Tuple

MIN_TOKENS_FOR_RELIABILITY = 20
PERPLEXITY_CLIP_MIN = 10.0
PERPLEXITY_CLIP_MAX = 100.0


def _tokenize(text: str) -> List[str]:
    """Minimal whitespace/punctuation tokenizer.

    This is a stand-in tokenizer good enough for stable token counts and
    reliability checks. Swap for the reference model's own tokenizer once
    a production LM is wired in (see ReferenceLanguageModel below).
    """
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


class ReferenceLanguageModel:
    """Placeholder reference LM used to score token predictability.

    TODO (before production): replace this with a real small reference
    model (e.g. distilgpt2 / gpt2-small) that returns per-token
    log-probabilities given preceding context. Everything else in this
    module only depends on the `token_logprobs` interface below, so
    swapping the implementation should not require any other changes.

    This placeholder uses a tiny built-in bigram frequency table so the
    pipeline is runnable end-to-end offline, without a network call or a
    model download. It has no real linguistic knowledge and is only
    meant to unblock the /submit endpoint and its tests in M3.
    """

    _SMALL_CORPUS = (
        "the quick brown fox jumps over the lazy dog "
        "this is a simple sentence used to seed a toy model "
        "it is not a substitute for a real language model "
        "please replace this placeholder with an actual reference model "
        "before this service is used in production"
    )

    def __init__(self):
        tokens = _tokenize(self._SMALL_CORPUS.lower())
        self._unigram_counts = {}
        self._bigram_counts = {}
        for tok in tokens:
            self._unigram_counts[tok] = self._unigram_counts.get(tok, 0) + 1
        for prev, cur in zip(tokens, tokens[1:]):
            key = (prev, cur)
            self._bigram_counts[key] = self._bigram_counts.get(key, 0) + 1
        self._vocab_size = max(len(self._unigram_counts), 1)
        self._total_unigrams = max(sum(self._unigram_counts.values()), 1)

    def token_logprobs(self, tokens: List[str]) -> List[float]:
        """Return a natural-log probability estimate for each token, given
        the immediately preceding token, with add-one (Laplace) smoothing
        over the toy corpus above.
        """
        logprobs = []
        for i, tok in enumerate(tokens):
            tok = tok.lower()
            if i == 0:
                count = self._unigram_counts.get(tok, 0)
                prob = (count + 1) / (self._total_unigrams + self._vocab_size)
            else:
                prev = tokens[i - 1].lower()
                bigram_count = self._bigram_counts.get((prev, tok), 0)
                prev_count = self._unigram_counts.get(prev, 0)
                prob = (bigram_count + 1) / (prev_count + self._vocab_size)
            logprobs.append(math.log(prob))
        return logprobs


# Reference model is small and stateless enough to load once at import time.
_reference_model = ReferenceLanguageModel()


def _perplexity(tokens: List[str]) -> float:
    logprobs = _reference_model.token_logprobs(tokens)
    if not logprobs:
        return PERPLEXITY_CLIP_MAX
    avg_neg_logprob = -sum(logprobs) / len(logprobs)
    return math.exp(avg_neg_logprob)


def _normalize_perplexity(perplexity: float) -> float:
    """Clip to [PERPLEXITY_CLIP_MIN, PERPLEXITY_CLIP_MAX], rescale linearly
    into [0, 1], then invert so low perplexity -> high score (AI-like).
    """
    clipped = max(PERPLEXITY_CLIP_MIN, min(PERPLEXITY_CLIP_MAX, perplexity))
    span = PERPLEXITY_CLIP_MAX - PERPLEXITY_CLIP_MIN
    normalized = (clipped - PERPLEXITY_CLIP_MIN) / span  # 0 (low ppl) .. 1 (high ppl)
    return 1.0 - normalized  # invert: low ppl -> 1.0, high ppl -> 0.0


def expectedness_score(text: str) -> Tuple[float, bool]:
    """Compute Signal 1 (predictability) for a piece of text.

    Returns:
        (score, reliable)
        score: float in [0, 1]; 1.0 = maximally predictable / AI-like,
            0.0 = maximally unpredictable / human-like.
        reliable: False if the text has fewer than
            MIN_TOKENS_FOR_RELIABILITY tokens (perplexity is noisy on
            very short output).
    """
    tokens = _tokenize(text)
    reliable = len(tokens) >= MIN_TOKENS_FOR_RELIABILITY

    if not tokens:
        return 0.0, False

    perplexity = _perplexity(tokens)
    score = _normalize_perplexity(perplexity)
    return score, reliable

 
MIN_TOKENS_FOR_RELIABILITY = 20
 
# Neutral placeholder score. Never actually used in combine_score()
# while reliable=False, since the combiner weights unreliable signals
# at 0 -- kept as 0.5 (rather than 0.0 or 1.0) so that if this ever gets
# wired in by mistake before the real LM swap, it doesn't silently bias
# the combined score toward "AI" or "human".
_PLACEHOLDER_SCORE = 0.5
 
 
def expectedness_score(text: str) -> Tuple[float, bool]:
    """Stub for Signal 1 (predictability). Always returns
    (0.5, False) -- i.e. "no real reading, don't trust this" --
    until a real reference LM is wired in. See module docstring.
    """
    return _PLACEHOLDER_SCORE, False