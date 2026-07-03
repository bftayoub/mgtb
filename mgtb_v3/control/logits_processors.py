from __future__ import annotations


def build_no_bad_words_from_ngrams(ngrams, tokenizer=None):
    bad_words_ids = [list(map(int, ngram)) for ngram in ngrams if ngram]
    try:
        from transformers import NoBadWordsLogitsProcessor

        eos_token_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None
        return NoBadWordsLogitsProcessor(bad_words_ids=bad_words_ids, eos_token_id=eos_token_id)
    except Exception:
        return bad_words_ids
