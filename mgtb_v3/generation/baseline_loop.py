from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from mgtb_v3.control.cache_utils import crop_hf_cache
from mgtb_v3.features.repetition import NgramTracker
from mgtb_v3.generation.hf_loop import _sample_token
from mgtb_v3.logging.trace_logger import TraceLogger


def generate_scheduled_backtracking(
    model,
    tokenizer,
    prompt: str,
    config,
    schedule: list[dict[str, int]],
    max_new_tokens: int,
    trace_log_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate with externally scheduled backtracks, without detector/calibrator access."""
    import torch

    model.eval()
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    prompt_len = int(input_ids.shape[-1])
    tokens = input_ids[0].tolist()
    generated = input_ids
    cache = None
    ngram_tracker = NgramTracker(
        config.window.ngram_min,
        config.window.ngram_max,
        prompt_tokens=tokens,
        exclude_prompt=config.window.exclude_prompt_ngrams,
    )
    temperature = 1.0
    repetition_penalty = 1.0
    bad_ngrams: list[tuple[int, ...]] = []
    next_event = 0
    backtracks: list[dict[str, Any]] = []
    started = time.time()

    with TraceLogger(trace_log_path) as logger:
        for _ in range(max_new_tokens):
            with torch.no_grad():
                if cache is None:
                    outputs = model(input_ids=generated, use_cache=True)
                else:
                    outputs = model(input_ids=generated[:, -1:], past_key_values=cache, use_cache=True)
            logits = outputs.logits[:, -1, :]
            cache = getattr(outputs, "past_key_values", None)
            token_id = int(
                _sample_token(
                    logits,
                    temperature=temperature,
                    generated_tokens=tokens,
                    repetition_penalty=repetition_penalty,
                    bad_ngrams=bad_ngrams,
                ).item()
            )
            tokens.append(token_id)
            generated = torch.tensor([[token_id]], device=model.device)
            token_pos = len(tokens) - 1
            # Scheduled controls need repeated n-grams for redecode masking, but
            # not entropy/log-probability features or detector windows.
            ngram_tracker.update([token_id], [0.0], token_pos)
            logger.log_token(token_pos, token_id, None, None)
            if token_id == getattr(tokenizer, "eos_token_id", None):
                break

            current_pos = len(tokens)
            generated_count = current_pos - prompt_len
            if next_event < len(schedule) and generated_count >= int(schedule[next_event]["trigger_at"]):
                requested = max(1, int(schedule[next_event]["rollback_tokens"]))
                rollback_pos = max(prompt_len, current_pos - requested)
                faulty = ngram_tracker.faulty_ngrams(rollback_pos, current_pos)
                cache = crop_hf_cache(cache, rollback_pos)
                ngram_tracker.truncate(rollback_pos)
                tokens = tokens[:rollback_pos]
                generated = torch.tensor([tokens], device=model.device)
                temperature = float(config.backtracking.redecode_temperature)
                repetition_penalty = float(config.backtracking.repetition_penalty)
                bad_ngrams = [tuple(map(int, ngram)) for ngram in faulty] if config.backtracking.use_no_bad_ngrams else []
                event = {
                    "applied": True,
                    "trigger_at": int(schedule[next_event]["trigger_at"]),
                    "rollback_pos": rollback_pos,
                    "requested_rollback_tokens": requested,
                    "actual_rollback_tokens": current_pos - rollback_pos,
                    "reroll_index": len(backtracks) + 1,
                    "decode_overrides": {
                        "temperature": temperature,
                        "repetition_penalty": repetition_penalty,
                        "use_no_bad_ngrams": bool(config.backtracking.use_no_bad_ngrams),
                    },
                }
                backtracks.append(event)
                logger.log_backtrack(event)
                next_event += 1

    return {
        "text": tokenizer.decode(tokens, skip_special_tokens=True),
        "tokens": tokens,
        "alerts": [],
        "backtracks": backtracks,
        "interventions": backtracks,
        "trace_log_path": str(trace_log_path) if trace_log_path else None,
        "latency": time.time() - started,
    }
