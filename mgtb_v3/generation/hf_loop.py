from __future__ import annotations

import time
from pathlib import Path

from mgtb_v3.config import MGTBV3Config
from mgtb_v3.control.backtracking import BacktrackingController
from mgtb_v3.detector.e_detector import EDetector
from mgtb_v3.features.window_features import TrajectoryMonitor, linear_window_score
from mgtb_v3.logging.trace_logger import TraceLogger
from mgtb_v3.types import AlertInfo, GenerationResult, WindowScore


def generate_with_mgtb_v3(
    model,
    tokenizer,
    prompt: str,
    config: MGTBV3Config,
    calibrator,
    threshold: float,
    max_new_tokens: int = 256,
    trace_log_path: str | Path | None = None,
    do_backtracking: bool = True,
):
    import torch

    model.eval()
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    prompt_len = int(input_ids.shape[-1])
    tokens = input_ids[0].tolist()
    generated = input_ids
    cache = None
    monitor = TrajectoryMonitor(config, prompt_tokens=tokens)
    detector = EDetector(
        threshold,
        gammas=config.detector.betting_gammas,
        p_clip=config.detector.p_clip,
        refractory_windows=config.detector.refractory_windows,
    )
    backtracker = BacktrackingController(config)
    alerts: list[AlertInfo] = []
    backtracks: list[dict] = []
    start_time = time.time()
    decode_temperature = 1.0
    repetition_penalty = 1.0
    bad_ngrams: list[tuple[int, ...]] = []

    with TraceLogger(trace_log_path) as logger:
        for _ in range(max_new_tokens):
            with torch.no_grad():
                if cache is None:
                    outputs = model(input_ids=generated, use_cache=True)
                else:
                    outputs = model(input_ids=generated[:, -1:], past_key_values=cache, use_cache=True)
            logits = outputs.logits[:, -1, :]
            cache = getattr(outputs, "past_key_values", None)
            next_token = _sample_token(
                logits,
                temperature=decode_temperature,
                generated_tokens=tokens,
                repetition_penalty=repetition_penalty,
                bad_ngrams=bad_ngrams,
            )
            token_id = int(next_token.item())
            tokens.append(token_id)
            generated = torch.tensor([tokens], device=model.device)
            monitor.update_token(token_id, logits[0])
            stat = monitor.token_stats[-1]
            logger.log_token(stat.position, token_id, stat.entropy, stat.logprob)

            if token_id == getattr(tokenizer, "eos_token_id", None):
                break

            while monitor.should_emit_window():
                features = monitor.compute_window_features()
                raw_score = linear_window_score(features, config.score)
                p_value = calibrator.p_value(raw_score, features.end_pos)
                update = detector.update(p_value)
                window_score = WindowScore(
                    features=features,
                    raw_score=raw_score,
                    p_value=p_value,
                    e_value=update["e_value"],
                    logE=update["logE"],
                    alert=update["alert"],
                )
                logger.log_window(window_score)
                if update["alert"]:
                    active_prompt_len = monitor.prompt_len
                    cp_window = detector.changepoint_window()
                    cp_token = active_prompt_len + cp_window * config.window.stride
                    rollback = max(active_prompt_len, cp_token - config.backtracking.margin_tokens)
                    alert = AlertInfo(
                        window_index=features.window_index,
                        token_pos=features.end_pos,
                        changepoint_window=cp_window,
                        rollback_token_pos=rollback,
                        score=raw_score,
                        p_value=p_value,
                        logE=update["logE"],
                    )
                    alerts.append(alert)
                    if do_backtracking:
                        event = backtracker.on_alert(alert, tokens, cache, monitor, detector, active_prompt_len)
                        if event.get("applied"):
                            tokens = event["tokens"]
                            cache = event["cache"]
                            injection_text = str(event.get("wait_injection_text") or "")
                            injection_tokens = _encode_injection_tokens(tokenizer, injection_text)
                            if injection_tokens:
                                tokens = [*tokens, *injection_tokens]
                                cache = None
                                monitor = TrajectoryMonitor(config, prompt_tokens=tokens)
                                event["injected_token_count"] = len(injection_tokens)
                                event["injected_tokens"] = injection_tokens
                            else:
                                event["injected_token_count"] = 0
                            generated = torch.tensor([tokens], device=model.device)
                            overrides = event.get("decode_overrides", {})
                            decode_temperature = float(overrides.get("temperature", decode_temperature))
                            repetition_penalty = float(overrides.get("repetition_penalty", repetition_penalty))
                            if overrides.get("use_no_bad_ngrams", False):
                                bad_ngrams = [tuple(map(int, ngram)) for ngram in event.get("bad_ngrams", [])]
                            backtracks.append({k: v for k, v in event.items() if k not in {"tokens", "cache"}})
                            logger.log_backtrack(backtracks[-1])
                    break

    text = tokenizer.decode(tokens, skip_special_tokens=True)
    result = GenerationResult(text=text, tokens=tokens, alerts=alerts, backtracks=backtracks, trace_log_path=str(trace_log_path) if trace_log_path else None)
    result.latency = time.time() - start_time  # dynamic field for eval convenience
    return result


def _sample_token(
    logits,
    temperature: float = 1.0,
    generated_tokens: list[int] | None = None,
    repetition_penalty: float = 1.0,
    bad_ngrams: list[tuple[int, ...]] | None = None,
):
    import torch

    adjusted = logits.detach().clone()
    generated_tokens = generated_tokens or []
    if repetition_penalty and repetition_penalty > 1.0 and generated_tokens:
        for token_id in set(int(t) for t in generated_tokens):
            if 0 <= token_id < adjusted.shape[-1]:
                value = adjusted[:, token_id]
                adjusted[:, token_id] = torch.where(value < 0, value * repetition_penalty, value / repetition_penalty)
    _mask_bad_ngram_completions(adjusted, generated_tokens, bad_ngrams or [])
    logits = adjusted / max(float(temperature), 1e-6)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _encode_injection_tokens(tokenizer, text: str) -> list[int]:
    if not text:
        return []
    try:
        encoded = tokenizer(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer(text)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def _mask_bad_ngram_completions(logits, generated_tokens: list[int], bad_ngrams: list[tuple[int, ...]]) -> None:
    if not generated_tokens:
        return
    for ngram in bad_ngrams:
        if not ngram:
            continue
        if len(ngram) == 1:
            logits[:, int(ngram[0])] = float("-inf")
            continue
        prefix = list(ngram[:-1])
        if len(generated_tokens) >= len(prefix) and generated_tokens[-len(prefix) :] == prefix:
            banned_token = int(ngram[-1])
            if 0 <= banned_token < logits.shape[-1]:
                logits[:, banned_token] = float("-inf")
