from __future__ import annotations

from copy import deepcopy

from mgtb_v3.config import MGTBV3Config


ABLATION_MODES = [
    "vanilla",
    "mgtb_v2_baseline",
    "mgtb_v3_window",
    "entropy_threshold",
    "direct_score_threshold",
    "e_detector",
    "random_trigger",
    "fixed_k_backtrack",
    "adaptive_cp_backtrack",
    "continue_penalty_no_backtrack",
    "repetition_penalty_always_on",
    "knockout_no_entropy",
    "knockout_no_logprob",
    "knockout_no_repetition",
    "knockout_no_confident_loop",
    "knockout_no_local_entropy",
    "global_calibration",
    "positional_calibration",
    "token_level",
    "window_level",
]


def config_for_ablation(base: MGTBV3Config, mode: str) -> MGTBV3Config:
    cfg = deepcopy(base)
    if mode == "knockout_no_entropy":
        cfg.score.w_entropy = 0.0
    elif mode == "knockout_no_logprob":
        cfg.score.w_logprob = 0.0
    elif mode == "knockout_no_repetition":
        cfg.score.w_repetition = 0.0
    elif mode == "knockout_no_confident_loop":
        cfg.score.w_confident_loop = 0.0
    elif mode == "knockout_no_local_entropy":
        cfg.score.w_local_entropy_pos = 0.0
        cfg.score.w_local_entropy_neg = 0.0
    elif mode == "fixed_k_backtrack":
        cfg.backtracking.use_adaptive_changepoint = False
        cfg.backtracking.fixed_rollback_tokens = cfg.backtracking.fixed_rollback_tokens or 128
    elif mode == "adaptive_cp_backtrack":
        cfg.backtracking.use_adaptive_changepoint = True
    elif mode == "token_level":
        cfg.window.window_size = 1
        cfg.window.stride = 1
    elif mode == "window_level":
        cfg.window.window_size = max(2, cfg.window.window_size)
    return cfg
