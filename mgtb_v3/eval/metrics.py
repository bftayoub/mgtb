from __future__ import annotations


def safe_mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def recovery(acc_int4_v3: float, acc_int4: float, acc_fp16: float) -> float:
    denom = acc_fp16 - acc_int4
    if abs(denom) < 1e-12:
        return 0.0
    return (acc_int4_v3 - acc_int4) / denom


def compute_metrics(results: list[dict]) -> dict:
    if not results:
        return {}
    return {
        "accuracy": safe_mean(r.get("correct", 0.0) for r in results),
        "tokens_generated": safe_mean(r.get("tokens_generated", len(r.get("tokens", []))) for r in results),
        "latency": safe_mean(r.get("latency", 0.0) for r in results),
        "number_of_alerts": safe_mean(r.get("number_of_alerts", len(r.get("alerts", []))) for r in results),
        "number_of_backtracks": safe_mean(r.get("number_of_backtracks", len(r.get("backtracks", []))) for r in results),
        "false_alert_rate": safe_mean(1.0 if r.get("false_alert", False) else 0.0 for r in results),
        "degeneration_rate": safe_mean(1.0 if r.get("degenerated", False) else 0.0 for r in results),
    }
