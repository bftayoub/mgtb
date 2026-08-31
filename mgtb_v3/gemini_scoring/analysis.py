from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from mgtb_v3.science_fast.io import atomic_write_json, load_json, sha256_json

from .audit import arbitration_candidates, individual_audit_candidates
from .config import ScoringConfig
from .runner import prepare_scope
from .store import ArtifactStore, utc_now


def _matrix(rows: list[dict[str, str]], left: str, right: str) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        matrix[row.get(left) or "N/A"][row.get(right) or "N/A"] += 1
    return {key: dict(value) for key, value in sorted(matrix.items())}


def _exact_mcnemar(corrections: int, regressions: int) -> float:
    discordant = corrections + regressions
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(corrections, regressions) + 1))
    return float(min(1.0, 2.0 * tail / (2**discordant)))


def _binary_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["verdict"] for row in rows)
    by_seed: dict[int, list[int]] = defaultdict(list)
    by_domain: dict[str, list[int]] = defaultdict(list)
    by_difficulty: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        correct = int(row["verdict"] == "TRUE")
        by_seed[int(row["replicate_seed"])].append(correct)
        for domain in row.get("domains") or ["unclassified"]:
            by_domain[str(domain)].append(correct)
        difficulty = row.get("difficulty")
        by_difficulty[str(difficulty if difficulty is not None else "unknown")].append(correct)
    total = len(rows)
    return {
        "total": total,
        "counts": {key: counts[key] for key in ("TRUE", "FALSE", "ABSTAIN")},
        "accuracy": counts["TRUE"] / total,
        "abstain_rate": counts["ABSTAIN"] / total,
        "abstain_is_incorrect_for_accuracy": True,
        "accuracy_by_replicate_seed": {
            str(seed): float(np.mean(values)) for seed, values in sorted(by_seed.items())
        },
        "accuracy_by_domain_descriptive": {
            domain: {"items": len(values), "accuracy": float(np.mean(values))}
            for domain, values in sorted(by_domain.items())
        },
        "accuracy_by_difficulty_descriptive": {
            difficulty: {"items": len(values), "accuracy": float(np.mean(values))}
            for difficulty, values in sorted(by_difficulty.items())
        },
    }


def _binary_paired_comparison(
    left: list[dict[str, Any]], right: list[dict[str, Any]], *, bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    left_by_key = {
        (row["source_item_id"], int(row["replicate_seed"])): row for row in left
    }
    right_by_key = {
        (row["source_item_id"], int(row["replicate_seed"])): row for row in right
    }
    if left_by_key.keys() != right_by_key.keys():
        raise ValueError("full Gemini results are not paired by source item and replicate seed")
    keys = sorted(left_by_key)
    baseline = np.array(
        [left_by_key[key]["verdict"] == "TRUE" for key in keys], dtype=np.int8,
    )
    method = np.array(
        [right_by_key[key]["verdict"] == "TRUE" for key in keys], dtype=np.int8,
    )
    difference = method - baseline
    corrections = int(np.sum((baseline == 0) & (method == 1)))
    regressions = int(np.sum((baseline == 1) & (method == 0)))
    clusters: dict[str, list[float]] = defaultdict(list)
    for key, value in zip(keys, difference):
        clusters[key[0]].append(float(value))
    cluster_means = np.array([np.mean(clusters[key]) for key in sorted(clusters)])
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.choice(
        cluster_means, size=(bootstrap_samples, len(cluster_means)), replace=True,
    ).mean(axis=1)
    return {
        "paired_units": len(keys),
        "problem_clusters": len(cluster_means),
        "baseline_accuracy": float(np.mean(baseline)),
        "method_accuracy": float(np.mean(method)),
        "difference": float(np.mean(difference)),
        "corrections": corrections,
        "regressions": regressions,
        "paired_cluster_bootstrap_95_ci": np.quantile(draws, [0.025, 0.975]).tolist(),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "mcnemar_exact_two_sided_p": _exact_mcnemar(corrections, regressions),
        "mcnemar_note": (
            "Exact unit-level McNemar is descriptive; the problem-cluster bootstrap is primary."
        ),
    }


def full_statistics(
    joined: list[dict[str, Any]], variants: tuple[str, ...] | list[str], *,
    bootstrap_seed: int = 20260831, bootstrap_samples: int = 10000,
) -> dict[str, Any]:
    """Compute complete full-scope statistics from authenticated, locally restored rows."""
    variants = list(variants)
    if "vanilla" not in variants:
        raise ValueError("full Gemini analysis requires vanilla as the declared baseline")
    by_variant = {variant: [row for row in joined if row["variant"] == variant] for variant in variants}
    if any(not rows for rows in by_variant.values()):
        raise ValueError("full Gemini analysis has an empty variant")
    summaries = {variant: _binary_summary(rows) for variant, rows in by_variant.items()}
    comparisons = {
        variant: _binary_paired_comparison(
            by_variant["vanilla"], rows, bootstrap_seed=bootstrap_seed,
            bootstrap_samples=bootstrap_samples,
        )
        for variant, rows in by_variant.items() if variant != "vanilla"
    }
    ordered = sorted(comparisons, key=lambda name: comparisons[name]["mcnemar_exact_two_sided_p"])
    running, count = 0.0, len(ordered)
    for rank, name in enumerate(ordered):
        raw = comparisons[name]["mcnemar_exact_two_sided_p"]
        running = max(running, min(1.0, (count - rank) * raw))
        comparisons[name]["mcnemar_holm_adjusted_p"] = running
    agreement_rows = [row for row in joined if row.get("old_omni_verdict") in {"TRUE", "FALSE"}]
    agreement = (
        sum(row["verdict"] == row["old_omni_verdict"] for row in agreement_rows)
        / len(agreement_rows) if agreement_rows else None
    )
    additional_pairwise = {}
    if {"full_mgtb", "matched_random"}.issubset(by_variant):
        additional_pairwise["full_mgtb_vs_matched_random"] = _binary_paired_comparison(
            by_variant["matched_random"], by_variant["full_mgtb"],
            bootstrap_seed=bootstrap_seed, bootstrap_samples=bootstrap_samples,
        )
    return {
        "methods": summaries,
        "comparisons_against_vanilla": comparisons,
        "additional_pairwise_comparisons_descriptive": additional_pairwise,
        "agreement_with_omni_judge": {
            "comparable_units": len(agreement_rows),
            "agreement": agreement,
            "matrix": _matrix(agreement_rows, "verdict", "old_omni_verdict"),
        },
        "primary_inference": (
            "paired bootstrap clustered by source problem; all replicate seeds remain in a cluster"
        ),
        "multiplicity": "Holm adjustment across comparisons against vanilla",
        "subgroup_note": "domain results are descriptive",
    }


def analyze_pilot(config: ScoringConfig) -> dict[str, Any]:
    pilot = prepare_scope(config, "pilot")
    grouped_store = ArtifactStore(config.output_root / "pilot")
    individual_store = ArtifactStore(config.output_root / "pilot_individual")
    arbitration_store = ArtifactStore(config.output_root / "pilot_arbitration")
    joined = []
    for row in pilot:
        decision = grouped_store.valid_decision(row["unit_id"])
        arbitration = arbitration_store.valid_decision(row["unit_id"])
        primary_final = decision["verdict"] if decision else None
        joined.append({
            "unit_id": row["unit_id"], "stratum": row["pilot_stratum"],
            "gemini": decision["verdict"] if decision and decision["source"].startswith("gemini") else None,
            "control": row["control"]["verdict"], "omni_judge": row["old_omni_verdict"],
            "arbitration": arbitration["verdict"] if arbitration else None,
            "final": arbitration["verdict"] if arbitration else primary_final,
        })
    final_counts = Counter(row["final"] or "PENDING" for row in joined)
    candidate_by_id = {row["unit_id"]: row for row in pilot}
    by_variant: dict[str, Any] = {}
    for variant in config.variants:
        variant_rows = [row for row in joined if candidate_by_id[row["unit_id"]]["variant"] == variant]
        counts = Counter(row["final"] or "PENDING" for row in variant_rows)
        by_variant[variant] = {"total": len(variant_rows), **dict(counts)}
    audits = individual_audit_candidates(config)
    comparisons = []
    for row in audits:
        grouped = grouped_store.valid_decision(row["unit_id"])
        individual = individual_store.valid_decision(row["unit_id"])
        comparisons.append({
            "unit_id": row["unit_id"],
            "grouped": grouped["verdict"] if grouped else None,
            "individual": individual["verdict"] if individual else None,
        })
    complete_pairs = [row for row in comparisons if row["grouped"] and row["individual"]]
    agreement = (sum(row["grouped"] == row["individual"] for row in complete_pairs) / len(complete_pairs)
                 if complete_pairs else None)
    symbolic_audit = []
    for row in pilot:
        if row["pilot_stratum"] != "symbolic":
            continue
        decision = grouped_store.valid_decision(row["unit_id"])
        symbolic_audit.append({
            "unit_id": row["unit_id"], "variant": row["variant"], "seed": row["replicate_seed"],
            "domain": (row.get("domains") or [None])[0], "problem": row["problem"],
            "reference_answer": row["reference_answer"], "candidate_answer": row["candidate_answer"],
            "verdict": decision["verdict"] if decision else "PENDING",
            "reason": decision["reason"] if decision else "not judged",
        })
    completed = sum(value for key, value in final_counts.items() if key != "PENDING")
    abstain_rate = final_counts["ABSTAIN"] / completed if completed else None
    pilot_complete = completed == len(pilot)
    audit_complete = len(complete_pairs) == config.individual_audit_count
    arbitration_target = arbitration_candidates(config)
    arbitration_complete = all(
        arbitration_store.valid_decision(row["unit_id"]) is not None for row in arbitration_target
    )
    certain_errors = []
    for row in arbitration_target:
        arbitration = arbitration_store.valid_decision(row["unit_id"])
        control = row["control"]["verdict"]
        if arbitration and control is not None and arbitration["verdict"] != control:
            certain_errors.append({
                "unit_id": row["unit_id"], "control_verdict": control,
                "gemini_secondary_verdict": arbitration["verdict"],
                "control_rule": row["control"]["rule"], "reason": arbitration["reason"],
            })
    go = bool(pilot_complete and audit_complete and arbitration_complete
              and agreement is not None and agreement >= 0.90
              and (abstain_rate or 0.0) <= 0.10 and not certain_errors)
    # At most 500 problem-group requests for the full run; deterministic/cache hits reduce this.
    full_index = config.output_root / "full" / "source_index.json"
    if full_index.is_file():
        from mgtb_v3.science_fast.io import load_json
        indexed = load_json(full_index)
        api_problem_count = len({
            row["content_sha256"] for row in indexed.get("candidates", [])
            if row.get("control", {}).get("verdict") is None
        })
    else:
        # Exactly 500 frozen problems: controls and cache hits can only lower this bound.
        api_problem_count = 500
    remaining_estimate = {
        "maximum_grouped_requests": api_problem_count,
        "minimum_rate_limited_minutes": api_problem_count / config.primary.rpm,
        "minimum_quota_days": (api_problem_count + config.primary.rpd - 1) // config.primary.rpd,
        "note": "before cache hits, retries, secondary arbitration, and daily reset timing",
    }
    grouped_remaining = len({
        row["content_sha256"] for row in pilot
        if row["control"]["verdict"] is None and grouped_store.valid_decision(row["unit_id"]) is None
    })
    individual_remaining = sum(
        individual_store.valid_decision(row["unit_id"]) is None for row in audits
    )
    projected_secondary = config.pilot_counts["numeric_contradictions"] + config.secondary_audit_count
    pilot_remaining_estimate = {
        "primary_requests_remaining": grouped_remaining + individual_remaining,
        "primary_minimum_rate_limited_minutes": (
            (grouped_remaining + individual_remaining) / config.primary.rpm
        ),
        "secondary_projected_minimum_requests": projected_secondary,
        "secondary_minimum_quota_days": (
            (projected_secondary + config.secondary.rpd - 1) // config.secondary.rpd
        ),
        "note": "ABSTAIN and grouped/individual conflicts may add secondary requests",
    }
    report = {
        "schema_version": 1, "created_at": utc_now(), "pilot_total": len(pilot),
        "completed": completed, "rates": {
            key: (final_counts[key] / completed if completed else None) for key in ("TRUE", "FALSE", "ABSTAIN")
        },
        "accuracy_main": ((final_counts["TRUE"]) / len(pilot) if pilot_complete else None),
        "abstain_is_incorrect_for_accuracy": True,
        "variant_comparison_after_local_restoration": by_variant,
        "agreement_matrices": {
            "gemini_vs_controls": _matrix(joined, "gemini", "control"),
            "gemini_vs_omni_judge": _matrix(joined, "gemini", "omni_judge"),
            "controls_vs_omni_judge": _matrix(joined, "control", "omni_judge"),
        },
        "grouped_vs_individual": {"target": config.individual_audit_count,
                                  "completed": len(complete_pairs), "agreement": agreement,
                                  "cases": comparisons},
        "secondary_arbitration": {
            "model": config.secondary.model, "target": len(arbitration_target),
            "completed": sum(arbitration_store.valid_decision(row["unit_id"]) is not None
                             for row in arbitration_target),
            "rpd": config.secondary.rpd,
            "minimum_quota_days": ((len(arbitration_target) + config.secondary.rpd - 1)
                                   // config.secondary.rpd),
        },
        "certain_gemini_errors": certain_errors, "symbolic_manual_audit": symbolic_audit,
        "pilot_remaining_estimate": pilot_remaining_estimate,
        "remaining_estimate": remaining_estimate,
        "recommendation": "GO" if go else "NO-GO",
        "recommendation_reason": (
            "pilot complete; grouped/individual agreement and abstention thresholds satisfied"
            if go else "pilot or individual audit incomplete, or quality threshold not satisfied"
        ),
    }
    report["report_sha256"] = sha256_json(report)
    atomic_write_json(config.output_root / "pilot" / "report.json", report)
    _write_markdown(config.output_root / "pilot" / "REPORT.md", report)
    return report


def analyze_full(config: ScoringConfig, *, finalize_without_pilot_go: bool = False) -> dict[str, Any]:
    candidates = prepare_scope(config, "full")
    manifest = load_json(config.manifest)
    expected_manifest_hash = manifest.get("manifest_sha256")
    actual_manifest_hash = sha256_json({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    if expected_manifest_hash != actual_manifest_hash:
        raise ValueError("Omni-MATH manifest hash mismatch during full analysis")
    metadata_by_hash = {row["content_sha256"]: row for row in manifest["roles"]["test"]}
    store = ArtifactStore(config.output_root / "full")
    joined: list[dict[str, Any]] = []
    missing = []
    for candidate in candidates:
        decision = store.valid_decision(candidate["unit_id"])
        if decision is None:
            missing.append(candidate["unit_id"])
            continue
        if decision.get("generation_artifact_sha256") != candidate["generation_artifact_sha256"]:
            raise ValueError("Gemini decision refers to a different generation artifact")
        if decision.get("verdict") not in {"TRUE", "FALSE", "ABSTAIN"}:
            raise ValueError(f"invalid Gemini verdict for {candidate['unit_id']}")
        joined.append({
            **candidate,
            "difficulty": metadata_by_hash[candidate["content_sha256"]].get("difficulty"),
            "verdict": decision["verdict"],
            "decision_source": decision.get("source"),
            "decision_sha256": decision["decision_sha256"],
        })
    if missing:
        raise ValueError(f"full Gemini analysis refuses incomplete decisions: {len(missing)} missing")
    expected = 500 * 3 * len(config.variants)
    if len(joined) != expected:
        raise ValueError(f"full Gemini analysis expected {expected} units, found {len(joined)}")

    statistics = full_statistics(joined, config.variants)
    pilot_path = config.output_root / "pilot" / "report.json"
    pilot_gate = {"recommendation": "UNAVAILABLE", "report_sha256": None}
    if pilot_path.is_file():
        pilot = load_json(pilot_path)
        expected_pilot_hash = pilot.get("report_sha256")
        actual_pilot_hash = sha256_json({key: value for key, value in pilot.items() if key != "report_sha256"})
        if expected_pilot_hash != actual_pilot_hash:
            raise ValueError("pilot report hash mismatch")
        pilot_gate = {
            "recommendation": pilot.get("recommendation"),
            "report_sha256": expected_pilot_hash,
            "secondary_arbitration_completed": pilot.get("secondary_arbitration", {}).get("completed"),
            "secondary_arbitration_target": pilot.get("secondary_arbitration", {}).get("target"),
        }
    pilot_go = pilot_gate["recommendation"] == "GO"
    result_final = pilot_go or finalize_without_pilot_go
    mgtb_comparison = statistics["comparisons_against_vanilla"].get("full_mgtb")
    mgtb_interval = mgtb_comparison["paired_cluster_bootstrap_95_ci"] if mgtb_comparison else None
    if mgtb_interval and mgtb_interval[0] > 0:
        conclusion = "full_mgtb améliore l’exactitude par rapport à Vanilla selon le scoring Gemini."
    elif mgtb_interval and mgtb_interval[1] < 0:
        conclusion = "full_mgtb réduit l’exactitude par rapport à Vanilla selon le scoring Gemini."
    else:
        conclusion = (
            "Aucun bénéfice d’exactitude de full_mgtb par rapport à Vanilla n’est détecté "
            "selon le scoring Gemini."
        )
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "scope": "full",
        "total": len(joined),
        "decision_sources": dict(Counter(row["decision_source"] for row in joined)),
        "pilot_gate": pilot_gate,
        "pilot_gate_enforced": not finalize_without_pilot_go,
        "report_status": (
            "CONFIRMATORY" if pilot_go else
            "FINAL_USER_ACCEPTED_JUDGE" if finalize_without_pilot_go else
            "PROVISIONAL_PENDING_PILOT_GO"
        ),
        "result_final": result_final,
        "confirmatory_ready": pilot_go,
        "finalization_basis": (
            "frozen_pilot_go" if pilot_go else
            "explicit_user_decision_to_accept_primary_gemini_judge_without_completed_secondary_gate"
            if finalize_without_pilot_go else "pending_frozen_pilot_go"
        ),
        "interpretation_guard": (
            "Les estimations full sont disponibles, mais restent provisoires tant que le rapport pilote "
            "gelé ne recommande pas GO. Relancer analyze-full après la fin du pilote."
            if not result_final else
            "Le résultat est définitif par décision explicite d’accepter le juge Gemini principal ; la "
            "porte d’arbitrage secondaire pré-déclarée n’a pas été terminée et n’est pas présentée comme validée."
            if finalize_without_pilot_go and not pilot_go else
            "Les estimations full complètes sont appuyées par un rapport pilote gelé recommandant GO."
        ),
        "conclusion": conclusion,
        **statistics,
    }
    report["report_sha256"] = sha256_json(report)
    atomic_write_json(config.output_root / "full" / "report.json", report)
    _write_full_markdown(config.output_root / "full" / "REPORT.md", report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    rates = report["rates"]
    lines = [
        "# Pilote Gemini Omni-MATH", "", f"Recommandation : **{report['recommendation']}**", "",
        f"Cas terminés : {report['completed']}/{report['pilot_total']}",
        f"TRUE : {rates['TRUE'] if rates['TRUE'] is not None else 'N/A'}",
        f"FALSE : {rates['FALSE'] if rates['FALSE'] is not None else 'N/A'}",
        f"ABSTAIN : {rates['ABSTAIN'] if rates['ABSTAIN'] is not None else 'N/A'}", "",
        "## Accord jugement groupé / individuel", "",
        f"{report['grouped_vs_individual']['completed']}/{report['grouped_vs_individual']['target']} paires; "
        f"accord={report['grouped_vs_individual']['agreement']}", "",
        "## Matrices d’accord", "", "```json",
    ]
    import json
    lines.extend([json.dumps(report["agreement_matrices"], ensure_ascii=False, indent=2), "```", "",
                  "## Erreurs certaines de Gemini", ""])
    if report["certain_gemini_errors"]:
        lines.extend(f"- `{row['unit_id']}`" for row in report["certain_gemini_errors"])
    else:
        lines.append("Aucune erreur certaine observée dans les cas terminés.")
    lines.extend(["", "## Audit symbolique lisible", ""])
    for row in report["symbolic_manual_audit"]:
        lines.extend([
            f"### {row['unit_id']} — {row['verdict']}", "",
            f"- Variante restaurée localement : `{row['variant']}`, seed `{row['seed']}`",
            f"- Domaine : {row['domain']}", f"- Référence : {row['reference_answer']}",
            f"- Candidat : {row['candidate_answer']}", f"- Motif : {row['reason']}", "",
        ])
    lines.extend(["## Estimation restante", "", "```json",
                  json.dumps({"pilot": report["pilot_remaining_estimate"],
                              "full": report["remaining_estimate"]}, ensure_ascii=False, indent=2),
                  "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f} %"


def _write_full_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Analyse Gemini Omni-MATH — scope full", "",
        f"Statut : **{report['report_status']}**", "",
        f"> {report['interpretation_guard']}", "",
        f"Conclusion : **{report['conclusion']}**", "",
        f"Unités authentifiées : {report['total']}/4500", "",
        "## Résultats principaux", "",
        "| Variante | Exactitude | TRUE | FALSE | ABSTAIN | Taux ABSTAIN |", 
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, summary in report["methods"].items():
        counts = summary["counts"]
        lines.append(
            f"| `{variant}` | {_percent(summary['accuracy'])} | {counts['TRUE']} | "
            f"{counts['FALSE']} | {counts['ABSTAIN']} | {_percent(summary['abstain_rate'])} |"
        )
    lines.extend([
        "", "`ABSTAIN` est compté comme incorrect dans l’exactitude principale.", "",
        "## Comparaisons appariées contre Vanilla", "",
        "| Variante | Différence | Corrections | Régressions | IC bootstrap clusterisé 95 % | McNemar | Holm |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for variant, comparison in report["comparisons_against_vanilla"].items():
        interval = comparison["paired_cluster_bootstrap_95_ci"]
        lines.append(
            f"| `{variant}` | {_percent(comparison['difference'])} | {comparison['corrections']} | "
            f"{comparison['regressions']} | [{_percent(interval[0])} ; {_percent(interval[1])}] | "
            f"{comparison['mcnemar_exact_two_sided_p']:.6g} | "
            f"{comparison['mcnemar_holm_adjusted_p']:.6g} |"
        )
    direct = report["additional_pairwise_comparisons_descriptive"].get(
        "full_mgtb_vs_matched_random"
    )
    if direct:
        interval = direct["paired_cluster_bootstrap_95_ci"]
        lines.extend([
            "", "## Comparaison descriptive full_mgtb contre matched_random", "",
            f"Différence : {_percent(direct['difference'])} ; corrections/régressions : "
            f"{direct['corrections']}/{direct['regressions']} ; IC bootstrap clusterisé 95 % : "
            f"[{_percent(interval[0])} ; {_percent(interval[1])}] ; "
            f"McNemar descriptif : {direct['mcnemar_exact_two_sided_p']:.6g}.",
        ])
    agreement = report["agreement_with_omni_judge"]
    lines.extend([
        "", "## Accord avec Omni-Judge", "",
        f"Accord sur {agreement['comparable_units']} unités : "
        f"{_percent(agreement['agreement']) if agreement['agreement'] is not None else 'N/A'}.", "",
        "```json", __import__("json").dumps(agreement["matrix"], ensure_ascii=False, indent=2), "```", "",
        "## Résultats par seed", "",
    ])
    for variant, summary in report["methods"].items():
        seeds = ", ".join(
            f"seed {seed}: {_percent(value)}"
            for seed, value in summary["accuracy_by_replicate_seed"].items()
        )
        lines.append(f"- `{variant}` — {seeds}")
    lines.extend([
        "", "## Cadre d’interprétation", "",
        "L’inférence principale est le bootstrap apparié clusterisé par problème. Les trois seeds "
        "d’un même problème restent dans le même cluster. McNemar est descriptif et les comparaisons "
        "contre Vanilla sont corrigées par Holm. Les résultats par domaine sont descriptifs.", "",
        f"Hash du rapport : `{report['report_sha256']}`", "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
