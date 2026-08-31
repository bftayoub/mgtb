#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mgtb_v3.gemini_scoring.analysis import analyze_full, analyze_pilot
from mgtb_v3.gemini_scoring.api import PINNED_GOOGLE_GENAI_VERSION, sdk_version
from mgtb_v3.gemini_scoring.audit import run_individual_audit, run_secondary_arbitration
from mgtb_v3.gemini_scoring.config import load_scoring_config
from mgtb_v3.gemini_scoring.runner import dry_run, run_scope, status


DEFAULT_CONFIG = "configs/science_campaign/omnimath_gemini_scoring_v1.yaml"


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Blind and resumable Gemini scoring for frozen Omni-MATH generations")
    root.add_argument("--config", default=DEFAULT_CONFIG)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("setup", help="check the pinned SDK and key environment")
    dry = commands.add_parser("dry-run", help="build and validate payloads without contacting Gemini")
    dry.add_argument("--scope", choices=["pilot", "full"], default="pilot")
    pilot = commands.add_parser("pilot", help="run/resume the 200-case pilot and 50 individual audits")
    pilot.add_argument("--stop-after", type=int)
    pilot.add_argument("--workers", type=int, default=1)
    pilot.add_argument("--resume", action="store_true")
    run = commands.add_parser("run", help="run scoring; full scope needs explicit post-pilot approval flag")
    run.add_argument("--scope", choices=["pilot", "full"], default="full")
    run.add_argument("--stop-after", type=int)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--approved-full", action="store_true")
    resume = commands.add_parser("resume", help="resume exactly the same artifact-backed scoring")
    resume.add_argument("--scope", choices=["pilot", "full"], default="pilot")
    resume.add_argument("--stop-after", type=int)
    resume.add_argument("--workers", type=int, default=1)
    resume.add_argument("--approved-full", action="store_true")
    stat = commands.add_parser("status")
    stat.add_argument("--scope", choices=["pilot", "full"], default="pilot")
    commands.add_parser("analyze", help="write the JSON and readable Markdown pilot reports")
    analyze_full_parser = commands.add_parser(
        "analyze-full",
        help="analyze all 4,500 decisions; remains provisional until the pilot recommendation is GO",
    )
    analyze_full_parser.add_argument(
        "--finalize-without-pilot-go", action="store_true",
        help="make the result final by explicitly accepting the primary judge without the pilot gate",
    )
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_scoring_config(args.config)
    if args.command == "setup":
        result = {
            "required_google_genai": PINNED_GOOGLE_GENAI_VERSION,
            "installed_google_genai": sdk_version(),
            "sdk_ready": sdk_version() == PINNED_GOOGLE_GENAI_VERSION,
            "gemini_api_key_set": bool(os.environ.get("GEMINI_API_KEY")),
            "install_command": f'{sys.executable} -m pip install "google-genai=={PINNED_GOOGLE_GENAI_VERSION}"',
            "secret_policy": "GEMINI_API_KEY is read from the environment only and never persisted",
        }
        print(json.dumps(result, indent=2))
        return 0 if result["sdk_ready"] else 2
    if args.command == "dry-run":
        print(json.dumps(dry_run(config, args.scope), indent=2))
        return 0
    if args.command == "status":
        print(json.dumps(status(config, args.scope), indent=2))
        return 0
    if args.command == "analyze":
        report = analyze_pilot(config)
        print(json.dumps({key: report[key] for key in (
            "completed", "rates", "grouped_vs_individual", "pilot_remaining_estimate",
            "remaining_estimate", "recommendation",
            "recommendation_reason", "report_sha256",
        )}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "analyze-full":
        report = analyze_full(config, finalize_without_pilot_go=args.finalize_without_pilot_go)
        print(json.dumps({key: report[key] for key in (
            "scope", "total", "report_status", "result_final", "confirmatory_ready", "pilot_gate",
            "pilot_gate_enforced", "finalization_basis", "conclusion",
            "methods", "comparisons_against_vanilla", "agreement_with_omni_judge",
            "additional_pairwise_comparisons_descriptive",
            "report_sha256",
        )}, ensure_ascii=False, indent=2))
        return 0
    if args.command in {"run", "resume"}:
        scope = args.scope
        if scope == "full" and not args.approved_full:
            raise SystemExit("full run is locked: obtain user approval after a GO report, then pass --approved-full")
        result = run_scope(
            config, scope, stop_after=args.stop_after, workers=args.workers,
            resume=args.resume or args.command == "resume",
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "pilot":
        grouped = run_scope(
            config, "pilot", stop_after=args.stop_after, workers=args.workers, resume=args.resume,
        )
        if grouped["remaining"]:
            report = analyze_pilot(config)
            print(json.dumps({
                "grouped": grouped,
                "individual": "deferred_until_grouped_pilot_is_complete",
                "arbitration": "deferred_until_grouped_pilot_is_complete",
                "recommendation": report["recommendation"],
            }, indent=2))
            return 0
        individual = run_individual_audit(config, stop_after=args.stop_after, resume=args.resume)
        arbitration = run_secondary_arbitration(config, stop_after=args.stop_after, resume=args.resume)
        report = analyze_pilot(config)
        print(json.dumps({"grouped": grouped, "individual": individual, "arbitration": arbitration,
                          "recommendation": report["recommendation"]}, indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
