from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from mgtb_v3.config import config_from_dict, load_config
from mgtb_v3.science_fast.io import sha256_file, sha256_json

SCHEMA_VERSION = 1
VARIANT_KINDS = {"vanilla", "controller", "matched_random", "sample_aggregate"}


def deep_merge(base: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_campaign(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw["_config_path"] = str(path)
    raw["_config_dir"] = str(path.parent)
    validate_campaign(raw)
    return raw


def _path(campaign: dict[str, Any], value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(campaign["_config_dir"]) / path
        if not path.exists():
            path = Path.cwd() / value
    return path.resolve()


def controller_mapping(campaign: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    base = asdict(load_config(_path(campaign, campaign["controller_base"])))
    return deep_merge(base, overrides)


def calibration_spec(campaign: dict[str, Any], key: str) -> dict[str, Any]:
    try:
        raw = deepcopy(campaign["calibrations"][key])
    except KeyError as exc:
        raise ValueError(f"unknown calibration {key!r}") from exc
    raw.setdefault("controller_overrides", {})
    raw.setdefault("feature_source", key)
    raw.setdefault("calibration_mode", "positional")
    raw.setdefault("accumulation_mode", "cusum_reset")
    raw["controller"] = controller_mapping(campaign, raw["controller_overrides"])
    raw["calibration_sha256"] = sha256_json(raw)
    return raw


def resolve_variant(campaign: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        variant = deepcopy(campaign["variants"][name])
    except KeyError as exc:
        raise ValueError(f"unknown variant {name!r}") from exc
    variant["name"] = name
    variant.setdefault("kind", "controller")
    variant.setdefault("controller_overrides", {})
    variant.setdefault("accumulation_mode", "cusum_reset")
    variant.setdefault("repair_group", None)
    variant["controller"] = controller_mapping(campaign, variant["controller_overrides"])
    if variant["kind"] == "controller" and not variant.get("calibration"):
        raise ValueError(f"controller variant {name!r} requires a calibration key")
    if variant["kind"] == "matched_random" and not variant.get("profile"):
        raise ValueError(f"matched_random variant {name!r} requires profile")
    if variant.get("profile"):
        variant["profile"] = str(_path(campaign, variant["profile"]))
        if Path(variant["profile"]).exists():
            variant["profile_sha256"] = sha256_file(variant["profile"])
    if variant["kind"] == "sample_aggregate":
        variant.setdefault("num_samples", 5)
        variant.setdefault("selection", "majority_answer")
        if int(variant["num_samples"]) < 2:
            raise ValueError("sample_aggregate num_samples must be at least 2")
    variant["variant_sha256"] = sha256_json(variant)
    return variant


def output_root(campaign: dict[str, Any]) -> Path:
    return _path(campaign, campaign["output_root"])


def manifest_path(campaign: dict[str, Any]) -> Path:
    return _path(campaign, campaign["manifest"])


def validate_campaign(campaign: dict[str, Any]) -> None:
    if int(campaign.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"campaign schema_version must be {SCHEMA_VERSION}")
    for key in ("campaign_id", "manifest", "output_root", "controller_base", "model", "generation", "variants"):
        if not campaign.get(key):
            raise ValueError(f"campaign requires {key}")
    if campaign.get("experimental_status") not in {"confirmatory", "exploratory"}:
        raise ValueError("experimental_status must be confirmatory or exploratory")
    if campaign.get("manifest_build") and campaign.get("manifest_derive"):
        raise ValueError("campaign cannot define both manifest_build and manifest_derive")
    if campaign["experimental_status"] == "confirmatory":
        design = campaign.get("confirmatory_design", "independent_test")
        if design not in {"independent_test", "frozen_evaluation"}:
            raise ValueError("confirmatory_design must be independent_test or frozen_evaluation")
        if design == "independent_test" and not campaign.get("exclude_manifests"):
            raise ValueError("independent confirmatory campaigns require exclude_manifests")
        if design == "frozen_evaluation":
            if campaign.get("no_retuning_from_prior_test") is not True:
                raise ValueError("frozen_evaluation requires no_retuning_from_prior_test=true")
            planned = [int(seed) for seed in campaign.get("planned_seeds", [])]
            executed = [int(seed) for seed in campaign.get("seeds", [0])]
            if not planned or not set(executed) <= set(planned):
                raise ValueError("frozen_evaluation seeds must be included in planned_seeds")
    revision = campaign.get("model", {}).get("revision")
    if campaign["experimental_status"] == "confirmatory" and (not revision or str(revision).startswith("REPLACE_")):
        raise ValueError("confirmatory campaigns require an immutable model revision")
    for name, variant in campaign["variants"].items():
        if variant.get("kind", "controller") not in VARIANT_KINDS:
            raise ValueError(f"unsupported variant kind for {name}: {variant.get('kind')}")
    for key, spec in campaign.get("calibrations", {}).items():
        if spec.get("calibration_mode", "positional") not in {"positional", "global"}:
            raise ValueError(f"invalid calibration_mode for {key}")
        if spec.get("accumulation_mode", "cusum_reset") not in {"cusum_reset", "no_reset"}:
            raise ValueError(f"invalid accumulation_mode for {key}")
