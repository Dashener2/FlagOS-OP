from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_environment_locks_are_valid_and_generated_index_is_current():
    from scripts.build_env_index import (
        OUTPUT_PATH,
        load_locks,
        render_index,
        validate_locks,
    )

    locks = load_locks()
    validate_locks(locks)
    assert OUTPUT_PATH.read_text(encoding="utf-8") == render_index(locks)


def test_environment_locks_match_device_profiles_and_cover_delivered_families():
    from scripts.build_env_index import load_locks, validate_locks
    from scripts.build_ops_index import load_catalog

    locks = load_locks()
    validate_locks(locks)

    catalog = load_catalog()
    delivered_profiles = {
        profile
        for meta in catalog["platform_families"].values()
        if meta["status"] == "partial"
        for profile in meta["current_profiles"]
    }
    delivered_families = {
        family
        for family, meta in catalog["platform_families"].items()
        if meta["status"] == "partial"
    }
    names = {lock["name"] for lock in locks}
    families = {lock["family"] for lock in locks}
    assert names == delivered_profiles
    assert families == delivered_families
    assert all(
        (ROOT / "configs" / "devices" / f"{name}.yaml").is_file()
        for name in names
    )


def test_p800_lock_declares_vendor_runtime_packages():
    from scripts.build_env_index import load_locks, validate_locks

    locks = load_locks()
    validate_locks(locks)
    p800 = next(lock for lock in locks if lock["name"] == "p800-kunlunxin")
    packages = p800["packages"]

    assert "torch_xmlir" in packages
    assert "xtorch_ops" in packages


def test_embedding_mlu_evidence_uses_matching_environment_lock():
    from scripts.build_env_index import load_locks, validate_locks

    locks = load_locks()
    validate_locks(locks)
    by_name = {lock["name"]: lock for lock in locks}
    evidence = {item["path"] for item in by_name["mlu590"]["evidence"]}

    assert "ops/embedding/reports/cambricon.md" in evidence
    assert any(
        "ops/embedding Cambricon backend" in item
        for item in by_name["mlu590"]["scope"]
    )
    assert all(
        "ops/embedding" not in item
        for item in by_name["cambricon"]["scope"]
    )


def test_environment_lock_rejects_one_sided_base_image_relation():
    from scripts.build_env_index import load_locks, validate_locks
    from scripts.build_base_image_index import load_catalog

    locks = load_locks()
    image_catalog = deepcopy(load_catalog())
    image = next(
        item for item in image_catalog["images"]
        if item["id"] == "kunlunxin-p800-operator-runtime-v2"
    )
    image["flagos_op_relation"] = "not-registered"
    image["related_profiles"] = []

    with pytest.raises(ValueError, match="does not declare profile"):
        validate_locks(locks, image_catalog=image_catalog)
