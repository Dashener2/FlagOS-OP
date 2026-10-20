from __future__ import annotations

from copy import deepcopy

import pytest


def test_base_image_catalog_is_valid_and_generated_index_is_current():
    from scripts.build_base_image_index import (
        OUTPUT_PATH,
        load_catalog,
        render_index,
        validate_catalog,
    )

    catalog = load_catalog()
    validate_catalog(catalog)
    assert OUTPUT_PATH.read_text(encoding="utf-8") == render_index(catalog)


def test_base_image_source_is_pinned_and_index_covers_all_families():
    from scripts.build_base_image_index import (
        load_catalog,
        render_index,
        validate_catalog,
    )
    from scripts.build_ops_index import load_catalog as load_operator_catalog

    catalog = load_catalog()
    validate_catalog(catalog)

    assert catalog["source"]["revision"].startswith("74197bf7d4b5dc8cdd3b2b89")
    assert catalog["source"]["revision"] in catalog["source"]["url"]
    assert {
        image["family"] for image in catalog["images"]
    } <= set(load_operator_catalog()["platform_families"])

    rendered = render_index(catalog)
    for family in load_operator_catalog()["platform_families"].values():
        assert family["display_name"] in rendered


def test_base_image_catalog_distinguishes_version_alignment_from_validation():
    from scripts.build_base_image_index import load_catalog, validate_catalog

    catalog = load_catalog()
    validate_catalog(catalog)
    p800 = next(
        image for image in catalog["images"]
        if image["id"] == "kunlunxin-p800-operator-runtime-v2"
    )

    assert p800["flagos_op_relation"] == "version-aligned"
    assert p800["related_profiles"] == ["p800-kunlunxin"]
    assert any("同镜像实例" in note for note in p800["notes"])


def test_public_base_image_references_include_registry_namespace():
    from scripts.build_base_image_index import load_catalog, validate_catalog

    catalog = load_catalog()
    validate_catalog(catalog)
    ppu = next(
        image for image in catalog["images"]
        if image["id"] == "thead-ppu-runtime-v1"
    )

    assert ppu["registry_reference"].startswith(
        "harbor.baai.ac.cn/flagtree/flagtree-ppu-"
    )


def test_base_image_catalog_rejects_one_sided_environment_relation():
    from scripts.build_base_image_index import load_catalog, validate_catalog
    from scripts.build_env_index import load_locks

    catalog = deepcopy(load_catalog())
    image = next(
        item for item in catalog["images"]
        if item["id"] == "kunlunxin-p800-operator-runtime-v2"
    )
    image["flagos_op_relation"] = "not-registered"
    env_locks = {lock["name"]: lock for lock in load_locks()}

    with pytest.raises(ValueError, match="does not declare"):
        validate_catalog(catalog, env_locks=env_locks)


def test_historical_base_image_requires_successor_or_reason():
    from scripts.build_base_image_index import load_catalog, validate_catalog

    catalog = deepcopy(load_catalog())
    catalog["images"][0]["current"] = False
    with pytest.raises(ValueError, match="historical entries need"):
        validate_catalog(catalog)


def test_historical_base_images_keep_stable_document_anchors():
    from scripts.build_base_image_index import load_catalog, render_index

    catalog = deepcopy(load_catalog())
    image = catalog["images"][0]
    image["current"] = False
    image["archival_reason"] = "外部源保留历史版本，仅作追溯。"

    rendered = render_index(catalog)
    assert f'<a id="{image["id"]}"></a>' in rendered
