#!/usr/bin/env python3
"""Validate the base-image catalog and generate the collaboration index."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "configs" / "images" / "catalog.yaml"
OUTPUT_PATH = ROOT / "docs" / "base-images.md"
EXPECTED_SCHEMA = 1
VALID_STATUSES = {"registered", "functional-repro", "partial-repro"}
VALID_DISTRIBUTIONS = {
    "public-registry", "internal-build", "offline-release"
}
VALID_RELATIONS = {
    "not-registered", "version-aligned", "validated-in-image", "platform-planned"
}
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def load_catalog(path: Path = CATALOG_PATH) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def _require_string(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_string_list(value, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{name} must only contain non-empty strings")
    return value


def validate_catalog(
    catalog: dict, root: Path = ROOT, env_locks: dict[str, dict] | None = None
) -> None:
    if catalog.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"schema must be {EXPECTED_SCHEMA}")
    if not DATE_PATTERN.match(_require_string(catalog.get("updated"), "updated")):
        raise ValueError("updated must use YYYY-MM-DD")

    source = catalog.get("source")
    if not isinstance(source, dict):
        raise ValueError("source must be a mapping")
    for field in ("repository", "revision", "path", "url", "retrieved_at"):
        _require_string(source.get(field), f"source.{field}")
    if not DATE_PATTERN.match(source["retrieved_at"]):
        raise ValueError("source.retrieved_at must use YYYY-MM-DD")
    if source["revision"] not in source["url"]:
        raise ValueError("source.url must pin source.revision")

    ownership = catalog.get("ownership")
    if not isinstance(ownership, dict):
        raise ValueError("ownership must be a mapping")
    for field in ("builder", "consumer", "boundary"):
        _require_string(ownership.get(field), f"ownership.{field}")

    operator_catalog = yaml.safe_load((root / "ops" / "catalog.yaml").read_text())
    families = operator_catalog["platform_families"]
    images = catalog.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("images must be a non-empty list")

    ids: set[str] = set()
    references: set[str] = set()
    current_by_series: set[tuple[str, str]] = set()
    for image in images:
        image_id = _require_string(image.get("id"), "image.id")
        if image_id in ids:
            raise ValueError(f"duplicate image id: {image_id}")
        ids.add(image_id)

        family = _require_string(image.get("family"), f"{image_id}.family")
        if family not in families:
            raise ValueError(f"{image_id}: unknown platform family {family!r}")
        series = _require_string(image.get("series"), f"{image_id}.series")
        version = _require_string(image.get("version"), f"{image_id}.version")
        _require_string(image.get("use"), f"{image_id}.use")
        reference = _require_string(
            image.get("registry_reference"), f"{image_id}.registry_reference"
        )
        if reference in references:
            raise ValueError(f"duplicate registry reference: {reference}")
        references.add(reference)

        status = _require_string(image.get("status"), f"{image_id}.status")
        if status not in VALID_STATUSES:
            raise ValueError(f"{image_id}: unknown status {status!r}")
        distribution = _require_string(
            image.get("distribution"), f"{image_id}.distribution"
        )
        if distribution not in VALID_DISTRIBUTIONS:
            raise ValueError(f"{image_id}: unknown distribution {distribution}")
        relation = _require_string(
            image.get("flagos_op_relation"), f"{image_id}.flagos_op_relation"
        )
        if relation not in VALID_RELATIONS:
            raise ValueError(f"{image_id}: unknown FlagOS-OP relation {relation!r}")
        _require_string_list(image.get("notes"), f"{image_id}.notes")

        current = image.get("current")
        if not isinstance(current, bool):
            raise ValueError(f"{image_id}.current must be a boolean")
        if current:
            key = (family, series)
            if key in current_by_series:
                raise ValueError(
                    f"{image_id}: duplicate current entry for {family}/{series}"
                )
            current_by_series.add(key)
        elif not image.get("superseded_by") and not image.get("archival_reason"):
            raise ValueError(
                f"{image_id}: historical entries need superseded_by or archival_reason"
            )

        for profile in image.get("related_profiles", []):
            if not isinstance(profile, str):
                raise ValueError(f"{image_id}: related profile must be a string")
            if not (root / "configs" / "devices" / f"{profile}.yaml").is_file():
                raise ValueError(f"{image_id}: unknown related profile {profile!r}")
            if not (root / "configs" / "env" / f"{profile}.lock.yaml").is_file():
                raise ValueError(
                    f"{image_id}: related profile {profile!r} lacks an environment lock"
                )
            if env_locks is None:
                env_lock = yaml.safe_load(
                    (root / "configs" / "env" / f"{profile}.lock.yaml").read_text()
                )
            else:
                env_lock = env_locks[profile]
            relations = {
                item.get("image_id"): item.get("relation")
                for item in env_lock.get("base_images", [])
            }
            if relations.get(image_id) != relation:
                raise ValueError(
                    f"{image_id}: {profile} environment lock does not declare "
                    f"this image with relation {relation!r}"
                )
        if relation in {"version-aligned", "validated-in-image"} and not image.get(
            "related_profiles"
        ):
            raise ValueError(
                f"{image_id}: {relation} entries need related_profiles"
            )
        if relation not in {"version-aligned", "validated-in-image"} and image.get(
            "related_profiles"
        ):
            raise ValueError(
                f"{image_id}: only aligned or validated entries may declare profiles"
            )
        if relation == "platform-planned" and families[family]["status"] != "planned":
            raise ValueError(
                f"{image_id}: platform-planned requires a planned operator family"
            )

    for image in images:
        successor = image.get("superseded_by")
        if successor is not None and successor not in ids:
            raise ValueError(
                f"{image['id']}: superseded_by must reference another image id"
            )
        if successor == image["id"]:
            raise ValueError(f"{image['id']}: superseded_by cannot reference itself")


def _display(value) -> str:
    if value is None or value == "":
        return "未登记"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _status_display(status: str) -> str:
    return {
        "registered": "登记式基座",
        "functional-repro": "外部可复现构建",
        "partial-repro": "部分可复现",
    }[status]


def _distribution_display(value: str) -> str:
    return {
        "public-registry": "公开 registry",
        "internal-build": "内部构建",
        "offline-release": "离线发布",
    }[value]


def _relation_display(value: str) -> str:
    return {
        "not-registered": "未接入 FlagOS-OP 环境锁",
        "version-aligned": "版本对齐；未声明同镜像实例",
        "validated-in-image": "已在锁定镜像实例内验证",
        "platform-planned": "平台算子矩阵仍为 planned",
    }[value]


def _family_display(family: str, operator_catalog: dict) -> str:
    return operator_catalog["platform_families"][family]["display_name"]


def render_index(catalog: dict) -> str:
    operator_catalog = yaml.safe_load((ROOT / "ops" / "catalog.yaml").read_text())
    source = catalog["source"]
    lines = [
        "# 基座镜像索引",
        "",
        "[← 返回文档中心](index.md)",
        "",
        "这一页索引各平台协作方提供的当前基座镜像、用途、分发状态和与 FlagOS-OP 环境锁的关系。",
        "FlagOS-OP 不构建、不分发这些镜像，也不把镜像登记等同于算子支持完成。",
        "本页由 `configs/images/catalog.yaml` 生成，修改后不要手工编辑。",
        "",
        f"外部索引来源：[{source['repository']} · {source['path']}]({source['url']})，",
        f"revision `{source['revision']}`，读取时间 {source['retrieved_at']}。",
        "",
        "维护分工：",
        "",
        f"- 基座构建与分发：{catalog['ownership']['builder']}。",
        f"- FlagOS-OP 记录内容：{catalog['ownership']['consumer']}。",
        f"- 边界：{catalog['ownership']['boundary']}。",
        "",
        "## 阅读边界",
        "",
        "| 问题 | 判断位置 |",
        "|---|---|",
        "| 基座镜像 tag、用途、外部复现状态 | 本页 |",
        "| FlagOS-OP 实测硬件、系统与软件栈 | [软硬件环境版本汇总](environments.md) |",
        "| 算子 / 平台是否已有交付证据 | [算子 / 平台矩阵](operator-catalog.md) |",
        "| 镜像构建过程、digest、已知限制 | 外部索引和其锁文件 |",
        "",
        "`version-aligned` 只表示包版本线一致。FlagOS-OP 环境锁未登记 image id 或 digest 时，",
        "不能宣称算子验证发生在同一镜像实例上。",
        "",
        "<a id=\"current\"></a>",
        "## 当前索引",
        "",
        "| 平台 | 用途 / 系列 | 镜像引用 | 外部状态 | 分发 | 与 FlagOS-OP 的关系 |",
        "|---|---|---|---|---|---|",
    ]

    current_images = [image for image in catalog["images"] if image["current"]]
    historical_images = [image for image in catalog["images"] if not image["current"]]
    for image in current_images:
        lines.append(
            "| {family} | {use}<br>`{series}` / `{version}` | `{reference}` | "
            "{status} | {distribution} | {relation} |".format(
                family=_family_display(image["family"], operator_catalog),
                use=_display(image["use"]),
                series=image["series"],
                version=image["version"],
                reference=_display(image["registry_reference"]),
                status=_status_display(image["status"]),
                distribution=_distribution_display(image["distribution"]),
                relation=_relation_display(image["flagos_op_relation"]),
            )
        )

    lines.extend([
        "",
        "### 平台说明",
        "",
    ])
    for image in catalog["images"]:
        if not image["current"]:
            continue
        lines.append(f"<a id=\"{image['id']}\"></a>")
        lines.append(f"#### {_display(image['use'])}")
        lines.append("")
        for note in image["notes"]:
            lines.append(f"- {note}")
        if image.get("related_profiles"):
            profiles = " · ".join(f"`{item}`" for item in image["related_profiles"])
            lines.append(f"- 关联环境锁：{profiles}。")
        else:
            lines.append("- 关联环境锁：未登记。")
        lines.append("")

    if historical_images:
        lines.extend([
            "<a id=\"historical\"></a>",
            "## 历史版本",
            "",
            "| 平台 | 系列 / 版本 | 镜像引用 | 状态 | 后继或归档原因 |",
            "|---|---|---|---|---|",
        ])
        for image in historical_images:
            successor = image.get("superseded_by")
            reason = image.get("archival_reason")
            outcome = f"`{successor}`" if successor else _display(reason)
            lines.append(
                f"| <a id=\"{image['id']}\"></a>"
                f"{_family_display(image['family'], operator_catalog)} | "
                f"`{image['series']}` / `{image['version']}` | "
                f"`{_display(image['registry_reference'])}` | "
                f"{_status_display(image['status'])} | {outcome} |"
            )
        lines.append("")

    lines.extend([
        "<a id=\"coverage\"></a>",
        "## 平台覆盖",
        "",
        "| 平台家族 | 外部基座线索 | FlagOS-OP 算子矩阵 |",
        "|---|---|---|",
    ])
    for family, meta in operator_catalog["platform_families"].items():
        images = [image for image in catalog["images"] if image["family"] == family]
        base = " · ".join(f"`{item['series']}`" for item in images) or "未登记"
        operator_status = "部分交付" if meta["status"] == "partial" else "规划中"
        lines.append(
            f"| {_family_display(family, operator_catalog)} | {base} | {operator_status} |"
        )

    lines.extend([
        "",
        "<a id=\"extend\"></a>",
        "## 平台方扩展方式",
        "",
        "1. 在 `configs/images/catalog.yaml` 的 `images` 列表追加一行，不要手工编辑本页。",
        "2. 填写平台家族、用途、镜像引用、外部状态、分发方式和备注。",
        "3. 如果该镜像只是版本对齐，`flagos_op_relation` 使用 `version-aligned`，并在备注中写明对齐项。",
        "4. `version-aligned` 表示包版本对齐，但环境锁尚未证明使用同一镜像实例。",
        "5. 只有环境锁登记 image id 或 digest 并完成算子验证时，才使用 `validated-in-image`。",
        "6. 运行生成与校验命令后提交 PR：",
        "",
        "```bash",
        "python3 scripts/build_base_image_index.py",
        "python3 scripts/build_base_image_index.py --check",
        "```",
        "",
        "常用字段：",
        "",
        "| 字段 | 含义 |",
        "|---|---|",
        "| `family` / `series` / `version` | 平台家族、镜像系列和版本号 |",
        "| `registry_reference` | 镜像 tag；digest 继续维护在外部基座仓库 |",
        "| `status` | 外部源声明的 `registered` / `functional-repro` / `partial-repro` |",
        "| `distribution` | 公开 registry、内部构建或离线发布 |",
        "| `current` | 是否当前推荐入口；同一平台同一系列只能有一个当前版本 |",
        "| `superseded_by` / `archival_reason` | 历史版本的后继镜像或归档原因 |",
        "| `related_profiles` | 已建立对应 FlagOS-OP 环境锁的 profile |",
        "| `flagos_op_relation` | 未接入、版本对齐、镜像内验证或平台仍为 planned |",
        "",
        "登记规则：",
        "",
        "- 不用 planned 平台预填镜像来推断算子支持完成；",
        "- 不用包版本相同推断镜像实例相同；",
        "- 不把外部 `functional-repro` 转化为 FlagOS-OP 的算子验证结论；",
        "- 镜像 digest、构建配方和已知限制维护在外部基座仓库；本仓库只保留索引和关联关系。",
        "",
        "<!-- AUTO-GENERATED FROM configs/images/catalog.yaml; do not edit. -->",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check the generated index")
    args = parser.parse_args()

    catalog = load_catalog()
    validate_catalog(catalog)
    rendered = render_index(catalog)
    if args.check:
        if OUTPUT_PATH.read_text(encoding="utf-8") != rendered:
            print(f"{OUTPUT_PATH} is out of date; run scripts/build_base_image_index.py")
            return 1
        print(f"{OUTPUT_PATH} is up to date.")
        return 0

    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
