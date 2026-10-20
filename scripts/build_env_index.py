#!/usr/bin/env python3
"""Validate environment lock files and generate the environment version index."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = ROOT / "configs" / "env"
OUTPUT_PATH = ROOT / "docs" / "environments.md"
EXPECTED_SCHEMA = 1
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def load_locks(env_dir: Path = ENV_DIR) -> list[dict]:
    locks = []
    for path in sorted(env_dir.glob("*.lock.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a mapping")
        data["_path"] = path
        locks.append(data)
    return locks


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


def validate_locks(
    locks: list[dict], root: Path = ROOT, image_catalog: dict | None = None
) -> None:
    if not locks:
        raise ValueError("configs/env must contain at least one environment lock")

    operator_catalog = yaml.safe_load((root / "ops" / "catalog.yaml").read_text())
    valid_families = set(operator_catalog["platform_families"])
    if image_catalog is None:
        image_catalog = yaml.safe_load(
            (root / "configs" / "images" / "catalog.yaml").read_text()
        )
    known_images = {item["id"]: item for item in image_catalog["images"]}
    names: set[str] = set()

    for lock in locks:
        path = lock["_path"]
        name = _require_string(lock.get("name"), f"{path}: name")
        if name != path.name.removesuffix(".lock.yaml"):
            raise ValueError(f"{path}: name must match the file name")
        if name in names:
            raise ValueError(f"duplicate environment name: {name}")
        names.add(name)

        if lock.get("schema") != EXPECTED_SCHEMA:
            raise ValueError(f"{path}: schema must be {EXPECTED_SCHEMA}")
        _require_string(lock.get("description"), f"{path}: description")
        collected_at = _require_string(
            lock.get("collected_at"), f"{path}: collected_at"
        )
        if not DATE_PATTERN.match(collected_at):
            raise ValueError(f"{path}: collected_at must use YYYY-MM-DD")
        if lock.get("hardware_collected_at") is not None:
            hardware_at = _require_string(
                lock["hardware_collected_at"],
                f"{path}: hardware_collected_at",
            )
            if not DATE_PATTERN.match(hardware_at):
                raise ValueError(
                    f"{path}: hardware_collected_at must use YYYY-MM-DD"
                )

        family = _require_string(lock.get("family"), f"{path}: family")
        if family not in valid_families:
            raise ValueError(f"{path}: unknown platform family {family!r}")
        if not (root / "configs" / "devices" / f"{name}.yaml").is_file():
            raise ValueError(f"{path}: matching device profile does not exist")

        _require_string_list(lock.get("scope"), f"{path}: scope")
        hardware = lock.get("hardware")
        if not isinstance(hardware, dict):
            raise ValueError(f"{path}: hardware must be a mapping")
        _require_string(hardware.get("model"), f"{path}: hardware.model")

        system = lock.get("system")
        if not isinstance(system, dict):
            raise ValueError(f"{path}: system must be a mapping")
        _require_string(system.get("os"), f"{path}: system.os")
        _require_string(lock.get("python"), f"{path}: python")

        packages = lock.get("packages")
        if not isinstance(packages, dict) or not packages:
            raise ValueError(f"{path}: packages must be a non-empty mapping")
        for package, version in packages.items():
            _require_string(package, f"{path}: package name")
            _require_string(version, f"{path}: packages.{package}")

        _require_string(lock.get("notes"), f"{path}: notes")

        evidence = lock.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"{path}: evidence must be a non-empty list")
        evidence_paths: set[str] = set()
        for item in evidence:
            if not isinstance(item, dict):
                raise ValueError(f"{path}: evidence items must be mappings")
            evidence_path = _require_string(item.get("path"), f"{path}: evidence.path")
            _require_string(item.get("note"), f"{path}: evidence.note")
            if not (root / evidence_path).exists():
                raise ValueError(f"{path}: evidence path does not exist: {evidence_path}")
            if evidence_path in evidence_paths:
                raise ValueError(f"{path}: duplicate evidence path {evidence_path}")
            evidence_paths.add(evidence_path)

        base_images = lock.get("base_images", [])
        if not isinstance(base_images, list):
            raise ValueError(f"{path}: base_images must be a list")
        for item in base_images:
            if not isinstance(item, dict):
                raise ValueError(f"{path}: base_images items must be mappings")
            image_id = _require_string(item.get("image_id"), f"{path}: image_id")
            relation = _require_string(
                item.get("relation"), f"{path}: base_images.relation"
            )
            if relation not in {"version-aligned", "validated-in-image"}:
                raise ValueError(f"{path}: unknown base-image relation {relation!r}")
            if image_id not in known_images:
                raise ValueError(f"{path}: unknown base image {image_id!r}")
            if known_images[image_id]["family"] != family:
                raise ValueError(f"{path}: base image family does not match {image_id}")
            image = known_images[image_id]
            if name not in image.get("related_profiles", []):
                raise ValueError(
                    f"{path}: base image {image_id!r} does not declare profile {name!r}"
                )
            if image.get("flagos_op_relation") != relation:
                raise ValueError(
                    f"{path}: base image {image_id!r} relation does not match "
                    f"{relation!r}"
                )
            if relation == "validated-in-image" and not (
                item.get("image_instance_id") or item.get("image_digest")
            ):
                raise ValueError(
                    f"{path}: validated-in-image requires image_instance_id or image_digest"
                )


def _display(value) -> str:
    if value is None or value == "":
        return "未登记"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _hardware_summary(hardware: dict) -> str:
    parts = [
        f"{_display(hardware.get('model'))} ×{_display(hardware.get('count', 1))}",
        f"{_display(hardware.get('memory_per_device'))}/卡",
    ]
    for key in ("compute", "driver", "firmware", "runtime", "monitoring", "visible_device"):
        if hardware.get(key) is not None:
            label = {
                "compute": "计算拓扑",
                "driver": "驱动",
                "firmware": "固件",
                "runtime": "运行时",
                "monitoring": "监控工具",
                "visible_device": "可见设备",
            }[key]
            parts.append(f"{label}: {_display(hardware[key])}")
    return " · ".join(parts)


def _base_images(items: list[dict]) -> str:
    if not items:
        return "未登记"
    relations = {
        "version-aligned": "版本对齐",
        "validated-in-image": "镜像内验证",
    }
    return " · ".join(
        f"[`{item['image_id']}`](base-images.md#{item['image_id']})"
        f"（{relations[item['relation']]}）"
        for item in items
    )


def _anchor(name: str) -> str:
    return name.replace("_", "-")


def render_index(locks: list[dict]) -> str:
    lines = [
        "# 软硬件环境版本汇总",
        "",
        "[← 返回文档中心](index.md)",
        "",
        "这一页汇总当前有交付证据的设备 profile、硬件、系统、Python 和核心软件栈版本。",
        "它是环境声明索引，不是兼容性承诺；性能与精度结论仍必须回到对应算子报告。",
        "本页由 `configs/env/*.lock.yaml` 生成，修改环境后不要手工编辑本页。",
        "",
        "<a id=\"sources\"></a>",
        "## 事实来源与使用方式",
        "",
        "| 事实 | 事实来源 | 用途 |",
        "|---|---|---|",
        "| 设备映射、dispatch key、框架参数 | [configs/devices/](../configs/devices/) | 运行测试和注册算子 |",
        "| 硬件、系统、Python、软件包版本 | [configs/env/](../configs/env/) | 环境声明与 `check_env` 核对 |",
        "| 单次精度、性能和应用验证 | 算子 `REPORT.md`、`reports/` 与性能 JSON | 限定每条结论的适用范围 |",
        "| 算子 / 平台覆盖状态 | [算子 / 平台矩阵](operator-catalog.md) | 判断某平台是否已有交付算子 |",
        "| 外部基座镜像线索 | [基座镜像索引](base-images.md) | 查看协作方镜像 tag 与分发状态 |",
        "",
        "核对当前运行时：",
        "",
        "```bash",
        "python3 scripts/check_env.py --device <profile>",
        "```",
        "",
        "修改环境声明后重新生成本页：",
        "",
        "```bash",
        "python3 scripts/build_env_index.py",
        "python3 scripts/build_env_index.py --check",
        "```",
        "",
        "<a id=\"current\"></a>",
        "## 已登记实测环境",
        "",
    ]

    for lock in locks:
        name = lock["name"]
        collected_at = lock["collected_at"]
        hardware_at = lock.get("hardware_collected_at", collected_at)
        lines.extend([
            f"<a id=\"{name}\"></a>",
            f"### {_display(lock.get('description'))}",
            "",
            "| 项 | 声明 |",
            "|---|---|",
            f"| profile | [`{name}`](../configs/devices/{name}.yaml) |",
            f"| 平台家族 | `{_display(lock.get('family'))}` |",
            f"| 硬件 | {_hardware_summary(lock['hardware'])} |",
            f"| 系统 / 内核 | {_display(lock.get('system', {}).get('os'))} / {_display(lock.get('system', {}).get('kernel'))} |",
            f"| Python | `{_display(lock.get('python'))}` |",
            f"| 声明采集时间 | 软件栈 {collected_at} · 硬件快照 {hardware_at} |",
            f"| 环境锁 | [`{name}.lock.yaml`](../configs/env/{name}.lock.yaml) |",
            f"| 基座镜像 | {_base_images(lock.get('base_images', []))} |",
            "",
            "适用范围：",
            "",
            *[f"- {item}" for item in lock["scope"]],
            "",
            "| 软件包 / 组件 | 锁定版本 |",
            "|---|---|",
            *[
                f"| `{package}` | `{_display(version)}` |"
                for package, version in lock["packages"].items()
            ],
            "",
            "证据入口：",
            "",
            *[
                f"- [{item['path']}](../{item['path']})：{item['note']}"
                for item in lock["evidence"]
            ],
            "",
            f"备注：{lock['notes'].strip().replace(chr(10), ' ')}",
            "",
        ])
        if lock.get("historical_note"):
            lines.extend([
                f"历史环境：{lock['historical_note'].strip().replace(chr(10), ' ')}",
                "",
            ])

    lines.extend([
        "<a id=\"not-validated\"></a>",
        "## 未登记为实测环境",
        "",
        "| 对象 | 当前状态 | 边界 |",
        "|---|---|---|",
        "| `cpu` profile | 语义参考与 CPU CI | 无固定硬件环境锁；GitHub runner 版本随 CI 变化，不能转化为加速器结论 |",
        "| `nvidia` profile | profile 就绪，未建立实测环境锁 | 未声明已验证的 NVIDIA 硬件、驱动或 CUDA 版本 |",
        "| A100 80GB SXM | 外部论文参照 | 数据来自 FlashAttention-2 论文图表读取，不是本仓库运行环境；见 [A100 对标](../ops/sdpa/reports/perf_a100.md) |",
        "",
        "规划中的厂商家族不预填环境声明。目标范围和当前覆盖以",
        "[算子 / 平台矩阵](operator-catalog.md) 为准；只有拿到实际硬件并完成",
        "设备 profile、环境锁和算子验证后，才加入本页的实测环境列表。",
        "",
        "<!-- AUTO-GENERATED FROM configs/env/*.lock.yaml; do not edit. -->",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check the generated index")
    args = parser.parse_args()

    locks = load_locks()
    validate_locks(locks)
    rendered = render_index(locks)

    if args.check:
        if OUTPUT_PATH.read_text(encoding="utf-8") != rendered:
            print(f"{OUTPUT_PATH} is out of date; run scripts/build_env_index.py")
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
