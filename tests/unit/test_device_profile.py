"""设备 profile 加载、模型路径覆盖与自动探测边界（纯 CPU）。"""
import sys
import types

import pytest


def test_model_path_env_override(monkeypatch):
    from common.device import load_profile
    monkeypatch.setenv("FLAGOS_MODEL_PATH", "/tmp/my-model")
    p = load_profile("cpu")
    assert p.engine_args()["model"] == "/tmp/my-model"


def test_model_path_profile_default(monkeypatch):
    from common.device import load_profile
    monkeypatch.delenv("FLAGOS_MODEL_PATH", raising=False)
    p = load_profile("cpu")
    assert p.engine_args()["model"].endswith("llama-3.1-8b-like")


def _write_profile(directory, name, vendor, module, enabled=None):
    lines = [
        f"name: {name}",
        f"vendor: {vendor}",
        "auto_detect:",
        f"  module: {module}",
    ]
    if enabled is not None:
        lines.append(f"  enabled: {enabled}")
    (directory / f"{name}.yaml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def test_auto_detect_module_path_respects_enabled(tmp_path, monkeypatch):
    import common.device as cd

    _write_profile(tmp_path, "aaa-disabled", "vx", "json", enabled=False)
    _write_profile(tmp_path, "zzz-enabled", "vy", "json")
    monkeypatch.setattr(cd, "DEVICES_DIR", tmp_path)

    # module 探测路径: 禁用项（排序更靠前、module 可 import）必须被跳过
    detected = cd.detect_profile()
    assert detected.name == "zzz-enabled"

    # 显式加载禁用项仍正常
    disabled = cd.load_profile("aaa-disabled")
    assert disabled.name == "aaa-disabled"
    assert disabled.auto_detect_enabled is False


def test_auto_detect_vendor_fallback_respects_enabled(tmp_path, monkeypatch):
    import common.device as cd

    _write_profile(
        tmp_path, "aaa-disabled", "vx", "missing_mod_for_detect",
        enabled=False,
    )
    _write_profile(tmp_path, "zzz-other", "vy", "missing_mod_for_detect")
    monkeypatch.setattr(cd, "DEVICES_DIR", tmp_path)

    def _module(name):
        mod = types.ModuleType(name)
        mod.__path__ = []
        return mod

    fg = _module("flag_gems")
    runtime = _module("flag_gems.runtime")
    backend = _module("flag_gems.runtime.backend")
    device = _module("flag_gems.runtime.backend.device")

    class _FakeDetector:
        @property
        def vendor_name(self):
            return "vx"

    device.DeviceDetector = _FakeDetector
    fg.runtime, runtime.backend, backend.device = runtime, backend, device
    for name, mod in [
        ("flag_gems", fg),
        ("flag_gems.runtime", runtime),
        ("flag_gems.runtime.backend", backend),
        ("flag_gems.runtime.backend.device", device),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    # vendor fallback 路径: 唯一 vendor=vx 的 profile 已禁用 → 必须报错
    with pytest.raises(RuntimeError, match="--device"):
        cd.detect_profile()
