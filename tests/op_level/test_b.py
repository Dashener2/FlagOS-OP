#!/usr/bin/env python3
"""B 算子层测试: vendor 注册 / 选择 / 计数 / 语义精度"""
from __future__ import annotations


def run(profile) -> bool:
    import json
    import os

    import torch
    import torch.nn.functional as F

    dev = profile.torch_device
    # dispatch 插件自加载（直接调用本文件时也生效）。
    # 显式赋值 + 重置 manager: --all 模式下同进程串跑多个测试，
    # 避免上一个测试的插件残留/OpManager 缓存污染。
    # VLLM_FL_OP_CONFIG（JSON 文件路径）优先级高于 VLLM_FL_PER_OP，
    # 且在 vllm_fl.utils import 期就被缓存——必须在导入 vllm_fl 之前
    # 清掉并刷新缓存，钉选才生效；结束后恢复原值防污染后续测试。
    _saved_env = {
        k: os.environ.pop(k, None)
        for k in ("VLLM_FL_PLUGIN_MODULES", "VLLM_FL_OP_CONFIG",
                  "VLLM_FL_PER_OP")
    }
    try:
        import vllm_fl.utils as _vllm_utils
        if hasattr(_vllm_utils, "_load_op_config_from_env"):
            _vllm_utils._load_op_config_from_env()
        os.environ["VLLM_FL_PLUGIN_MODULES"] = \
            "routes.b_vendor.backend.register_ops"
        os.environ["VLLM_FL_PER_OP"] = "silu_and_mul=vendor:audit|reference"
        from vllm_fl.dispatch import (
            get_default_manager,
            call_op,
            reset_default_manager,
        )
        from vllm_fl.dispatch.policy import (
            reset_global_policy,
            with_preference,
            with_allowed_vendors,
        )
        reset_global_policy()
        reset_default_manager()

        print("=" * 60)
        print(f"B op-level [{profile.name}]: audit vendor @ {dev}")
        print("=" * 60)

        m = get_default_manager()
        m.ensure_initialized()
        impls = m.registry.snapshot().impls_by_op.get("silu_and_mul", [])
        ids = [i.impl_id for i in impls]
        assert "vendor.audit" in ids, f"vendor.audit 未注册: {ids}"
        print(f"  silu_and_mul impls : {ids}")

        x = torch.randn(128, 8192, dtype=torch.bfloat16, device=dev)
        d = x.shape[-1] // 2
        ref = (F.silu(x[..., :d].float()) * x[..., d:].float()).to(x.dtype)

        # 算子层用 reference 委托（语义明确）
        os.environ["AUDIT_DELEGATE"] = "reference"
        with with_preference("vendor"), with_allowed_vendors("audit"):
            out = call_op("silu_and_mul", None, x)
            used = m._called_ops["silu_and_mul"]
        assert used == "vendor.audit", f"选中了 {used}"
        err = (out.float() - ref.float()).abs().max().item()
        print(f"  selected impl      : {used}")
        print(f"  accuracy           : max_err={err:.3e}")
        assert err < 1e-1

        # 分片感知读取（audit 写 pid 分片; 兼容旧单文件格式）
        import glob as _glob
        base = os.environ.get("AUDIT_VENDOR_COUNT_FILE", "/tmp/audit_vendor_counts")
        counts: dict = {}
        for p_ in _glob.glob(f"{base}.*") + [f"{base}.json"]:
            try:
                with open(p_) as f:
                    for k, v in json.load(f).items():
                        counts[k] = counts.get(k, 0) + v
            except (OSError, ValueError):
                continue
        assert counts.get("silu_and_mul", 0) >= 1
        print(f"  call counter       : {counts}")
        print("  => B op-level PASS")
        return True
    finally:
        for _k, _v in _saved_env.items():
            if _v is None:
                os.environ.pop(_k, None)
            else:
                os.environ[_k] = _v


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from common.device import load_profile
    name = sys.argv[1] if len(sys.argv) > 1 else "p800-kunlunxin"
    run(load_profile(name))
