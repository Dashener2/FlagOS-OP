#!/usr/bin/env python3
"""Ascend A1 算子真实 vLLM 前向验证（C2 证据采集）。

用途: 在 Ascend910 + vllm_fl 环境上，对一个 A1 aten 算子做「注册 + 真实 vLLM
前向」，输出命中计数与生成 token，用于与 baseline 对比，判定 C2。

用法:
  # baseline（不注册）
  VLLM_ENABLE_V1_MULTIPROCESSING=0 TASK_QUEUE_ENABLE=0 \
  ASCEND_RT_VISIBLE_DEVICES=1 \
  python3 scripts/e2e_vllm_c2_ascend.py --op __baseline__ --model /path/to/tiny

  # 注入某算子（进程内引擎，保证注册在 forward 的同一进程）
  VLLM_ENABLE_V1_MULTIPROCESSING=0 TASK_QUEUE_ENABLE=0 \
  ASCEND_RT_VISIBLE_DEVICES=1 \
  python3 scripts/e2e_vllm_c2_ascend.py --op empty_strided --platform ascend \
      --model /path/to/tiny

输出: 一行 JSON {"op","ids","count"}；baseline 与注入的 ids 逐位一致且 count>0
即达 C2。参考结果见 docs/known-issues.md #26。
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True,
                    help="aten 算子/目录名；__baseline__ 表示不注册")
    ap.add_argument("--model", required=True)
    ap.add_argument("--platform", default="ascend")
    ap.add_argument("--dispatch-key", default="AutogradPrivateUse1")
    ap.add_argument("--max-tokens", type=int, default=8)
    args = ap.parse_args()

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("TASK_QUEUE_ENABLE", "0")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import torch  # noqa: F401
    import torch_npu  # noqa: F401

    regmod = None
    if args.op != "__baseline__":
        regmod = importlib.import_module(f"ops.{args.op}.register")
        regmod.register_a1(args.dispatch_key, platform=args.platform)

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, tokenizer=args.model, enforce_eager=True,
              max_model_len=256, tensor_parallel_size=1,
              gpu_memory_utilization=0.5, seed=0, disable_log_stats=True)
    out = llm.generate(["the future of AI is"],
                       SamplingParams(max_tokens=args.max_tokens,
                                      temperature=0.0))
    ids = list(out[0].outputs[0].token_ids)
    cnt = None
    if regmod is not None:
        cnt = regmod.CALL_COUNT.get(args.op, 0)
    print("C2_RESULT " + json.dumps({"op": args.op, "ids": ids,
                                     "count": cnt}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
