# Ascend910 真实 vLLM C2 证据 — empty_like

- **级别**: **C2**（真实 vLLM 前向实际命中，输出与基线逐位一致）
- **方法**: vLLM **EngineCore（多进程）** baseline/injected 双跑；直接注册本算子到
  `AutogradPrivateUse1 + PrivateUse1`（`platform=ascend`）；在 EngineCore 进程退出时
  一次性 dump `CALL_COUNT`（避免热路径 I/O 扰动）。
- **引擎**: `vllm 0.20.2+empty` + `vllm-plugin-fl`（`PlatformFL`, `device_type=npu`）
- **模型**: `/tmp/flagos_tiny_llama`（hidden=64, heads=4, layers=2, head_dim=16）
- **输入**: `the future of AI is`；max_tokens=8, temperature=0
- **结果**: baseline/injected token 均为 `[7,21,1,18,10,23,21,1]`（逐位一致）；
  EngineCore 进程命中 **count=97**
- **环境**: driver 25.5.0 / CANN 9.0.0 / torch 2.10.0+cpu / torch_npu 2.10.0 /
  flagtree 3.5.1 / vllm 0.20.2+empty / vllm_fl `g8b059122e` / SOC Ascend910_9382
- **证据 JSON**: [e2e_vllm_c2_ascend910.json](e2e_vllm_c2_ascend910.json)
- **复现**: `scripts/e2e_vllm_c2_ascend.py`（in-process）或 `scratch/repro_mp`（EngineCore，仓库外）
