# 性能分册: result_type

## 口径

短采样 warmup 20 + iters 100 + MLU 同步；result_type 为元数据/同步操作，
延迟绝对值小、受调度影响大，仅作可用性采样，不作门禁。

```bash
python3 script/bench_perf.py --device mlu
```

## 结果（MLU590-M9）

| 实现 | 延迟 |
|---|---|
| torch 级 | 0.0013ms |
| reference（原生） | 同量级 |

## P800（p800-kunlunxin）复验（2026-10-26）

环境：torch 2.9.0+cu129 / torch_xmlir 1.0.0.1 / Triton 3.6.0 / FlagGems 4.2.1rc0；测试设备 `cuda:1`。口径与 MLU 相同（warmup 20 + iters 100 + 设备同步）。

| 实现 | 延迟 (ms) |
|---|---:|
| reference | 0.0004 |
| torch | 0.0004 |

纯 dtype 元数据，reference 0.00042 ms / torch 0.00042 ms。

基线已写入 `perf/baselines/p800-kunlunxin.json`；`python3 scripts/perf_compare.py --device p800-kunlunxin` → **OK**（FAIL 0 / WARN 0 / NEW 0）。
