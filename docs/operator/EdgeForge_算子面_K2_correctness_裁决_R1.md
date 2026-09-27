# EdgeForge · 算子面 K2 correctness 裁决 · R1

> R1 历史终态：本文记录的双 oracle 不可满足失败保持有效。owner 后续授权的透明 K2-R2 已完成并因 Day-1 性能负结果关闭；当前裁决见 `docs/operator/EdgeForge_算子面_K2_R2_执行裁决_R1.md` 与 `docs/operator/EdgeForge_算子面_阶段台账_R6.md`。

- 执行日期：2026-09-16（UTC+8）
- 正式载体：C02-S.O + bitsandbytes NF4 double quant，RTX 4060 Laptop / sm_89 / WSL2
- 冻结预注册：`configs/operator/k2_nf4_decode_gemv_prereg.json`
- 预注册 SHA-256：`bb497bf4be20a922abbe6edbf8e4812d55b01e14d8db088371ec332ad61f299f`
- 机器 receipt：`results/operator/pool/k2_correctness_gate_receipt_20260916.json`
- receipt SHA-256：`c25e744b5377f4f7254a24d3ab1139d842d6cf424f4d2d16dafe6034f422e04b`

## 1. 裁决

K2 candidate 已按冻结设计实现，但 K2 不能进入第一个日历日的正式性能测量。原因不是“暂时还没调快”，而是冻结 correctness 合同同时要求 candidate 在 `atol=0.03125, rtol=0.02` 下贴合两个彼此不兼容的 oracle。

对全部 14 个代表变体 × 3 分布 × 3 seeds，只读审计得到：

- 126 个冻结 case 全部执行；
- 35 个 case 中，eager 与 explicit-reference 给 candidate 的允许区间存在不相交元素；
- 共 297 个输出元素的两个允许区间不相交；
- 最大未覆盖间隙为 `0.10308104753494263`；
- 矛盾集中于冻结的 `wide_normal` 分布，覆盖 13/14 个变体和全部 3 个 seeds。

因此，在实数域上都不存在能同时通过两个 allclose 断言的 candidate 输出；BF16 离散输出更不可能满足。事后放宽阈值、删掉一个 oracle 或更换已看过的分布，都会改写正式预注册，本轮不允许这样做。

K2 当前状态为 `BLOCKED_FROZEN_CORRECTNESS_ORACLE_CONFLICT`，不是性能 `NEGATIVE`，因为正式性能窗口从未开始。

## 2. 实际执行

1. 实现 `scripts/operator/k2_nf4_gemv.py`：Triton fused high/low nibble decode、nested scale reconstruction、FP32 accumulation 和 BF16 output；未物化完整权重。
2. 实现 `scripts/operator/run_k2_day.py`：冻结 source/hash 核验、semantic microcase、343 真实权重、14 × 3 × 3 参照、fallback matrix，以及在 correctness PASS 后才可进入的三路性能窗口。
3. 第一次启动在完整工件哈希期间遭 WSL 宿主重启；该轮未进入 correctness，单独留证。
4. 第一个完整正式 correctness 轮通过 semantic microcase 并继续到 343 实际权重之后，在 `k_proj_full_nonshared / wide_normal / seed 20260916` 出现 `max_abs=0.125`，按冻结阈值失败并留证。
5. 由于 candidate 同时偏离两个 oracle，执行不依赖 candidate 输出的全量 oracle 一致性审计，确认门本身不可满足。

## 3. 不可满足性证明

对任一 oracle 值 `o`，冻结 allclose 要求 candidate 值 `c` 满足：

`abs(c - o) <= 0.03125 + 0.02 * abs(o)`

这为 `c` 定义一个闭区间。审计直接计算 eager 和 explicit-reference 两个区间的交集。只要交集为空，就无需考察任何 candidate 实现：没有任何实数输出可以同时通过。

本轮 explicit-reference 严格依冻结定义执行：bitsandbytes `dequantize_4bit` 按 C02 `QuantState.dtype=torch.bfloat16` 得到权重，再将 activation 与解码权重转 FP32 做显式 `F.linear`，最后 cast BF16。eager 为同一真实 packed weight 的 bitsandbytes fused `gemm_4bit`。

例如首个失败 case 已定位出 3 个区间不相交元素；全量审计将这一事实扩展为 35 case / 297 elements，因而不是单一 kernel config 或归约顺序的局部 bug。

## 4. 证据索引

| 证据 | SHA-256 | 结论 |
|---|---|---|
| `scripts/operator/k2_nf4_gemv.py` | `149b4b8c651cfa6530f320779687132f0bdb31f511282473250ffc80257d7c65` | K2 candidate 实现 |
| `scripts/operator/run_k2_day.py` | `c9e9af9550f4be4bc275b5696c75549c4cc7b2e8244f7aaf2d7a5731550e3625` | correctness/day runner；correctness 不过不进性能窗口 |
| `results/operator/failures/k2_correctness_external_host_restart_20260916T160001.json` | `af3bba5c323b7609ec168cd6bda2567e7b5ba1587c0c043a41c688435a5ee7a2` | 宿主重启；无正式数据 |
| `results/operator/failures/k2_day_failure_20260916T160332.json` | `670627b9a38ef60be8c80160245d07bfd482182939b40a05321a74b1e0102f6d` | 首个完整 correctness 轮按冻结阈值失败 |
| `scripts/operator/audit_k2_oracle_consistency.py` | `3e2807a8d945b71475fad8cfa9e5b7c6a8ac9a7613e9732a12fcb9120033a697` | 126-case oracle 不可满足性审计 runner |
| `results/operator/failures/k2_oracle_consistency_audit_20260916.json` | `0158290477f8f8d45cb0419e7e0f4daa5edc9048d239ce9edc585d9aa3901573` | 35 case / 297 element 区间不相交 |
| `results/operator/pool/k2_correctness_gate_receipt_20260916.json` | `c25e744b5377f4f7254a24d3ab1139d842d6cf424f4d2d16dafe6034f422e04b` | K2 correctness 机器裁决 |

## 5. 未执行项与下一步边界

本轮没有生成 K2 day-1 性能数字，没有 ncu Tier-B，没有 `torch.library` 回连，也没有 5 盘/11 请求 A/B。这些不能标记 PASS。

后续需 owner 在两条路径中明确选择：

1. 关闭本次 K2，保留不可满足预注册与全部失败证据，转入 backlog 下一项 GQA decode attention 的独立 B probe；或
2. 明确授权新的、透明版本化的 K2 协议；必须只保留一个内部一致的正确性语义，改用未见的 holdout inputs，并明确标注 R1 失败已影响 R2 设计。

未获得该选择前，不自动开始 GQA、不修改 R1 预注册，也不将当前 candidate 的非正式烟测解读为性能结论。
