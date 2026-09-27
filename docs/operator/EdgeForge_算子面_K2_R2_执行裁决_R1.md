# EdgeForge · 算子面 K2-R2 执行裁决 · R1

- 执行日期：2026-09-16（UTC+8）
- 正式载体：C02-S.O + bitsandbytes NF4 double quant，RTX 4060 Laptop / sm_89 / WSL2
- 透明版本化协议：`configs/operator/k2_nf4_decode_gemv_prereg_r2.json`
- 协议 SHA-256：`86c3e13edaca148b4c7bcceddfe9d86806997155b285ca39408876e721e739ae`
- 最终状态：`K2_R2_CLOSED_DAY1_NOT_POSITIVE`

## 1. 裁决

K2-R2 的单一 FP32 NF4 语义正确性门 **PASS**，但第一个有效日历日的冻结性能门为 **NEGATIVE**。因此 K2-R2 已按预注册规则关闭，不得继续 Day-2、ncu Tier-B 或 C02 回连。

这不是 R1 的重写或独立复现：R1 的双 oracle 不可满足失败保持原样；R2 明示使用了 R1 已见事实设计单一语义 oracle 与新鲜 holdout。candidate 在 R2 冻结前已经存在，整个 R2 只使用同一份 candidate 身份：

`149b4b8c651cfa6530f320779687132f0bdb31f511282473250ffc80257d7c65`

R2 的结果应准确表述为：**candidate 对选定 FP32 数学语义正确，但在本机正式流式工作负载上远慢于 vendor eager，未达到继续投资门。**

## 2. 正确性门

单一 hard oracle 按 R2 合同执行：在 FP32 中显式还原 nested scales 与 NF4 权重，BF16 activation 转 FP32 后做 `F.linear`，最后 cast BF16。bitsandbytes fused eager 只作必报的非 gating 数值对照和性能基线。

正式结果：

| 覆盖 | 结果 | 最大绝对误差 |
|---|---:|---:|
| 343 个真实 C02 权重，unit-normal seed 20261001 | 343/343 allclose | 0.0078125 |
| 14 代表变体 × 4 分布 × 3 新 seeds | 168/168 allclose | 0.125 |
| fallback 拒绝矩阵 | 22/22 exact | — |
| source tensor、对象身份、版本与输出合同 | PASS | — |

非 gating 对照如实保留：新鲜 holdout 中 candidate 与 vendor eager 为 90/168 case allclose，vendor eager 与主 oracle 同样为 90/168 case allclose。这与 R1 暴露的实现特定舍入路径差异一致，不影响 R2 的单一 hard oracle 判定，也不能被省略成“candidate 与 vendor 数值完全相同”。

首次 R2 runner 在完成 candidate correctness 主体后，于构造 `scale_offset_size_mismatch` fallback case 时对真实 0-D offset 执行 `offset[:0]`，触发 `IndexError`。该失败原样留证；只修复 runner 的空张量构造为 `offset.reshape(1)[:0]`，协议和 candidate 均未改变。最终 receipt 因此记录 `candidate_identity_attempt=1`、`harness_execution_attempt=2`。

## 3. Day-1 性能门

性能窗口严格使用完整 343-weight 模型顺序，eager / fullgraph compile / candidate 三路各 50 个全序列预热，并执行 200 个三路旋转的全序列测量。没有重复单一权重获取 L2 热缓存数字。

有效性门全部通过：

- 同日 512 MiB float32 triad ceiling：241.904639 GB/s；
- memory clock p50：8000 MHz，OP0 比率 1.000，阈值 0.95；
- 温度 63–65°C；无 software thermal、hardware thermal 或 hardware power-brake throttle；
- 正式窗口 57 条 telemetry；343 个权重和 14 个变体全部完成 200 轮；
- candidate 与协议 SHA 从运行开始到 receipt 写入保持不变。

核心结果：

| 指标 | eager | compile | candidate |
|---|---:|---:|---:|
| invocation-weighted p50 / call | 89.935859 µs | 203.643897 µs | 511.205873 µs |
| 343-call sequence p50 | 32.415743 ms | 75.582977 ms | 176.124352 ms |

- best baseline：eager；
- candidate 对 best baseline 加权 speedup：`0.1759288463×`；
- candidate Tier-A 实测带宽：11.717869 GB/s；同日 ceiling 相对分 `0.048440`；
- 三个 MLP 变体 speedup：down `0.150442×`、gate `0.170155×`、up `0.170155×`；
- 冻结继续门要求 aggregate ≥1.05× 且三个 MLP 均 ≥0.95×，两项均明显失败；分类为 `NEGATIVE`。

结果远离阈值，不属于噪声边界或跨日待判情形。R2 协议规定只有正向 Day-1 才可进入第二日；因此不执行第二日来“争取翻盘”，也不以 ncu 或回连延长已经关闭的 candidate。

## 4. 证据索引

| 证据 | SHA-256 | 结论 |
|---|---|---|
| `configs/operator/k2_nf4_decode_gemv_prereg_r2.json` | `86c3e13edaca148b4c7bcceddfe9d86806997155b285ca39408876e721e739ae` | 透明 adaptive R2 冻结协议 |
| `results/operator/pool/k2_r2_prereg_receipt_20260916.json` | `cd35f54790f6b8f2ed2a8a929fb407114fd52c793d8b1f79cefe210853e2dd6a` | R2 协议/source/candidate 身份核验 PASS |
| `scripts/operator/k2_nf4_gemv.py` | `149b4b8c651cfa6530f320779687132f0bdb31f511282473250ffc80257d7c65` | 唯一锁定 candidate |
| `results/operator/failures/k2_r2_correctness_failure_20260916T164041.json` | `cad059a35b9171158219d343b45d96817876423433849d3f8d31a0736eee40ad` | 首次 runner fallback-harness 中断；非 candidate failure |
| `scripts/operator/run_k2_r2_correctness.py` | `7127cb08890ebb9ec864a9d2df8ff658fe1187d2e30533ea15d8e18a34637701` | 修复后 R2 correctness runner |
| `results/operator/pool/k2_r2_correctness_20260916.json` | `890bef7a0200ce3a012427190a1b1e7d03ab1355f3e4e135dc14f4d9e29d9429` | 343 + 168 正确性明细 |
| `results/operator/pool/k2_r2_correctness_receipt_20260916.json` | `32512ce603321d59f34d71a6d8cc9ca5272afc6f441731f291cb0579cb990360` | R2 correctness PASS |
| `scripts/operator/run_k2_r2_day1.py` | `eefd7f7acc85628271368946622c668d4f4c5f81baa0cb6505c442206de802f6` | bounded Day-1 runner |
| `results/operator/pool/k2_r2_day1_20260916.json` | `07c7006651254246a3023e8fad8b4652697f3b7c2c5711eebb89b5e82561dff0` | 有效 Day-1 全量结果 |
| `results/operator/pool/k2_r2_day1_receipt_20260916.json` | `8bef2b9e9385869211d826b47750bac1cec16b35c976aa142b569dd2059dc6bb` | `K2_R2_CLOSED_DAY1_NOT_POSITIVE` |

## 5. 下一授权边界

K2 的 R1 与 R2 均已形成终态：R1 是协议不可满足失败；R2 是语义正确、性能负结果。不得继续调整这个已见 holdout 和性能数据的 candidate 后仍称其为同一 R2 尝试。

OP5 backlog 的下一项是 **GQA decode attention（SWA 512 + full ≤10k 双形态）**。它必须先完成自己的 Authenticity/Bottleneck 证据与测前冻结协议，不能继承 K2 的性能余量、correctness oracle 或 candidate 身份。P0-3b、OP-E、训练和模型侧 M4–M7 仍未因本次裁决获得授权。
