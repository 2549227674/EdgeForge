# EdgeForge · 算子面·板端 OPE5-① Attention 序列正式段报告（2026-09-21）

## 1. 最终裁决

`VALID_PRIMARY_OPE5_01_SINGLE_SEGMENT_OWNER_WAIVER_PARTIAL_TIER_B_PROTOCOL_CASE_ID_DEFECT`

OPE5-① 已用唯一板端正式段完成完整 decode-attention sequence 的 capability、correctness、SWA/full 主计时和 SWA Tier-B。主结果有效，但原预登记并非严格全满足：Tier-B 把 correctness-only 的 `full_k3123_d512` 写成性能 case，binary 在参数解析阶段拒绝该未知 case，导致 runner 最终退出 90，full Tier-B 未执行。

依 owner 的“只测一次、取消同日复测”指令，没有修改 source-lock 后补跑、没有第二正式段。后验裁决只豁免这一个辅助门：保留已独立完成的主计时与 SWA Tier-B，删除所有 full-context PMU 声明，并明确标记 `strict_original_prereg_complete=false`。

## 2. 正确性和能力门

- 13 个性能 case 的 grouped QK/PV：26/26 core0 FP16→FP32 capability cells PASS。
- 6 个代表 shape × 2 分布 × 2 seeds × 3 实现：270,336/270,336 输出元素 PASS，零 mismatch。
- 三实现各 90,112 个元素；最大绝对误差均为 0.0009765625。最大相对误差 0.090909 出现在近零元素，仍满足冻结的 `atol=0.03125, rtol=0.02` 联合门。
- 最大 FP32 softmax pre-round row-sum error 为 `3.38e-8`，最大 FP16 probability row-sum error 为 `2.6071e-4`。
- GQA 头映射、SWA ring wrap、full prefix append、QK mask padding、PV zero padding、source mutation boundary、输出 shape/dtype/finite/contiguous：8/8 PASS。

因此 grouped M=4 carrier 在本板 Runtime 2.3.2 / RKNPU core0 上是可运行且数值合格的，不再是 OPE2 的 zero-attempt standby。

## 3. 主计时结果

每条路径覆盖同一个完整计时边界：KV append → active view → QK transpose/V packing 与 padding → QK → FP32 softmax → FP16 probability carrier → PV → FP16 output。13 case × 3 实现共 39/39 主计时格完成；每格至少 50 次且累计至少 500 ms，SWA 快格最多 249 次。

以下“全人口”是 `sum(projected_invocations × case_p50) / 911148`；“42-call forward-equivalent”只是 census-weighted isolated counterfactual。

| 完整序列 | SWA p50 | full 加权 case-p50 | 全人口加权 case-p50 | 42-call equivalent | OPE4 TPOT fraction |
|---|---:|---:|---:|---:|---:|
| CPU scalar | 4.232375 ms | 105.505379 ms | 21.111209 ms | 886.670776 ms | 198.748842% |
| CPU NEON | 2.080458 ms | 28.755624 ms | 6.526319 ms | 274.105397 ms | 61.441215% |
| Hybrid RKNPU | 3.747917 ms | 37.493569 ms | 9.372192 ms | 393.632077 ms | 88.233335% |

核心比较：

- Hybrid 相对 CPU NEON 的全人口 latency ratio 为 **1.436061×**，即 NEON 对 hybrid 为 `0.696349×` speedup；hybrid 明确更慢。
- 分族同向：SWA hybrid/NEON = **1.801486×**，full hybrid/NEON = **1.303869×**。
- Hybrid 相对 scalar 加速 **2.252537×**；CPU NEON 相对 scalar 加速 **3.234780×**。
- 分类：`NEGATIVE_HYBRID_SLOWER_THAN_NEON`。

本 carrier 中 RKNPU 只承担四次小 M grouped matmul，而 KV materialization/packing、四次同步 API 边界与 softmax 留在 host；结果说明该组合没有胜过单 A76 NEON 完整序列。现有计时没有子阶段 breakdown，不能把差额单独归因给 packing、API overhead、softmax 或 matmul 中任一项。

三路径最大 IQR/p50 分别为 scalar 0.256592、NEON 0.342189、hybrid 0.282568。预登记没有 IQR 硬阈值，所有 count/elapsed 门均满足；这些离散度随原始 case 行完整披露，不据此补跑。

## 4. Tier-B 与协议缺陷

SWA `K=512,D=256` 的三实现 process-block 五事件均 numeric：

| 实现 | repetitions | task-clock | cycles:u | instructions:u | context switches | migrations |
|---|---:|---:|---:|---:|---:|---:|
| CPU scalar | 140 | 570.70 ms | 1,293,695,751 | 2,919,499,249 | 3 | 1 |
| CPU NEON | 312 | 549.76 ms | 1,244,632,716 | 1,608,166,651 | 5 | 1 |
| Hybrid RKNPU | 163 | 501.62 ms | 1,077,855,970 | 1,745,602,338 | 696 | 0 |

这些是整个 `perf stat` 进程块的 CPU counter，含 context create、warmup 与正式重复，不能除成纯单步 NPU 事件，也不是 NPU bytes/ops。Hybrid 的 context switches 高与同步 host↔runtime 调用现象相容，但不能仅凭这张表归因具体 runtime 机制。

full Tier-B 的第一个请求在 benchmark work 前即失败：

`OPE5_ATTENTION_ERROR=unknown performance case: full_k3123_d512`

原因是 3123 属于 correctness 代表长度，13 个性能 case 使用 2897/3349 等 K3 等调用量分箱。处理为 `PARTIAL_SWA_ONLY_OWNER_WAIVER`；不把 3123 偷换成相邻 case，也不补跑。

## 5. 频率、温度和恢复

- 唯一段板端 monotonic duration：205.001717 s。
- 190 个 1 Hz 系统样本；policy6 全程 2.352 GHz / performance，NPU 全程 1.0 GHz / performance。
- policy4 governor 全程 ondemand，只读未改。
- 10 Hz NPU load 共 2,047 个 numeric 样本：core0 mean/p50/max = 0.3273% / 0% / 4%，core1/core2 max 均为 0%。该低值是 CPU/NPU 三路径交错、单步小 matmul 和 10 Hz 采样共同形成的粗粒度现象，不解释为 NPU 峰值能力。
- 全局最高温 54.538°C；最小 thermal headroom 21.385°C。
- runner exit 90 如实保留；governor restore exit 0，policy6 恢复 ondemand，NPU 恢复 `rknpu_ondemand`。

板端 wall-clock 仍是 advisory 域；段时长使用 `CLOCK_MONOTONIC_RAW`，controller 窗口为 `2026-09-21T18:23:37.095589070+08:00` 至 `2026-09-21T18:27:57.278515145+08:00`。

## 6. 声明边界

1. 这是 FP16 synthetic host-orchestrated sequence carrier，不是闭源 RKLLM attention 内部计时或 patch。
2. 42-call equivalent 不能从 OPE4 TPOT 相减，也不能解释成模型提速；其超过或接近 TPOT 只表明 synthetic carrier 代价高。
3. PC BF16 K3 不与本板 FP16 数字合并。
4. 没有 prefill、多核 NPU、token exact、质量、能耗或成本结论。
5. 原预登记的 full Tier-B 门没有满足；不得写成“完整 prereg PASS”或“runner exit-zero”。
6. 只有一个正式段；无 performance smoke、自动 retry、同日复测或跨日复现，replication=`WAIVED_BY_OWNER_NOT_SATISFIED`。

## 7. 权威证据

- `configs/operator/board/ope5_attention_sequence_scope.json`
- `configs/operator/board/ope5_attention_sequence_prereg.json`
- `results/operator/board/ope5/ope5_attention_sequence_source_lock_20260921.json`
- `results/operator/board/ope5/ope5_attention_sequence_segment1_20260921T182337+0800/`
- `configs/operator/board/ope5_attention_sequence_tier_b_adjudication_20260921.json`
- `results/operator/board/ope5/ope5_attention_sequence_receipt_20260921.json`：14/14 post-run checks PASS。

没有为本阶段重复构建 ZIP 证据包；原始段、source-lock、后验裁决和最终 receipt 构成可审计入口。

## 8. 下一任务

OPE5-① 主问题已闭合，不复测。按 backlog 转 OPE5-⑦ `ggml aarch64 q8_0 external anchor`，先解决 `DEFER_EXTERNAL_LINEAGE_GOVERNANCE`：冻结新的 carrier identity、quantization lineage、允许/禁止比较矩阵和最小可交付人口；新 scope/prereg 通过前不构建、不运行。
