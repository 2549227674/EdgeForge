# EdgeForge · 算子面·板端 OPE5-⑤ A55 FP16 GEMV 正式段报告（2026-09-21）

## 1. 结论

OPE5-⑤ 以唯一四 A55 正式段有效闭合，终态：

`VALID_OPE5_05_SINGLE_SEGMENT_OWNER_WAIVER_ISOLATED_4XA55_FP16_GEMV_COMPARATOR`

独立收据 32/32 checks PASS，69,120/69,120 元素正确性全通过。A55 explicit NEON candidate p50 为 **7.254041 ms**，相对 A55 autovec 为 **4.649954×**，解析带宽 **7.231765 GB/s**。

对不可变 OPE5-② A76 primary，A55 candidate 的 raw latency 为 **2.481640×**，analytic bandwidth 为 **0.402959×**，cycles/invocation 为 **1.539714×**，分类 `A55_SLOWER`。这是非并发的系统级 cluster comparator，同时包含频率、微架构、cache/memory service、compiler target 和 OS housekeeping 差异，不是单变量微架构因果。

owner 指令“**不需要复测，取消同日复测，只测一次**”已执行：无 performance smoke、无 A76 重跑、无 A55 同日复测、无自动 retry，replication=`WAIVED_BY_OWNER_NOT_SATISFIED`。

## 2. 测前实现锁与唯一段身份

A55 source 从 SHA-256 `7843a9f7…2c9e` 的 OPE5-② A76 source 派生，严格只含三类变更：`kFirstCpu 4→0`、主 worker 错误文案 `CPU4→CPU0`、诊断 tag `OPE5_FP16_GEMV→OPE5_A55_FP16_GEMV`。内核、常数、RNG/分布/seeds、FP64 oracle、容差、四等分、warmup/时间下限和六排列均未改。

板端预锁审计在任何 benchmark 执行前完成，收据 13/13 PASS：

- policy0 `related_cpus=0 1 2 3`、max=1,800,000 kHz，CPU0–3 单核亲和性全 PASS；
- first-core feature 含 `asimd/fphp/asimdhp`；
- `cycles:u/instructions:u/l2d_cache_refill:u/bus_access:u` 以非 benchmark `true` 进程验证为 numeric；
- `-mcpu=cortex-a55` 只编译/反汇编一次，scalar vector FMLA=0，NEON `fcvtl`=16、vector FMLA=8，临时 binary 已删除；
- 预锁阶段 benchmark execution count=0。

source-lock SHA-256 为 `40fe76d2…85483`，时间早于控制器正式启动；PC 与板端的 candidate、runner、payload、telemetry、prelock receipt、A76 receipt 和 A76 source parent 逐项同 SHA。

唯一正式段为 `ope5_a55_fp16_gemv_segment1_20260921T174008+0800`，控制器窗口 88.574 s，SSH/runner/governor restore 退出均为 0。板端 RTC 仍是非权威 wall-clock 域；时长仅来自板端 `CLOCK_MONOTONIC_RAW`，不与 PC wall time 相减。

## 3. 正确性

三实现 × 三分布 × 三 seeds × 2,560 outputs = **69,120 个元素**，27/27 case row PASS，零 mismatch：

- `quarter_grid_exact`：三实现在 `atol=rtol=0` 下 exact；
- 两组 clipped random：均满足 `atol=0.05,rtol=0.001`；
- 所有输出 finite，激活/权重 buffer 字节不变，输出 shape/contiguity、CPU0–3 亲和性和四个 640-row 分区的无重叠完整覆盖均 PASS。

## 4. A55 计时结果

三实现各取得 421 个完整 52,428,800 B 权重矩阵样本，每实现累计时间均超过 3 s。

| A55 实现 | p50 | IQR/p50 | GOPS | analytic GB/s | shared OPE0 context |
|---|---:|---:|---:|---:|---:|
| scalar FP32 | 33.731833 ms | 0.002750 | 1.554283 | 1.555193 | 0.067048 |
| GCC autovec | 33.730958 ms | 0.003580 | 1.554323 | 1.555234 | 0.067050 |
| explicit NEON | **7.254041 ms** | 0.017530 | **7.227530** | **7.231765** | **0.311779** |

NEON 相对 autovec/scalar 为 **4.649954× / 4.650075×**，分类 `POSITIVE`。OPE0 relative 列仅保留冻结的项目共享 A76-derived ceiling 参考，不是 A55-specific roofline 或通过门。

为遵守三处 source allowlist，原始 `timings.csv` 仍保留了 A76 source schema 中的 legacy 42-call 字段；A55 收据、本报告和结论都不消费它，不产生 A55 42-call RKLLM 组件声明。

## 5. A55/A76 comparator

| candidate 指标 | A55 CPU0–3 | 不可变 A76 CPU4–7 | A55/A76 |
|---|---:|---:|---:|
| p50 | 7,254.041 µs | 2,923.083 µs | **2.481640×** |
| analytic GB/s | 7.231765 | 17.946641 | **0.402959×** |
| cycles/invocation | 45,370,197.924 | 29,466,634.995 | **1.539714×** |
| instructions/invocation | 33,305,109.251 | 30,352,204.901 | descriptive only |
| IPC | 0.734075 | 1.030053 | descriptive only |
| recorded max frequency | 1.800 GHz | 2.352 GHz | A76/A55 = **1.306667×** |

latency 分类为 `A55_SLOWER`。频率比只相邻披露，不用它除 raw latency 后制造“纯微架构得分”。A76 锚来自 owner-waived 单段且与 A55 非并发，因此不声称 paired replication 或环境单旋钮。

## 6. Tier-B、频率与温度

| A55 实现 | cycles/inv | instructions/inv | IPC | L2D refill/inv | bus_access/inv |
|---|---:|---:|---:|---:|---:|
| scalar | 316,716,700.525 | 276,296,922.070 | 0.872379 | 1,033,055.740 | 8,272,366.945 |
| autovec | 316,305,854.820 | 276,296,919.995 | 0.873512 | 1,032,748.240 | 8,270,612.970 |
| NEON | 45,370,197.924 | 33,305,109.251 | 0.734075 | 893,881.260 | 7,153,280.228 |

四事件在三实现上全为 numeric。A55 `bus_access` 仅保留 raw count 与 count/invocation，**不乘 8、不乘 64**，不做 analytic-byte 门；L2D/bus 不做 A55/A76 直接比。

policy0 在所有 timed-window 样本中均为 1.800 GHz / performance，最小频率比 1.0；policy4、policy6 和 NPU governor 全程只读且保持 ondemand/ondemand/`rknpu_ondemand`。退出后 policy0 从 performance 恢复 ondemand。

79 个遥测样本的最大间隔为 1.10 s。timed-window 全局最高温为 54.538°C；最紧的 SoC zone 为 54.538/75°C，headroom **20.462°C**，所有 zone 均未触及最低启用 trip。

## 7. 声明边界

- A55 放置是 synthetic isolated comparator，不证明闭源 W8A8 RKLLM 将该 GEMV 发往 A55 或 CPU。
- A55/A76 行只固定语义、shape、dtype/layout、source lineage、线程数、分区和工作集；不固定频率、微架构、cache/memory service、compiler target 或 OS 调度环境。
- 本结果不解释 OPE4 NPU/runtime gap，不作 RKLLM 组件时间、部署收益、硬件榜、质量、能耗或成本结论。
- 证据只有一个 owner-waived A55 正式段；replication 明确未满足。

## 8. 权威证据与后续

- prelock receipt：`results/operator/board/ope5/ope5_a55_fp16_gemv_prelock_receipt_20260921.json`，SHA-256 `5919fbde…d0b49`。
- source-lock：`results/operator/board/ope5/ope5_a55_fp16_gemv_source_lock_20260921.json`，SHA-256 `40fe76d2…85483`。
- 唯一原始段：`results/operator/board/ope5/ope5_a55_fp16_gemv_segment1_20260921T174008+0800/`。
- 机器收据：`results/operator/board/ope5/ope5_a55_fp16_gemv_receipt_20260921.json`，SHA-256 `cde7f31b…e0544`。

OPE5-⑤ 已闭合。按原 backlog 顺序，下一候选转 OPE5-① decode attention dual-form sequence；其当前 disposition 仍是 `DEFER_DISTINCT_SEQUENCE_CARRIER_REQUIRED`，所以必须重新过四问并先解决 distinct sequence carrier，不自动实现或上板。
