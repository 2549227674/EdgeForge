# EdgeForge · 算子面·板端 OPE5-③ RMSNorm 正式段报告（2026-09-21）

## 1. 结论

OPE5-③ 已按冻结协议完成唯一一个 A76 正式段，终态为：

`VALID_OPE5_SINGLE_SEGMENT_OWNER_WAIVER_ISOLATED_A76_RMSNORM`

独立收据 23/23 checks PASS，随机化正确性 144/144 PASS。完整 302-call decode population 的 p50 为：scalar `1.716458 ms`、`-O3` autovec `1.721125 ms`、NEON FP16 `0.732084 ms`。NEON 相对 autovec 为 `2.350994×`，按预登记归类为 `POSITIVE`。

NEON 的 `0.732084 ms` 是单核 A76 上对 census population 的隔离投影，仅为 OPE4 C02 `446.126261 ms/token` 的 `0.164098%`，归类 `SUPPORTS_MINOR_CPU_RMSNORM_INSTRUMENT_BOUND`。它不是 RKLLM 内部 RMSNorm 时间，不能从 TPOT 中相减，也不构成模型提速声明。

## 2. 执行身份与治理

- 冻结预登记：`configs/operator/board/ope5_rmsnorm_prereg.json`，SHA-256 `971f9720…a4cb224`。
- source-lock：`results/operator/board/ope5/ope5_rmsnorm_source_lock_20260921.json`，在控制器正式启动前写入；三份实现和遥测依赖的 PC/板端 SHA 全部一致。
- 正式段：`results/operator/board/ope5/ope5_rmsnorm_segment1_20260921T140743+0800/`。
- 控制器窗口：`2026-09-21T14:07:43.365958442+08:00` 至 `14:08:20.187662043+08:00`，SSH exit 0。
- 板端 RTC 仍处于独立且不可信的 wall-clock 域；板端保留 wall time，但所有时长使用 `CLOCK_MONOTONIC_RAW`。控制器与板端 wall time 未相减。
- primary 为 CPU7 / A76 单线程；policy4、policy6 在窗口内设为 performance，退出后均从 performance 精确恢复到 ondemand。
- NPU governor 全程保持 `rknpu_ondemand`，未执行写操作。
- owner 已取消同日复测；本阶段只有一个正式段，replication 明确记为 `WAIVED_BY_OWNER_NOT_SATISFIED`。
- 没有 A55 lane、NPU kernel/model run、RKLLM run、3-core、质量评测、driver/kernel/共享库改动，也没有独立性能 smoke。

## 3. 正确性

八个冻结语义 case × 三个输入分布 × 三个 seed × 两个非 oracle 实现，共 144 次 tensor comparison：

| 实现 | comparison | PASS | mismatch elements | max abs error | max rel error |
|---|---:|---:|---:|---:|---:|
| autovec_o3 | 72 | 72 | 0 | 0 | 0 |
| neon_fp16 | 72 | 72 | 0 | 0.125 | 0.000952381 |

所有输出有限；输入与 scale buffer 均逐字节未变；两个 scale-free `v_norm` case 均以 null scale pointer 执行。NEON 最大绝对误差发生在大幅值输出上，仍满足冻结的 `atol=0.015625, rtol=0.01` 双容差。

## 4. Tier-A 计时

population 使用八 case 的 `211/35/7/20/20/4/4/1` 调用频次，合计 302 calls、3,840,000 analytic B/forward；16 个独立 frame 轮转，对应 61,440,000 B 解析工作集。三实现按六种排列循环交错，最终各 4,046 个 population sample，最快实现累计计时不少于 3 秒。

| 实现 | p25 / p50 / p75 (ms) | IQR/p50 | analytic GB/s | OPE0 相对分 | 占 OPE4 TPOT |
|---|---:|---:|---:|---:|---:|
| scalar_ref | 1.713250 / **1.716458** / 1.721708 | 0.004928 | 2.237165 | 0.096450 | 0.384747% |
| autovec_o3 | 1.717625 / **1.721125** / 1.728125 | 0.006101 | 2.231099 | 0.096188 | 0.385793% |
| neon_fp16 | 0.727416 / **0.732084** / 0.738500 | 0.015140 | 5.245300 | 0.226137 | **0.164098%** |

全部 primary population 的 IQR/p50 远低于 0.30 稳定性门。NEON 相对 scalar 为 `2.344619×`，相对 autovec 为 `2.350994×`。

八个 per-case 的 autovec/NEON p50 如下；这些微秒级行受计时分辨率影响，primary 结论仍以完整 population 为准。

| case | calls/forward | autovec p50 (µs) | NEON p50 (µs) | speedup |
|---|---:|---:|---:|---:|
| decoder_hidden_scaled | 211 | 6.125 | 2.333 | 2.625× |
| q_norm_sliding_scaled | 35 | 4.959 | 2.041 | 2.430× |
| q_norm_full_scaled | 7 | 10.208 | 3.792 | 2.692× |
| k_norm_sliding_scaled | 20 | 1.167 | 0.583 | 2.002× |
| v_norm_sliding_scale_free | 20 | 1.166 | 0.292 | 3.993× |
| k_norm_full_scaled | 4 | 2.625 | 0.875 | 3.000× |
| v_norm_full_scale_free | 4 | 2.042 | 0.875 | 2.334× |
| ple_projection_norm_scaled | 1 | 26.541 | 9.917 | 2.676× |

解析 GB/s 只是 `analytic bytes / p50`。RMSNorm 含 FP32 reduction 与 reciprocal square root，故低于 OPE0 STREAM 天花板不代表实现缺陷，也不能据此证明 DRAM bottleneck。

## 5. Tier-B 与 `bus_access` 纪律

每个实现只对完整 streaming population 采一次 `cycles:u,instructions:u,l2d_cache_refill:u,bus_access:u`；四事件均为 numeric。不同实现按最短 3 秒窗口运行，因此 raw total 不直接横比，收据同时给出 per-sequence 归一值。

| 实现 | sequences | cycles | instructions | IPC | L2D refill | bus_access | `bus_access×8 / analytic bytes` |
|---|---:|---:|---:|---:|---:|---:|---:|
| scalar_ref | 1,742 | 7,779,101,146 | 20,927,255,721 | 2.6902 | 12,426,336 | 869,293,232 | 1.03963 |
| autovec_o3 | 1,652 | 7,779,641,681 | 19,938,144,135 | 2.5629 | 11,522,643 | 563,207,851 | 0.71026 |
| neon_fp16 | 4,340 | 7,682,310,618 | 9,750,994,954 | 1.2693 | 28,868,369 | 1,417,489,745 | 0.68044 |

`bus_access×8` 仅保留为测前声明的 beat-size hypothesis diagnostic。它不承担 PASS 门，也不证明解析字节闭合；OPE3 已否决的 `bus_access×64` 没有重引入、回写或归一化。

## 6. 遥测与恢复

- 1 Hz 遥测 31 个样本，其中正式计时/Tier-B 窗口 27 个；最大采样间隔 1.07 秒。
- CPU7 所属 policy6 在全部 timed-window 样本均为 2,352,000/2,352,000 kHz，最低频率比 1.0。
- timed-window 最高温度 51.769°C；最低启用 trip 为 75°C，最小 thermal headroom 25.077°C。
- governor 轨迹为 `ondemand/ondemand → performance/performance → ondemand/ondemand`；NPU 始终为 `rknpu_ondemand`。
- build、correctness、timing、三个 perf 进程及 governor restore 均 exit 0。

## 7. 裁决与边界

本项回答的是：若把 census 的完整 302-call FP16 RMSNorm population 放到单核 A76 上执行，一个明确实现的隔离成本量级是多少。结果显示 NEON 仪器约 `0.732 ms`，对 OPE4 的 `446.126 ms/token` 只占 `0.164%`，支持将其作为较小 CPU 侧候选排除，而不是 OPE4 大 gap 的主要解释。

本项没有证明 RKLLM 在 CPU 上执行同样的 RMSNorm 数量、dtype、布局或调度；因此不得写成“RKLLM RMSNorm 实测 0.732 ms”，不得外推模型收益，也不得与 PC K1 的 sm_89/BF16 数字作硬件胜负比较。

OPE5-③ 至此完成。按已冻结 backlog，下一候选为 OPE5-② NEON FP16 GEMV；应先用本次结果和 OPE2/OPE4 证据重新过四问、冻结单独预登记，不自动启动下一次板测。

