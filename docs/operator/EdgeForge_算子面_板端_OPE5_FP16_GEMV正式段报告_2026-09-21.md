# EdgeForge · 算子面·板端 OPE5-② FP16 GEMV 正式段报告（2026-09-21）

## 1. 结论

OPE5-② 已按冻结合同完成唯一一个四 A76 正式段，终态为：

`VALID_OPE5_02_SINGLE_SEGMENT_OWNER_WAIVER_ISOLATED_4XA76_FP16_GEMV`

独立收据 28/28 checks PASS，69,120/69,120 个元素比较全部通过。`M=1,K=10240,N=2560` 的单次完整矩阵调用 p50 为：scalar `5.836396 ms`、GCC autovec `5.836980 ms`、显式 NEON FP16/FP32-accumulate `2.923083 ms`。NEON 相对 autovec 为 `1.996857×`，按预登记归类为 `POSITIVE`。

`42 × 2.923083 ms = 122.769486 ms`，为 OPE4 C02 `446.126261 ms/token` 的 `27.519000%`，按预登记归类为 `MATERIAL` counterfactual。它是 isolated p50 乘法，不是实测完整 linear population、不是 RKLLM 组件时间，也不能从 TPOT 中相减。

## 2. 执行身份与治理

- scope：`configs/operator/board/ope5_fp16_gemv_scope.json`，SHA-256 `51c6d942…c07068`。
- 冻结预登记：`configs/operator/board/ope5_fp16_gemv_prereg.json`，SHA-256 `b89dc6e0…4fb514`。
- source-lock：`results/operator/board/ope5/ope5_fp16_gemv_source_lock_20260921.json`，SHA-256 `cee44b4f…ad5b`；记录时间早于控制器正式启动，PC/板端四份锁定文件逐 SHA 一致。
- 正式段：`results/operator/board/ope5/ope5_fp16_gemv_segment1_20260921T145232+0800/`。
- 控制器窗口：`2026-09-21T14:52:32.731171628+08:00` 至 `14:53:15.538703367+08:00`，SSH exit 0。
- 板端 RTC 保留为独立 wall-clock 域；所有测量时长使用 `CLOCK_MONOTONIC_RAW`，两种 wall time 没有互减。
- primary 为 CPU4–7 四个 A76 worker，每核固定 640 行；policy4/policy6 在窗口内为 performance，退出后均恢复为 ondemand。
- NPU governor 全程保持 `rknpu_ondemand`，未执行写操作，也没有 NPU/RKLLM 进程。
- owner 指令为“不需要复测，取消同日复测，只测一次”。本阶段只有一个正式段，无自动重跑、无同日复测，replication=`WAIVED_BY_OWNER_NOT_SATISFIED`。
- 没有 performance smoke、缩小 shape timing、A55/NPU/gate-up/prefill/model/quality lane，也没有 driver、kernel 或共享库改动。

板端 compile/disassembly-only 审查最初只因工作区缺少 `tmp/` 父目录而在编译器启动前退出；随后创建 `/home/orangepi/edgeforge_ope_r1/tmp`，完成相同的纯编译审查。该过程 benchmark execution count 为 0，且发生在 source-lock 与唯一正式段之前，不构成额外测量段。

## 3. 正确性与 codegen

独立 FP64 scalar oracle 覆盖三实现 × 三分布 × 三 seed × 2,560 输出，共 27 个 case row、69,120 个元素：

| 实现 | 元素比较 | PASS case | mismatch | max abs error | max rel error |
|---|---:|---:|---:|---:|---:|
| scalar_fp32 | 23,040 | 9/9 | 0 | 0.0174730 | 0.0793261 |
| autovec_o3 | 23,040 | 9/9 | 0 | 0.0174730 | 0.0793261 |
| neon_fp16_f32acc | 23,040 | 9/9 | 0 | 0.00184857 | 0.0387507 |

quarter-grid exact corpus 在 `atol=rtol=0` 下逐元素 exact；两个随机 corpus 均满足冻结的 `atol=0.05,rtol=0.001` 联合判据。冻结门使用绝对/相对联合容差，单列 max relative error 不独立判失败；实际 mismatch 为 0。全部输出有限，输入/权重逐字节未变，输出形状、四核亲和性及 `[0,2560)` 分区覆盖均通过。

反汇编门也通过：scalar inner kernel 的 vector `fmla` 为 0；candidate 检出 16 个 `fcvtl/fcvtl2` 与 8 个 vector `fmla`。GCC autovec 未生成 vector FMA；冻结协议已明确它仍是合法 compiler baseline，不把“必须自动向量化”当成通过条件。

## 4. Tier-A 计时

三个实现共享同一 worker pool、亲和性、分区与同步路径；warmup 50，随后使用六种排列循环交错。各实现最终得到 1,028 个完整 52.4 MB 权重遍历样本，分别满足 ≥200 次和累计 ≥3 秒。

| 实现 | p25 / p50 / p75 (ms) | IQR/p50 | GOPS | analytic GB/s | OPE0 相对分 |
|---|---:|---:|---:|---:|---:|
| scalar_fp32 | 5.820718 / **5.836396** / 5.846459 | 0.004410 | 8.983079 | 8.988342 | 0.387509 |
| autovec_o3 | 5.821958 / **5.836980** / 5.847333 | 0.004347 | 8.982180 | 8.987443 | 0.387470 |
| neon_fp16_f32acc | 2.915135 / **2.923083** / 2.928042 | 0.004415 | 17.936131 | 17.946641 | 0.773723 |

三路 IQR/p50 均远低于 0.30。NEON 相对 scalar 为 `1.996657×`、相对 autovec 为 `1.996857×`。单次解析帐为 52,459,520 B、52,428,800 ops；candidate p50 是 OPE0 compulsory-byte floor `2.261656 ms` 的约 `1.2925×`，解析带宽达到 OPE0 天花板的 `77.3723%`。

## 5. 同 shape 锚与 42-call 反事实

| lineage | 合同 | p50 | 解析字节 |
|---|---|---:|---:|
| OPE3 A76 int8 | 四 A76、row-major、`neon_sdot` | 0.944708 ms | 26,234,880 B |
| OPE5-② A76 FP16 | 四 A76、row-major、FP16 load + FP32 FMA | 2.923083 ms | 52,459,520 B |
| OPE2 NPU FP16 | `rknn_matmul_run` core0/normal | 4.728208 ms | 52,459,520 B |

OPE5-② candidate/OPE3 int8 延迟比为 `3.094166×`，同时 FP16/int8 解析字节比为 `1.999610×`。两行虽然同几何、同四 A76 拓扑和 row-major 布局，仍同时改变 dtype、指令路径和实现；该比值不能命名为纯 dtype 因果。OPE2 行使用不同硬件与 runtime，只是复用上下文，不构成硬件榜。

candidate 的 42-call 反事实为 `122.769486 ms`，占 OPE4 TPOT `27.519000%`，分类 `MATERIAL`。该数假定同一单矩阵 p50 重复 42 次；没有测 42 份不同层权重、完整 linear population 或闭源 RKLLM 调度，所以只说明“若同等 FP16 CPU 工作存在，其 isolated 量级不可忽略”，不证明这类工作实际出现在 W8A8 carrier 中。

## 6. Tier-B

每个实现只在同一正式段内采一次 `cycles:u,instructions:u,l2d_cache_refill:u,bus_access:u`，各运行至少 3 秒；四事件全部 numeric。raw total 因 invocation 数不同不直接横比，以下使用每调用归一值：

| 实现 | invocations | cycles/call | instructions/call | IPC | L2D refill/call | bus_access/call | `bus_access×8 / analytic` |
|---|---:|---:|---:|---:|---:|---:|---:|
| scalar_fp32 | 511 | 60,836,636 | 235,811,846 | 3.8761 | 57,522 | 7,159,493 | 1.09181 |
| autovec_o3 | 512 | 60,945,445 | 235,760,918 | 3.8684 | 57,707 | 7,154,301 | 1.09102 |
| neon_fp16_f32acc | 1,019 | 29,466,635 | 30,352,205 | 1.0301 | 12,699 | 6,850,613 | 1.04471 |

`bus_access×8` 只保留为预登记的 transferred-beat hypothesis diagnostic。它不是解析字节 acceptance gate；OPE3 已否决的 `bus_access×64` 没有重引入、回写或事后替换门限。

## 7. 遥测与恢复

- 1 Hz 遥测共 36 个样本，其中正式计时/Tier-B 窗口 26 个；最大间隔 1.12 秒。
- policy4 与 policy6 在全部 timed-window 样本均为 2,352,000/2,352,000 kHz，最低频率比均为 1.0。
- timed-window 全局最高温为 bigcore zone 的 80.384°C，其最低启用 trip 为 115°C。
- 最小逐 zone thermal headroom 来自 SoC zone：最高 73.923°C、最低 trip 75°C，余量 1.077°C；未触门。
- governor 轨迹为 `ondemand/ondemand → performance/performance → ondemand/ondemand`；NPU 始终为 `rknpu_ondemand`。
- build、correctness、timing、三个 perf 进程与 governor restore 均 exit 0。

## 8. 裁决与边界

本项补齐了 backlog 中缺失的“四 A76、真实 down-projection 几何、FP16 storage/FP32 accumulate”隔离行。显式 NEON 相对未向量化的 GCC baseline 接近 2×，candidate 分类为 positive；相对 OPE3 int8，同 shape 延迟约 3.09×，而解析字节约 2.00×。

这些结果没有证明闭源 W8A8 RKLLM 将 FP16 down-projection 派发到 CPU，也不解释 OPE4 的 160.471% analytic-vs-stage gap。不得据此声称部署收益、质量变化、CPU/NPU 胜负或可相减的 TPOT 组件。

OPE5-② 至此闭合。按冻结 backlog，下一未完成候选是 OPE5-⑤ A55 cluster comparator；它必须先基于已经闭合的 A76 primary 重新完成四问、确定比较对象与独立预登记，不自动复制本段到 A55，也不自动启动板测。
