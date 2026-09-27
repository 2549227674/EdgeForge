# EdgeForge · 算子面 lm_head 执行裁决 · R1

- 执行日期：2026-09-16（UTC+8）
- 正式载体：C02-S.O + bitsandbytes NF4 double quant，RTX 4060 Laptop / sm_89 / WSL2
- 最终状态：`LM_HEAD_C02_LOCKSTEP_RECONNECTION_PASS_LOCAL_PATH_AUTHORIZED_AFTER_CLOCK_GATE_ADJUDICATION`
- OP5 总状态：`OP5_CURRENT_BACKLOG_CLOSED_NO_REMAINING_EXECUTABLE_CANDIDATE`
- SHA 策略：按 owner 指示，自本阶段起不计算、不比对 SHA；所有新证据显式写为 `SKIPPED_BY_OWNER_DIRECTION`

## 1. 最终裁决

post-K3 剩余 backlog 审计确认：Inductor 已作为 K2/K3 supporting baseline 覆盖，不能独立通过 Authenticity；KV 反量化在当前 BF16 KV 载体上仍无真实 dequant op，保持 `BLOCKED_AUTHENTICITY`。唯一可继续验证的项目是原处于载体可交付性 HOLD 的 `lm_head 262k vocab + softcap`。

lm_head 的 bounded deliverability、正式 A/B/C/D 选择和全人口 C02 lockstep 回连均已完成。最终授权的是一个**外置、未注册的 BF16 CUDA weight mirror + eager `F.linear` + 原 softcap + argmax**路径，不是自定义 kernel：

- 原 CPU `lm_head.weight` 与 `embed_tokens.weight` 始终保持同一 storage，Parameter、state_dict、module、NF4 topology 和 device map 均未改变。
- 全量 5 盘、11 请求、21,705 个真实 token 决策中，CPU tied head 与 CUDA mirror 为 `21,705/21,705 exact`。
- 配对反事实 aggregate incremental TPOT 为 `143.354129 → 119.798383 ms`，A/B speedup `1.196628×`，通过冻结的 `≥1.05×` 门。
- peak CUDA allocated `6,364,221,440` bytes，低于 7.5 GiB；11/11 cache、3,333 条 telemetry、结构和热学门均有效。
- 不开发 custom lm_head kernel：选择探针的最佳 eager CUDA Tier-A relative score 为 `1.037999`，高于预注册 `≤0.80` 的 Bottleneck PASS 上限，表明现有路径已接近同日带宽尺子。

授权边界只适用于冻结的本地 C02 PyTorch 路径。不得表述为跨日/独立复现、板端、llama.cpp、sm_100 或 sm_120 收益。

## 2. bounded deliverability

bounded probe 在最长 prompt（5,142 tokens）的 live cache 存续期间分配一份 `1,342,177,280`-byte BF16 CUDA mirror，并保持原 tied CPU storage 不变。结果状态为 `LM_HEAD_DELIVERABILITY_HOLD_LIFTED_SELECTION_AUTHORIZED`：

| 项目 | 结果 |
|---|---:|
| 原始 tied CPU head/embedding | identity 与结构前后不变 |
| CPU/CUDA token | 同为 `35434` |
| softcapped logit max abs | `0.0625` |
| CPU head p50 | `42.401599 ms` |
| CUDA head p50 | `6.047262 ms` |
| 诊断 speedup | `7.0117×` |
| live peak CUDA allocated | `3,727,292,416` bytes |
| 投影 R5 peak + mirror | `6,365,310,976` bytes `< 7.5 GiB` |

该 probe 只解除载体 HOLD，不构成正式选择或部署授权。

## 3. 正式选择

正式选择覆盖 11 个 prompt prefill 与其同加载首个 decode，共 22 个真实 hidden state。候选范围在开跑前冻结为现有 CPU tied head、CUDA eager/compile softcap 路径，以及仅作诊断的 raw-argmax comparator；没有实现或计时 custom candidate。

首次执行在 22 个状态采集完成后因 prompt-dependent leading stride 触发 `torch.compile(fullgraph=True)` recompile limit，按 harness-only `EXECUTION_FAILURE` 保留。重试前将尺寸为 1 的 leading dimensions 通过 `clone(memory_format=torch.contiguous_format)` 规范化；样本、阈值、候选和决策门均未改变。

成功选择结果：

| 项目 | 结果 |
|---|---:|
| CUDA eager softcap exact | `22/22` |
| CUDA compile softcap exact | `22/22` |
| CPU p50 | `30.983353 ms` |
| CUDA eager p50 | `5.345280 ms` |
| CUDA compile p50 | `17.137153 ms` |
| eager speedup vs CPU | `5.796395×` |
| 同日 triad ceiling | `241.904639 GB/s` |
| eager measured bandwidth | `251.096745 GB/s` |
| Tier-A relative score | `1.037999` |
| peak CUDA allocated | `5,023,468,544` bytes |

因此 custom-kernel Bottleneck 门为 FAIL；现有 eager CUDA mirror 获得独立 C02 回连授权，compile 和 raw-softcap-elision 均不进入部署路径。

## 4. 全人口 lockstep 回连

为避免重复执行两遍完全相同的 42 层 Transformer，正式协议使用逐 token 同隐藏态 lockstep A/B：每个真实 hidden state 同时执行 A（CPU tied head）和 B（CUDA mirror），CPU-first/CUDA-first 按全局 token 奇偶交替。如果 token 不同，立即记录首分歧并 fail-close；只要此前 token 一直相同，下一输入与 Transformer/cache 轨迹就相同，因此全 21,705 个决策一致可由归纳推出独立 greedy 序列一致。

最终没有触发 fail-fast：

| 门 | 结果 |
|---|---:|
| 请求 / 完整预算 | `11/11` |
| A/B head calls | `21,705 / 21,705` |
| token exact | `21,705/21,705` |
| cache 结构 | `11/11 PASS` |
| CPU head p50 | `30.318492 ms` |
| CUDA head p50 | `7.145948 ms` |
| A aggregate incremental TPOT | `143.354129 ms` |
| B aggregate incremental TPOT | `119.798383 ms` |
| A/B speedup | `1.196628×` |
| peak CUDA allocated | `6,364,221,440` bytes |
| max NVML device-used | `8,023,179,264` bytes |
| telemetry | 3,333 samples；最高 70°C；无 thermal/hardware throttle |

TPOT 是每一步共同 Transformer 时间加对应 head 时间得到的配对反事实，不是两次独立 wall-clock replay。它直接在同一 hidden state、同一热态和交错顺序下隔离 lm_head 替换的增量；不构成 repeatability 证据。

## 5. memory-clock 门重判

冻结 runner 最初把 isolated selection 的 `memory clock ≥0.95 × 8,000 MHz` 复制为回连硬门。原始结果因此在其余门全部 PASS 时写为 `LM_HEAD_C02_RECONNECTION_INVALID_NO_DECISION`：lockstep p50 为 6,000 MHz，P-state 主要为 P4。

该门与协议本身不相容。lockstep 每 token 故意执行约 30 ms CPU head，制造 GPU idle gap；项目早于本次结果冻结的 OP3 规则已经明确：`hybrid CPU lm_head creates expected GPU-idle gaps unlike the OP0 continuous bandwidth loop`，memory clock 在该情形只报告、不作硬门。证据也支持这是预期降频而非硬件异常：

- 正式 CUDA head p50 / isolated 8,000 MHz head p50 = `1.336871×`；8,000/6,000 = `1.333333×`，几乎完全按显存频率缩放。
- 低频实测已经包含在 `1.196628×` 整模收益内，因此重判不会夸大速度，反而更保守。
- 无 thermal/hardware throttle，最高 70°C；所有 token、cache、结构和显存门独立 PASS。

原始 INVALID result 与 receipt 保持不改写；单独的 post-run adjudication 将 `memory_clock_gate` 标为 `NOT_APPLICABLE_PASS_HYBRID_LOCKSTEP_GPU_IDLE_GAPS`。这是结果后适用性修正，必须与原始 INVALID 一并披露，不能说成预注册硬门原样 PASS。

## 6. 显存与使用约束

formal peak CUDA allocated 比 7.5 GiB 仍有约 1.57 GiB 余量，但 NVML max device-used 距 7.5 GiB 只剩 `29,884,416` bytes。两者口径不同，后者反映驱动与非 allocator 占用；因此本地授权附带两个强约束：

1. 仅在独占 GPU 或确认无 co-resident workload 时建立 mirror，并在建立前做 runtime free-memory guard。
2. mirror 必须保持外置、未注册；不得移动或替换原 CPU tied Parameter，不得改变 embedding tie 或 state_dict。

若不满足这两个条件，应保留 CPU head，而不是把本裁决外推为无条件默认配置。

## 7. 证据索引

以下新证据均按 owner 指示不做 SHA 核验：

| 证据 | 结论 |
|---|---|
| `configs/operator/op5_lm_head_deliverability_probe_r1.json` | bounded deliverability 冻结协议 |
| `results/operator/pool/op5_lm_head_deliverability_probe_r1_receipt_20260916.json` | HOLD lifted，正式选择获授权 |
| `configs/operator/op5_lm_head_selection_r1.json` | A/B/C/D 选择协议 |
| `results/operator/failures/op5_lm_head_selection_r1_2026-09-16T205933_0800.json` | 首轮 stride/recompile harness failure |
| `results/operator/pool/op5_lm_head_selection_r1_receipt_20260916.json` | existing eager CUDA mirror 获回连授权；custom 不获授权 |
| `configs/operator/op5_lm_head_reconnect_r1.json` | 全人口 lockstep 回连协议 |
| `results/operator/pool/op5_lm_head_reconnect_r1_20260916.json` | 原始全量结果；仅 memory-clock 硬门导致 INVALID |
| `results/operator/pool/op5_lm_head_reconnect_r1_adjudication_receipt_20260916.json` | clock gate 适用性重判；本地路径最终授权 |
| `results/operator/pool/op5_current_backlog_close_receipt_20260916.json` | OP5 当前 backlog 全部处置完成 |

## 8. 终态边界

OP5 当前 backlog 已关闭：K2 与 K3 均终态关闭，Inductor 仅作 supporting baseline，lm_head 获得上述本地路径授权且 custom kernel 关闭，KV dequant 保持 Authenticity BLOCK。当前没有剩余可执行 candidate。

重开 OP5 需要新的 census/载体事实与新预注册；例如只有观测到真实量化 KV dequant op 后，KV 项才可重新通过 Authenticity。OP-E 仍为 `BLOCKED_NO_EXECUTION`，不因本裁决自动开启。
