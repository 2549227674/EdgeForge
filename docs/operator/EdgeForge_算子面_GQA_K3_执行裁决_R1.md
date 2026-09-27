# EdgeForge · 算子面 GQA K3 执行裁决 · R1

- 执行日期：2026-09-16（UTC+8）
- 正式载体：RTX 4060 Laptop / sm_89 / WSL2，Torch 2.13.0+cu130，Triton 3.7.1
- 目标人口：Gemma4 decode GQA，8Q/2KV；SWA `K=512,D=256` 与 full `K=807..7359,D=512`
- 冻结协议：`configs/operator/k3_gqa_decode_attention_prereg.json`
- 协议 SHA-256：`fe427c9650aa929d18f72daaaa22ed918b7d33fd5b0af4da12386e0a86229626`
- 当前状态：`K3_C02_RECONNECTION_FAIL_CLOSE_DEPLOYMENT_PATH`

## 1. 裁决

GQA decode attention 已完成独立选择探针、候选预注册、锁源、正式正确性、首个有效日历日的隔离性能门、ncu 归因与 C02 整模回连。K3 isolated correctness **PASS**，Day-1 冻结性能门也为 **POSITIVE**：candidate 对逐 case 最优正确性合格基线的 invocation-weighted speedup 为 `1.275002×`，SWA 与 full 两族分别为 `1.116531×`、`1.714064×`。

原 R1 结论在 Day-1 后为暂定正向。随后 owner 在已见结果后明确取消复测并取消 SHA 核验，授权直接执行 ncu；该 adaptive override 不追溯改写 R1，也不能被称为跨日或独立复现。R3 ncu PASS 曾授权 C02 回连，但 R5 部署等价门最终为 0/11 请求 token exact，且端到端 decode TPOT 变慢 2.260164%。因此 isolated 与 ncu 证据只保留为 bounded diagnostics，K3 部署路径现已终态关闭。

## 2. 选择探针

OP1 census 对账得到 911,148 次 decode attention 调用：SWA `K=512,D=256` 为 759,290 次（83.3%）；full `D=512,K=807..7359` 为 151,858 次。full 长度按调用数冻结为 12 个等量分箱，连同 SWA 共 13 个性能 case。

四条既有路径在 13 case × 3 seeds 上先做正确性：

| 路径 | 对冻结 eager 的结果 | 是否可作性能基线 |
|---|---:|---|
| Transformers eager | reference | 是 |
| fullgraph compile eager | 39/39 allclose，最大绝对误差 0 | 是 |
| native SDPA GQA | 1/39 allclose，最大绝对误差 0.3974609375 | 否 |
| compiled SDPA GQA | 1/39 allclose，最大绝对误差 0.3974609375 | 否 |

正确性合格的最优现有基线加权 p50 为 272.178712 µs，Tier-A 相对分仅 `0.048083`，SWA/full 分别为 `0.021455`/`0.092254`，因此 Authenticity 与 Bottleneck 均 PASS。SDPA 的 peaked-softmax 数值路径不满足冻结 eager 合同，不能用作部署捷径；选择结论为 `CUSTOM_GQA_DESIGN_PREREG_AUTHORIZED`。

选择 runner 前两次中断均保留为 harness-only 证据：第一次在任何 attention baseline 执行前读取 inference tensor version counter；第二次在正确性阶段复用被 CUDAGraph 覆盖的输出。修复只涉及输入创建位置与正确性输出 clone，未改变协议、正式计时路径或 candidate 状态；成功执行为 harness attempt 3。

## 3. 锁定候选与正确性

候选在正式留出集揭示前锁定为：

`3d3d57ac20e390e9f117ac763af7d5ffcfabe084fce861461e8e62e183ac76f0`

实现由三段 Triton kernel 构成：

1. 直接按 `query_head // 4` 读取两个 KV head，计算 BF16 QK、FP32 accumulation，并保留 eager 的 BF16 carrier 舍入；不物化 2→8 KV repeat。
2. 对每个 query head 做 FP32 softmax，再显式舍入为 BF16 probability。
3. 直接对两个 KV head 做 PV，FP32 accumulation 后输出连续 BF16 `[1,1,8,D]`。

正式 correctness 在锁源后首次执行：

| 覆盖 | 结果 |
|---|---:|
| 冻结全长度点 | 26/26 PASS |
| 4 个代表 case × 3 分布 × 3 新鲜 seeds | 36/36 PASS |
| 4 个代表 case × 3 additive-mask 情形 | 12/12 PASS |
| GQA 头映射微例 | PASS |
| fallback 拒绝矩阵 | 13/13 exact eager fallback |
| source tensor 内容、指针与版本不变 | PASS |
| 冻结 Triton 搜索空间 | PASS |

总计 75 个 candidate trials 全部满足 `atol=0.03125, rtol=0.02`；其中 64 个逐元素完全一致，最大绝对误差 `0.00390625`。正式正确性轮没有 candidate 或 harness failure，也没有启动性能计时。

## 4. Day-1 性能门

正式窗口对 13 个冻结 case 分别执行 current eager / fullgraph compile eager / candidate 三路各 50 次预热和 200 次轮转测量。13/13 case 的 best correctness-eligible baseline 均为 compile eager。

有效性门：

- 同日 512 MiB float32 triad ceiling：241.904639 GB/s；
- memory clock p50：8000 MHz，OP0 比率 1.000，阈值 0.95；
- 温度 63–67°C；无 software thermal、hardware thermal 或 hardware power-brake throttle；
- 协议与 candidate SHA 从启动到 receipt 写入保持不变。

核心结果：

| 人口 | best baseline p50 | candidate p50 | candidate speedup |
|---|---:|---:|---:|
| 全部 911,148 次调用加权 | 273.188497 µs | 214.265078 µs | `1.275002×` |
| SWA，759,290 次 | 210.943997 µs | 188.928001 µs | `1.116531×` |
| full，151,858 次 | 584.410994 µs | 340.950464 µs | `1.714064×` |

冻结继续门要求 aggregate ≥1.05×，且任一 family 不得低于 0.95×；两项均 PASS。因此原始 R1 结果为 `K3_PROVISIONAL_DAY1_POSITIVE`。后续复测门如何被 owner 覆盖及 ncu 结果见 §6。

## 5. 证据索引

| 证据 | SHA-256 | 结论 |
|---|---|---|
| `configs/operator/op5_gqa_decode_selection_probe.json` | `121ae0b4b4ad6858f1b497c1f6895073823519f17fe7ead93db94703f3d2617d` | GQA 选择协议，测前冻结 |
| `results/operator/pool/op5_gqa_prereg_receipt_20260916.json` | `6fd79a4fcaa47bd652b3e7f8b2174827ff348851e22e027d4ebd13f955c74c8f` | 选择协议预注册 PASS |
| `results/operator/failures/op5_gqa_selection_failure_20260916T170832.json` | `c3b542ecfa12e414e20d05afacf13d8ccb2656283f2a8499dbd112bb40809fe9` | harness-only inference-version 中断 |
| `results/operator/failures/op5_gqa_selection_failure_20260916T170952.json` | `091548bdb95f95186dc492beebe166e44fbb65abb8201e41021c08dc2d0c05a4` | harness-only CUDAGraph output-lifetime 中断 |
| `scripts/operator/run_op5_gqa_selection.py` | `9869b85b405527b408f33b3867f843b451573a0d3e6dc56de385fd7808914f17` | 成功的选择 runner |
| `results/operator/pool/op5_gqa_selection_20260916.json` | `0b83a210be8a061ce7572e9a71fc15d3814e7bcf050b09836826cdf41717c12d` | 选择明细；custom design authorized |
| `results/operator/pool/op5_gqa_selection_receipt_20260916.json` | `9a5f231cedbe128973cad0c2ab09b888da9fff1eb4e5cb1e9168817f64c705bc` | `PASS_CUSTOM_DESIGN_AUTHORIZED` |
| `configs/operator/k3_gqa_decode_attention_prereg.json` | `fe427c9650aa929d18f72daaaa22ed918b7d33fd5b0af4da12386e0a86229626` | K3 算法、留出集和 bounded gate 冻结协议 |
| `results/operator/pool/k3_gqa_prereg_receipt_20260916.json` | `7516c79668bffb164101d3a80af2c2270dd69ff7a17b598cda57fd1193a57ebf` | K3 预注册 PASS |
| `scripts/operator/k3_gqa_attention.py` | `3d3d57ac20e390e9f117ac763af7d5ffcfabe084fce861461e8e62e183ac76f0` | 唯一锁定 candidate |
| `results/operator/pool/k3_gqa_candidate_lock_receipt_20260916.json` | `261e6b443bb9adfb36403edc49d2328f6ac2426dc4af967c417794884ca8eb39` | 正式 holdout 前锁源 |
| `scripts/operator/run_k3_correctness.py` | `75aee702b7431516ff3e394616b695cd7231ce43c7410cde7ffeb4fc59e50d25` | 正式 correctness runner |
| `results/operator/pool/k3_gqa_correctness_20260916.json` | `8ba0bca4ee44af54d7da368b72a61833adaba0a0ebcfbe8fa700d376129df514` | 75-trial correctness 明细 PASS |
| `results/operator/pool/k3_gqa_correctness_receipt_20260916.json` | `ac8e9834ccfd32d731ef6374c962989486b3779caa9fa90e848a1392b8f7cf32` | K3 correctness PASS |
| `scripts/operator/run_k3_day1.py` | `0d26582d7ec5bdbd6477f81198d04d7178de2ba08b573eb339e3df79eb933ed1` | 冻结 Day-1 runner |
| `results/operator/pool/k3_gqa_day1_20260916.json` | `17fead290b82258c49a530424bb907999b6b1665a4819b542c2cd084ba208ef3` | 有效正向 Day-1 全量结果 |
| `results/operator/pool/k3_gqa_day1_receipt_20260916.json` | `06a39a84261592896dbf298702a7f376796e17658ad759b9654d696830f00b1f` | `K3_PROVISIONAL_DAY1_POSITIVE` |
| `configs/operator/k3_gqa_replication_override_r2.json` | 未核验（owner 取消） | adaptive 同日复测规则；未执行 |
| `results/operator/pool/k3_gqa_replication_override_r2_receipt_20260916.json` | 未核验（owner 取消） | `SUPERSEDED_UNEXECUTED_BY_OWNER` |
| `configs/operator/k3_gqa_ncu_override_r3.json` | 未核验（owner 取消） | 跳过复测并直接 ncu 的授权边界 |
| `scripts/operator/run_k3_ncu_r3.py` | 未核验（owner 取消） | 三 case / 九 kernel ncu runner |
| `results/operator/pool/k3_gqa_ncu_r3_20260916.json` | 未核验（owner 取消） | ncu 归因 PASS |
| `results/operator/pool/k3_gqa_ncu_r3_receipt_20260916.json` | 未核验（owner 取消） | 回连获得授权（历史中间状态） |
| `configs/operator/k3_gqa_reconnect_override_r4.json` | 未核验（owner 取消） | R4 正式协议；执行后判 invalid |
| `results/operator/failures/k3_gqa_reconnect_r4_2026-09-16T181918_0800.json` | 未核验（owner 取消） | `candidate_calls=0`，不作部署裁决 |
| `configs/operator/k3_gqa_reconnect_override_r5.json` | 未核验（owner 取消） | same-load A/B、None-mask 与长度桶规则 |
| `scripts/operator/run_k3_reconnect_r5.py` | 未核验（owner 取消） | 5 盘/11 请求、A/B 22-run 正式 runner |
| `results/operator/pool/k3_gqa_reconnect_r5_20260916.json` | 未核验（owner 取消） | R5 正式明细；token exact 0/11 |
| `results/operator/pool/k3_gqa_reconnect_r5_receipt_20260916.json` | 未核验（owner 取消） | `K3_C02_RECONNECTION_FAIL_CLOSE_DEPLOYMENT_PATH` |
| `results/operator/pool/k3_gqa_reconnect_r5_adjudication_receipt_20260916.json` | 未核验（owner 取消） | cache validator 更正；终态不变 |

## 6. Owner 覆盖与 ncu

owner 在正向 Day-1 后依次作出两项显式覆盖：先取消 SHA 计算/比对并允许同日独立复测，随后在复测启动前再次确认取消复测、直接进入 ncu。R2 measurement 从未启动，也没有复测 result；因此证据口径始终是“单日正向 + replication waiver”，不是“两日一致性 PASS”。

R3 对 SWA `K=512,D=256`、full `K=2897,D=512`、full `K=6455,D=512` 各预热 50 次，并在 profiler 窗口内各执行一次三段 candidate。9/9 kernel launch、四项 metrics、三 case byte account 与 telemetry 全部 PASS：

| case | Tier-A bytes | ncu DRAM bytes | ncu / analytic | 处置 |
|---|---:|---:|---:|---|
| SWA 512/256 | 1,057,792 | 1,093,760 | 1.034003 | Tier-A 由 ncu 背书 |
| full 2897/512 | 11,888,290 | 12,038,656 | 1.012648 | Tier-A 由 ncu 背书 |
| full 6455/512 | 26,468,974 | 26,808,320 | 1.012821 | Tier-A 由 ncu 背书 |

最大偏差 3.4003%，全部小于 15%，无 case 需要 Tier-B measured bytes 覆盖。ncu 窗口 memory clock p50 8000 MHz、最高 66°C、无 thermal/hardware throttle；ncu duration 只作诊断，不替代 Day-1 CUDA-event p50。

R3 据此授权了 C02 output-attentions-false 回连。授权边界要求保持 prefill eager、dispatch miss 回退 eager，并在 5 盘/11 请求上验证 exact greedy token IDs、每 token 42 次候选命中、343 个 NF4 模块及 Parameters/state_dict/KV cache 不变、peak VRAM <7.5 GiB；实际回连与终态见 §7。

## 7. C02 回连与终态

R4 首次尝试完整跑完一个请求，但 candidate 调用为 0。真实 decode 调用的 Q/K/V 均符合 contiguous BF16 合同，唯一拦截条件是 Transformers 对 Q=1 优化掉全零 causal mask，使 `attention_mask=None`；同时清理逻辑误从 registry 的 local mapping 删除全局注册项。该尝试被中断并标记为 `INVALID_EXECUTION_ATTEMPT_NO_DECISION`，不得进入性能或部署结论。

R5 在正式回放前冻结两项适配：decode `mask=None` 规范化为连续 BF16 零 mask；full-attention K 向上补到已过 correctness 的冻结长度桶，K/V 补零、mask 后缀填 BF16 最小值。为排除 NF4 重载导致的跨进程 token 漂移，oracle 改为一次模型加载内先跑完整 A eager，再跑完整 B K3。正式结果如下：

| 门 | R5 结果 |
|---|---:|
| 范围 | 5 盘、11 请求、A/B 共 22 run；每路径 21,705 completion token |
| candidate dispatch | 911,148/911,148；42 个模块各 21,694 次 |
| family 对账 | SWA 759,290；full 151,858；prefill eager 462 |
| mask / padding | 合成零 mask 911,148；full 补桶 151,123；结构违规 0 |
| greedy token exact | **0/11，FAIL** |
| A / B incremental TPOT | 138.568327 / 141.700198 ms；B 慢 2.260164% |
| 模型结构 | 343 个 NF4、Parameters、state_dict、device map 与 registry restore PASS |
| 显存 / 遥测 | peak allocated 5,023,133,696 bytes；6,150 样本；最高 74°C；无 thermal/hardware throttle |

11 个请求均从冻结 messages 独立渲染，A/B prompt token IDs 相同。5 个 tape 的首轮请求分别在生成 index 6、3、8、3、8 首次分叉，证明 decode candidate 会改变 greedy 轨迹；另有 5/11 请求在虽被 dispatch 计为 eager 的 prefill 首 token 即不一致。这是额外的部署等价失败事实，但现有证据不足以把首 token 分叉单独归因于某一内部子路径，故不作超出 receipt 的因果表述。

原始 receipt 把 42 个 attention module 直接当作 42 个持久 KV cache layer，因而错误记录 0/22。模型配置为 42 层、其中 18 层 shared KV，正确持久层数是 24；post-run adjudication 对原始记录逐层复核后得到 cache 22/22 PASS、11 对 A/B 结构完全一致，且没有几何、逻辑长度、dtype 或 device 错误。原始 result/receipt 保留不改，补充 receipt 单列修正；token 硬门仍独立失败，所以总裁决不变。

最终状态为 `K3_C02_RECONNECTION_FAIL_CLOSE_DEPLOYMENT_PATH`。不得把 Day-1 isolated speedup 或 ncu byte account 改写为 C02、板端、prefill、llama.cpp、sm_100/sm_120 部署收益，也不得声称跨日或独立复现 PASS。K3 不再按同一 candidate/协议续跑；OP5 下一入口是剩余 backlog 处置审计。现行重排中 Inductor 不是独立 kernel、lm_head 仍为 carrier-deliverability HOLD、KV 反量化仍为 Authenticity BLOCK；只有新的 bounded deliverability probe 能决定是否重开 lm_head 选择。
