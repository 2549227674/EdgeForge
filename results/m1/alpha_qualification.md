# EdgeForge M1 · α 资格判定书

- 判定日期：2026-08-11（Asia/Shanghai）
- 执行口径：`docs/m1/EdgeForge_M1_执行卡_R1.1.md` §2A，门槛测后未修改
- 结论：**激活检查点失格，进入分支②（SFT 4,096 + 轮级分段）**
- 阻塞解除范围：线 α 已完成；允许进入 T1 分支②重渲染，但不得以本次检查点配置运行 T2/OPD
- 云端工作区：`/root/autodl-tmp/edgeforge-m1/workspace`
- 云端大工件目录：`/root/autodl-tmp/edgeforge-m1/workspace/results/m1/alpha`

## 1. 判定摘要

检查点的显存收益成立，但训练轨迹不满足冻结的 loss 等价门。2,048-token 配置从第 3 step 起越过 `1e-3`，最大相对差 14.9867%；4,096-token 配置从第 4 step 起越门，最大相对差 0.1563%。固定 32 序列的初始 forward/logits 指纹完全相同，说明问题不是静态前向，而是检查点反向重算后产生了不同的参数更新轨迹。

R1.1 规定任一语义门破即失格并关闭检查点议题。因此没有执行使用失格检查点配置的 8K/16K 阶梯；这不是漏项，而是预注册停止规则的执行结果。

| 门 | 冻结判据 | 实测 | 判定 |
|---|---|---|---|
| G1 loss 等价 @2,048 | 20 step，逐 step 相对差 ≤1e-3 | step 1 相等；step 3 首次越门；最大差 0.1498665841 | **失败** |
| G2 loss 等价 @4,096 | 5 step，逐 step 相对差 ≤1e-3 | step 1 相等；step 4 首次越门；最大差 0.0015631681 | **失败** |
| G3 logits 指纹 | top-1 ≥99.9%；mean \|Δlogprob\| ≤5e-3 | 32 序列、16,384 token；top-1 =100%；mean 差=0 | 通过 |
| G4 显存收益 @4,096 | checkpoint-on peak allocated ≤40 GiB | 37.5462 GiB（off 为 86.0588 GiB） | 通过 |
| G5 吞吐 | 只记录 | 2K：3,026.02→2,424.04 tok/s（-19.89%）；4K：2,500.64→2,055.20 tok/s（-17.81%） | 已记录 |

总体判定为 **FAIL / 分支②**，因为 G1、G2 为语义门且均失败。G4 的显存收益不能覆盖语义失格。

## 2. 机器、模型与 shared-KV 事实

- GPU：NVIDIA RTX PRO 6000 Blackwell Server Edition，97,887 MiB，compute capability 12.0。
- 栈：PyTorch 2.11.0+cu128；Transformers 5.14.1；PEFT 0.20.0；NF4 double quant / bf16 / SDPA / LoRA r=8 alpha=16 all-linear。
- 权重：15,992,595,884 bytes；SHA-256 `cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503`，与冻结身份相等。
- 模板 SHA-256：`0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5`，与冻结身份相等。
- 实际 text config：`num_hidden_layers=42`、`hidden_size=2560`、`intermediate_size=10240`、`head_dim=256`、`global_head_dim=512`、`sliding_window=512`、`num_kv_shared_layers=18`。

“42 层”和“18 个 shared-KV 层”不冲突：42 是 decoder block 总数。当前 Transformers 实现计算 `first_kv_shared_layer_idx = 42 - 18 = 24`，所以 0–23 层是非共享层，24–41 层不含自己的 K/V 投影。后 18 层按 attention 类型复用共享点之前最后的 K/V：sliding 路径来自 layer 22，full-attention 路径来自 layer 23。

这也修正了评审粗估中的 35 层、hidden 2048、sliding 256；这些数值不得继续用于本模型显存推导。

## 3. 显存与 SDPA 归因

4,096-token、无检查点的单步 forward/backward profiler 结果：

- peak allocated/reserved：85.9066/86.8125 GiB；
- 35 个 sliding block（head_dim 256）全部进入 `aten::_scaled_dot_product_efficient_attention`；
- 7 个 full-attention block（global head_dim 512）也全部进入同一 efficient kernel；
- 没有观测到 SDPA math kernel，V-2 的“全局层 math 回落”嫌疑被实测否定；
- 累计显存分配热点主要是 `[1,4096,10240]` MLP/LoRA 张量和 `[1,4096,2560]` 残差/归一化张量。

因此 18.03 MiB/token 不能归因于 math attention 回落；主要来源是无检查点时跨 42 block 保留的大量训练激活。

### logits/CE 分块探针

128-token 分块并对 CE 使用非重入重算，loss 绝对差为 `2.384185791015625e-07`：

| 配置 @4,096 | 标准 loss peak | 分块 CE peak | Δpeak |
|---|---:|---:|---:|
| 无检查点（可独立消费的归因结果） | 85.9066 GiB | 66.4926 GiB | **-19.4139 GiB** |
| 检查点开启（仅诊断，不得进主线） | 37.3939 GiB | 18.9244 GiB | -18.4695 GiB |

分块 CE 是有效的独立显存杠杆，但尚未通过完整训练轨迹与 merge 合同，不能用它推翻本次分支②判定；若后续立项，应单独预注册资格门。

## 4. 旧 8K OOM 审计补齐

真实样本：

- UID：`Crownelius__Complete-FABLE.5-traces-2M:d794d4fdf4153c0819341559b78bdedcd454dca7bd891272c87a3d8ef314615a:0`
- 实际长度：8,153；监督 token：5,892。

无检查点第 1 step 在 **forward** 阶段 OOM：

- peak allocated/reserved：93.5670/94.2520 GiB；
- 尚需申请：320 MiB；
- 精确 Python/算子栈：Gemma4 decoder layer → MLP `down_proj(act(gate_proj(x)) * up_proj(x))` → bitsandbytes LoRA `lora_B(lora_A(dropout(x))) * scaling`；
- 该结果补齐了原讨论稿缺失的“forward/backward 与精确算子阶段”审计。

16,251 样本未执行：旧阶梯在 8,153 已 OOM；新检查点阶梯又因 G1/G2 语义失格被预注册规则关闭。

## 5. 出口与下一阶段

线 α 出口固定为分支②：

1. T1 以 `max_seq_length=4096` 实现轮级分段；前文只作上下文，loss 只落本窗新增轮次。
2. 重渲染报告必须按 family × task_type 给出可训练记录保留率和监督 token 保留率。
3. 必须通过 D-9：每家族记录保留率 ≥90%，监督 token 保留率 ≥70%；否则不得生成赛马计划。
4. 旧 `rendered_4096_r2`、24,628,652 token/配方计划和 raw-uniform 4 step 继续作废。
5. D-9 通过后重新计算 1/8 token budget、重新冻结两份 Parquet SHA，并从 step 0 重启赛马。
6. 第二张 Pro 6000 可在 owner 接受本判定书后为 T2 DDP 租用；DDP 只用于吞吐，不用于提高单序列上限。

多-rank token 预算一致性没有在 α 单卡阶段伪测。由于 16K 阶梯已关闭，它改为 T2 DDP smoke 的强制验收项。

## 6. 工件与命令审计

已回收入仓的小工件位于 `results/m1/alpha/`：

- `g1_pair_2048.json`
- `g2_g4_pair_4096.json`
- `g3_logits_fingerprint.json`
- `fingerprint_manifest.json`、`fingerprint_samples.jsonl`
- `alpha2_memory_attention_profile.json`
- `alpha2_chunked_ce_probe.json`
- `alpha2_chunked_ce_probe_no_ckpt.json`
- `legacy_8153_oom_audit.json`

每个 GPU JSON 的 `environment.command` 保存了完整 argv、cwd、hostname、模型/模板 SHA、torch/CUDA/GPU 身份和实际模型 config。

云端大工件保留而不纳入 git：

| 工件 | bytes | SHA-256 |
|---|---:|---|
| `results/m1/alpha/alpha2_memory_snapshot.pickle` | 10,745,571 | `0bc1fa8235030ef35d3ae66d9b3a21959f703f8d46ce106570b2ade86dde63b3` |
| `results/m1/alpha/legacy_8153_oom_snapshot.pickle` | 5,066,489 | `972b5709e2178a8eeda689464e2313123261c1c7b82b0e8cae460447a7f7e318` |

云端回传路径已验证；上述 JSON/JSONL 已由该目录通过 SCP 回收到仓库。快照可按 SHA 定点回传，不依赖实例系统盘。

## 7. R1.1 未复核项更新

- V-1：**阴性销号**。shared-KV E4B 检查点省显存但训练轨迹不等价，M1 主线失格。
- V-2：**阴性销号**。35 个 sliding + 7 个 full block 均为 efficient SDPA，无 math 回落。
- V-3：**不再消费**。16K 检查点显存/吞吐估算因 V-1 失格不实测。
- V-4：保持未复核；分段方案未触发 D-9 失败前不碰 CP。
- V-5：转入下一阶段 T1 分支②重渲染。
- V-6：保持 T3 首日复核。
- V-7：身份缺口声明不变。
