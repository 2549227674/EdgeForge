# M0 §5 · 数据管线七门（冻结事实）

> 文件名沿用 R1.0，便于现有链接稳定；本文内容已于 2026-08-06 升级为 R1.1 事实版。
>
> 状态：**M0 线 C 已完成**。9 个来源均已按冻结 revision 完整获取、七门（含 ④b）已执行、人工复核已签署，形成 154,097 条规范训练池。本文只陈述实际产物和可验证边界；历史操作规格不再作为当前执行指令。
>
> 范围：数据处理在独立 `.venv-data` 内完成；不修改冻结测试题、测试 manifest、B0 配置或评测结果；不涉及 M1 训练或全量渲染。

## 版本与结论

| 版本 | 日期 | 定位 |
|---|---|---|
| R1.0 | 2026-08-06 | 开工规格：基于样本和上游元数据提出七门执行口径。 |
| **R1.1** | **2026-08-06** | **事实版：以已生成的 manifest、门报告、数据卡、`mix.yaml` 和人工复核记录为准。** |

线 C 的 M0 交付已经闭环：完整性与来源冻结、IR 解析、退化审查、去重、安全与去污染、规范池和抽样渲染均有版本化证据。以下事项不是未完成的 M0 缺陷：M1 选择训练时采样配方、全量渲染、以及 M5 的 SWE-bench Pro 去污染。

## 1. 输入冻结与完整性

九个来源通过原始仓库文件与 Hugging Face `refs/convert/parquet` 两种入口冻结到 [manifest](../../manifests/data_archive_sha256.json)。该 manifest 保存每个文件的 SHA-256、来源 revision、上游/本地行数、许可和文件角色；[数据源布局说明](../../data/SOURCES.md) 解释 `archive` 与 `archive_parquet` 的差异并列出每个来源的规范输入。默认 Dataset Viewer revision 的观察值另存于 `data/pipeline/gate1_upstream_metadata.json`，不与冻结 export revision 混用。

| 来源 | 上游 / 本地行数 | 完整度 | 许可 | 线 C 处置 |
|---|---:|---:|---|---|
| AletheiaResearch/GLM-5.2-Agent | 319 / 319 | 100% | 未标注 | 训练候选 |
| Crownelius/Complete-FABLE.5-traces-2M | 228,968 / 228,968 | 100% | MIT | 训练候选；保留聚合来源链 |
| Glint-Research/Fable-5-traces | 4,665 / 4,665 | 100% | AGPL-3.0 | 最终保留 1 条 |
| Infatoshi/kernelbench-hard-traces | 383 / 383 | 100% | MIT | 永不进训练 |
| Infatoshi/kernelbench-mega-traces | 75 / 75 | 100% | MIT | 永不进训练 |
| WithinUsAI/claude_mythos_distilled_25k | 25,000 / 25,000 | 100% | Apache-2.0 | 退化后整集剔除 |
| armand0e/claude-opus-4.8-pi-traces | 4 / 4 | 100% | 未标注 | 门②后无保留记录 |
| armand0e/qwen3.7-max-pi-traces | 47 / 47 | 100% | 未标注 | 训练候选 |
| lambda/hermes-agent-reasoning-traces | 14,701 / 14,701 | 100% | Apache-2.0 | 训练候选 |

关键口径已经以实测纠正：Crownelius 仓库名中的 `2M` 不是行数证据，冻结 export 为 228,968 行；Glint 的 4,665 行是前缀展开记录，不能视为 4,665 个独立会话；Hermes 是数据 harness 名，其 config provenance 对应 Kimi-K2.5 和 GLM-5.1，而非单一 teacher 身份。

第二介质备份本轮**未做**，该预声明风险保留；它不影响本地 manifest、完整性对账和本次管线结果。

## 2. 七门事实漏斗

门①以 export 行数计；门④将行解析为 IR 并标注源会话；门②才会折叠前缀展开和跨源重复。因此各门分母不能直接相加。完整漏斗以 [数据卡](../../data/data_card.md) 为准。

| 数据集 | 门①行数 | 门④/④b 合格 IR | 门②（含 L2） | 门③ | 门⑤ | 门⑥规范池 | 门⑦ |
|---|---:|---:|---:|---:|---:|---:|---|
| GLM-5.2 | 319 | 319 | 234 | 234 | 232 | 232 | 已覆盖 |
| Crownelius aggregate | 228,968 | 201,786 | 151,364 | 151,364 | 150,921 | 150,919 | 已覆盖 |
| Glint Fable | 4,665 | 4,665 | 1 | 1 | 1 | 1 | 无合格多轮工具轨迹 |
| Mythos-25k | 25,000 | 25,000 | 12 | 12 | 12 | 0 | 整集排除 |
| Opus pi traces | 4 | 4 | 0 | 0 | 0 | 0 | 无记录 |
| Qwen pi traces | 47 | 47 | 38 | 38 | 38 | 38 | 已覆盖 |
| Hermes agent traces | 14,701 | 14,701 | 2,910 | 2,910 | 2,907 | 2,907 | 已覆盖 |
| KernelBench hard / mega | 383 / 75 | 不适用 | 不适用 | 不适用 | 不适用 | 0 | 永不进训练 |

### 门①：完整性与文件角色

- 九个来源均已完成上游行数对账，详细结论见 `data/pipeline/gate1_completeness.md`。
- Crownelius 已从原有单分片补齐为 4 个 frozen export 分片；Hermes 顶层分片与 config 分片按唯一 `id` 对账，避免重复计分。
- `data/pipeline/file_roles.tsv` 区分训练会话、子代理、CLI 历史、粘贴缓存、工具结果和仓库元数据。KernelBench 文件登记 checksum 但标为 `kernelbench_excluded`。

### 门④与门④b：结构、退化与人工复核

- 七个可训练来源均成功解析；Crownelius 的门④输入 202,514 条中有 201,786 条合格，失败率 0.36%（`no_messages=684`、`missing_user_or_assistant=33`、`empty_assistant=11`）；其他来源失败率为 0。见 `data/pipeline/gate4_parse_rates.md`。
- 门④b 的退化诊断显示 Mythos 最高频前缀覆盖 100%、集内精确重复率 99.14%。剥样板和去重后仅余 12/25,000，低于预声明 2% 阈值，因此整集排除。
- 人工复核从预置样本中纳入全部 5 条正则阳性并补足 15 条阴性；5 条均为 Mythos 的真实质量问题（桩实现或编造评测统计），其余 15 条未改变处置。签署记录见 [门④b 人工复核](../../data/pipeline/gate4b_manual_review.md)。

### 门②：会话折叠与字面去重

- L0/L1 依次折叠同源会话、首轮 user 精确重叠和剥样板后的 assistant 精确重复；其中同源会话折叠删除 4,447 条、首轮 user 精确重叠删除 85,997 条、assistant 精确重复删除 195 条。
- L2 采用 128-entry bottom-k 5-gram MinHash 产生候选、完整 Jaccard ≥0.8 验证：137,958 个候选对，1,372 个验证阳性，删除 1,324 条。4 个过大 band 被留档跳过，意为这些 bucket 的候选组合过大、未进入近重比较；不是数据下载缺失，也不是已确认的重复样本。
- Glint 的最终 1 条来自会话折叠和跨源去重，不是家族配比压缩。完整证据见 `data/pipeline/gate2_dedup.md` 与 `data/pipeline/gate2_l2_summary.json`。

### 门③：安全清扫与人工抽检

- 原始 archive 与 parquet export 的 TruffleHog 结果只版本化检测器和文件计数，不写候选值；IR 中有 515 条可复用脱敏规则的 uid/rule 映射。基础脱敏规则在最终规范池共命中 10,904 个记录—规则对。
- 抽检 50/50 条；42 条仍在规范池，其余 8 条均是已整集排除的 Mythos。记录见 [门③人工抽检](../../data/pipeline/gate3_manual_50.md)。
- 最终安全复核只扫描训练模板消费的 `messages` 与 `tools` 载荷，明确排除 UID、provenance、source hash 和路径等非训练 token。215 个未验证候选被精确替换 3,850 处，覆盖 361 条规范池记录；另有 2 条无法可靠精确替换的记录被保守排除。重建载荷后 TruffleHog 复扫为 0。见 [最终载荷复核](../../data/pipeline/gate3_final_mix_trufflehog.md) 与 `data/pipeline/gate3_security_exclusions.json`。

### 门⑤：冻结测试集去污染

训练侧在脱敏前扫描，命中只删除训练样本；所有冻结测试题、manifest 和 B0 配置均未修改。

| 测试集 | 分母 | canary 命中 | 13-gram 阳性训练记录 | 删除 |
|---|---:|---:|---:|---:|
| MMLU | 14,042 | 0 | 11 | 11 |
| GSM8K | 1,319 | 0 | 370 | 370 |
| HumanEval | 164 | 0 | 6 | 6 |
| Terminal-Bench 2.1 | 20 | 0 | 61 | 61 |

门⑤由 154,559 条输入删除 448 条，保留 154,111 条。它证明未检出冻结测试集的字面 canary 或归一化 13-gram 重叠；不证明不存在语义污染，也不追溯上游预训练污染。完整声明见 `data/pipeline/gate5_decontamination.md`。

### 门⑥：规范池与训练时配方

门⑥不再物理 cap 家族。Mythos 规则剔除 12 条、门③最终安全规则排除 2 条后，所有其余硬门合格记录都保留：**154,097 条**。

| 规范池来源 | 记录数 |
|---|---:|
| Crownelius（仅按来源标签标记为 `claimed_anthropic_from_source_label`） | 150,919 |
| Hermes | 2,907 |
| GLM-5.2 | 232 |
| Qwen3.7-Max | 38 |
| Glint Fable | 1 |
| **合计** | **154,097** |

原始均匀池中 Anthropic-style 来源标签为 150,920 / 154,097（97.94%）；这只是描述性分布，不等同于已验证的 teacher 身份。`data/mix.yaml` 定义 raw-uniform、家族 80/20、家族 60/40 三种**训练时**采样配方。60% 不是数据删除阈值，M1 必须在相同 optimizer-step 与 token 预算下比较后再选择默认配方。KernelBench 排除断言通过。

### 门⑦：Gemma 4 原生渲染与 loss mask

- HF Jinja、B0 Q4_K_M GGUF 内嵌模板和官方 QAT Q4_0 GGUF 内嵌模板逐字节一致：18,569 bytes，SHA-256 `0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5`。
- 以 `enable_thinking=true`、`preserve_thinking=true`、`add_generation_prompt=false` 进行确定性抽样渲染。20/20 条渲染成功、tokenizer 往返一致、源 reasoning 完整保留；20/20 同时包含 tool call 和多 assistant turn。
- 人工复核 20 条的所有边界：69 个 thought、219 个 tool-call、153 个 tool-response、141 个 turn 标记全部通过。loss 目标为 assistant 内容、thought 和 tool-call；system、user、工具声明、tool response 与 turn 标记掩除。见 [门⑦人工复核](../../data/pipeline/gate7_mask_review.md) 与 `data/pipeline/gate7_render_validation.md`。

## 3. 当前规范资产与发布边界

| 资产 | 当前状态 | 边界 |
|---|---|---|
| `data/pipeline/mix_records/` | 154,097 条规范训练记录 | 可再生中间产物，受 `.gitignore` 保护，不入库。 |
| `data/mix.yaml` | 规范池统计、来源 provenance 与三种采样配方 | 版本化；不包含训练文本。 |
| `data/data_card.md` | 完整漏斗、来源、去污染与发布限制 | 版本化的总览。 |
| `data/pipeline/*.md` / `*.json` / `*.tsv` | 各门报告、断言、规则、人工复核 | 版本化，且不保存 TruffleHog 候选值或原始训练文本。 |
| `data/pipeline/gate3_scan_payloads/`、`gate7_render_samples/` | 安全投影及逐 token 人工审核材料 | **2026-08-11 已删除**；可由脚本与冻结输入再生，不得当成冻结证据。 |

2026-08-11 资产收敛后，`ir/`、`ir_dedup/`、`ir_l2/`、`ir_gate5/`、安全投影、渲染样本、SQLite 去重库和 cluster 中间文件已删除。冻结 parquet export、`mix_records/`、处理脚本、manifest、规则和报告保留，因此 M1 可直接消费规范池，也可从 export 重建完整管线。

规范池仍含 1 条 AGPL-3.0 Glint 记录。依项目发布边界，**不得发布任何权重**；可发布的是报告、脚本、manifest 和不含训练文本的审计元数据。

## 4. 已验证项与顺延项

已执行的验证包括：脚本 `py_compile`、规范池计数与安全排除断言、KernelBench 排除断言、最终训练载荷零候选复扫、20 条 Gate 7 渲染/掩码审核，以及 `git diff --check`。

| 项目 | 状态 | 原因或下一步 |
|---|---|---|
| M1 采样配方选择 | 顺延 M1 | raw-uniform、80/20、60/40 需等 optimizer-step 与 token 预算对照。 |
| 全量渲染 | 顺延 M1 | M0 的 20 条边界抽样已覆盖本卡验收范围。 |
| SWE-bench Pro 去污染 | 顺延 M5 | M0 没有消费该测试集。 |
| 第二介质备份 | 本轮未做 | 风险已记录；不伪称已完成。 |
| 子代理权重 | 顺延 M1 | 若要限制，应采用采样参数而非物理删除。 |

## 5. 事实证据索引

| 主题 | 权威产物 |
|---|---|
| 来源、完整度与许可证 | `data/pipeline/gate1_completeness.md`、`manifests/data_archive_sha256.json` |
| 解析与退化处置 | `data/pipeline/gate4_parse_rates.md`、`data/pipeline/gate4b_degeneracy.md`、`data/pipeline/gate4b_manual_review.md` |
| 去重 | `data/pipeline/gate2_dedup.md`、`data/pipeline/gate2_l2_summary.json` |
| 安全和去污染 | `data/pipeline/gate3_security.md`、`data/pipeline/gate3_final_mix_trufflehog.md`、`data/pipeline/gate5_decontamination.md` |
| 最终混合与渲染 | `data/mix.yaml`、`data/pipeline/gate6_balance.md`、`data/pipeline/gate7_render_validation.md`、`data/pipeline/gate7_mask_review.md` |
| 全局事实总览 | `data/data_card.md` |
