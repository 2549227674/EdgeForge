# M1 T1 分支②重渲染与 D-9 判定

- 判定日期：2026-08-11
- 结论：**通过 D-9，T1 分支②重渲染关闭；赛马计划已按新预算从 step 0 重建。**
- 正式云端工件：`/root/autodl-tmp/edgeforge-m1/workspace/results/t1/segmented_4096_r6/`
- 失败诊断工件：`segmented_4096_r3/`，含 17 条字面量 tool-call 边界误判，仅作诊断，禁止训练消费。

## 冻结分段合同

- 窗口上限：4,096 token。
- 基本单元：用户消息起始的完整消息轮；在可容纳时加入最大完整前置轮后缀。
- loss：只落目标轮助手 span；所有上下文 label 为 `-100`。
- 超长单轮：输入窗口按 1/4 窗长重叠，目标 label 互斥分配，保证任一监督 token 最多消费一次。
- UID：`source_uid + target_round + target_part`；正式结果无重复 UID。
- reasoning：保持完整冻结模板对历史 reasoning 的可见性语义。

## 输入身份

`manifests/mix_records_sha256.json` 中五个 JSONL 的字节数、非空行数和 SHA-256 全匹配，合计 154,097 条；冻结模板 SHA-256 为 `0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5`。

清单内历史 `mix_yaml_sha256=8cbe…` 与当前受版本控制的 `data/mix.yaml=aca2…` 不同；YAML 不是渲染输入，按告警登记，不覆盖五个规范 JSONL 与模板的硬门结果。配方配置继续引用当前 `aca2…` 身份。

## 正式渲染结果

| 指标 | 结果 |
|---|---:|
| 源记录 seen / rendered / retained | 154,097 / 154,097 / 154,097 |
| 丢弃源记录 / 渲染失败 | 0 / 0 |
| 唯一窗口 | 188,095 |
| 最大窗口长度 | 4,096 |
| 窗口输入 token | 300,334,422 |
| 原始监督 token | 198,632,672 |
| 保留监督 token | 198,630,117 |
| Parquet 体积 | 663 MiB / 189 shards |
| 墙钟 | 259.71 s |

Parquet 流式复核确认：188,095 个 UID 全唯一；每行 `len(input_ids)=len(labels)=kept_tokens≤4096`；每行监督计数与字段一致；按 source 汇总与 gzip 审计一致；总计与渲染报告一致。

## D-9

| family | 记录保留率 | 监督 token 保留率 | 判定 |
|---|---:|---:|---|
| claimed_anthropic_from_source_label | 100% | 99.998521% | PASS |
| fable-5 | 100% | 100% | PASS |
| glm-5.1 | 100% | 100% | PASS |
| glm-5.2 | 100% | 100% | PASS |
| kimi-k2.5 | 100% | 100% | PASS |
| qwen3.7-max | 100% | 100% | PASS |

D-9 门槛为每 family 记录 ≥90%、监督 token ≥70%。全部 53 个 family × task_type 诊断组也通过；最低诊断组为 `claimed_anthropic_from_source_label|tool_agent`，监督保留率 99.972063%。未使用一次性阈值校准权。

2,555 个未保留监督 token 集中于 18 条 Anthropic 标签 tool-agent 源记录：17 条的正文含字面量 `<|tool_call>`，冻结全量正则口径额外计入边界 token；另 1 条为模板历史轮可见性差异。它们均保留为可训练记录，不影响过门。

## Holdout 与新预算

- 原 1,500 个 source UID 全部从训练计划排除；对应 1,948 个渲染窗口、3,483,799 input token。
- `manifests/m1_holdout_long_gt8192.json` 冻结其中 72 条源长度 >8,192 的长上下文探针。
- 训练池：152,597 个 source、186,147 个窗口、296,850,623 input token。
- 每配方 1/8 等 token 预算：**37,106,327**；旧 24,628,652 预算与 raw-uniform 4 步继续作废。
- raw-uniform：23,267 行、24,492,290 shifted supervised token，计划 SHA `310460f7…9483`。
- family-80/20：20,385 行、22,154,013 shifted supervised token，计划 SHA `15137a7d…de07`。
- 两个计划均为 1,024 optimizer steps、每步监督非空；必须从 optimizer step 0 启动，不得续接旧 checkpoint。

## 冻结 SHA

| 工件 | SHA-256 |
|---|---|
| 正式 `render_report.json` | `0694e87c5b75185adff702929d6f03f8fdd3098d7059e16fb95db83866b2bb02` |
| 正式 `segment_audit.jsonl.gz` | `01de75a2c26a80d55676fe76997a7435ca876601f5dcbdfa7a3787f391696d15` |
| `manifests/m1_recipe_plans.json` | `4161035534a486be093413c88a081b05e2ee253444de1f1a3573476a0046e00f` |
| `manifests/m1_holdout_long_gt8192.json` | `8dfc17d2af0edf52ba2aff940e126bf1299b25a232764bb5f5c89890e2d97af6` |

当前出口：**重渲染与 D-9 已完成，赛马输入已就绪；下一执行动作是 raw-uniform 与 family-80/20 从 step 0 开始 1/8 预算赛马。按 R2.1 冻结拓扑，实际开跑前应租第二张 Pro 6000 组成 DDP；第二卡只用于吞吐，不用于提高 4,096 单序列上限。**
