# M0 §4 · Agent 指标（冻结事实）

- 完成日期：2026-08-05
- 状态：**完成，schema B 已固定**
- runner：`metrics.py`（`edgeforge-agent-metrics/v2`）

## 1. 输入与结果

100 条 B0 trajectory 已按 `traces/trajectories_sha256.txt` 逐项校验。v2 runner 保留 v1 回归断言，必须复现 parser 四数 149/498/836/125；临时将 hard count 改为 148 时，断言已实测以退出码 2 失败。

| 指标 | 冻结值 |
|---|---:|
| success | 0/100 |
| F1/F2/F3/F4 | 92/7/1/0 |
| hard parser error | 149/836（17.823%） |
| soft format warning | 498/836（59.569%） |
| hard error per-trial median / zero-hard trials | 0 / 64 |
| task-cluster bootstrap SE / design effect | 5.929pp / 20.064× |
| trial turns median / restricted mean | 6 / 8.36 |
| censored at 30 turns / timeout | 7 / 1 |
| parser recovery events / median turns / lock-in trials | 42 / 1 / 3 |
| no-reasoning responses | 125/836（14.952%） |
| completion / prompt tokens per trial median | 5,801 / 19,831.5 |
| cached-token share median | 81.613%（n=835） |
| thinking / message / command-content token median | 273（n=711） / 159 / 4 |
| tool calls | 1,340 across 559 turns |

## 2. 解释红线

- hard parser error 与 soft warning 是两个不可相加的集合。
- 0/100 不使用 Wald SE=0；按 20 个锁定任务的 rule of three，95% 上界为 15%。
- 7 个 trial 被 30 轮上限截停，真实需要轮数只能写 `≥30`；30 不是模型自然结束轮数。
- 125 个缺失 reasoning 的响应中，111 个被 parser 接受；它是模型行为信号，不是日志丢失。
- `premature_complete_rate` 不输出，因 trajectory 没有结构化 `task_complete` 字段。F1 只表示 zero reward、未耗尽轮数、未超时。
- thinking/message/command 是 Harbor 规范化视图，不冒充原始 OpenAI 响应 JSON。

## 3. 机制分层

| 机制 | runner / 结果 |
|---|---|
| terminus 文本 parser | `metrics.py` → `agent_metrics.json` |
| BFCL OpenAI tools API | `scripts/m0/eval/run_bfcl.py` → `baseline_bfcl.json` |
| lm-eval | `scripts/m0/eval/run_lm_eval.py` → `baseline_lmeval.json` |

三者不得混在一个「tool accuracy」指标中。

## 4. 复用接口

后续 candidate 仅在有新里程碑授权时使用：

1. 以同一 GGUF tokenizer 端点运行 `scripts/m0/eval/count_response_tokens.py`，sidecar 只保存计数和端点指纹。
2. 离线运行 `metrics.py`，强制检查 trajectory manifest 和 v1 回归断言。
3. 以 20 个锁定任务做配对比较，同时比较 pooled rate、per-trial 分布、上限截停数和超时数。

M0 本身不再重跑该链路。

## 5. 保留资产

- `metrics.py`、`scripts/m0/eval/count_response_tokens.py`。
- `results/baseline_e4b_q4km/agent_metrics.json`、`token_counts.json`、`parser_metrics.json`。
- `logs/m0/m0_input_probe.{json,md}` 与 `m0_token_counts.log`；空的 `m0_agent_metrics.log` 已删除。
- 100 条原始 trajectory 与 SHA-256 manifest；不保留 cast、pane 或任意复制的响应正文。
