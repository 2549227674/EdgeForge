# M0 §2 · Terminal-Bench 2.1 自家端点链路（冻结事实）

- 完成日期：2026-08-02
- 状态：**完成**

## 1. 过门结果

`terminal-bench/kv-store-grpc` 的正式探测完成以下四段：

1. terminus-2 向本地 llama.cpp 端点发送请求。
2. 端点返回回复，`reasoning_content` 与最终 JSON `content` 分离。
3. agent 在任务容器中执行 7 条命令。
4. verifier 完成 7 项检查并给出 reward。

reward=0.0 的原因是模型提交的 `Server` 类名和 protobuf `value` 字段不符合 verifier，不是 endpoint、harness 或 parser 故障。该分数只用于链路探测，不进入锁题或基线分数。

## 2. 冻结协议

| 项 | 值 |
|---|---|
| Harbor | 0.18.0 |
| agent | terminus-2，LiteLLM 协议名 `openai/gemma4-e4b`，后端仍是 llama.cpp |
| parser | `json` |
| reasoning | `auto`，thinking 与 final content 分离 |
| 摘要 | `enable_summarize=false`，上下文溢出计失败 |
| 轮数 | `max_turns=30`，耗尽计失败 |
| 格式错误上限 | `null`；Harbor 0.18.0 未实现 `max_format_errors` |
| 单轮生成上限 | 探测时 4,096；基线阶段为 32,768 |
| 任务超时 | task-defined 900 s，不放大 |
| setup | `agent_setup_timeout_multiplier=3.0` |

JSON 和 XML hello-world 在任何 TB 真题之前各跑一次：JSON reward=1；XML 因 `keystrokes` 缺少末尾换行而 reward=0。两者无硬 parser error，按预声明规则选 JSON，后续不再换 parser。

## 3. 探测分布

| 量 | 实测值 |
|---|---:|
| completion tokens p50 / p95 | 465 / 895 |
| 单轮墙钟 p50 / p95 | 9.43 / 19.07 s |
| agent / 完整 trial 墙钟 | 81.65 / 110.90 s |
| 峰值上下文 | 5,770 tokens |
| 4,096 触顶 | 0/7 |

因 4,096 在 7 轮中从未触顶，这组分布被作为模型行为分布；基线阶段以 `finish_reason=length` 作为反证计数器，最终为 0。

## 4. 证据与资产边界

- 锁定快照：`results/m0_tb_probe/2026-08-02__16-31-43/lock.json`。
- 端点日志：`logs/m0/llama_server_tb_probe_c131072_q8.log`。
- 协议自检：`logs/m0/v1_models_tb_probe_c131072.json`、`logs/m0/reasoning_format_check.json`。
- 成功探测的小型原始目录保留；首次 setup 超时、hello-world、allowlist 和 prewarm 过程目录已删除。
