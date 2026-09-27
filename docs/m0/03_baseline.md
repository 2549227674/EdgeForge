# M0 §3 · 锁题、官方基线与回放（冻结事实）

- 完成日期：2026-08-05
- 状态：**完成，HumanEval 仅做 5 题 smoke**

## 1. 锁题与失败口径

- Terminal-Bench 2.1 锁定 20 题，每题 5 次，串行。
- sanity 最终 0/5，只发生 1 轮换题后即按预声明上限锁定，不为抬高基线继续换题。
- 失败分为 F1 verifier-zero、F2 30 轮耗尽、F3 agent timeout、F4 基础设施重跑，互斥计数为 92/7/1/0。
- `finish_reason=length` 为 0，基线阶段将单轮上限提到 32,768 没有被观测到触发。

锁定载体是 `configs/m0/m0_baseline_job.yaml` 和 `results/baseline_e4b_q4km/lock.json`；不再从历史执行文档复制命令作为新授权。

## 2. B0 基线

| 类别 | 结果 |
|---|---:|
| TB 2.1 | 0/100 |
| hard / soft parser | 149/836（17.823%） / 498/836（59.569%） |
| BFCL v4 `simple_python` | 363/400（90.75%） |
| MMLU fast-500 | 0.598 ± 0.0200 |
| GSM8K fast-200 strict / flexible | 0.840 / 0.845 |
| HumanEval | 5 题完成评分，0 题通过（pass@1=0）；仅 smoke |
| cache-warm TTFT p50 / p95 | 349.459 / 1762.491 ms |
| cache-warm TPOT p50 | 19.220 ms/token |
| cache-warm throughput p50 | 52.029 tok/s |

MMLU 使用同一 GGUF、同一 llama.cpp commit 和同一 131K/Q8/CUDA 合同的 continuation-token loglikelihood 兼容层，因 OpenAI-compatible completions 端点不返回 prompt token logprobs。它仍是 MCQ loglikelihood，不是生成式判题。

HumanEval 将 `TMPDIR/TMP/TEMP` 放到 Linux `/tmp` 后，官方执行器完成 5 题并得到 0/5；样本量不足，不得称为正式全量基线。

## 3. 官方 QAT-Q4_0 锚

| 项 | 结果 |
|---|---:|
| endpoint / tools | 通过 |
| BFCL `simple_python` | 364/400（91.00%） |
| GSM8K strict / flexible | 0.850 / 0.865 |
| cache-warm TTFT p50 / p95 | 377.423 / 848.977 ms |
| cache-warm TPOT / throughput p50 | 19.161 ms/token / 52.188 tok/s |

该锚只进锚区，不冒充 B0 before 行。它的 TB 20×5 慢档按既定边界延至 M6 三点同场比较。

## 4. 磁带与系统口径

- 从 B0 原始 trajectory 冻结 5 盘、每遍 11 请求的 `traces/tapes/`。
- 同一批磁带回放所有 candidate，不按 candidate 自己的成功轨迹重选负载。
- 每盘连续回放两遍，仅统计第二遍 cache-warm。
- B0 完整 22/22 成功，第二遍 11/11；中断的 17/22 记录不进正式值。

## 5. 保留资产

- `configs/m0/m0_baseline_job.yaml`、`eval_config.yaml`、固定评测 manifest。
- `results/baseline_e4b_q4km/` 中已入库的 lock/result/metrics/replayer 快照。
- 100 条原始 `agent/trajectory.json` 与 `traces/trajectories_sha256.txt`；cast、pane、trial 附件已删除。
- BFCL/lm-eval 聚合结果与冻结日志；samples JSONL 与旧 smoke 目录已删除。
