# M0 §1 · 本地推理端点（冻结事实）

- 完成日期：2026-08-02 前
- 状态：**完成**
- 权威合同：`eval_config.yaml`

## 1. 冻结结果

| 项 | 实测值 |
|---|---|
| 基线 | `B0-PTQ-Q4KM`，Gemma 4 E4B IT |
| 谱系 / 量化 | `google_bf16_instruct` / PTQ `Q4_K_M`，无 imatrix |
| 文件 | `models/gguf/gemma4-e4b-it-Q4_K_M.gguf` |
| SHA-256 | `953b94c6a89960ab9363720d14bf3ed266058dff31f3d35d2f91e68efdf8989a` |
| llama.cpp | build 9987，commit `ad8d8219915df8e423768d082d1dccfccb6e8437` |
| GPU | RTX 4060 Laptop，8,187 MiB |
| 端点 | alias `gemma4-e4b`，`127.0.0.1:8080`，131,072 context，单 slot，43/43 层 CUDA offload |
| KV | K/V 均为 `q8_0`；non-SWA 1,088.00 MiB + SWA 21.25 MiB = 1,109.25 MiB |
| 模板 | Jinja 开，thinking 开，`reasoning_format=auto` |
| 采样 | temperature 1.0，top-p 0.95，top-k 64，min-p 0 |

端点完成过两次 5,121-token 提示请求，最长运行到 7,206 tokens，`truncated=0`。这证明 131K 的分配、加载和中等长度请求闭环，不等于已做接近 131,072 tokens 的满窗压测。

## 2. 谱系边界

`gemma-4-E4B_q4_0-it.gguf` 是 Google 官方 QAT-Q4_0 部署锚，不是项目自转 PTQ Q4_0。B0 Q4_K_M 与官方 QAT-Q4_0 必须始终分开 `training_lineage`、`quantization_method` 和 `quantization_format`。

## 3. 证据与保留

- 转换/量化日志：`logs/m0/convert_gemma4_e4b_bf16_gguf.log`、`logs/m0/quantize_gemma4_e4b_q4_k_m.log`。
- 基线服务日志：`logs/m0/llama_server_baseline_c131072_q8_n32768.log`。
- TB 探测服务日志：`logs/m0/llama_server_tb_probe_c131072_q8.log`。
- 两条 GGUF 和 HF 权重作为后续纵向比较输入保留，不入 Git。

## 4. 不再声称

- 不声称已做 131K 满窗压测。
- 不把历史 Pi 日志当成 TB endpoint flag 的权威证据。
- 不把 tools-API 路径与 terminus-2 的纯文本 JSON parser 混为同一机制。
