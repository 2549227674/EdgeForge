# M0 脚本索引

M0 已冻结；本目录中的脚本是可复现方法和后续纵向比较接口，不是当前执行授权。

## `eval/`

- `count_response_tokens.py`：生成不含响应正文的 token sidecar。
- `edgeforge_llamacpp_loglikelihood.py` / `llama_loglikelihood.cpp`：llama.cpp continuation-loglikelihood 兼容层。
- `freeze_b0_tapes.py` / `replay_tapes.py`：冻结和回放 B0 磁带。
- `freeze_lmeval_fast_manifests.py`：固定 lm-eval 快档题单。
- `run_bfcl.py` / `run_lm_eval.py`：两套独立评测 runner。

## `data/`

保留完整的线 C 重建链：下载与上游元数据、文件角色、IR、退化分析、L0/L1/L2 去重、安全投影与精确脱敏、去污染、配平、渲染/掩码审计和 archive manifest。

可再生的运行中间层已删除；因此这些脚本不得再被当作「过程垃圾」删掉。它们与冻结 export、`data/mix.yaml`、`data/pipeline/requirements.lock.txt` 和门报告共同构成可重建边界。

## 通用脚本

`scripts/generate_local_project_index.py` 不属于 M0，继续位于 `scripts/` 根目录。
