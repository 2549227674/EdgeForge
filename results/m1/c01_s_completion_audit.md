# EdgeForge M1 C01-S 完成审计

- 结论：**PASS**
- 终验时间：2026-08-12 00:07:40 UTC（08:07:40 Asia/Shanghai）
- 云端训练目录：`/root/autodl-tmp/edgeforge-m1/workspace/results/t2/c01_s_family_80_20_half_r1`
- 云端 candidate：`/root/autodl-tmp/edgeforge-m1/workspace/models/C01-S-bf16-r1`
- 机器审计：`results/m1/c01_s_completion_audit.json`
- 机器审计 SHA-256：`d1c2a2ca192b5ffe6f1637e83759a75388e49d356d5611a52d49355ca2902a7d`
- 修正版 verifier SHA-256：`a7a58bcad8edd4a4d741c59eb9cf5d5ab761cd0f7db4c8a5a5beebd343927a61`

## 1. 冻结输入与训练完成性

正式 run 使用 2-rank DDP，rank seeds 为 20260811 / 20260812。冻结计划、holdout、部署模板 SHA 分别为：

- family-80/20 1/2 plan：`61ece8bad05f4156b4b20fa734c2b7e1f804fc879aa50d7a28cceb790042820e`
- holdout parquet：`0a28a7f7b52a01f7a73ccdf927f98ac21902f8037ef236db4f86d5024a67d73b`
- chat template：`0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5`

训练 report SHA-256 为 `a950394e75a5f36db5030aac2c9b94c20f370229dada2e4fdc35d309cfbb7512`。终态精确计数如下：

| 项目 | 结果 |
|---|---:|
| optimizer steps | 4,096 / 4,096 |
| records | 81,757 |
| input tokens | 148,425,311 |
| shifted supervised tokens | 88,486,262 |
| report elapsed | 33,340.9857 s |
| rank 0 records / input / supervised | 41,901 / 75,994,420 / 45,422,524 |
| rank 1 records / input / supervised | 39,856 / 72,430,891 / 43,063,738 |

4,096 个 optimizer-step 日志连续，无缺步；全部训练 loss 有限且非负。末 100 步 weighted loss mean / median 为 0.8434061 / 0.8351436，末步学习率与冻结 cosine schedule 均精确为 0。

## 2. 完整 holdout

每次评测均消费相同的 1,948 个窗口、3,483,799 input token、1,964,016 supervised token；训练 source 与 1,500 个冻结 holdout source 交集为 0。

| Step | Sample mean loss | SE | Token-weighted loss |
|---:|---:|---:|---:|
| 0 | 1.8861029 | 0.0220918 | 1.6302139 |
| 1,024 | 0.9195983 | 0.0122779 | 0.8784115 |
| 2,048 | 0.8918359 | 0.0121011 | 0.8543855 |
| 3,072 | 0.8762867 | 0.0119866 | 0.8415895 |
| 4,096 | 0.8727379 | 0.0119659 | 0.8385551 |

五次完整评测 loss 全有限，token-weighted loss 单调下降。该结论是训练健康证据，不替代后续任务能力评测。

## 3. Checkpoint 与数值路径

step 512 / 1,024 / 1,536 / 2,048 / 2,560 / 3,072 / 3,584 七个中间 checkpoint 均有 `READY`，且逐一通过精确五文件集合、完整文件 SHA、加载性、scheduler step、双 rank RNG 与累计消费核验。根目录 step-4,096 终态也通过相同核心状态核验，并额外确认 run 顶层精确包含 `READY`、`adapter`、`checkpoints`、`continuation_state.pt`、`report.json`、`train_log.jsonl`。

step-512 与 step-4,096 的数值实物复核一致：

- 1,184 个 LoRA tensor，共 25,249,792 个 FP32 元素，非有限值 0；
- Adam `exp_avg` / `exp_avg_sq` 各 19,439,616 个 FP32 元素，非有限值均为 0；
- BF16 底座计算 + FP32 LoRA/optimizer + 最终 BF16 merge，与 owner 已接受路径一致；
- 端到端恢复续跑 smoke 仍按 owner 单次决定标记 `WAIVED_BY_OWNER`，未以加载性检查冒充实际续跑。

## 4. C01-S BF16 工件

candidate manifest 为 schema 3，SHA-256 `89fed9d97eeb95847d41a11ad55b751a69d6ce1d9df827852f56dc12bf0998d0`。manifest 外 11 个文件共 15,914,883,568 bytes，完整 artifact-set SHA-256 为 `1eebd3e1302a502ce41a9416267219e70cec63ac69ea20429dd97bd47ba9a133`。

- 4 个权重 shard；
- 2,076 tensors / 7,941,101,802 BF16 elements；
- 全部 tensor 为 BF16；
- `lora_` tensor 数为 0；
- CPU 独立重载类为 `Gemma4ForConditionalGeneration`；
- 1,106 parameter tensors / 7,941,100,832 BF16 parameter elements；
- LoRA parameter 数为 0；
- `chat_template.jinja`、`processor_config.json`、`tokenizer.json`、`tokenizer_config.json` 与基座逐字节 SHA 相同。

因此 C01-S 已形成可冻结的 BF16 HF candidate；Q4_K_M 评测载体和快/慢档仍是后续任务，未在本审计中宣称完成。

## 5. Verifier 缺陷与处置

merge watcher 的第一次终验在权重 merge 和 manifest 生成成功后失败。原因是 verifier 对最终 run 根目录使用了中间 checkpoint 的递归“五文件集合”规则，因而把合法的 `checkpoints/`、`report.json`、`train_log.jsonl` 错判为额外文件。

修复将“核心 checkpoint 五文件”与“最终 run 顶层精确集合”分开验证；先分别回归 step-512 与 step-4,096 根目录，再运行完整终验。修复没有修改训练计划、checkpoint、adapter、merge 脚本或 candidate 权重。重跑 exit 0，机器审计 `status=PASS`；原失败未被隐藏，也未触发重训或重新 merge。

## 6. 终态资源状态

终验后两个训练 rank、merge 和 verifier 进程均已退出；两张 RTX PRO 6000 均为 0 MiB / 0% utilization。`/root/autodl-tmp` 使用 108 GiB / 250 GiB，剩余约 143 GiB。Codex monitor 保持停用；本次检查未设置定时关机。
