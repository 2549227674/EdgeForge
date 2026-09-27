# C01-S step-00512 checkpoint 审计

- 审计对象：云端 `results/t2/c01_s_family_80_20_half_r1/checkpoints/step-00512`
- checkpoint 形成时间：2026-08-11 23:53 CST
- 审计结论：**PASS WITH OWNER-ACCEPTED DTYPE CORRECTION**
- owner 裁决（2026-08-12）：接受 **FP32 LoRA trainables + BF16 底座计算 + 最终 BF16 merge**；当前 4,096-step run 原地继续，不因原执行卡的 dtype 误记而重启。
- 边界：本审计证明 checkpoint 文件完整、状态可加载且身份/计数一致；owner 已显式豁免本次实际恢复续跑演练，因此不声称执行过 end-to-end resume smoke。
- 机器审计 receipt：`results/m1/c01_s_checkpoint_00512_machine_audit.json`，SHA-256 `fdd0d416221311387973161f3ac679e876aff747476b551600754b59d2cbd040`；由通用中间 checkpoint verifier 独立重载并复验本页门项。

## 1. 身份与计划一致性

`continuation_state.pt` 成功加载并给出：

- `completed_steps = 512`
- `full_schedule_optimizer_steps = 4096`
- plan SHA-256：`61ece8bad05f4156b4b20fa734c2b7e1f804fc879aa50d7a28cceb790042820e`
- holdout SHA-256：`0a28a7f7b52a01f7a73ccdf927f98ac21902f8037ef236db4f86d5024a67d73b`
- template SHA-256：`0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5`
- 已登记 evaluation step：`[0]`
- scheduler `last_epoch = 512`
- optimizer state 条目数：686
- RNG rank：`[0, 1]`

逐 rank 审计值与冻结计划前 512 step 的独立重算完全一致：

| rank | records | input tokens | supervised tokens |
|---:|---:|---:|---:|
| 0 | 5,213 | 9,566,248 | 5,707,184 |
| 1 | 4,965 | 8,988,519 | 5,316,093 |

## 2. 文件与 SHA-256

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `READY` | 9 | `37a40f08d8548dba289b9b0bb35bcf63b359f6d37ee86044ebc6b6da080b9ec1` |
| `adapter/README.md` | 5,196 | `6a369ffd0c2d8faf04f11e374228788bdb6d63f9d006215a273f8ad032301ec9` |
| `adapter/adapter_config.json` | 10,162 | `c3b3d2588b0d1eeea5aa84df262ed3cfca605306990e9c00aead78ef65c1fe4a` |
| `adapter/adapter_model.safetensors` | 101,178,872 | `a7658d0e2f9b9660b234d94d66fce485f7c170a1a7d3a99be7a1526e2af0aba7` |
| `continuation_state.pt` | 156,128,677 | `a61e981cd56d20a3b3771845c388d8e99f32afa63a49d02fbfbea2adb185860e` |

`READY` 内容为 `complete`。adapter 配置为 r=8、alpha=16、dropout=0、bias=none；权重共 1,184 个 tensor、25,249,792 个元素，全部有限。

## 3. dtype 偏差、原因与处置

原 R2.2 写为“bf16 LoRA”，但 checkpoint 实物显示：

- adapter 权重 25,249,792 个元素全部为 FP32；
- Adam `exp_avg` 与 `exp_avg_sq` 各 19,439,616 个元素，全部为 FP32；
- 底座 NF4 反量化计算仍为 BF16。

运行环境为 PyTorch 2.11.0+cu128、PEFT 0.20.0、Transformers 5.14.1、bitsandbytes 0.50.0。原因是 PEFT 默认 `autocast_adapter_dtype=True` 会将 FP16/BF16 adapter 提升至 FP32，以采用其标准稳定训练路径；这不是训练中途发生的 dtype 漂移。

owner 于 2026-08-12 接受该实际数值路径并批准修正 R2.2，故本 checkpoint 的 HOLD 已解除。最终 C01-S 仍须满足独立 merge 门：完整权重全部 BF16、无 `lora_` tensor 残留、文件集合 SHA-256 完整登记；在该门通过前不得把训练 checkpoint 本身称为 BF16 candidate。

## 4. BF16 merge 路径预检

2026-08-12 在同一云实例内，以 step-00512 adapter 做低优先级、4 CPU 线程、纯 CPU、不落盘 merge 预检；未占用训练 GPU，也未写入正式 `models/C01-S-bf16-r1` 路径。资源判断采用容器 cgroup 与实际挂载点，不采用 `free -h` 暴露的宿主机内存视图：

- CPU quota：44 cores（`cpu.max = 4400000 100000`）；
- memory limit：236,223,201,280 bytes，即 220 GiB；
- GPU：2× NVIDIA RTX PRO 6000 Blackwell Server Edition，各 97,887 MiB；
- 数据盘：250 GiB，预检时剩余 218 GiB。

PEFT `safe_merge=True` 实物结果：

- merged model class：`Gemma4ForConditionalGeneration`；
- parameter tensors：1,106；
- BF16 parameter elements：7,941,100,832；
- residual `lora_` parameters：0；
- 结论：**PASS**。

原始日志：`results/m1/c01_s_merge_preflight_step00512.log`，7,253 bytes，SHA-256 `6974bc1e9617767d9b2633db7da046aa35152928b898b9bd5a77aed732886a48`。该预检证明当前 FP32 adapter 可安全并入 BF16 底座；正式 C01-S 仍须在 step 4,096 后重新 merge、完整落盘并通过逐文件 SHA 与逐 tensor dtype 门，不能用本次预检代替终态验收。
