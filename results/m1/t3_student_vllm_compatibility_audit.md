# T3 student vLLM compatibility audit

- Audit date: 2026-08-12 (Asia/Shanghai)
- Student: `models/C01-S-bf16-r1`, config SHA-256 `6b23bfd41dafe39fd7d61e858c156370bb23d531de46190e4457748ef3e7176c`
- Runtime: Python 3.12.3, PyTorch 2.13.0+cu130, vLLM 0.27.0, FlashInfer 0.6.16.post3, TRL 1.9.2
- GPU1: RTX PRO 6000 Blackwell Server Edition, SM120, UUID `GPU-39160c7c-0272-9254-c171-be00298b0dfc`

## Verdict

Student colocated vLLM is **rejected** for the formal T3 run. The qualified path remains Transformers generation plus Transformers forward/backward. This verdict does not apply to the already-qualified GPU0 BF16 teacher server, whose model has no shared-KV upper block and whose numerical path was separately gated.

## Findings

1. The initial `FlashInfer requires GPUs with sm75 or higher` message was misleading. PyTorch reports SM120 and CUDA 13.0, but FlashInfer JIT followed `/usr/local/cuda`, whose `nvcc` is 12.8. Its SM12 normalization requires CUDA >=12.9, caught that exception, produced an empty architecture set, and then emitted the generic SM75 error.
2. The venv contains CUDA toolkit 13.3 at `.../site-packages/nvidia/cu13`. With both `CUDA_HOME` and `CUDA_PATH` set to it, FlashInfer correctly resolves target `(12, '0f')` and emits `compute_120f`. vLLM also supports disabling only the FlashInfer sampler with `VLLM_USE_FLASHINFER_SAMPLER=0`. Therefore the sampler failure is repairable and is not the decisive blocker.
3. C01-S has 42 language layers and `num_kv_shared_layers=18`. Layers 24--41 intentionally contain Q projection/Q norm/output projection but no K projection, V projection, or K norm. The last 18 layers reuse K/V from the last non-shared layer of the same attention type.
4. vLLM 0.27's native `Gemma4Attention` registers a KV sharing target but still constructs QKV projection and K norm for every layer, loads the absent weights as required, and unconditionally splits its projection output into Q/K/V. The observed shared-layer projection output is Q-only width 2048, while the forward attempts a `[2048, 512, 512]` split totaling 3072. This fails during vLLM's dummy forward before TRL LoRA synchronization or student generation.
5. `model_impl=transformers` is not a working escape hatch in this stack. It fails during initialization because a global `head_dim` read is rejected for the heterogeneous 256/512 per-layer configuration.
6. The attention API itself can accept `key=None, value=None` for KV-sharing layers, so an upstream engine fix is possible. However it requires coordinated changes to Gemma4 module construction, weight loading, and forward behavior followed by HF/vLLM logits alignment. A local monkey-patch that merely passes generation is not sufficient evidence for formal OPD.
7. TRL 1.9.2 declares vLLM support only through 0.25.1; this project already records 0.27.0 as an owner-requested exploration exception. Extending that exception with an unreviewed model implementation patch would add a second unsupported axis.

## Evidence chain

| Receipt | Failure |
|---|---|
| `t3_student_vllm_nf4_rslora_smoke_receipt.json` | heterogeneous global `head_dim` access |
| `t3_student_vllm_native_nf4_rslora_bridge_smoke_receipt.json` | 18 shared-layer `k_norm` weights not initialized |
| `t3_student_vllm_native_nf4_rslora_sharedkv_smoke_receipt.json` | Q-only 2048 output split as QKV 3072 |
| `t3_student_vllm_native_nf4_rslora_sharedkv_forward_smoke_receipt.json` | reached sampling only after exploratory shared-KV forward patch, then FlashInfer architecture error |
| `t3_student_vllm_native_nf4_rslora_sharedkv_forward_native_sampler_smoke_receipt.json` | sampler fallback works; unpatched native model returns to heterogeneous `head_dim` failure |
| `t3_student_vllm_transformers_nf4_rslora_native_sampler_smoke_receipt.json` | Transformers model implementation also fails on heterogeneous `head_dim` |

All receipts are stored beside this audit under `results/m1/`.

## Frozen implementation hashes

- vLLM `gemma4.py`: `b2bf27dfb914d3afa62f4a126f3f7aff08845fb4be46f2469659097d7ff19100`
- vLLM attention API: `dae6d1f09448adc1d67c776089e77e8b75378332166844a234cd8fa45c18195e`
- FlashInfer compilation context: `65d3afe9c069f9d685a9f99a15e307f02d966a7112681e2d1d5ca70c054336cd`
- vLLM sampler selection: `bc8c1d91ad624368d2bd87a53c0361e550d28b060bcd460462d2f86565c26426`

## Revisit condition

Reconsider student vLLM only after an upstream release natively loads Gemma 4 E4B shared-KV checkpoints, runs without global heterogeneous-field overrides, supports the selected adapter synchronization path, and passes a fixed-token HF/vLLM logits/top-1 alignment gate before any throughput comparison.
