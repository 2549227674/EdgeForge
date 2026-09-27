# Gate 7 mask review

审核人：Codex（2026-08-06）。逐条查看已保存的 `token` / `id` / `mask` / 字符偏移，并以边界跨度复算作交叉检查。

结论：20/20 通过。thought 与 tool-call 范围内 token 均为目标（mask=1）；tool-response 与 turn 控制标记范围内 token 均为非目标（mask=0）。边界上跨越目标内容的 tokenizer token 按既定保守规则记为 1。

汇总：thought 69 段、tool-call 219 段、tool-response 153 段、turn 标记 141 个。

| # | UID | Dataset | thought | tool call | tool response | turn | 往返 | thinking | 审核 |
|---:|---|---|---:|---:|---:|---:|---|---|---|
| 1 | `AletheiaResearch__GLM-5.2-Agent:019ee987-5910-762c-9c9e-c4175135adda:270` | `AletheiaResearch__GLM-5.2-Agent` | 3 | 36 | 3 | 7 | pass | 3/3 | pass |
| 2 | `Crownelius__Complete-FABLE.5-traces-2M:74acbf6e-edf5-4771-bec6-958106b7587e@48d97c9f8ee7:0` | `Crownelius__Complete-FABLE.5-traces-2M` | 1 | 7 | 7 | 16 | pass | 1/1 | pass |
| 3 | `armand0e__qwen3.7-max-pi-traces:019e4e51-c2f5-7603-bf45-0ee45f2146cc:0` | `armand0e__qwen3.7-max-pi-traces` | 2 | 10 | 2 | 5 | pass | 2/2 | pass |
| 4 | `lambda__hermes-agent-reasoning-traces:861b26b4-86e3-4cb4-8417-b69d62707fc5:0` | `lambda__hermes-agent-reasoning-traces` | 2 | 21 | 21 | 4 | pass | 2/2 | pass |
| 5 | `AletheiaResearch__GLM-5.2-Agent:019ee965-9e95-774b-9c7e-c2b19e55f9d6:252` | `AletheiaResearch__GLM-5.2-Agent` | 2 | 22 | 2 | 7 | pass | 2/2 | pass |
| 6 | `Crownelius__Complete-FABLE.5-traces-2M:e735714a-93cf-4ccb-9ce4-6be482b2e736@48d97c9f8ee7:0` | `Crownelius__Complete-FABLE.5-traces-2M` | 4 | 4 | 4 | 17 | pass | 4/4 | pass |
| 7 | `armand0e__qwen3.7-max-pi-traces:019e4e0d-903d-755c-8314-4b7e3e2f595c:0` | `armand0e__qwen3.7-max-pi-traces` | 2 | 2 | 2 | 5 | pass | 2/2 | pass |
| 8 | `lambda__hermes-agent-reasoning-traces:7403aa8a-1435-48ae-9213-eacf1261f026:0` | `lambda__hermes-agent-reasoning-traces` | 3 | 18 | 18 | 4 | pass | 3/3 | pass |
| 9 | `AletheiaResearch__GLM-5.2-Agent:019ee1b0-a92d-70ce-8d35-a2ea793b536a:24` | `AletheiaResearch__GLM-5.2-Agent` | 3 | 4 | 3 | 7 | pass | 3/3 | pass |
| 10 | `Crownelius__Complete-FABLE.5-traces-2M:fc35e99c6cb080fbfb918cef72d64cbb2e3192942ef55da4db0832c2eb3fb1fa:0` | `Crownelius__Complete-FABLE.5-traces-2M` | 4 | 4 | 4 | 8 | pass | 4/4 | pass |
| 11 | `lambda__hermes-agent-reasoning-traces:f23ee4d9-dc85-460b-b278-e6962c7892ab:0` | `lambda__hermes-agent-reasoning-traces` | 2 | 18 | 18 | 4 | pass | 2/2 | pass |
| 12 | `AletheiaResearch__GLM-5.2-Agent:019ee94f-c2dd-737e-9259-8baad7e42dd2:244` | `AletheiaResearch__GLM-5.2-Agent` | 3 | 4 | 3 | 7 | pass | 3/3 | pass |
| 13 | `Crownelius__Complete-FABLE.5-traces-2M:50029a5fae5ca9fc4766ad3b3d2fab11366ff140d692bafabffc454e35936188:0` | `Crownelius__Complete-FABLE.5-traces-2M` | 4 | 4 | 4 | 7 | pass | 4/4 | pass |
| 14 | `lambda__hermes-agent-reasoning-traces:007aa203-d6b0-4936-adeb-4e58fe0fe6b5:0` | `lambda__hermes-agent-reasoning-traces` | 2 | 17 | 17 | 4 | pass | 2/2 | pass |
| 15 | `AletheiaResearch__GLM-5.2-Agent:019ee94a-ebfe-70de-9d96-1e2de598d5fc:240` | `AletheiaResearch__GLM-5.2-Agent` | 3 | 2 | 2 | 7 | pass | 3/3 | pass |
| 16 | `Crownelius__Complete-FABLE.5-traces-2M:d16afda113f515d99fb646f001cd78c3d53ba1a412ff197bececac5e7fa3e31f:0` | `Crownelius__Complete-FABLE.5-traces-2M` | 4 | 4 | 4 | 8 | pass | 4/4 | pass |
| 17 | `lambda__hermes-agent-reasoning-traces:896d8974-eef0-4038-9d0e-85aea6eabc2a:0` | `lambda__hermes-agent-reasoning-traces` | 15 | 17 | 15 | 5 | pass | 15/15 | pass |
| 18 | `Crownelius__Complete-FABLE.5-traces-2M:1168fc07e57829b879b2dccc7ab0708963cb38d21c6275c1d1f888ab0267dbca:0` | `Crownelius__Complete-FABLE.5-traces-2M` | 4 | 4 | 4 | 8 | pass | 4/4 | pass |
| 19 | `lambda__hermes-agent-reasoning-traces:f28f00b9-4923-4452-ae23-11686f6be554:0` | `lambda__hermes-agent-reasoning-traces` | 2 | 17 | 16 | 3 | pass | 2/2 | pass |
| 20 | `Crownelius__Complete-FABLE.5-traces-2M:739b407b4e5edfebc6caac56665b690798c0fc006bab3de233956b4517a92c7b:0` | `Crownelius__Complete-FABLE.5-traces-2M` | 4 | 4 | 4 | 8 | pass | 4/4 | pass |
