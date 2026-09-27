# 门④b 人工复核

审核人：Codex（2026-08-06）。从 124 条预置样本中先纳入全部 5 条正则阳性，再按数据集轮转补足 15 条阴性样本，共 20 条；原文不复写入本报告。

结论：通过。5 条阳性均为真实质量问题，全部来自 Mythos：2 条以“完整实现将需 200+ LOC”代替实现，3 条虚构“internal Anthropic eval”统计，且同批样本另可见模型身份式前缀。这支持、但不新增 Mythos 因退化规则整集排除的既有决定。其余 15 条未见会改变处置的未检出身份自述、控制标记或桩代码；Crown/Glint 的本地路径随后由门③脱敏，Glint 的前缀展开和跨源重叠已由门②处理。

| # | UID | 数据集 | 正则命中 | 结论 |
|---:|---|---|---|---|
| 1 | `WithinUsAI__claude_mythos_distilled_25k:mythos-distilled-00001:0` | Mythos | stub | 真阳性；整集已排除 |
| 2 | `WithinUsAI__claude_mythos_distilled_25k:mythos-distilled-00010:0` | Mythos | invented_eval | 真阳性；整集已排除 |
| 3 | `WithinUsAI__claude_mythos_distilled_25k:mythos-distilled-00013:0` | Mythos | stub | 真阳性；整集已排除 |
| 4 | `WithinUsAI__claude_mythos_distilled_25k:mythos-distilled-00018:0` | Mythos | invented_eval | 真阳性；整集已排除 |
| 5 | `WithinUsAI__claude_mythos_distilled_25k:mythos-distilled-00019:0` | Mythos | invented_eval | 真阳性；整集已排除 |
| 6 | `AletheiaResearch__GLM-5.2-Agent:019ee172-2f9f-7341-8125-4e5d535d9e56:0` | GLM | — | 通过 |
| 7 | `Crownelius__Complete-FABLE.5-traces-2M:b4e6394ad9928c6f25c3bb2363e332718045add32a3e533114c192c2ceb38f1a:0` | Crown | — | 通过；路径由门③脱敏 |
| 8 | `Glint-Research__Fable-5-traces:f956721a-0af7-4bdc-8678-3a493d8fcd39:0` | Glint | — | 通过；前缀展开由门②折叠 |
| 9 | `WithinUsAI__claude_mythos_distilled_25k:mythos-distilled-00000:0` | Mythos | — | 整集已排除 |
| 10 | `armand0e__claude-opus-4.8-pi-traces:019e9f68-3075-7136-b429-c6b2c871ed67:0` | Opus pi | — | 通过；后续门②无保留记录 |
| 11 | `armand0e__qwen3.7-max-pi-traces:019e529f-a2f9-70da-ad9e-ee24c934c4a4:0` | Qwen | — | 通过 |
| 12 | `lambda__hermes-agent-reasoning-traces:0c699abf-bc77-454a-8197-d56a2294098a:0` | Hermes | — | 通过 |
| 13 | `AletheiaResearch__GLM-5.2-Agent:019ee173-6fff-7300-ab06-cf89fc97744b:1` | GLM | — | 通过 |
| 14 | `Crownelius__Complete-FABLE.5-traces-2M:c5f5a7b4487080da4d0c009940927264123a309a524fe6d24e453bfb9ba40b76:0` | Crown | — | 通过；路径由门③脱敏 |
| 15 | `Glint-Research__Fable-5-traces:f956721a-0af7-4bdc-8678-3a493d8fcd39:1` | Glint | — | 通过；前缀展开由门②折叠 |
| 16 | `WithinUsAI__claude_mythos_distilled_25k:mythos-distilled-00002:0` | Mythos | — | 整集已排除 |
| 17 | `armand0e__claude-opus-4.8-pi-traces:019e9f7c-683a-73c1-a3c8-b746ac4fac9c:1` | Opus pi | — | 通过；后续门②无保留记录 |
| 18 | `armand0e__qwen3.7-max-pi-traces:019e4e0d-903d-7033-99da-258aa42061bf:0` | Qwen | — | 通过 |
| 19 | `lambda__hermes-agent-reasoning-traces:67584368-7a5c-4d7a-b682-40ee44ff8c61:0` | Hermes | — | 通过 |
| 20 | `AletheiaResearch__GLM-5.2-Agent:019ee174-28a3-76b2-bae6-f6679665530c:2` | GLM | — | 通过 |
