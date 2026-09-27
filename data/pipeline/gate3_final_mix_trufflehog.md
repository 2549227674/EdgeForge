# 门③最终 mix TruffleHog 复核

扫描范围为实际送入训练模板的 messages 与 tools 载荷字段；UID、provenance、source hash 和文件路径均不是训练 token，明确排除。扫描 JSON 与投影载荷保留在忽略路径，本报告只保留检测器和计数。

- 扫描候选：215；验证为真：0；均按高风险候选处理。
- 精确候选值已替换：3,850 处，覆盖 361 条规范池记录。
- 替换占位符：`<REDACTED_TRUFFLEHOG_CANDIDATE>`；原始候选值未写入版本化产物。
- 已重建投影并复扫：0 个候选。门③最终训练输入安全检查通过。

| 检测器 | 扫描候选 | 实际替换 |
|---|---:|---:|
| Atlassian | 1 | 10 |
| Box | 3 | 3 |
| CloudflareApiToken | 5 | 10 |
| Dockerhub | 150 | 2,197 |
| Eraser | 1 | 37 |
| Gitlab | 10 | 148 |
| GoogleGeminiAPIKey | 1 | 1 |
| Mockaroo | 2 | 1 |
| NpmToken | 30 | 918 |
| Shortcut | 1 | 15 |
| SlackWebhook | 1 | 1 |
| Stripe | 1 | 1 |
| TLy | 1 | 1 |
| TravisCI | 1 | 493 |
| URI | 4 | 4 |
| UnifyID | 1 | 1 |
| YoutubeApiKey | 2 | 9 |
