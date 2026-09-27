# M1 配方赛马中止事实记录

- 日期：2026-08-11
- 裁决来源：owner override
- 状态：**赛马中止，未产生赢家**

## 已发生事实

- raw-uniform 正式臂启动后完成初始 holdout，并训练到 15/1,024 step 后被取消。
- 初始 holdout：1,948 窗口，sample mean loss `1.886102917247`，SE `0.0220918051393`，token-weighted loss `1.63021391393`。
- raw-uniform 仅保留部分 `train_log.jsonl`；没有正式 checkpoint、没有最终 report，不得恢复为赛马结果或用于胜负判断。
- family-80/20 赛马臂尚未正式启动即被取消。
- 原先“raw-uniform vs family-80/20 各 1/8”的 D-1 判据没有执行完毕，不能声称 family-80/20 是统计赢家。

## 后续裁决

owner 直接指定 family-80/20 作为 C01-S 配方，并进一步批准把训练预算从 296,850,623 token 降为 1/2，即 148,425,311 token。该选择是项目决策，不是赛马结论。

云端中止材料保留在：

- `results/t1/race_branch2/race_raw_uniform_r1/train_log.jsonl`
- 旧 1/8 计划：`results/t1/recipe_plans_branch2_r1/`

上述材料只作审计，不属于当前训练输入。
