# 研究来源与原始记录

本目录保存讨论所依据的原始资料和环境检查输出。当前项目定义及路线选择见上层文档；原稿保留当时方案，后续修订见 [讨论与决策](../DECISIONS.md)。

## 飞书文档

- 标题：KDA 有限记忆的选择性维护：研究问题、算法方案与实验计划。
- 来源：[飞书原文](https://bytedance.sg.larkoffice.com/docx/AipDdksJ4oS2jXxW5rUlwZm8gJg)。
- 文档 ID：`AipDdksJ4oS2jXxW5rUlwZm8gJg`；本次读取 revision：`12`。
- 原稿整理日期：2026 年 10 月 12 日；本次归档日期：2026 年 10 月 13 日。
- [完整 Markdown 快照](feishu-kda-rev12.md)保留原稿的 13 个章节，文件末尾换行允许规范化。
- [来源与校验元数据](feishu-kda-rev12.metadata.json)记录读取命令、UTC 时间、版本与 SHA-256。

使用已鉴权的 bytedcli，以用户身份只读获取：

```bash
bytedcli --json lark docs fetch --as user \
  --doc https://bytedance.sg.larkoffice.com/docx/AipDdksJ4oS2jXxW5rUlwZm8gJg \
  --scope full --detail simple --doc-format markdown
```

本次操作通过 bytedcli MCP 的 `run_command` 执行相同命令参数，没有修改线上飞书文档。后续如原文更新，保留此快照并另存新 revision，不覆盖历史。

## 环境原始输出

[2026 年 10 月 13 日远端检查](environment-20261013.json)包含只读命令、检查时间、退出码、GPU 查询、占卡 PID 查询和共享原稿文件的 SHA-256。

该输出只说明检查时刻的状态。当前 GPU 可用性、PID 和利用率会变化，实验前应重新检查。

## 证据使用

原稿中的 A/B/C 提升幅度、容量目标和 GPU 小时为研究假设或暂定目标，不是 SteadyMem 已有实验结果。原稿中的文献评价也需要结合后来补充的 MoM 等工作理解，详见 [文献评估](../LITERATURE.md)及 [技术路线](../ROUTES.md)。

当前项目记录是讨论整理，不是逐字聊天转录。正式结论、提出的候选与开放问题均在对应专题文档中标明。
