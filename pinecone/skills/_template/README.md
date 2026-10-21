# 新增技能

1. 复制本目录到 `plugins/` 下，目录名随意（不要以下划线开头）：

   ```bash
   cp -r pinecone/skills/_template plugins/hello
   ```

2. 修改 `skill.py`：`name`、`description`、`patterns`、`handle`，可选 `tools`。
3. 可选：在 `config/config.yaml` 中添加私有配置：

   ```yaml
   skills:
     settings:
       hello:
         greeting: 你好，我是小派
   ```

4. 重启程序，日志中出现 `已加载技能：... hello` 即生效。

## 约定

- 技能只依赖 `SkillContext`（`ctx`），不要直接导入音频、对话等底层模块。
- `handle` 要尽快返回；耗时工作用 `ctx.spawn(...)` 放到后台托管，这样用户可以说
  “取消XX”“停止所有任务”终止它。
- 需要清理的资源写在 `try/finally` 或 `on_cancel` 中。
- 正则匹配的是归一化文本（无标点、无空格、英文小写）。
