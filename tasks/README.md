# 自定义任务

在本目录放一个 YAML 文件，就能新增一个语音指令，不用写 Python。程序启动时加载，修改后重启生效。
文件名以 `_` 开头的会被忽略，可以用来临时禁用某个任务。

## 字段

| 字段 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- |
| `name` | 是 | | 任务名，用于“取消XX”和播报 |
| `description` | | `""` | 给 AI 看的说明 |
| `keywords` | | `[]` | 包含匹配的触发词 |
| `patterns` | | `[]` | 正则触发，命名分组成为变量 |
| `background` | | `false` | 是否后台执行 |
| `timeout` | | `30` | 超时秒数（包含 speak 动作的播报时间），`null` 不限 |
| `cancellable` | | `true` | 是否允许取消 |
| `reply` | | | 回复文本（后台任务启动时 / 前台任务完成时） |
| `done_reply` | | | 后台任务完成后的播报 |
| `expose_to_agent` | | `true` | 是否作为 AI 工具 |
| `actions` | 是 | | 顺序执行的动作 |
| `on_error` | | `[]` | 失败后执行的动作，变量 `{{error}}` |
| `on_cancel` | | `[]` | 取消 / 超时后执行的动作，变量 `{{reason}}` |

## 变量

动作参数中用 `{{变量名}}` 引用：`text`（用户原话）、正则命名分组、`output`（上一个动作的输出）。
`${ENV}` / `${ENV:默认值}` 在加载时展开为环境变量，适合放密钥。

## 动作类型

- `shell`：`command`、`cwd`、`env`、`grace_s`（取消时 SIGTERM 到 SIGKILL 的宽限期）、`check`
- `http`：`method`、`url`、`headers`、`params`、`json`、`data`、`timeout_s`、`result_path`
- `python`：`callable`（`模块:函数`，可放在本目录下）、`args`；函数有 `token` 参数时会注入取消令牌
- `speak`：`text`
- `sleep`：`seconds`

新增动作类型见 `pinecone/tasks/actions/__init__.py`。

## 取消

- 前台任务：说“停止”、唤醒打断、“取消<任务名>”都会取消；
- 后台任务：只有“取消<任务名>”或“停止所有任务”会取消；
- 关键词前面带“取消 / 停止 / 不要”时不会触发任务，例如“取消备份照片”只会取消，不会再启动一次。
