# PineconeAgent

运行在 **树莓派 5（8G）** 上的智能音箱，类似小爱音箱：

- **语音唤醒**：自定义中文唤醒词（默认“小派小派”），完全离线，无需训练
- **常用技能**：询问时间日期、设置 / 查询 / 取消闹钟（支持重复、贪睡、重启不丢）、播放本地音乐和网络电台
- **自定义任务**：在 `tasks/` 放一个 YAML，“关键词 → shell / http / python / 播报”，不用写代码
- **对话打断**：播报过程中说唤醒词立即打断；回声消除生效时还可以开启 `audio.barge_in.vad`，直接开口打断
- **任务可终止**：“停止”“取消备份”“停止所有任务”“现在在做什么”，AI 也能调用
- **AI 对话**：LangChain v1 + LangGraph，OpenAI 兼容接口（DeepSeek / 通义 / Kimi / Ollama），
  流式按句播报，带短期记忆（会话 + 摘要压缩）和长期记忆（记住你的称呼和喜好），
  能通过工具调用全部技能和自定义任务，并可接入任意 MCP 服务

识别、唤醒、VAD、断网时的语音合成都在本地运行；只有 AI 对话和在线语音合成（edge-tts）需要联网。

## 工作流程

```mermaid
flowchart LR
    Mic[麦克风] --> AEC[回声消除] --> KWS[唤醒词] --> VAD[VAD 断句] --> ASR[SenseVoice 识别]
    ASR --> Router{意图路由}
    Router -->|自定义任务关键词| Task[custom_task]
    Router -->|停止 / 取消| Ctl[task_control]
    Router -->|技能正则| Skills[时间 / 闹钟 / 音乐 / 插件]
    Router -->|未命中| Agent[AI Agent<br/>工具 = 全部技能 + MCP]
    Task & Ctl & Skills & Agent --> TTS[edge-tts / 本地 Matcha] --> Spk[扬声器]
    Spk -.参考信号.-> AEC
```

所有“正在执行的事情”（技能、自定义任务、AI 推理、播报、音乐、闹钟响铃）都由
`TaskManager` 统一托管：唤醒打断只取消 **前台** 任务；音乐、备份这类 **后台** 任务
只有被点名取消或“停止所有任务”时才会停止。

## 硬件

| 项目 | 说明 |
| --- | --- |
| 主机 | 树莓派 5（4G 也能跑，8G 更从容），Raspberry Pi OS Bookworm 64 位 |
| 麦克风 | 任意 USB 麦克风 / 带麦的 USB 耳机或摄像头 |
| 扬声器 | USB 声卡、HDMI 显示器、蓝牙音箱均可（树莓派 5 没有 3.5mm 口） |
| 可选 | ReSpeaker USB Mic Array v2.0 等自带硬件回声消除的麦克风阵列 |

回声消除（AEC）是配置项，初期不买任何额外硬件也能用：

| `audio.aec.mode` | 说明 |
| --- | --- |
| `none` | 不做回声消除，只能用唤醒词打断（播报期间自动提高唤醒阈值防误唤醒） |
| `software`（默认） | 进程内 WebRTC APM（livekit），没有时退回 speexdsp；能消除播报的回声，不能消除 mpv 音乐的回声 |
| `pipewire` | 系统级 `module-echo-cancel`，能消除包括音乐在内的所有声音；`install_pi.sh --pipewire-aec` 一键配置 |
| `hardware` | 麦克风阵列自带 AEC，程序内跳过处理 |

## 在树莓派上安装

```bash
git clone <仓库地址> ~/PineconeAgent && cd ~/PineconeAgent

# 一键安装：系统依赖、Python 环境、语音模型、开机自启服务
bash scripts/install_pi.sh
# 国内网络加速：GH_PROXY=https://ghfast.top/ bash scripts/install_pi.sh
# 需要消除音乐回声：bash scripts/install_pi.sh --pipewire-aec

# 填入大模型密钥（可选，不填只使用本地技能）
nano .env
systemctl --user restart pinecone

# 查看日志
journalctl --user -u pinecone -f
```

音频设备不是系统默认时，先列出设备，再在 `config/config.local.yaml` 中按名称关键字指定：

```bash
.venv/bin/python -m pinecone.audio.devices
```

```yaml
audio:
  input_device: "USB"        # 设备名包含的关键字
  output_device: "USB Audio"
```

## 开发环境（Mac / Linux 电脑）

```bash
uv sync --extra dev --extra aec       # 安装依赖
bash scripts/download_models.sh        # 下载语音模型
uv run pytest                          # 运行测试
uv run ruff check . && uv run ruff format .

# 文本模式：终端输入代替说话，回复仍从扬声器播报，适合调试技能和 AI
OPENAI_API_KEY=sk-xxx uv run python -m pinecone.main --text
# 语音模式
uv run python -m pinecone.main --debug
```

音乐技能需要 libmpv：macOS `brew install mpv`，树莓派由安装脚本处理。

## 配置

- `config/config.yaml`：全部配置项，每一项都有中文注释和默认值；
- `config/config.local.yaml`：本机覆盖（设备名、密钥等），不提交到 git；
- 字符串中可以用 `${环境变量}` 或 `${环境变量:默认值}`，密钥建议放在 `.env` 中。

常用配置：

```yaml
wake:
  keywords: ["小派小派"]      # 唤醒词，任意中文，可以配置多个
  threshold: 0.25             # 越小越灵敏
tts:
  edge_voice: zh-CN-XiaoxiaoNeural
agent:
  base_url: https://api.deepseek.com/v1
  api_key: ${OPENAI_API_KEY:}
  model: deepseek-chat
  persona: "说话简短，偶尔幽默"
skills:
  disabled: [music]           # 禁用某个技能
  settings:
    music: {library: [~/Music]}
```

## 语音指令示例

| 功能 | 说法 |
| --- | --- |
| 时间 | “现在几点”“今天几号”“明天星期几” |
| 闹钟 | “明天早上7点叫我起床”“10分钟后提醒我关火”“每个工作日7点半的闹钟”“我有哪些闹钟”“取消明天7点的闹钟”“再睡五分钟” |
| 音乐 | “播放周杰伦的歌”“我想听晴天”“来首歌”“暂停”“下一首”“大声点”“这是什么歌”“停止播放” |
| 任务控制 | “停止”“别说了”“取消备份照片”“停止所有任务”“现在在做什么” |
| 自定义任务 | “打开客厅的灯”“开始备份”“你的IP是多少”（见 `tasks/`） |
| AI | 其他任何问题；“记住我喜欢听周杰伦”“我叫什么”“屋里太暗了”（AI 会调用开灯任务） |

## 扩展

### 自定义任务（不写代码）

在 `tasks/` 下新建 YAML，重启生效。完整字段见 [tasks/README.md](tasks/README.md)：

```yaml
name: 开灯
keywords: ["开灯"]
patterns: ["打开(?P<room>客厅|卧室)的?灯"]   # 命名分组可在动作中用 {{room}} 引用
background: false       # true：后台执行，不阻塞对话
timeout: 15             # 超时自动取消
actions:
  - type: http
    method: POST
    url: "${HA_URL}/api/services/light/turn_on"
    headers: {Authorization: "Bearer ${HA_TOKEN}"}
    json: {entity_id: "light.living_room"}
  - type: speak
    text: "好的，{{room}}灯已打开"
on_cancel:
  - type: speak
    text: "开灯已取消"
```

动作类型：`shell`（取消时终止整个进程组）、`http`、`python`（可注入取消令牌）、`speak`、`sleep`；
新增动作类型见 `pinecone/tasks/actions/`。

### 技能插件（写 Python）

```bash
cp -r pinecone/skills/_template plugins/hello
```

```python
@register_skill
class HelloSkill(Skill):
    name = "hello"
    description = "打招呼"
    patterns = {"greet": [r"你好(?P<who>.*)"]}  # 命名分组成为 intent.slots

    async def handle(self, ctx: SkillContext, intent: Intent) -> SkillResult:
        return SkillResult(reply=f"你好{intent.slot('who', '')}")

    def tools(self) -> list[ToolSpec]:  # 可选：同时作为 AI 工具
        return [ToolSpec(name="say_hello", description="打招呼", intent="greet")]
```

技能只依赖 `SkillContext`：`ctx.say()` 播报、`ctx.spawn()` 启动可被取消的后台任务、
`ctx.settings` 私有配置、`ctx.data_dir` 数据目录、`ctx.on_state_change()` 订阅状态。
详见 [pinecone/skills/_template/README.md](pinecone/skills/_template/README.md)。

### MCP 服务

```yaml
agent:
  mcp_servers:
    weather:
      transport: stdio
      command: uvx
      args: [mcp-weather-server]
```

启动时自动加载，工具名为 `服务名_工具名`；某个服务连不上只会告警，不影响其他功能。

## 目录结构

```
config/        配置文件（config.yaml + 本机覆盖 config.local.yaml）
tasks/         自定义任务，一个任务一个 YAML
plugins/       用户技能插件（复制 pinecone/skills/_template）
models/        语音模型（scripts/download_models.sh 下载，不提交）
data/          运行数据：闹钟、对话历史、长期记忆（SQLite，不提交）
scripts/       install_pi.sh 安装脚本、download_models.sh 模型下载
deploy/        systemd 服务、PipeWire 回声消除配置
pinecone/
  main.py app.py         入口与组装
  dialog.py              对话流程（状态机驱动：唤醒 → 聆听 → 思考 → 播报 → 追问）
  core/                  配置、事件总线、状态机、任务托管（TaskManager / CancelToken）
  audio/                 麦克风、扬声器、设备选择、回声消除（aec/）
  speech/                唤醒词（wake/）、VAD（vad/）、识别（asr/）、合成（tts/）
  nlu/                   文本归一化、中文时间解析、意图路由
  skills/                技能框架 + 内置技能（builtin/）+ 模板（_template/）
  tasks/                 自定义任务引擎（加载、动作注册表、执行器）
  agent/                 AI Agent：模型、提示词、记忆（memory/）、工具（tools/）
tests/         单元测试（pytest，假模型 / 假声卡，不需要硬件和网络）
```

## 常见问题

- **唤醒不灵敏 / 误唤醒**：调整 `wake.threshold`（越小越灵敏）和 `wake.score`；唤醒词建议 4 个字。
- **播放音乐时很难唤醒**：软件 AEC 拿不到 mpv 的声音，可改用 `--pipewire-aec`，或调低音乐音量。
- **断网**：识别、唤醒、技能、自定义任务照常工作；播报自动降级到本地 Matcha 合成；AI 会提示连不上。
- **任务停不下来**：shell 动作取消时会对整个进程组 SIGTERM，3 秒后 SIGKILL；
  python 动作中的同步函数需要检查 `token.cancelled`。
