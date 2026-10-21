# PineconeAgent

运行在 **树莓派 5（8G）** 上的智能音箱，类似小爱音箱：

- 语音唤醒（自定义中文唤醒词，离线）
- 询问时间、设置闹钟、播放音乐
- 自定义关键词触发自定义任务（`tasks/*.yaml`，放进去就生效）
- 对话打断（播报过程中说唤醒词即可打断）
- 正在执行的任务可随时终止（语音“停止”“取消备份”“停止所有任务”）
- 接入 AI 对话（LangChain / LangGraph，OpenAI 兼容接口，带短期与长期记忆）

## 目录结构

```
config/      配置文件（config.yaml）
tasks/       自定义任务，一个任务一个 YAML
plugins/     用户自定义技能（复制 pinecone/skills/_template 即可）
models/      语音模型（scripts/download_models.sh 下载）
scripts/     安装与模型下载脚本
deploy/      systemd 服务文件
pinecone/    源代码
tests/       单元测试
```

## 开发环境快速开始

```bash
# 1. 安装依赖（推荐 uv）
uv sync --extra dev

# 2. 运行测试
uv run pytest
```
