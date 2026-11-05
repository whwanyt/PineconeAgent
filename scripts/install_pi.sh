#!/usr/bin/env bash
# =============================================================================
# 树莓派 5 一键安装脚本（Raspberry Pi OS Bookworm 64 位）
# -----------------------------------------------------------------------------
# 用法（在项目根目录或任意目录执行，不要用 sudo，脚本内部按需调用 sudo）：
#   bash scripts/install_pi.sh                    # 完整安装并注册开机自启
#   bash scripts/install_pi.sh --pipewire-aec     # 同时安装 PipeWire 回声消除
#   bash scripts/install_pi.sh --no-service       # 不注册 systemd 服务（手动运行）
#   bash scripts/install_pi.sh --no-models        # 跳过模型下载（已手动下载时）
#
# 国内网络：
#   GH_PROXY=https://ghfast.top/ bash scripts/install_pi.sh       # 加速 GitHub 模型下载
#   UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash scripts/install_pi.sh
#
# sherpa-onnx 的 Linux wheel 不含 libonnxruntime.so，脚本会按扩展模块的符号版本补上。
# 脚本可以重复执行：已完成的步骤会被跳过或安全覆盖。
# =============================================================================
set -euo pipefail

# ---- 参数解析 ----
WITH_PIPEWIRE_AEC=0
INSTALL_SERVICE=1
DOWNLOAD_MODELS=1
for arg in "$@"; do
  case "$arg" in
    --pipewire-aec) WITH_PIPEWIRE_AEC=1 ;;
    --no-service) INSTALL_SERVICE=0 ;;
    --no-models) DOWNLOAD_MODELS=0 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "未知参数：$arg（--help 查看用法）" >&2; exit 1 ;;
  esac
done

# 切换到项目根目录
cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd)"

step() { echo; echo "==== $* ===="; }

# ---- 0. 环境检查 ----
step "检查运行环境"
if [[ "$(id -u)" -eq 0 ]]; then
  echo "请不要用 sudo / root 运行本脚本：服务需要以普通用户身份运行才能使用 PipeWire 音频" >&2
  exit 1
fi
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "本脚本用于树莓派（Linux）。在电脑上开发请参考 README 的“开发环境”一节" >&2
  exit 1
fi
echo "系统：$(uname -sm)，用户：$USER，项目目录：$PROJECT_DIR"

# ---- 1. 系统依赖 ----
step "安装系统依赖（需要 sudo）"
# libportaudio2   sounddevice 访问声卡
# libmpv-dev mpv  音乐播放（python-mpv 需要 libmpv）
# pipewire-*      音频服务（Bookworm 默认已安装，这里确保齐全）
# alsa-utils      arecord / aplay / alsamixer 调试声卡
# rsync bzip2 curl 示例任务与模型下载
sudo apt-get update
sudo apt-get install -y \
  python3-dev libportaudio2 libmpv-dev mpv \
  pipewire pipewire-pulse wireplumber pipewire-alsa \
  alsa-utils rsync bzip2 curl git

# 把当前用户加入 audio 组，保证能直接访问声卡（重新登录后生效）
if ! id -nG "$USER" | grep -qw audio; then
  sudo usermod -aG audio "$USER"
  echo "已把 $USER 加入 audio 组（重新登录后生效）"
fi

# ---- 2. Python 环境（uv）----
step "安装 Python 依赖"
if ! command -v uv >/dev/null 2>&1; then
  echo "安装 uv（Python 包管理器）"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
# 优先安装 WebRTC 软件回声消除（livekit）；该平台没有对应的预编译包时退回 speexdsp
# --extra tui：SSH 前台调试时的全屏界面。systemd 没有终端，不画界面，只跑语音
if ! uv sync --extra aec --extra tui; then
  echo "livekit 安装失败，改用 speexdsp 作为软件回声消除"
  sudo apt-get install -y libspeexdsp-dev
  uv sync --extra aec-speex --extra tui
fi
# uv sync 会重装 sherpa-onnx，放在它后面，避免刚拷好的动态库被清掉
bash scripts/install_onnxruntime.sh

# ---- 3. 语音模型 ----
if [[ "$DOWNLOAD_MODELS" -eq 1 ]]; then
  step "下载语音模型（约 300MB）"
  bash scripts/download_models.sh
fi

# ---- 4. 本地配置与密钥模板 ----
step "生成本地配置"
# config.local.yaml 会覆盖 config.yaml，且不会提交到 git，适合放机器相关的设置
if [[ ! -f config/config.local.yaml ]]; then
  cat > config/config.local.yaml <<'YAML'
# 本机配置：覆盖 config/config.yaml 中的同名字段（不会提交到 git）
# 查看可用的音频设备：.venv/bin/python -m pinecone.audio.devices
audio:
  # 设备名关键字，例如 "USB"；null 为系统默认
  input_device: null
  output_device: null
YAML
  echo "已生成 config/config.local.yaml"
fi
# .env 存放密钥，由 systemd 服务的 EnvironmentFile 读取
if [[ ! -f .env ]]; then
  cat > .env <<'ENV'
# 大模型接口（OpenAI 兼容），不配置则只使用本地技能
OPENAI_API_KEY=
# OPENAI_BASE_URL=https://api.deepseek.com/v1
# OPENAI_MODEL=deepseek-chat
# Home Assistant（tasks/open_light.yaml 示例使用）
# HA_URL=http://homeassistant.local:8123
# HA_TOKEN=
ENV
  chmod 600 .env
  echo "已生成 .env（请填入 OPENAI_API_KEY）"
fi

# ---- 5. PipeWire 回声消除（可选）----
if [[ "$WITH_PIPEWIRE_AEC" -eq 1 ]]; then
  step "安装 PipeWire 回声消除"
  mkdir -p "$HOME/.config/pipewire/pipewire.conf.d"
  cp deploy/pipewire/echo-cancel.conf "$HOME/.config/pipewire/pipewire.conf.d/echo-cancel.conf"
  systemctl --user restart pipewire pipewire-pulse wireplumber || true
  sleep 2
  if pactl list short sources 2>/dev/null | grep -q echo-cancel-source; then
    echo "✔ Echo-Cancel 虚拟设备已就绪"
  else
    echo "⚠ 没有检测到 Echo-Cancel 设备，请检查：journalctl --user -u pipewire"
  fi
  cat <<'HINT'
请在 config/config.local.yaml 中加入：
  audio:
    input_device: "Echo-Cancel Source"
    output_device: "Echo-Cancel Sink"
    aec: {mode: pipewire}
  skills:
    settings:
      music: {audio_device: "pipewire/echo-cancel-sink"}
HINT
fi

# ---- 6. systemd 用户服务（开机自启）----
if [[ "$INSTALL_SERVICE" -eq 1 ]]; then
  step "注册开机自启服务"
  mkdir -p "$HOME/.config/systemd/user"
  sed "s#__PROJECT_DIR__#$PROJECT_DIR#g" deploy/pinecone.service \
    > "$HOME/.config/systemd/user/pinecone.service"
  # linger：不登录桌面 / SSH 也会在开机时启动用户服务
  sudo loginctl enable-linger "$USER"
  systemctl --user daemon-reload
  systemctl --user enable pinecone
  systemctl --user restart pinecone
  echo "服务已启动。查看日志：journalctl --user -u pinecone -f"
fi

step "安装完成"
cat <<EOF
下一步：
  1. 编辑 .env 填入 OPENAI_API_KEY（可选，不填只使用本地技能）
  2. 查看音频设备：.venv/bin/python -m pinecone.audio.devices
     必要时在 config/config.local.yaml 中指定 input_device / output_device
  3. 前台调试运行：.venv/bin/python -m pinecone.main --debug
     （先停止服务：systemctl --user stop pinecone）
  4. 说「小派小派」唤醒试试（唤醒词在 config.yaml 的 wake.keywords 中修改）
EOF
