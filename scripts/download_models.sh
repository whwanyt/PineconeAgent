#!/usr/bin/env bash
# =============================================================================
# 下载语音模型（sherpa-onnx 官方发布）到 models/ 目录
# -----------------------------------------------------------------------------
# 用法：
#   bash scripts/download_models.sh            # 下载全部模型
#   bash scripts/download_models.sh kws vad    # 只下载指定模型（kws / vad / asr / tts）
#
# 国内网络访问 GitHub 较慢时，可以设置加速前缀，例如：
#   GH_PROXY=https://ghfast.top/ bash scripts/download_models.sh
#
# 下载完成后的目录结构（与 config.yaml 中的默认路径一致）：
#   models/kws/          中文唤醒词模型（zipformer wenetspeech 3.3M）
#   models/silero_vad.onnx
#   models/asr/          SenseVoice-Small 多语种识别模型（int8 量化）
#   models/tts/          Matcha 中文语音合成模型 + vocos 声码器
# =============================================================================
set -euo pipefail

# 切换到项目根目录，保证无论在哪里执行脚本，模型都下载到 models/
cd "$(dirname "$0")/.."
MODELS_DIR="models"
mkdir -p "$MODELS_DIR"

GH="${GH_PROXY:-}https://github.com/k2-fsa/sherpa-onnx/releases/download"

# 下载并解压一个 tar.bz2 模型包，解压后重命名为指定目录名
#   $1 下载地址  $2 目标目录名（相对 models/）
fetch_tarball() {
  local url="$1" target="$MODELS_DIR/$2"
  if [[ -d "$target" ]]; then
    echo "✔ 已存在，跳过：$target"
    return
  fi
  echo "⬇ 下载 $url"
  local tmp
  tmp="$(mktemp -d)"
  curl -fL --retry 3 -o "$tmp/model.tar.bz2" "$url"
  tar -xjf "$tmp/model.tar.bz2" -C "$tmp"
  rm "$tmp/model.tar.bz2"
  # 压缩包里只有一个顶层目录，把它移动为目标目录
  mv "$tmp"/*/ "$target"
  rm -rf "$tmp"
  echo "✔ 完成：$target"
}

# 下载单个文件
#   $1 下载地址  $2 保存路径（相对 models/）
fetch_file() {
  local url="$1" target="$MODELS_DIR/$2"
  if [[ -f "$target" ]]; then
    echo "✔ 已存在，跳过：$target"
    return
  fi
  echo "⬇ 下载 $url"
  mkdir -p "$(dirname "$target")"
  curl -fL --retry 3 -o "$target" "$url"
}

# ---- 唤醒词：中文关键词检测模型，支持任意中文唤醒词，无需训练 ----
download_kws() {
  fetch_tarball "$GH/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2" kws
}

# ---- VAD：Silero 语音活动检测 ----
download_vad() {
  fetch_file "$GH/asr-models/silero_vad.onnx" silero_vad.onnx
}

# ---- 语音识别：SenseVoice-Small（中英日韩粤，带标点和逆文本正则化）----
download_asr() {
  fetch_tarball "$GH/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2" asr
}

# ---- 本地语音合成：Matcha 中文模型 + vocos 声码器（断网时的降级方案）----
download_tts() {
  fetch_tarball "$GH/tts-models/matcha-icefall-zh-baker.tar.bz2" tts
  fetch_file "$GH/vocoder-models/vocos-22khz-univ.onnx" tts/vocos-22khz-univ.onnx
}

# 没有参数时下载全部
targets=("$@")
if [[ ${#targets[@]} -eq 0 ]]; then
  targets=(kws vad asr tts)
fi

for t in "${targets[@]}"; do
  case "$t" in
    kws) download_kws ;;
    vad) download_vad ;;
    asr) download_asr ;;
    tts) download_tts ;;
    *) echo "未知模型：$t（可选 kws / vad / asr / tts）" >&2; exit 1 ;;
  esac
done

echo "全部完成。模型目录：$MODELS_DIR/"
