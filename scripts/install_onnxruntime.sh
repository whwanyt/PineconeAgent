#!/usr/bin/env bash
# =============================================================================
# 为 sherpa-onnx 补上匹配版本的 libonnxruntime.so
# -----------------------------------------------------------------------------
# sherpa-onnx 的 Linux wheel 只有约 4MB，不打包推理库。扩展模块的 NEEDED 项是
# 不带版本号的 libonnxruntime.so，并且符号版本必须一致（1.13.8 是 VERS_1.28.2）。
# 树莓派系统包是 libonnxruntime.so.1.21，文件名和 ABI 都对不上，直接做软链接会在
# 导入时报 undefined symbol。
#
# 动态库放到扩展模块旁边：该 .so 的 RPATH 含 $ORIGIN，加载时会先搜这个目录，
# 不必改 LD_LIBRARY_PATH，也不会抢系统里其它程序的 onnxruntime。
# uv sync 会覆盖 site-packages，所以本脚本必须在 uv sync 之后再跑一遍。
#
# 用法（在项目根目录，或任意目录）：
#   bash scripts/install_onnxruntime.sh
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -x .venv/bin/python ]]; then
  echo "未找到 .venv，请先运行 bash scripts/install_pi.sh" >&2
  exit 1
fi

shopt -s nullglob
candidates=(.venv/lib/python*/site-packages/sherpa_onnx/lib/_sherpa_onnx*.so)
shopt -u nullglob
if ((${#candidates[@]} == 0)); then
  echo "未找到 sherpa-onnx 扩展模块，请先安装 Python 依赖" >&2
  exit 1
fi
so="${candidates[0]}"
dest="$(dirname "$so")"

# 已经能导入就不用再下载。导入失败若不是缺这个库，原样把错误抛出。
if .venv/bin/python -c "import sherpa_onnx" >/dev/null 2>"/tmp/pinecone-sherpa-import.err"; then
  echo "sherpa-onnx 可以导入，跳过 onnxruntime"
  exit 0
fi
import_err="$(cat /tmp/pinecone-sherpa-import.err)"
if [[ "$import_err" != *"libonnxruntime"* && "$import_err" != *"VERS_"* ]]; then
  echo "$import_err" >&2
  echo "sherpa-onnx 导入失败，但不是缺少 libonnxruntime.so" >&2
  exit 1
fi

# 从扩展模块的未定义符号里读出版本，例如 VERS_1.28.2。多个版本说明 wheel 对不上。
mapfile -t vers < <(grep -a -o 'VERS_[0-9][0-9.]*' "$so" | sed 's/^VERS_//' | sort -u)
if ((${#vers[@]} != 1)); then
  echo "无法从 sherpa-onnx 确定唯一的 onnxruntime 版本：${vers[*]:-无}" >&2
  exit 1
fi
ver="${vers[0]}"

case "$(uname -m)" in
  aarch64 | arm64) platform="linux-aarch64" ;;
  x86_64 | amd64) platform="linux-x64" ;;
  *)
    echo "没有预编译包：$(uname -m) 的 onnxruntime ${ver}" >&2
    exit 1
    ;;
esac

# 与 sherpa-onnx v1.13.8 的 cmake/onnxruntime-*.cmake 使用同一份 Release 包。
# 新版本若还没写进这张表，仍会下载，只是跳过校验。
declare -A known_sha256=(
  ["linux-aarch64-1.28.2"]="28083273643f40fa6477ef9d53986fe216304710c4b1c743b217a4effd016339"
  ["linux-x64-1.28.2"]="c4f8994d56191d9d2c92a961b39fe790459f2c5d155f912b239506ea31359534"
)

name="onnxruntime-${platform}-glibc2_17-Release-${ver}.zip"
url="https://github.com/csukuangfj/onnxruntime-libs/releases/download/v${ver}/${name}"
cache=".cache/onnxruntime"
mkdir -p "$cache"
zip="$cache/$name"

# 直连 GitHub Release 在国内经常只有几 KB/s。按顺序尝试，慢于约 8KB/s 持续 30 秒就换下一个。
# 已设置 http_proxy / https_proxy 时 curl 会自动走代理。
if [[ ! -f "$zip" ]]; then
  echo "下载 onnxruntime ${ver}（${platform}）"
  rm -f "${zip}.partial"
  downloaded=0
  for candidate in \
    "$url" \
    "https://gh-proxy.com/${url}" \
    "https://gh.nxnow.top/${url}" \
    "https://gh.catmak.name/${url}"
  do
    echo "尝试 ${candidate}"
    if curl -fL --retry 2 --retry-delay 1 --connect-timeout 15 \
      --speed-time 30 --speed-limit 8000 \
      -o "${zip}.partial" "$candidate"
    then
      mv "${zip}.partial" "$zip"
      downloaded=1
      break
    fi
    rm -f "${zip}.partial"
    echo "这个地址失败或太慢，换下一个"
  done
  if [[ "$downloaded" -ne 1 ]]; then
    echo "onnxruntime ${ver} 下载失败" >&2
    exit 1
  fi
fi

expected="${known_sha256[${platform}-${ver}]:-}"
if [[ -n "$expected" ]]; then
  echo "${expected}  ${zip}" | sha256sum -c -
else
  echo "未内置 onnxruntime ${ver} 的校验值，下载后直接安装"
fi

tmp="$(mktemp -d)"
cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT
# 用标准库解压，避免再装 unzip
.venv/bin/python -m zipfile -e "$zip" "$tmp"

copied=0
while IFS= read -r lib; do
  cp -a "$lib" "$dest/"
  copied=1
done < <(find "$tmp" -name 'libonnxruntime*.so*' \( -type f -o -type l \))
if [[ "$copied" -eq 0 ]]; then
  echo "压缩包里没有 libonnxruntime.so：${url}" >&2
  exit 1
fi

if ! .venv/bin/python -c "import sherpa_onnx" >/dev/null; then
  echo "已放入 ${dest}，但 sherpa-onnx 仍然无法导入。依赖：" >&2
  ldd "$so" || true
  exit 1
fi
echo "已安装 libonnxruntime.so ${ver} -> ${dest}"
