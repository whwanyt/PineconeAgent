"""配置加载与校验。

职责：
    1. 读取 ``config/config.yaml``（主配置，入库）；
    2. 如果存在 ``config/config.local.yaml``（本地覆盖，git 忽略），深度合并覆盖主配置，
       用来放 API Key、设备名这类因机器而异的配置；
    3. 展开 ``${ENV_NAME}`` / ``${ENV_NAME:默认值}`` 形式的环境变量引用；
    4. 用 pydantic 校验并转换成强类型的 :class:`AppConfig`。

其他模块只接收各自那一段子配置（例如 ``AudioConfig``），不直接读文件，
这样单元测试时可以直接构造配置对象。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

# 项目根目录：pinecone/core/config.py 往上三级
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
LOCAL_CONFIG_NAME = "config.local.yaml"

# 匹配 ${NAME} 或 ${NAME:default}
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


# =============================================================================
# 各模块的子配置
# -----------------------------------------------------------------------------
# 每个模块一个 BaseModel，全部字段都有默认值，因此 config.yaml 可以只写需要修改的部分。
# =============================================================================


class LogConfig(BaseModel):
    """日志配置。"""

    level: str = "INFO"
    """日志级别：DEBUG / INFO / WARNING / ERROR。"""
    file: str | None = "logs/pinecone.log"
    """日志文件路径（相对项目根目录）；设为 null 只输出到控制台。"""
    max_bytes: int = 5 * 1024 * 1024
    """单个日志文件的最大字节数，超过后滚动。"""
    backup_count: int = 3
    """保留的历史日志文件个数。"""


class AecConfig(BaseModel):
    """回声消除（AEC）配置。"""

    mode: Literal["none", "software", "pipewire", "hardware"] = "software"
    """回声消除模式：
    - none：不做回声消除，只能用唤醒词打断（播报期间会提高唤醒阈值）；
    - software：进程内软件回声消除（WebRTC APM / speexdsp），无需额外硬件；
    - pipewire：系统级 module-echo-cancel，需把输入/输出设备设为 Echo-Cancel 虚拟设备；
    - hardware：麦克风阵列自带 AEC（如 ReSpeaker USB Mic Array v2.0），程序不做处理。
    """
    backend: Literal["auto", "webrtc", "speex"] = "auto"
    """software 模式的算法后端：auto 优先 webrtc（pip 包 livekit），不可用时用 speex。"""
    stream_delay_ms: int = 60
    """扬声器播放到麦克风采集之间的大致延迟（毫秒），作为 WebRTC AEC 的初始估计。"""
    filter_ms: int = 200
    """speex 自适应滤波器长度（毫秒），需覆盖回声路径的延迟与混响尾巴。"""
    noise_suppression: bool = True
    """webrtc 后端是否同时开启降噪。"""


class BargeInConfig(BaseModel):
    """对话打断配置。"""

    wake_word: bool = True
    """播报 / 思考时说唤醒词即可打断（任何 AEC 模式下都可用）。"""
    vad: bool = False
    """播报时直接开口即可打断。必须在回声消除生效时才会启用，否则会被自己的声音打断。"""
    wake_threshold_boost: float = Field(0.15, ge=0.0, le=0.7)
    """播报期间唤醒阈值的提高量，防止扬声器声音误唤醒。"""


class AudioConfig(BaseModel):
    """音频输入输出配置。"""

    input_device: str | int | None = None
    """麦克风：None 表示系统默认；整数为设备序号；字符串为设备名关键字（模糊匹配）。"""
    output_device: str | int | None = None
    """扬声器：取值规则同 input_device。"""
    sample_rate: int = 16000
    """送给语音模型的采样率。唤醒词 / VAD / 识别模型都要求 16kHz，一般不要修改。"""
    frame_ms: int = 20
    """每帧时长（毫秒）。越小延迟越低、CPU 开销越大；10/20/30 均可。"""
    volume: float = Field(0.8, ge=0.0, le=1.0)
    """语音播报音量（0~1）。"""
    aec: AecConfig = Field(default_factory=AecConfig)
    barge_in: BargeInConfig = Field(default_factory=BargeInConfig)


class WakeConfig(BaseModel):
    """唤醒词配置（sherpa-onnx 关键词检测）。"""

    model_dir: str = "models/kws"
    """唤醒模型目录（scripts/download_models.sh kws 下载）。"""
    keywords: list[str] = Field(default_factory=lambda: ["小派小派"])
    """唤醒词列表，任意中文均可，建议 3~4 个字、发音清晰、不常出现在日常对话中。"""
    threshold: float = Field(0.25, gt=0.0, lt=1.0)
    """触发阈值，越大越不容易误唤醒、但也越难唤醒。"""
    score: float = 1.5
    """唤醒词路径加分，越大越容易唤醒。"""
    num_threads: int = 1
    """推理线程数。唤醒模型很小，1 个线程足够。"""
    use_int8: bool = True
    """是否使用 int8 量化模型（树莓派上更快，精度几乎无损）。"""


class VadConfig(BaseModel):
    """语音活动检测（Silero VAD）配置。"""

    model: str = "models/silero_vad.onnx"
    """VAD 模型路径。"""
    threshold: float = Field(0.5, gt=0.0, lt=1.0)
    """语音概率阈值，环境嘈杂时可调高到 0.6~0.7。"""
    min_silence_ms: int = 700
    """静音持续多久认为一句话说完（毫秒）。太短会把句中停顿截断，太长响应变慢。"""
    min_speech_ms: int = 250
    """短于该时长的声音不算说话（过滤咳嗽、敲击声）。"""
    max_speech_s: float = 15.0
    """单句最长时长（秒），超过强制截断。"""
    no_speech_timeout_s: float = 6.0
    """唤醒后多久没开口就放弃本次聆听（秒）。"""


class AsrConfig(BaseModel):
    """语音识别配置。"""

    engine: Literal["sensevoice"] = "sensevoice"
    """识别引擎。目前提供本地 SenseVoice；新增云端引擎时在 speech/asr/ 下加实现并扩展此处。"""
    model_dir: str = "models/asr"
    """模型目录（scripts/download_models.sh asr 下载）。"""
    language: Literal["auto", "zh", "en", "ja", "ko", "yue"] = "auto"
    """识别语言，auto 自动判断。只说普通话时设为 zh 略微更准。"""
    use_itn: bool = True
    """逆文本正则化：把“七点半”输出为“7点半”、加标点，便于后续解析时间。"""
    num_threads: int = 2
    """推理线程数。树莓派 5 有 4 核，2 线程时 5 秒语音约 0.3 秒识别完。"""


class TtsConfig(BaseModel):
    """语音合成配置。"""

    engine: Literal["edge", "local"] = "edge"
    """主引擎：edge（微软在线语音，免费、音色自然，需联网）/ local（本地 Matcha，离线）。"""
    fallback: Literal["local", "none"] = "local"
    """主引擎失败（如断网）时的降级引擎。"""
    edge_voice: str = "zh-CN-XiaoxiaoNeural"
    """edge 音色。可选如 zh-CN-XiaoyiNeural、zh-CN-YunxiNeural，`edge-tts --list-voices` 查看全部。"""
    edge_rate: str = "+0%"
    """edge 语速，例如 "+10%" 更快、"-10%" 更慢。"""
    edge_timeout_s: float = 4.0
    """edge 合成超时（秒），超时后改用降级引擎，避免断网时长时间无响应。"""
    local_model_dir: str = "models/tts"
    """本地模型目录（scripts/download_models.sh tts 下载）。"""
    local_speed: float = 1.0
    """本地合成语速，>1 更快。"""
    num_threads: int = 2
    """本地合成推理线程数。"""
    cache_size: int = 64
    """缓存最近合成过的短句（如“好的”“已为你设置闹钟”），重复播报时零延迟。"""


class DialogConfig(BaseModel):
    """对话流程配置。"""

    follow_up_s: float = 6.0
    """播报结束后的追问窗口（秒），窗口内无需唤醒词可直接继续说话；0 表示关闭。"""
    wake_beep: bool = True
    """唤醒成功时播放提示音。"""
    wake_reply: str = ""
    """唤醒成功时的语音应答（如“我在”），为空则只播提示音。"""


class SkillsConfig(BaseModel):
    """技能配置。"""

    enabled: list[str] = Field(default_factory=lambda: ["*"])
    """启用的技能名列表，"*" 表示全部（内置 + plugins/ 下的插件）。"""
    disabled: list[str] = Field(default_factory=list)
    """禁用的技能名列表（优先级高于 enabled）。"""
    plugin_dirs: list[str] = Field(default_factory=lambda: ["plugins"])
    """插件技能目录，目录下每个子目录（含 __init__.py）是一个技能包。"""
    settings: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """各技能的私有配置，键为技能名，例如 settings.music.library: ~/Music。"""


class AppConfig(BaseModel):
    """整个应用的配置根对象。"""

    name: str = "小派"
    """音箱的名字，用于自我介绍和 AI 人设。"""
    data_dir: str = "data"
    """运行时数据目录（SQLite 数据库等），相对项目根目录。"""
    log: LogConfig = Field(default_factory=LogConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    wake: WakeConfig = Field(default_factory=WakeConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    asr: AsrConfig = Field(default_factory=AsrConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    dialog: DialogConfig = Field(default_factory=DialogConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)

    def resolve_path(self, path: str | Path) -> Path:
        """把配置中的相对路径解析为基于项目根目录的绝对路径。

        Args:
            path: 配置中填写的路径，可以是绝对路径或相对项目根目录的路径。

        Returns:
            绝对路径。
        """
        p = Path(path).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def data_path(self) -> Path:
        """数据目录的绝对路径（自动创建）。"""
        p = self.resolve_path(self.data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


# =============================================================================
# 加载逻辑
# =============================================================================


def expand_env(value: Any) -> Any:
    """递归展开配置中的环境变量引用。

    只处理字符串；dict / list 会递归处理。未设置且没有默认值的变量会被替换成空字符串，
    这样缺少 API Key 时程序依然能启动（只是 AI 功能不可用），而不是直接崩溃。
    """
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """深度合并两个字典，``override`` 中的值优先。

    嵌套字典会逐层合并；列表和标量直接整体覆盖（合并列表往往不是用户想要的语义）。

    Args:
        base: 基础配置。
        override: 覆盖配置。

    Returns:
        合并后的新字典，不修改入参。
    """
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 文件，空文件返回空字典。"""
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是映射（字典）：{path}")
    return data


def load_config(path: str | Path | None = None) -> AppConfig:
    """加载并校验配置。

    查找顺序：参数 ``path`` > 环境变量 ``PINECONE_CONFIG`` > ``config/config.yaml``。
    同目录下的 ``config.local.yaml`` 会被深度合并到主配置之上。

    Args:
        path: 主配置文件路径，可选。

    Returns:
        校验后的 :class:`AppConfig`。

    Raises:
        pydantic.ValidationError: 配置字段类型或取值不合法。
        ValueError: YAML 顶层不是字典。
    """
    cfg_path = Path(path or os.environ.get("PINECONE_CONFIG") or DEFAULT_CONFIG_PATH)
    raw: dict[str, Any] = _read_yaml(cfg_path) if cfg_path.exists() else {}

    local_path = cfg_path.parent / LOCAL_CONFIG_NAME
    if local_path.exists():
        raw = deep_merge(raw, _read_yaml(local_path))

    return AppConfig.model_validate(expand_env(raw))
