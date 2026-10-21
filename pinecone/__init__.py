"""PineconeAgent：运行在树莓派 5 上的智能音箱。

整体由以下几层组成（自下而上）：

- ``core``：配置、日志、事件总线、状态机、任务托管（TaskManager）；
- ``audio``：麦克风采集、扬声器播放、回声消除（AEC）；
- ``speech``：唤醒词、语音活动检测（VAD）、语音识别（ASR）、语音合成（TTS）；
- ``skills`` / ``tasks``：内置技能与 YAML 自定义任务；
- ``agent``：基于 LangChain / LangGraph 的 AI 对话；
- ``app``：把以上模块组装成完整的语音交互流程。
"""

__version__ = "0.1.0"
