"""语音合成（TTS）。

- ``base.py``：引擎接口 + 带降级的组合引擎 + 工厂函数；
- ``edge.py`` / ``local.py``：具体引擎；
- ``text.py``：分句与朗读前的文本清洗；
- ``speaker.py``：面向业务的“说话”接口（流式分句合成 + 播放 + 可取消）。
"""

from pinecone.speech.tts.base import TextToSpeech, create_tts
from pinecone.speech.tts.speaker import Speaker

__all__ = ["Speaker", "TextToSpeech", "create_tts"]
