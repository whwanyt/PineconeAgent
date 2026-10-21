"""技能模板包。复制整个目录到 plugins/ 下后，这里的导入会触发技能注册。"""

from .skill import HelloSkill

__all__ = ["HelloSkill"]
