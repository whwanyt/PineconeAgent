"""自定义任务技能：把 ``tasks/*.yaml`` 接入技能体系（规则路由 + AI 工具）。"""

from .skill import CustomTaskSkill

__all__ = ["CustomTaskSkill"]
