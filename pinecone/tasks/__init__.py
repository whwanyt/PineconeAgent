"""自定义任务引擎。

用户在项目根目录的 ``tasks/`` 下放 YAML 文件，就能把“关键词 → 一串动作”接入音箱，
不需要写 Python。例如说“打开客厅灯”就调用 Home Assistant 的接口，再播报“好的”。

模块分工：
    - :mod:`.model`   任务 YAML 的数据结构与校验（pydantic）；
    - :mod:`.loader`  扫描 ``tasks/*.yaml``，校验后生成 :class:`TaskSpec` 列表；
    - :mod:`.runner`  顺序执行一个任务的动作，处理变量替换；
    - :mod:`.actions` 动作类型（shell / http / python / speak / sleep），用 ``@register_action`` 扩展。

任务如何被触发、如何后台运行、如何被取消，见 ``skills/builtin/custom_task``。
"""

from .loader import load_tasks
from .model import ActionSpec, TaskSpec
from .runner import TaskRunner

__all__ = ["ActionSpec", "TaskRunner", "TaskSpec", "load_tasks"]
