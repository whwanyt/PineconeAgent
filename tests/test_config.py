"""配置加载的单元测试。"""

from pathlib import Path

import pytest

from pinecone.core.config import deep_merge, load_config


def test_deep_merge_nested() -> None:
    """嵌套字典逐层合并，列表整体覆盖。"""
    base = {"a": {"b": 1, "c": 2}, "l": [1, 2]}
    override = {"a": {"c": 3}, "l": [9]}
    assert deep_merge(base, override) == {"a": {"b": 1, "c": 3}, "l": [9]}


def test_local_override_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """config.local.yaml 覆盖主配置，并能展开环境变量和默认值。"""
    (tmp_path / "config.yaml").write_text("name: 主配置\nlog:\n  level: INFO\n", encoding="utf-8")
    (tmp_path / "config.local.yaml").write_text(
        "name: ${SPEAKER_NAME}\ndata_dir: ${MISSING:fallback}\n", encoding="utf-8"
    )
    monkeypatch.setenv("SPEAKER_NAME", "本地名字")

    cfg = load_config(tmp_path / "config.yaml")

    assert cfg.name == "本地名字"
    assert cfg.data_dir == "fallback"
    assert cfg.log.level == "INFO"


def test_missing_file_uses_defaults(tmp_path: Path) -> None:
    """配置文件不存在时全部使用默认值。"""
    cfg = load_config(tmp_path / "nope.yaml")
    assert cfg.name == "小派"
