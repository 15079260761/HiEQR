import ast
from pathlib import Path

from decgc.config import load_config


def test_configs_use_portable_project_root():
    source = Path(__file__).parents[1]
    yaml_text = "\n".join(path.read_text(encoding="utf-8") for path in (source / "configs").rglob("*.yaml"))
    assert 'root: "."' in yaml_text
    assert "/root/autodl-tmp/DECGC" not in yaml_text
    windows_drive_marker = chr(68) + chr(58)
    assert windows_drive_marker not in yaml_text


def test_merge_interpolation_and_override(config):
    assert config.paths.dataset_root == config.project.root / "data/MultiTQ"
    assert config.model_extra["demo"]["train_max_samples"] == 8


def test_cli_set_option_does_not_shadow_builtin_set():
    cli_path = Path(__file__).parents[1] / "decgc/cli.py"
    tree = ast.parse(cli_path.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert all(argument.arg != "set" for function in functions for argument in function.args.args)
