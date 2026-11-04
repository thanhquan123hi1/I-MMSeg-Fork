"""Exercise the actual test.py key-remapping loop with both checkpoint formats."""
import ast
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("prefix", ["", "module."])
def test_original_evaluator_loads_single_and_data_parallel_weights(prefix):
    tree = ast.parse((Path(__file__).resolve().parents[1] / "test.py").read_text(encoding="utf-8"))
    loop = next(node for node in ast.walk(tree)
                if isinstance(node, ast.For) and isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Attribute)
                and isinstance(node.iter.func.value, ast.Name)
                and node.iter.func.value.id == "checkpoint"
                and node.iter.func.attr == "items")
    reference = torch.nn.Linear(2, 3)
    checkpoint = OrderedDict((prefix + name, tensor.clone()) for name, tensor in reference.state_dict().items())
    keys_before = list(checkpoint)
    buffer = BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    scope = {"checkpoint": torch.load(buffer, weights_only=True), "new_state_dict": OrderedDict()}
    exec(compile(ast.Module(body=[loop], type_ignores=[]), "test.py checkpoint loader", "exec"), scope)
    restored = torch.nn.Linear(2, 3)
    restored.load_state_dict(scope["new_state_dict"], strict=True)
    assert list(scope["checkpoint"]) == keys_before
    for name, expected in reference.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)
    sample = torch.tensor([[0.25, 0.75]])
    torch.testing.assert_close(restored(sample), reference(sample), rtol=0, atol=0)
