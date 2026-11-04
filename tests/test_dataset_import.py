"""The repo's dataset loader must win over unrelated installed datasets packages."""
from pathlib import Path
import subprocess
import sys


def test_repo_dataset_package_takes_precedence(tmp_path):
    root = Path(__file__).resolve().parents[1]
    external = tmp_path / "site-packages"
    package = external / "datasets"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("SOURCE = 'unrelated installed package'\n", encoding="utf-8")
    script = """
from pathlib import Path
import sys
root, external = map(Path, sys.argv[1:])
sys.path[:0] = [str(root), str(external)]
import datasets
assert Path(datasets.__file__).resolve() == (root / 'datasets' / '__init__.py').resolve()
from datasets.dataset_Myops import Myops_dataset, RandomGenerator
assert Myops_dataset.__module__ == 'datasets.dataset_Myops'
assert RandomGenerator.__module__ == 'datasets.dataset_Myops'
print('Local dataset import succeeded despite an installed name collision')
"""
    result = subprocess.run([sys.executable, "-B", "-c", script, str(root), str(external)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
