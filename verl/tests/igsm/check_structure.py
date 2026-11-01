"""Check syntax and internal module references against the upstream manifest."""
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
manifest = json.loads((ROOT / "slimming-manifest.json").read_text())


def module_name(path):
    name = path[:-3].replace("/", ".")
    return name[:-9] if name.endswith(".__init__") else name


old_modules = {module_name(p) for p in manifest["retained"] + manifest["removed"] if p.startswith("verl/") and p.endswith(".py")}
new_modules = {module_name(p.relative_to(ROOT).as_posix()) for p in (ROOT / "verl").rglob("*.py")}
errors = []
count = 0
for path in ROOT.rglob("*.py"):
    if any(p in (".git", ".venv") for p in path.relative_to(ROOT).parts):
        continue
    relative = path.relative_to(ROOT).as_posix()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    count += 1
    name = module_name(relative)
    pkg = name if path.name == "__init__.py" else name.rpartition(".")[0]
    for node in ast.walk(tree):
        refs = []
        if isinstance(node, ast.Import):
            refs = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = pkg.split(".")
                base = ".".join(parts[:len(parts) - node.level + 1] + ([base] if base else []))
            refs = [base] + [base + "." + a.name for a in node.names]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and re.fullmatch(r"verl(?:\.[A-Za-z_]\w*)+", node.value):
            refs = [node.value]
        for ref in refs:
            if not ref.startswith("verl."):
                continue
            candidate = ref
            while candidate and candidate not in old_modules and candidate not in new_modules:
                candidate = candidate.rpartition(".")[0]
            if candidate in old_modules and candidate not in new_modules:
                errors.append(f"{relative}:{node.lineno}: deleted internal module {ref}")
for name, expected in manifest["unchanged_core_sha256"].items():
    actual = hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if actual != expected:
        errors.append(f"Core training source changed unexpectedly: {name}")
if errors:
    raise SystemExit("\n".join(errors))
print(f"STRUCTURE_OK python_files={count} internal_modules={len(new_modules)} unchanged_core={len(manifest['unchanged_core_sha256'])}")
