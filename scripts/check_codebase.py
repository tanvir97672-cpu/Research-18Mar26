from __future__ import annotations

import compileall
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MODULES = [
    "rffi.config",
    "rffi.data.iq_dataset",
    "rffi.models.backbones",
    "rffi.models.jrffp_sc_plus",
    "rffi.train_loop",
]


def main() -> None:
    report: dict[str, object] = {}
    compile_ok = True
    compile_ok = compile_ok and compileall.compile_dir(str(ROOT / "src"), force=True, quiet=1)
    compile_ok = compile_ok and compileall.compile_dir(str(ROOT / "scripts"), force=True, quiet=1)
    report["compileall_ok"] = bool(compile_ok)

    imports = {}
    for module_name in MODULES:
        try:
            importlib.import_module(module_name)
            imports[module_name] = "ok"
        except Exception as ex:
            imports[module_name] = f"error: {ex}"
    report["imports"] = imports

    failed_imports = [k for k, v in imports.items() if not str(v).startswith("ok")]
    report["overall_ok"] = bool(compile_ok) and len(failed_imports) == 0

    print(json.dumps(report, indent=2))
    if not report["overall_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
