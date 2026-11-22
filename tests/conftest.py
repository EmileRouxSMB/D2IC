from __future__ import annotations

import sys
from pathlib import Path

# Ensure the src-layout package is importable without requiring an editable install.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
IM_JAX_ROOT = ROOT.parent / "im_jax"
for package_root in (SRC, IM_JAX_ROOT):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
