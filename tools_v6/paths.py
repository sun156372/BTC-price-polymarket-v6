"""Locate the offline dataset, once, for every V6 tool.

    from paths import resolve_data_root
    root = resolve_data_root(args.root)

Resolution order, first hit wins:

1. an explicit ``--root`` argument
2. the ``V6_DATA_ROOT`` environment variable (``V4_DATA_ROOT`` also honoured, so an
   existing setup keeps working)
3. candidates around the package: ``../micro_v3``, ``../../micro_v3``, ``./micro_v3``
   and the same for ``data``

Nothing is cached, so relocating the data mid-session is picked up immediately. When
nothing is found the error lists every path tried and says how to fix it, because a
tool that starts, finds nothing and prints "no data" wastes far more time than one
that refuses to start.
"""
from __future__ import annotations

import os

PKG = os.path.dirname(os.path.abspath(__file__))          # tools_v6
V6_SRC = os.path.dirname(PKG)                             # v6_src
RESEARCH = os.path.dirname(V6_SRC)                        # research tree
WORKSPACE = os.path.dirname(RESEARCH)                     # holds micro_v3

DEFAULT_NAMES = ("micro_v3", "data")


def candidates(name: str = "micro_v3") -> list[str]:
    out = []
    for variable in ("V6_DATA_ROOT", "V4_DATA_ROOT"):
        value = os.environ.get(variable)
        if value:
            out.append(value)
    for base in (RESEARCH, WORKSPACE, V6_SRC):
        out.append(os.path.join(base, name))
    return out


def resolve_data_root(explicit: str | None = None, name: str = "micro_v3",
                      required: bool = True) -> str | None:
    """Return the dataset root, or None when ``required`` is False and none exists."""
    if explicit:
        if os.path.isdir(explicit):
            return explicit
        if required:
            raise SystemExit(
                f"--root {explicit} is not a directory.\n"
                f"Point it at the extracted dataset, e.g. "
                f"--root <workspace>/{name}")
        return None
    for path in candidates(name):
        if path and os.path.isdir(path):
            return path
    if not required:
        return None
    tried = "\n  ".join(candidates(name))
    raise SystemExit(
        "Cannot find the offline dataset. Looked in:\n  "
        f"{tried}\n\n"
        "Fix it with either:\n"
        f"  --root <path to the extracted dataset>\n"
        f"  set V6_DATA_ROOT=<path>\n\n"
        "The live paper loop (run_live_v6.py) needs no local dataset.")


def describe() -> str:
    found = resolve_data_root(required=False)
    if found:
        return f"data root: {found}"
    return ("data root: NOT FOUND - offline tools need --root or V6_DATA_ROOT; "
            "the live loop does not")
