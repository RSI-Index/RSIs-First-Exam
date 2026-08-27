"""Helpers for editing Apptainer ``container.def`` ``%files`` sections."""
from __future__ import annotations

from typing import List


def inject_files_section(def_text: str, files_section: str) -> str:
    """Insert a ``%files`` block immediately before the first ``%post`` line."""
    files_section = files_section.rstrip() + "\n"
    lines = def_text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.lstrip().lower().startswith("%post"):
            return "".join(lines[:i]) + files_section + "\n" + "".join(lines[i:])
    return def_text.rstrip() + "\n\n" + files_section


def replace_apptainer_files_section(def_text: str, new_files_section: str) -> str:
    """Swap the first ``%files`` block for ``new_files_section`` (or inject if none)."""
    new_block = new_files_section.rstrip() + "\n\n"
    lines = def_text.splitlines(keepends=True)
    result: List[str] = []
    i = 0
    replaced = False
    while i < len(lines):
        line = lines[i]
        if line.lstrip().lower().startswith("%files"):
            if not replaced:
                result.append(new_block)
                replaced = True
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("%"):
                i += 1
            continue
        result.append(line)
        i += 1
    if replaced:
        return "".join(result)
    return inject_files_section(def_text, new_files_section)
