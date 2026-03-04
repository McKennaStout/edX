from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================
# CONFIG
# ============================================================
DOWNLOADS_DIR = Path(r"C:\Users\mstout\Downloads")
COURSE_PREFIX = "OMSA_ISyE6501"
OUTPUT_SUFFIX = "_ALL.txt"

# If True, inserts a short header before each source file's content.
ADD_FILE_BOUNDARIES = True


# ============================================================
# Parsing + sorting
# ============================================================
# Matches:
#   OMSA_ISyE6501_M1L1_IntroAnalyticsModeling_refresh_-en.txt
#   OMSA_ISyE6501_M2L1_IntroClassification_refresh_ver-en.txt
#
# Captures:
#   module = 1
#   lecture = 1
#   rest = everything after _M{module}L{lecture}_
NAME_RE = re.compile(
    rf"^(?P<prefix>{re.escape(COURSE_PREFIX)})_M(?P<module>\d+)L(?P<lecture>\d+)(?P<rest>_.*)?\.txt$",
    re.IGNORECASE,
)

def parse_module_lecture(filename: str) -> Tuple[int, int] | None:
    m = NAME_RE.match(filename)
    if not m:
        return None
    return int(m.group("module")), int(m.group("lecture"))

def group_files_by_module(folder: Path) -> Dict[int, List[Path]]:
    groups: Dict[int, List[Path]] = {}
    for p in folder.glob(f"{COURSE_PREFIX}_M*.txt"):
        parsed = parse_module_lecture(p.name)
        if not parsed:
            continue
        module, _lecture = parsed
        groups.setdefault(module, []).append(p)
    return groups

def sort_key(p: Path) -> Tuple[int, int, str]:
    parsed = parse_module_lecture(p.name)
    if not parsed:
        return (10**9, 10**9, p.name.lower())
    module, lecture = parsed
    return (module, lecture, p.name.lower())


# ============================================================
# Combine
# ============================================================
def combine_module_files(module: int, files: List[Path], out_dir: Path) -> Path:
    out_path = out_dir / f"{COURSE_PREFIX}_M{module}{OUTPUT_SUFFIX}"

    files_sorted = sorted(files, key=sort_key)

    with out_path.open("w", encoding="utf-8", newline="\n") as out_f:
        for i, src in enumerate(files_sorted, start=1):
            if ADD_FILE_BOUNDARIES:
                out_f.write(f"\n===== BEGIN FILE {i}/{len(files_sorted)}: {src.name} =====\n\n")

            # Read as-is; write as-is. No deletion, no filtering.
            text = src.read_text(encoding="utf-8", errors="replace")
            out_f.write(text)

            # Ensure a trailing newline between files
            if not text.endswith("\n"):
                out_f.write("\n")

            if ADD_FILE_BOUNDARIES:
                out_f.write(f"\n===== END FILE {i}/{len(files_sorted)}: {src.name} =====\n")

    return out_path


def main() -> None:
    if not DOWNLOADS_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {DOWNLOADS_DIR}")

    groups = group_files_by_module(DOWNLOADS_DIR)
    if not groups:
        print(f"No matching files found in: {DOWNLOADS_DIR}")
        return

    for module in sorted(groups.keys()):
        out_path = combine_module_files(module, groups[module], DOWNLOADS_DIR)
        print(f"[OK] Module M{module}: wrote {out_path.name} from {len(groups[module])} files")

    print("Done.")


if __name__ == "__main__":
    main()