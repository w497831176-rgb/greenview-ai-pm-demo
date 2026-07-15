"""Static syntax gate for frontend/index.html.

Extracts every inline <script> block (including type="module"), writes module
blocks to a temporary .mjs file, and runs `node --check`. Non-module blocks are
checked as classic scripts via a temporary .js file. Temp files are deleted
after the check so they are never committed.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HTML_PATH = REPO_ROOT / "frontend" / "index.html"


def check_block(code: str, is_module: bool) -> int:
    suffix = ".mjs" if is_module else ".js"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=suffix, delete=False, dir=REPO_ROOT
    ) as tmp:
        tmp.write(code)
        tmp_path = Path(tmp.name)
    try:
        result = subprocess.run(
            ["node", "--check", str(tmp_path)], capture_output=True, text=True
        )
        if result.returncode != 0:
            label = "module" if is_module else "classic"
            print(f"Syntax error in {label} script:")
            print(result.stdout)
            print(result.stderr)
        return result.returncode
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> int:
    html = HTML_PATH.read_text(encoding="utf-8")
    # Match <script ...>...</script>; allow attributes in any order.
    script_blocks = re.findall(r"<script([^>]*)>(.*?)</script>", html, re.DOTALL | re.IGNORECASE)
    if not script_blocks:
        print(f"No inline scripts found in {HTML_PATH}")
        return 1

    exit_code = 0
    for attrs, code in script_blocks:
        # Skip external scripts (they have a src attribute).
        if re.search(r'\bsrc\s*=\s*["\']', attrs, re.IGNORECASE):
            continue
        is_module = bool(re.search(r'\btype\s*=\s*["\']module["\']', attrs, re.IGNORECASE))
        block_exit = check_block(code, is_module)
        if block_exit != 0:
            exit_code = block_exit

    if exit_code == 0:
        print("All inline scripts pass node --check")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
