import re
import subprocess
import sys
from pathlib import Path

html = Path("frontend/index.html").read_text(encoding="utf-8")
scripts = re.findall(r"<script>(.*?)</script>", html, re.DOTALL)
if not scripts:
    print("No inline script found")
    sys.exit(1)

js = scripts[0]
# node --check expects a file path; write to temp file.
temp = Path("frontend/index.js")
temp.write_text(js, encoding="utf-8")
result = subprocess.run(["node", "--check", str(temp)], capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
sys.exit(result.returncode)
