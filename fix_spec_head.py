from pathlib import Path
import re

path = Path("SPEC.md")
content = path.read_text()

# Remove the bullet point at the top if it's there
if content.startswith("- Removed `ET.indent`"):
    content = content.split("\n", 1)[1]
    path.write_text(content)
    print("Fixed SPEC.md top bullet point")
