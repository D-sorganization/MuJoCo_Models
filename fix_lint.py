with open("src/mujoco_models/shared/utils/mjcf_helpers.py", "rb") as f:
    content = f.read().decode('utf-8')

content = content.replace(
    "# Avoiding ET.indent() beforehand saves a full O(N) tree traversal pass just to add whitespace.",
    "# Avoiding ET.indent() saves a full O(N) tree traversal pass just to add whitespace."
)

with open("src/mujoco_models/shared/utils/mjcf_helpers.py", "wb") as f:
    f.write(content.encode('utf-8'))
