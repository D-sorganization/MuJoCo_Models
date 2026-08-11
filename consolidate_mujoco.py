import subprocess
import re
import os

branches = [
    ("340", "origin/bolt-optimize-weld-constraint-format-14615539126553769549"),
    ("341", "origin/jules/optimize-tuple-string-formatting-5679378907059453899"),
    ("345", "origin/bolt/optimize-relpose-formatting-3355454498572605647"),
    ("346", "origin/bolt-optimize-polygon-distance-16529009910224475553"),
    ("347", "origin/bolt-optimize-balance-cost-2756212974544875236"),
    ("348", "origin/bolt/optimize-mjcf-serialization-1684493969845787934"),
    ("349", "origin/feat/optimize-compute-balance-cost-12392362000669155053"),
    ("350", "origin/bolt-opt-balance-cost-16678318763549801902"),
]

def run(cmd, check=True):
    print(f"> {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and res.returncode != 0:
        print(f"STDOUT: {res.stdout}")
        print(f"STDERR: {res.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")
    return res

def resolve_conflict_file(filepath):
    if not os.path.exists(filepath):
        return
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    if "<<<<<<<" not in content:
        return
    lines = content.splitlines()
    new_lines = []
    in_conflict = False
    head_lines = []
    other_lines = []
    in_other = False
    
    for line in lines:
        if line.startswith("<<<<<<<"):
            in_conflict = True
            in_other = False
            head_lines = []
            other_lines = []
        elif line.startswith("======="):
            in_other = True
        elif line.startswith(">>>>>>>"):
            in_conflict = False
            # Deduplicate or combine
            combined = head_lines + [l for l in other_lines if l not in head_lines]
            new_lines.extend(combined)
        else:
            if in_conflict:
                if in_other:
                    other_lines.append(line)
                else:
                    head_lines.append(line)
            else:
                new_lines.append(line)
                
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")

for pr_num, branch in branches:
    print(f"\n--- Merging PR #{pr_num} ({branch}) ---")
    res = run(f"git merge --no-ff {branch} -m 'Merge PR #{pr_num} into consolidated batch'", check=False)
    if res.returncode != 0:
        print(f"Conflict encountered for PR #{pr_num}, resolving...")
        status = run("git diff --name-only --diff-filter=U", check=False)
        conflicts = status.stdout.strip().splitlines()
        for f in conflicts:
            print(f"Resolving conflict in: {f}")
            resolve_conflict_file(f)
        run("git add -A")
        run(f"git commit -m 'Merge PR #{pr_num} into consolidated batch (resolved conflicts)'")

print("\nAll PR branches merged successfully!")
