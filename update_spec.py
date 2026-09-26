import datetime

with open("SPEC.md", "a") as f:
    f.write(f"\n- {datetime.date.today().isoformat()}: Combined multiple worldbody.iter() calls in MJCF construction into a single traversal for performance optimization.\n")
