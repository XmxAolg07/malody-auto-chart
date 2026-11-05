import json
from pathlib import Path

root = Path(".screen/neural")
index_path = root / "index_combined.json"
data = json.loads(index_path.read_text(encoding="utf-8"))
keep, removed = [], []
for record in data:
    name = str(record.get("file", ""))
    if name.startswith("download_") and int(record.get("frames", 0)) < 1500:
        removed.append(name)
        npz = root / f"{name}.npz"
        if npz.exists():
            npz.unlink()
        continue
    keep.append(record)
index_path.write_text(json.dumps(keep, ensure_ascii=False, indent=1), encoding="utf-8")
print("removed", len(removed), "total", len(keep))
