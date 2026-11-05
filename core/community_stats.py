"""Build community statistical priors from the 315 filtered charts.

Produces:
  * mask transition probabilities P(next mask | prev mask, chord size)
  * chord-size distribution
  * beat-spacing distribution
  * anchor-length distribution

These priors are style-agnostic and can be blended into the rule generator.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "community_clean" / "index.json"
OUT = ROOT / "assets" / "community_stats.json"
MIRROR = {0: 0, 1: 8, 2: 4, 3: 12, 4: 2, 5: 10, 6: 6, 7: 14,
          8: 1, 9: 9, 10: 5, 11: 13, 12: 3, 13: 11, 14: 7, 15: 15}


def _beat(value) -> float:
    return value[0] + value[1] / value[2]


def main() -> None:
    records = json.loads(INDEX.read_text(encoding="utf-8"))
    transitions = {}
    sizes = Counter()
    spacings = Counter()
    anchors = Counter()
    charts = 0
    for record in records:
        try:
            data = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = {}
        for note in data.get("note") or []:
            if isinstance(note, dict):
                beat_list, lane = note.get("beat"), note.get("column")
            else:
                beat_list, lane = note[0], note[1]
            if beat_list is None or lane is None:
                continue
            beat = round(_beat(beat_list) * 48) / 48
            rows[beat] = rows.get(beat, 0) | (1 << int(lane))
        keys = sorted(rows)
        if len(keys) < 10:
            continue
        charts += 1
        prev_mask = None
        prev_beat = None
        runs = {lane: 0 for lane in range(4)}
        for beat in keys:
            mask = rows[beat]
            sizes[mask.bit_count()] += 1
            if prev_beat is not None:
                gap = round((beat - prev_beat) * 48) / 48
                if 0 < gap <= 4:
                    spacings[gap] += 1
            if prev_mask is not None:
                for lane in range(4):
                    if mask & (1 << lane):
                        runs[lane] += 1
                        anchors[runs[lane]] += 1
                    else:
                        runs[lane] = 0
                size = mask.bit_count()
                for a, b in ((prev_mask, mask), (MIRROR[prev_mask], MIRROR[mask])):
                    key = f"{a}|{size}"
                    bucket = transitions.setdefault(key, {})
                    bucket[b] = bucket.get(b, 0) + 1
            prev_mask, prev_beat = mask, beat

    normalized = {}
    for key, bucket in transitions.items():
        total = sum(bucket.values())
        normalized[key] = {str(mask): count / total for mask, count in bucket.items()}
    size_total = sum(sizes.values())
    spacing_total = sum(spacings.values())
    anchor_total = sum(anchors.values())
    payload = {
        "charts": charts,
        "trans": normalized,
        "sizes": {str(k): v / size_total for k, v in sizes.items()},
        "spacings": {str(k): v / spacing_total for k, v in spacings.items()},
        "anchors": {str(k): v / anchor_total for k, v in anchors.items()},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print("charts", charts, "transitions", len(normalized), "saved", OUT)
    print("sizes", {k: round(v / size_total, 3) for k, v in sizes.items()})
    print("top spacings", [(k, round(v / spacing_total, 3)) for k, v in spacings.most_common(5)])
    print("anchor p95", sorted(anchors.elements())[int(anchor_total * 0.95)])


if __name__ == "__main__":
    main()
