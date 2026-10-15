"""Generates JSON test fixtures from data/cases.csv (stdlib only)."""
import csv
import json
import pathlib

out = pathlib.Path("tests/fixtures")
out.mkdir(parents=True, exist_ok=True)
for i, row in enumerate(csv.DictReader(open("data/cases.csv")), 1):
    (out / f"case_{i:03d}.json").write_text(json.dumps(row))
    print(f"generated fixture case_{i:03d}.json ... ok")
