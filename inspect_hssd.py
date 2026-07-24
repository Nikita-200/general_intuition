"""
inspect_hssd.py — dumps the actual structure of your HSSD download's
metadata/semantics files, so if HSSDRetriever's category matching still
isn't finding things on your machine, you (or whoever's fixing it) can see
exactly what's on disk instead of guessing blind.

Usage:
    python inspect_hssd.py --root ./hssd-hab

Run this any time HSSDRetriever prints "couldn't parse a handle->category
mapping" or "zero categories resolved" — paste this script's output back
to whoever's debugging it (or use it yourself: HSSDRetriever._parse_semantic_lexicon
in asset_retrieval.py is the single place that would need a matching
branch added for whatever shape this reveals).
"""

import argparse
import csv
import json
import os


def _preview_file(path, fn):
    """Shared preview for both metadata/ and semantics/ — the original
    version of this script only did this for files under metadata/, which
    meant semantics/objects.csv (a real, sizeable file in at least one
    confirmed real download) never got shown at all. Uses csv.DictReader
    rather than raw line-splitting for .csv files: a plain readline-based
    preview mis-renders any column whose header contains an embedded
    newline inside quotes (a real, confirmed case: hssd_obj_semantics_
    condensed.csv's header wraps across multiple lines) — DictReader
    parses that correctly and reports the real column names, which is
    exactly what asset_retrieval.py's id_col/cat_col matching cares about.
    """
    if fn.endswith(".csv"):
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                print(f"    columns ({len(fieldnames)}): {fieldnames}")
                for i, row in zip(range(2), reader):
                    preview = {k: (v[:60] + "…" if isinstance(v, str) and len(v) > 60 else v)
                               for k, v in row.items()}
                    print(f"    row {i}: {preview}")
        except Exception as e:
            print(f"    (failed to parse as CSV: {e})")
    elif fn.endswith(".json"):
        try:
            with open(path) as f:
                data = json.load(f)
            print(_shape(data, depth=2))
        except Exception as e:
            print(f"    (failed to parse as JSON: {e})")


def _shape(data, depth=0, max_depth=5):
    indent = "  " * depth
    if depth >= max_depth:
        return f"{indent}...(truncated at depth {max_depth})"
    if isinstance(data, dict):
        lines = [f"{indent}dict, {len(data)} keys: {list(data.keys())[:10]}"]
        for k in list(data.keys())[:3]:
            lines.append(f"{indent}  [{k!r}] ->")
            lines.append(_shape(data[k], depth + 1, max_depth))
        return "\n".join(lines)
    if isinstance(data, list):
        lines = [f"{indent}list, {len(data)} items"]
        if data:
            lines.append(f"{indent}  first item ->")
            lines.append(_shape(data[0], depth + 1, max_depth))
        return "\n".join(lines)
    return f"{indent}{type(data).__name__}: {data!r:.150s}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./hssd-hab", help="path to your HSSD download")
    args = ap.parse_args()
    root = args.root

    print(f"Inspecting {root!r}\n")

    if not os.path.isdir(root):
        print(f"'{root}' doesn't exist or isn't a directory. Nothing to inspect.")
        return

    print("=== top-level contents ===")
    for name in sorted(os.listdir(root)):
        print(f"  {name}")

    print("\n=== metadata/ directory ===")
    meta_dir = os.path.join(root, "metadata")
    if os.path.isdir(meta_dir):
        for fn in sorted(os.listdir(meta_dir)):
            path = os.path.join(meta_dir, fn)
            print(f"  {fn} ({os.path.getsize(path)} bytes)")
            _preview_file(path, fn)
    else:
        print("  (not found — did you download it? see download_hssd.py, which now "
              "fetches metadata/* by default)")

    print("\n=== semantics/ directory ===")
    sem_dir = os.path.join(root, "semantics")
    if os.path.isdir(sem_dir):
        for fn in sorted(os.listdir(sem_dir)):
            path = os.path.join(sem_dir, fn)
            if os.path.isfile(path):
                print(f"  {fn} ({os.path.getsize(path)} bytes)")
                _preview_file(path, fn)
            else:
                print(f"  {fn}/ (directory)")
    else:
        print("  (not found — did you download it? see download_hssd.py, which now "
              "fetches semantics/* by default)")

    print("\n=== one sample object_config.json (from objects/) ===")
    objects_dir = os.path.join(root, "objects")
    sample_cfg = None
    if os.path.isdir(objects_dir):
        for dirpath, _dirnames, filenames in os.walk(objects_dir):
            for fn in filenames:
                if fn.endswith(".object_config.json"):
                    sample_cfg = os.path.join(dirpath, fn)
                    break
            if sample_cfg:
                break
    if sample_cfg:
        print(f"  {sample_cfg}")
        with open(sample_cfg) as f:
            data = json.load(f)
        print(_shape(data, depth=2))
    else:
        print("  (no *.object_config.json found under objects/ — did the download "
              "actually complete?)")

    print("\nDone. If HSSDRetriever still isn't matching things after seeing this, "
          "the fix is almost always a small addition to "
          "HSSDRetriever._parse_semantic_lexicon or ._parse_metadata_table in "
          "asset_retrieval.py, guided by whatever this printed above.")


if __name__ == "__main__":
    main()
