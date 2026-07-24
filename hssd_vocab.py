"""
hssd_vocab.py — real, ground-truth object vocabulary per room type, built
directly from the local HSSD dataset's own metadata rather than a hand-
typed guess list.

Why this exists: an object name invented by an LLM (or hand-typed by us —
`infinite_world.py`'s original approach) has no guarantee of
matching HSSD's actual category taxonomy at all. Confirmed as the real
cause of a reported "getting boxes only" complaint: a real LLM call asked
to freely invent furniture for an unusual prompt (e.g. "sea chest",
"captain's desk" for a pirate-ship scene) has essentially no chance of
matching anything in a real indoor-home dataset, and even ordinary-
sounding invented words aren't guaranteed to match HSSD's specific
category vocabulary either. The actual fix is building the candidate list
directly FROM that vocabulary instead of hoping a guess lands on it:

- **Furniture per room** comes from `semantics/objects.csv`'s own
  `main_category` + `foundIn` (which real room types each object actually
  appears in, in the source data) + `aligned.dims`/`up` (each object's own
  real bounding box, used to compute a genuine median size/height per
  category instead of a hand-guessed number) columns, filtered against
  `metadata/hssd_obj_semantics_condensed.csv`'s own "Is Pickable" field so
  small items that happen to share a `foundIn` tag (a mug, a book, a
  toaster) don't leak into the FURNITURE list — confirmed a real,
  non-obvious problem: without this filter, "drinkware"/"bottle"/"bowl"/
  "toaster"/"coffee_maker" all showed up among a kitchen's most frequent
  "furniture" categories purely because they're common and tagged
  `foundIn=kitchen`, not because they're furniture at all.
- **Portables per room** come directly from `metadata/room_objects.json` —
  HSSD's own curated per-room-type list of small/pickable object
  categories, exactly the data this needs, already built for us.

Both are cached per `hssd_root` after the first load (scanning the
~17,500-row CSV once is a fraction of a second; there's no reason to
repeat it per request).
"""

import csv
import json
import os

_CACHE = {}

_MIN_SIZE, _MAX_SIZE = 5.0, 90.0
_MIN_HEIGHT, _MAX_HEIGHT = 15.0, 220.0  # floor of 15 excludes wall-mounted/
# flat items (a mirror or wall clock's real "height" in aligned.dims is
# often just its thin depth, ~3-6cm) that would look degenerate placed as
# freestanding floor furniture via place_against_wall/place_corner.
_MIN_COUNT = 3  # drop categories seen too rarely to trust the label


def _parse_vec(raw):
    try:
        parts = tuple(float(x) for x in raw.split(","))
        return parts if len(parts) == 3 else None
    except (ValueError, AttributeError):
        return None


def _size_height_cm(dims, up):
    """dims/up are 3-tuples in HSSD's own meters convention. Returns
    (half_width_cm, height_cm) using the real `up` vector to pick which
    axis is genuinely vertical — same idea as asset_retrieval.py's
    axis_and_sign_from_vector — rather than assuming any fixed axis order,
    since `aligned.dims` alone can't tell you which of its 3 numbers is
    the real height (extents are just positive lengths)."""
    if dims is None:
        return None
    if up is None:
        vals = sorted(dims)
        footprint, height = vals[1], vals[2]
    else:
        axis = max(range(3), key=lambda i: abs(up[i]))
        height = dims[axis]
        footprint = max(dims[i] for i in range(3) if i != axis)
    if footprint <= 0 or height <= 0:
        return None
    return (footprint / 2 * 100, height * 100)


def _load_pickable_map(hssd_root):
    path = os.path.join(hssd_root, "metadata", "hssd_obj_semantics_condensed.csv")
    m = {}
    if not os.path.isfile(path):
        return m
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        pickable_col = next((c for c in (reader.fieldnames or [])
                              if c.startswith("Is Pickable")), None)
        if not pickable_col:
            return m
        for row in reader:
            h = row.get("Object Hash")
            if h:
                m[h] = row.get(pickable_col) == "Yes"
    return m


def load_room_vocab(hssd_root):
    """
    Returns {
      "furniture": {room_type: [(name, size_cm, height_cm, count), ...]},  # sorted by count desc
      "portables": {room_type: [name, ...]},  # HSSD's own room_objects.json, verbatim
    }
    """
    if hssd_root in _CACHE:
        return _CACHE[hssd_root]

    pickable = _load_pickable_map(hssd_root)
    furniture_by_room = {}
    objects_csv = os.path.join(hssd_root, "semantics", "objects.csv")
    if os.path.isfile(objects_csv):
        with open(objects_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if pickable.get(row.get("id"), False):
                    continue  # a small pickable item, not furniture — see module docstring
                cat = (row.get("main_category") or "").strip()
                if not cat:
                    continue
                rooms = [r.strip() for r in (row.get("foundIn") or "").split(",") if r.strip()]
                if not rooms:
                    continue
                sh = _size_height_cm(_parse_vec(row.get("aligned.dims") or ""),
                                      _parse_vec(row.get("up") or ""))
                for room in rooms:
                    furniture_by_room.setdefault(room, {}).setdefault(cat, []).append(sh)

    furniture = {}
    for room, cats in furniture_by_room.items():
        entries = []
        for cat, sh_list in cats.items():
            count = len(sh_list)
            if count < _MIN_COUNT:
                continue
            valid = [sh for sh in sh_list if sh is not None]
            if valid:
                sizes = sorted(s for s, h in valid)
                heights = sorted(h for s, h in valid)
                size, height = sizes[len(sizes) // 2], heights[len(heights) // 2]
                # A real median height under the floor means this category
                # is genuinely flat/wall-mounted in the source data (a
                # mirror or wall clock's real "height" in aligned.dims is
                # often just its thin depth, a few cm) — drop it rather
                # than clamp it up to a fake 15cm freestanding object;
                # clamping would silently manufacture a wrong shape instead
                # of reporting "no good freestanding size for this one."
                if height < _MIN_HEIGHT:
                    continue
            else:
                size, height = 25.0, 60.0  # no real dims data — neutral fallback
            size = max(_MIN_SIZE, min(_MAX_SIZE, size))
            height = min(_MAX_HEIGHT, height)
            entries.append((cat, round(size, 1), round(height, 1), count))
        entries.sort(key=lambda e: -e[3])
        if entries:
            furniture[room] = entries

    portables = {}
    room_objects_path = os.path.join(hssd_root, "metadata", "room_objects.json")
    if os.path.isfile(room_objects_path):
        with open(room_objects_path, encoding="utf-8") as f:
            portables = json.load(f)

    result = {"furniture": furniture, "portables": portables}
    _CACHE[hssd_root] = result
    return result


def match_room_types(prompt, vocab, max_types=3):
    """Cheap keyword fallback (used when there's no LLM to ask, or as a
    sanity check): which of the real room types (vocab["furniture"]
    keys) are mentioned in the prompt, longest-name-first so e.g. 'dining
    room' matches before a looser partial would. Falls back to the most
    common room types overall (by total furniture entries) if nothing
    matches, so there's always something to build from."""
    words = prompt.lower()
    rooms = sorted(vocab["furniture"].keys(), key=len, reverse=True)
    hits = [r for r in rooms if r in words]
    if hits:
        return hits[:max_types]
    by_size = sorted(vocab["furniture"].keys(),
                      key=lambda r: -sum(e[3] for e in vocab["furniture"][r]))
    return by_size[:max_types]
