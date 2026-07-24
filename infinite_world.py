"""
infinite_world.py — deterministic, chunk-based "infinite corridor"
generation for the web app's Infinite mode (see webapp/server.py).

Design (confirmed with the user before building this): ONE real LLM call
up front turns the prompt into a THEME (which room categories/vocabulary
to draw from); after that, every individual chunk is generated
DETERMINISTICALLY — no LLM, no network, no per-chunk latency — reusing the
exact same DSL (`AddAsset`/`place_*`/`resolve_layout`) and HSSD asset
retrieval as everything else in this harness. This is also just how
procedural world generation actually works elsewhere (Minecraft's own
terrain isn't AI-authored per chunk either): creativity comes from the
prompt-derived theme, scale and continuity come from a fast deterministic
generator that can never stall waiting on a network call mid-stream.

A "chunk" is an ordinary `dsl3d.Scene3D(chunk_width, chunk_depth)` — same
`resolve_layout()`-guaranteed no-overlap furniture layout as any other
generated scene — that `chunk_to_json` then offsets in world space
(z += chunk_index * chunk_depth) so consecutive chunks tile into one
continuous corridor with no wall between them and no overall bounding
room at all. Deliberately imports `dsl3d.Scene3D` directly rather than
`physics_backend.Scene3D`: a chunk is static furniture-only content with
no agent and no need for real-time rigid-body simulation, so there's no
reason to pay PyBullet's per-instance physics-client connection cost
(and, worse, its cleanup burden — `dsl3d_pybullet.Scene3D.__init__` opens
a real client handle that only `.disconnect()` releases; a long-running
server generating many chunks over time with no explicit disconnect would
leak one client per chunk forever) for something that never touches
PyBullet-specific functionality at all.
"""

import json
import random
import re

import hssd_vocab
import viewer3d
from dsl3d import Scene3D
from synthesizer3d import _call_llm

_DEFAULT_PORTABLES = ["mug", "book", "key", "remote"]

# Ultimate fallback ONLY — used solely if the local HSSD dataset's own
# metadata can't be read at all (see derive_theme/hssd_vocab.load_room_vocab).
# The real, dataset-grounded path below replaced this as the primary
# source: an earlier version of this file asked the LLM to freely INVENT
# furniture names (or, before that, picked among these six hand-typed
# buckets) — reported back as "getting boxes only", because an invented
# name (or even a plausible-sounding hand-typed one) has no guarantee of
# matching HSSD's actual category taxonomy, so most of it silently fell
# back to plain colored primitives. `hssd_vocab.py` fixes this at the
# root: it builds the REAL candidate vocabulary directly from the local
# HSSD dataset's own metadata (which categories genuinely exist, per real
# room type, with genuine measured sizes), so anything selected from it is
# guaranteed retrievable.
_FALLBACK_LIBRARY = {
    "kitchen": {
        "furniture": [("counter", 45, 90), ("island", 55, 90), ("cabinet", 25, 90),
                      ("refrigerator", 32, 175), ("dishwasher", 30, 85), ("stove", 30, 90)],
        "portables": ["can", "mug", "apple", "bowl"],
    },
    "living room": {
        "furniture": [("sofa", 60, 75), ("coffee table", 32, 40), ("tv stand", 40, 45),
                      ("bookshelf", 22, 180), ("armchair", 30, 75), ("side table", 20, 55)],
        "portables": ["mug", "book", "remote", "ball"],
    },
    "bedroom": {
        "furniture": [("bed", 70, 50), ("nightstand", 20, 55), ("dresser", 30, 80),
                      ("wardrobe", 35, 200)],
        "portables": ["book", "key", "mug"],
    },
    "home office": {
        "furniture": [("desk", 38, 75), ("bookshelf", 22, 180), ("cabinet", 25, 90),
                      ("side table", 20, 55)],
        "portables": ["mug", "book", "key", "remote"],
    },
}


def _sanitize_name(name, fallback):
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(name)).strip("_").lower()
    return slug or fallback


def _fallback_theme(prompt):
    """Only reached if hssd_vocab can't read the local dataset at all
    (see derive_theme) — keeps the app running, just without dataset-
    grounded accuracy."""
    valid = list(_FALLBACK_LIBRARY.keys())
    hits = [c for c in valid if c in prompt.lower()] or valid
    furniture, portables = [], []
    for cat in hits:
        furniture.extend(_FALLBACK_LIBRARY[cat]["furniture"])
        portables.extend(_FALLBACK_LIBRARY[cat]["portables"])
    return {"style": prompt.strip() or "a home", "furniture": furniture,
            "portables": portables or _DEFAULT_PORTABLES}


def _select_from_vocab(prompt, vocab, max_furniture=16, max_portables=8):
    """Keyword-based selection directly from the REAL vocabulary — used
    for --mock (no LLM to ask) and as the fallback if a real LLM call
    fails/returns something unusable. Every name this returns is
    guaranteed to be a real HSSD category (unlike a free-text LLM
    invention), so this alone already fixes "boxes only" even without a
    live API key — it just can't be as prompt-specific as the real LLM
    selection below."""
    rooms = hssd_vocab.match_room_types(prompt, vocab, max_types=3)
    furniture, portables, seen_f, seen_p = [], [], set(), set()
    for room in rooms:
        for name, size, height, _count in vocab["furniture"][room]:
            if name not in seen_f and len(furniture) < max_furniture:
                seen_f.add(name)
                furniture.append((name, size, height))
        for p in vocab["portables"].get(room, []):
            if p not in seen_p and len(portables) < max_portables:
                seen_p.add(p)
                portables.append(p)
    return {"style": prompt.strip() or "a home", "furniture": furniture,
            "portables": portables or _DEFAULT_PORTABLES}


def derive_theme(prompt, use_mock, hssd_root, model="gpt-4o"):
    """
    Returns {"style": "<echoes the prompt>", "furniture": [(name, size,
    height), ...], "portables": [name, ...]}. Exactly ONE call reaches
    this — gives the LLM the REAL object vocabulary available in the
    local HSSD dataset (`hssd_vocab.load_room_vocab`, built from the
    dataset's own per-room-type category and portable-item metadata, with
    genuine measured sizes — not a hand-typed guess) and asks it to SELECT
    (never invent) whichever real names best fit the described scene. This
    is what actually fixes "getting boxes only": a selection constrained
    to real categories is guaranteed retrievable, where a free-text
    invention (or even a plausible hand-typed guess) is not. Every chunk
    afterward samples from this SAME pool deterministically (see
    build_chunk) — no further LLM calls, no per-chunk network latency.

    Falls back to `_select_from_vocab`'s keyword-based selection (still
    dataset-grounded, just less prompt-specific) in --mock mode or if the
    real call fails/returns nothing usable, and to `_fallback_theme`'s
    hand-typed library only if the local HSSD metadata can't be read at
    all.
    """
    vocab = hssd_vocab.load_room_vocab(hssd_root)
    if not vocab["furniture"]:
        print(f"[infinite_world] couldn't read HSSD metadata under {hssd_root!r}; "
              f"falling back to a small hand-typed furniture library.")
        return _fallback_theme(prompt)
    if use_mock:
        return _select_from_vocab(prompt, vocab)

    lines = []
    for room, entries in vocab["furniture"].items():
        names = [e[0] for e in entries[:18]]
        portables = vocab["portables"].get(room, [])[:10]
        lines.append(f"{room}: furniture={names}; portables={portables}")
    vocab_text = "\n".join(lines)
    system = (
        "You select real furniture for a procedural indoor 3D world "
        "generator, from a fixed real-object vocabulary. You may ONLY "
        "choose names that appear in the vocabulary below — never invent "
        "new ones, they won't have real 3D models available. This is a "
        "real, fixed indoor-home dataset, not a fantasy asset library: "
        "for a scene that doesn't literally match any room type, pick "
        "your closest, most evocative real matches anyway rather than "
        "leaving the selection sparse."
    )
    user = (
        f"Scene: {prompt!r}\n\n"
        f"Real object vocabulary, by room type:\n{vocab_text}\n\n"
        "Reply with ONLY a JSON object, no markdown fences, no prose:\n"
        '{"furniture": ["name", ...], "portables": ["name", ...]}\n\n'
        "furniture: 10-16 names from the vocabulary above (from whichever "
        "room type(s) best fit — mixing rooms is fine if the scene calls "
        "for it) that would realistically furnish this exact space.\n"
        "portables: 4-8 small item names from the same vocabulary."
    )
    try:
        raw = _call_llm(system, user, model=model).strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
        data = json.loads(raw)
        by_name = {name: (size, height) for entries in vocab["furniture"].values()
                   for name, size, height, _count in entries}
        all_portables = {p for lst in vocab["portables"].values() for p in lst}
        furniture = []
        for name in data.get("furniture", []):
            name = str(name).strip().lower()
            if name in by_name:
                size, height = by_name[name]
                furniture.append((name, size, height))
        portables = [str(p).strip().lower() for p in data.get("portables", [])
                     if str(p).strip().lower() in all_portables]
        if not furniture:
            raise ValueError("LLM selected no names that exist in the real vocabulary")
        return {"style": prompt.strip() or "a home", "furniture": furniture,
                "portables": portables or _DEFAULT_PORTABLES}
    except Exception as e:
        print(f"[infinite_world] LLM vocabulary selection failed ({e}); "
              f"falling back to keyword-based selection from the same real vocabulary.")
        return _select_from_vocab(prompt, vocab)


def build_chunk(theme, chunk_index, retriever=None, chunk_width=400, chunk_depth=500):
    """
    Deterministic given the same (theme, chunk_index): seeds its own RNG
    from chunk_index so re-requesting the same index (e.g. a client retry)
    reproduces the same chunk rather than a different random one. No
    agent, no goal — chunks are pure furniture content; the persistent web
    page owns the one agent across the whole corridor.

    Samples a random SUBSET of the theme's own furniture pool each chunk
    (rather than placing everything every time) — this is what gives
    consecutive chunks natural variety while every single piece placed
    still comes from the SAME theme-specific vocabulary derive_theme
    built for this exact prompt, not a generic fallback list.
    """
    rng = random.Random(chunk_index)
    scene = Scene3D(chunk_width, chunk_depth)
    scene.retriever = retriever

    pool = theme["furniture"]
    n = rng.randint(2, min(4, len(pool)))
    chosen = rng.sample(pool, n)

    walls = ["back", "front", "left", "right"]
    rng.shuffle(walls)
    placed = []
    for i, (name, size, height) in enumerate(chosen):
        # _sanitize_name defensively, regardless of source: the LLM path
        # already sanitizes (see derive_theme), but _FALLBACK_LIBRARY's
        # own entries ("coffee table") still carry a literal space —
        # applied here too rather than trusting every caller already did it.
        var_name = f"{_sanitize_name(name, 'item')}_{chunk_index}_{i}"
        jitter = rng.uniform(0.85, 1.15)
        asset = scene.AddAsset(var_name, kind="box", size=size * jitter, height=height * jitter)
        if i < len(walls):
            scene.place_against_wall(asset, walls[i])
        else:
            scene.place_corner(asset, rng.choice(
                ["front_left", "front_right", "back_left", "back_right"]))
        placed.append(asset)

    # There's no real pathfinding for infinite mode's auto-forward
    # controller (see JS_STREAMING_DRIVER's autoForwardTick) — it just
    # walks straight at whatever's directly ahead and slides around
    # collisions locally, with no A* to route around an obstacle. A piece
    # of furniture placed against the front/back wall defaults to being
    # CENTERED (place_against_wall's offset=0 means x=chunk_width/2) —
    # exactly where the agent's straight-ahead path travels — and if it's
    # wide enough, that's a real, confirmed deadlock: the agent walks into
    # it, gets pushed back to the near edge, and walks straight back into
    # the same spot next tick, forever (reproduced: an agent frozen at the
    # exact same (x, z) for 1500+ simulated ticks). Rather than add real
    # steering/pathfinding to the controller, guarantee the deadlock can't
    # happen in the first place: shove anything that would overlap a
    # walkable center lane out to whichever side is closer.
    lane_half = 55
    center = chunk_width / 2
    for asset in placed:
        half = asset.half_xz
        lane0, lane1 = center - lane_half, center + lane_half
        x0, x1 = asset.pos[0] - half, asset.pos[0] + half
        if x1 <= lane0 or x0 >= lane1:
            continue
        target_x = (lane0 - half) if asset.pos[0] < center else (lane1 + half)
        m = scene.wall_margin
        target_x = max(half + m, min(chunk_width - half - m, target_x))
        asset.set_pose(x=target_x)

    if placed and theme.get("portables"):
        item_name = rng.choice(theme["portables"])
        item = scene.AddAsset(f"{item_name}_{chunk_index}", kind="cylinder",
                               size=5, height=10, portable=True)
        scene.place_on(item, rng.choice(placed))

    scene.resolve_layout()
    return scene, theme["style"]


def find_spawn_point(scene, agent_half_xz=15):
    """
    A genuinely clear (x, z) for the agent to start at within `scene`
    (chunk 0's local coordinates — zero z-offset, so this doubles directly
    as world coordinates). Reuses Scene3D._find_free_spot exactly the way
    every other scene in this harness already picks the agent's starting
    position, rather than a hand-picked formula: confirmed as a real bug
    that a fixed `(width/2, wallMargin+20)` spawn point can land INSIDE a
    large piece of furniture placed against chunk 0's front wall (a sofa's
    own half-extent alone can exceed 20+20), which then shoves the agent
    straight backward into the front wall via collision push-out instead
    of leaving it free to walk forward — the auto-forward controller
    looked "stuck" for a reason that had nothing to do with the controller
    itself.
    """
    return scene._find_free_spot(agent_half_xz)


def chunk_to_json(scene, chunk_index, chunk_depth, category=None, draw_front_wall=False):
    """
    Exports this chunk's assets in the exact shape viewer3d.py's static
    exporter already emits (color, embedded real-mesh data, up-axis
    correction, ...) — see `viewer3d.assets_to_json`, the single source of
    truth for that shared by both code paths — offset into world space so
    chunks tile continuously with no gap or overlap. `draw_front_wall`
    should only be True for chunk 0 (a sense of a starting point); no
    chunk ever draws the wall facing the next one — that seam is
    deliberately left open, which is what makes the corridor read as
    "no boundaries" rather than a chain of separately-walled rooms.
    """
    state = scene.to_state()
    assets = viewer3d.assets_to_json(state, agent_name=None, goal_name=None)
    z_offset = chunk_index * chunk_depth
    for a in assets.values():
        a["z"] += z_offset
    return dict(
        index=chunk_index, category=category,
        width=scene.width, depth=scene.depth, wallMargin=scene.wall_margin,
        zOffset=z_offset, assets=assets, drawFrontWall=draw_front_wall,
    )


def generate_chunk_json(theme, chunk_index, retriever=None,
                         chunk_width=400, chunk_depth=500):
    """Convenience wrapper combining build_chunk + chunk_to_json — the
    entry point webapp/server.py actually calls per chunk request."""
    scene, category = build_chunk(theme, chunk_index, retriever, chunk_width, chunk_depth)
    return chunk_to_json(scene, chunk_index, chunk_depth, category=category,
                          draw_front_wall=(chunk_index == 0))
