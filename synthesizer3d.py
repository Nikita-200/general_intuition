"""
synthesizer3d.py — turns a text prompt into an executable 3D scene program.

Same Programmer + Debugger loop as the 2D version (synthesizer.py): an LLM
proposes a build_scene(scene) function against documented Scene3D calls,
and a retry loop feeds back tracebacks / ground-truth failures (furniture
overlap, unreachable task) exactly like 2D did. Nothing about that loop is
dimension-specific, so it's ported essentially unchanged.

Asset retrieval (real 3D meshes, not just primitives) lives in
asset_retrieval.py: FallbackRetriever there tries a real Objaverse lookup
first and degrades to a primitive shape automatically on any failure
(no network, no match, corrupt download). It's optional — pass
retriever=FallbackRetriever() to synthesize_scene to turn it on; the
AssetRetriever/PrimitiveRetriever classes below stay as the always-
available default and the documented interface contract. See README.md's
step-by-step guide for how this plugs in, and for HSSD/3D-FRONT (both need
gated account access this can't script for you, unlike Objaverse).
"""

import os
import random
import re
import textwrap
import traceback

from physics_backend import Scene3D, BACKEND

DSL_DOCS = textwrap.dedent("""
You are writing a Python function:

    def build_scene(scene):
        ...

`scene` is a Scene3D instance (x = right, z = depth, y = up; floor is y=0):

  scene.AddAsset(name, kind: "box"|"cylinder", size: float, height: float=None,
                 dynamic: bool=False, portable: bool=False) -> Asset3D
      `size` is the floor-plane half-extent (half-width for box, radius for
      cylinder) — so an object's real footprint width is 2x `size`. `height`
      defaults to 1.4x size if omitted — ALWAYS override it with a real
      value from the chart below instead of relying on that default, which
      produces inconsistent, "haywire"-looking proportions.

      Units are centimeters (the agent is 170 tall, like a real person —
      every other object's size should read as plausible next to that).
      (size, height) reference chart — size is HALF-width, so double it for
      full width when eyeballing proportions against neighboring objects:

        agent/person:      size 14-18,  height 170
        dining chair:       size 18,  height  45   (seat-height box; a person "sits" at this level)
        armchair:            size 30,  height  75
        stool / ottoman:     size 16,  height  40
        sofa / couch (2-3 seat): size 55-70, height  75
        dining table:        size 42,  height  75
        coffee table:        size 32,  height  40
        desk:                size 38,  height  75
        side table / nightstand: size 20, height 55
        kitchen counter / countertop: size 45, height  90
        kitchen island:       size 55,  height  90
        bookshelf:            size 22,  height 180
        low shelf:            size 22,  height 120
        dresser:              size 30,  height  80
        wardrobe / closet:    size 35,  height 200
        cabinet:              size 25,  height  90
        tv stand:             size 40,  height  45
        bed (single/double):  size 65-80, height  50
        fridge / refrigerator: size 32, height 175
        stove / oven:         size 30,  height  90
        sink (kitchen/bath):  size 25,  height  90
        washer / dishwasher:  size 30,  height  85
        floor lamp:           size 10,  height 150
        potted plant:         size 15,  height  90
        trash bin / basket:   size 22,  height  35
        portable items (can, mug, book, key, remote, apple, cup):
                              size  4-8, height  8-14
      For any object not listed, interpolate from the nearest listed
      category rather than guessing a number from scratch — e.g. an
      "end table" is a side table (20, 55), a "loveseat" is a small sofa
      (45, 75). The goal: every object's height should look correct RELATIVE
      to the agent (170) and to whatever it sits next to (a chair must read
      as shorter than the table beside it, a bookshelf must read as taller
      than a person, etc.) — comparative correctness matters more than any
      single absolute number.
      dynamic=False (default): STATIC/immovable — use for ALL furniture.
      portable=True: a small handheld item (can, mug, key, book). Never
      collides with anything and is placed with place_on, not place_relative.

  scene.place_at(asset, x, z, yaw=0.0)
  scene.place_relative(asset, anchor, direction, distance=60)
      direction: front, back, left, right, front_left, front_right, back_left, back_right
  scene.place_around(list_of_assets, anchor, radius=80, start_angle=0.0)
  scene.place_grid(list_of_assets, rows, cols, spacing=70, origin=(x, z))
  scene.place_against_wall(asset, wall, offset=0)   # wall: front/back/left/right
  scene.place_corner(asset, corner)   # corner: front_left/front_right/back_left/back_right
  scene.place_on(item, surface, jitter=0.6)
      Puts `item` genuinely ON TOP of `surface` (item's y becomes the
      surface's top + item's half-height). This is how "the mug is on the
      table" / "the can is on the counter" should be expressed — always use
      this for anything resting on furniture, never place_relative. Safe to
      call multiple times onto the same surface (a table with several
      items on it) — it automatically keeps items from overlapping each
      other on that surface.
  scene.place_on_wall(asset, wall, height, offset=0)
      For decor that's conceptually MOUNTED ON a wall, not resting on the
      floor or another surface — a wall-mounted TV, framed photos, wall
      art, a wall clock. `wall` is front/back/left/right like
      place_against_wall; `height` is how far up the wall its CENTER sits
      (e.g. a TV over a console: height ~120-140). This is the ONLY
      correct way to depict something mounted above floor level — see the
      strict rule below about never setting height any other way.

  scene.add_container(name, size=22, height=35) -> Asset3D
      A solid, static container (bin/basket) — defaults already match the
      size chart above, only override for a deliberately larger/smaller
      container. A real obstacle like any furniture — the agent walks up
      to it, not through it.

  scene.set_agent(asset)   # exactly one call required
  scene.set_goal(asset)    # required UNLESS the task is pick-and-place

  scene.width, scene.depth   # floor bounds (x, z) — already walled in

Rules:
- Always create the agent (kind='cylinder', size 14-18, height~170, name 'agent') and call scene.set_agent(...).
- Simple "reach it" task: one goal object, portable=True, placed with
  place_on(...) onto a piece of furniture, then scene.set_goal(...). If the
  prompt describes a whole room rather than one obvious target, still pick
  ONE sensible portable object and call scene.set_goal(...) on it — don't
  leave the scene with no goal at all.
- Pick-and-place task: item (portable=True, placed with place_on) AND
  scene.add_container(...) for it to go into, both with clear literal names
  ("can", "trash_bin") since a task instruction is matched against these
  names. Do NOT call scene.set_goal(...) in this case.
- Furniture must not overlap other furniture (it's static now, nothing
  separates it automatically) — space place_around/place_grid/place_relative
  distances generously (e.g. a place_around radius should clear the anchor's
  own half-extent by 40+ units). The harness also runs an automatic geometry
  repair pass after your function returns, so don't obsess over exact
  numbers — reasonable, generous spacing is enough.
- NEVER set an asset's position via anything other than the place_* methods
  above (e.g. never write `asset.pos[1] = ...` or otherwise reach into an
  Asset3D's internals directly) — every placement need, including something
  mounted on a wall, has a documented place_* method. Position math done by
  hand instead of through these methods is exactly what produces objects
  that render floating or embedded in the floor.
- Do not call scene.settle() — the harness handles validity checks after your function returns.
- Output ONLY the function definition, no imports, no explanation, no markdown fences.
""")

FEWSHOT = textwrap.dedent('''
def build_scene(scene):
    agent = scene.AddAsset("agent", kind="cylinder", size=15, height=170)
    scene.set_agent(agent)

    table = scene.AddAsset("table", kind="box", size=42, height=75)
    scene.place_at(table, 400, 300)

    chairs = [scene.AddAsset(f"chair_{i}", kind="box", size=18, height=45) for i in range(4)]
    scene.place_around(chairs, table, radius=100)

    goal = scene.AddAsset("goal_mug", kind="cylinder", size=6, height=10, portable=True)
    scene.place_on(goal, table)
    scene.set_goal(goal)
''')

FEWSHOT_PICK_PLACE = textwrap.dedent('''
def build_scene(scene):
    agent = scene.AddAsset("agent", kind="cylinder", size=15, height=170)
    scene.set_agent(agent)

    counter = scene.AddAsset("counter", kind="box", size=45, height=90)
    scene.place_against_wall(counter, "back")

    can = scene.AddAsset("can", kind="cylinder", size=8, height=12, portable=True)
    scene.place_on(can, counter)

    trash_bin = scene.add_container("trash_bin", size=22, height=35)
    scene.place_corner(trash_bin, "front_right")
    # no scene.set_goal(...) — the pick-and-place task references
    # "can" and "trash_bin" by name directly
''')


class AssetRetriever:
    """
    Swap-in point for real object retrieval (InteriorAgent-style). Given a
    semantic name, return (kind, size, height) — or, for a real dataset
    integration, a mesh path plus its footprint. PrimitiveRetriever (below)
    is the default and is all this harness uses today.

    To wire in a real one:
      - Objaverse: `pip install objaverse`, use their annotation index to
        search by category/tag, download the .glb, measure its bounding
        box for (size, height). Needs network access to Objaverse's asset
        host (not available in this sandbox).
      - 3D-FRONT / HSSD: both ship precomputed per-object bounding boxes
        alongside their meshes, so retrieval is a category lookup against
        their metadata JSON, no need to load the mesh just to get a size.
    Either way, the integration point is exactly this class — nothing in
    dsl3d.py, nav_agent_3d.py, or the viewer needs to change.
    """
    def retrieve(self, semantic_name, size_hint=None, height_hint=None):
        raise NotImplementedError


class PrimitiveRetriever(AssetRetriever):
    """Default: reasonable proportions for common furniture words, box/
    cylinder fallback otherwise. This is what makes the harness runnable
    today with zero external dependencies."""
    # (size, height) in cm, matching the size chart in DSL_DOCS above —
    # kept in sync deliberately so primitives and any real-mesh retriever
    # agree on what "a table" or "a dresser" should look like.
    # NOTE: order matters — this is a substring match, so any key that is
    # itself a substring of a more specific key (e.g. "table" is contained
    # in "coffee table") MUST come after that more specific key, or the
    # generic entry wins every time and the specific one is dead code.
    TABLE = {
        "dining table": (42, 75), "coffee table": (32, 40),
        "side table": (20, 55), "end table": (20, 55), "tv stand": (40, 45),
        "table": (42, 75),
        "countertop": (45, 90), "counter": (45, 90), "island": (55, 90),
        "desk": (38, 75), "armchair": (30, 75), "chair": (18, 45),
        "stool": (16, 40), "ottoman": (16, 40),
        "loveseat": (45, 75), "sofa": (60, 75), "couch": (60, 75),
        "bed": (70, 50), "bookshelf": (22, 180), "shelf": (22, 120),
        "wardrobe": (35, 200), "closet": (35, 200), "dresser": (30, 80),
        "sideboard": (35, 85), "cabinet": (25, 90), "nightstand": (20, 55),
        "refrigerator": (32, 175), "fridge": (32, 175),
        "dishwasher": (30, 85), "washer": (30, 85),
        "stove": (30, 90), "oven": (30, 90), "sink": (25, 90),
        "lamp": (10, 150), "plant": (15, 90),
        "trash": (22, 35), "basket": (22, 35), "bin": (22, 35),
    }

    def retrieve(self, semantic_name, size_hint=None, height_hint=None):
        key = semantic_name.lower().replace("_", " ")
        for k, (size, height) in self.TABLE.items():
            if k in key:
                return ("box", size, height)
        return ("box", 25, 60)  # neutral mid-sized generic furniture guess


def _extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def _estimate_room_size(prompt):
    """
    Rough, deterministic room sizing from the PROMPT TEXT alone, before the
    synthesis LLM ever runs. A fixed 800x600 for every prompt regardless of
    how much furniture it describes is a real contributor to two separate
    complaints: FURNITURE OVERLAP synthesis failures on elaborate, many-
    object prompts (a "lived-in" room with a sectional, TV console, coffee
    table, floor lamp, several bookshelves, an armchair, and a side table
    genuinely needs more floor than a one-counter, one-appliance kitchen),
    and the room reading as overcrowded even when it technically fits —
    "not too crowded nor too empty" is partly just a function of how much
    square footage the described contents got to begin with.

    Counts how many distinct pieces of furniture vocabulary (the same
    words PrimitiveRetriever.TABLE already recognizes) appear in the
    prompt as a proxy for described complexity, and scales the default up
    (never down — a simple prompt keeps the original 800x600 every
    existing example already relies on) accordingly. Deliberately a plain
    heuristic rather than a second LLM call: this only decides the
    CONTAINER'S size, not any actual furniture choice or layout, which
    stays entirely up to the synthesis LLM.
    """
    furniture_words = {tok for k in PrimitiveRetriever.TABLE for tok in k.split()}
    words = set(re.findall(r"[a-z]+", prompt.lower()))
    hits = len(words & furniture_words)
    if hits <= 4:
        return 800, 600
    if hits <= 8:
        return 1000, 750
    return 1200, 900


def _call_llm(system_prompt, user_prompt, model="gpt-4o"):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt},
                   {"role": "user", "content": user_prompt}],
        temperature=0.4,
    )
    return resp.choices[0].message.content


def _mock_llm(system_prompt, user_prompt, model=None):
    if re.search(r"\b(bin|trash|basket|container|pick up|place it|pick the)\b", user_prompt, re.I):
        m = re.search(r"\b(can|mug|book|ball|key|remote|apple|cup)\b", user_prompt, re.I)
        item = (m.group(1) if m else "can").lower()
        return FEWSHOT_PICK_PLACE.replace('"can"', f'"{item}"')
    m = re.search(r"\b(mug|can|book|ball|key|remote|apple)\b", user_prompt, re.I)
    target = (m.group(1) if m else "mug").lower()
    return FEWSHOT.replace("goal_mug", f"goal_{target}")


def get_llm(use_mock):
    return _mock_llm if use_mock else _call_llm


# Rotating library for --mock's infinite/curriculum mode: a genuinely
# different room type + task style each call, cycling forever without any
# unbounded growth. {item} gets filled from _MOCK_ITEMS, which is exactly
# the vocabulary _mock_llm's own regex already recognizes (can/mug/book/
# ball/key/remote/apple/cup) — so these prompts route through _mock_llm's
# EXISTING, already-tested logic unchanged; this only varies the prompt
# TEXT that reaches it, not _mock_llm itself.
_MOCK_PROMPT_LIBRARY = [
    "a kitchen with a counter, an island, and a {item} on the counter",
    "a kitchen with a {item} on the counter and a trash bin in the corner",
    "a living room with a sofa, a coffee table, and a {item} on the table",
    "a living room with a coffee table holding a {item}, and a basket by the sofa",
    "a bedroom with a bed, a nightstand, and a {item} on the nightstand",
    "a home office with a desk, a bookshelf, and a {item} on the desk",
    "a home office with a {item} on the desk and a trash bin by the chair",
    "a dining room with a dining table, chairs, and a {item} on the table",
]
_MOCK_ITEMS = ["mug", "can", "book", "ball", "key", "remote", "apple", "cup"]


def _mock_next_prompt(prev_prompt):
    # Excludes an exact repeat of the immediately-previous prompt (not a
    # rigorous no-two-alike-in-a-row guarantee across item substitutions,
    # just enough to make consecutive levels visibly different in the
    # common case) rather than tracking template identity separately.
    candidates = [t.format(item=item) for t in _MOCK_PROMPT_LIBRARY for item in _MOCK_ITEMS]
    candidates = [c for c in candidates if c != prev_prompt] or candidates
    return random.choice(candidates)


def next_prompt_text(prev_prompt, prev_success, use_mock, model="gpt-4o", level_idx=0):
    """
    Produces the prompt for the NEXT level of a --levels/--infinite run.
    Deliberately generates a genuinely NEW environment each call rather
    than mutating the previous prompt's text — an earlier version did
    `prev_prompt.replace("a ", "a bigger, more cluttered ", 1)`, which
    looked fine across a handful of --levels but is fundamentally unsound
    for a long or infinite run: since the replacement text itself starts
    with "a ", every call re-matches its OWN previous output, so the
    prompt grows by one "a bigger, more cluttered " chain link per call
    forever and stops describing anything real within a dozen iterations
    (confirmed: 8 iterations produced a 250+ character prefix of repeated
    "bigger, more cluttered" with no new content, and it only gets worse).

    use_mock=True cycles through a small library of distinct room-type/
    task-style prompts (see _MOCK_PROMPT_LIBRARY) — bounded, varied, and
    reuses _mock_llm's existing tested item/pick-place detection unchanged
    (only the prompt TEXT varies, not _mock_llm itself). The real-LLM path
    explicitly asks for a DIFFERENT room and objects, not a bigger version
    of the same one, which is both more sustainable over a long run and
    more genuinely useful as "diverse training environments" (the actual
    point of infinite generation) than a single room slowly inflating.
    """
    if use_mock:
        return _mock_next_prompt(prev_prompt)
    sys_p = "You design 3D indoor environment prompts for an embodied-agent benchmark."
    user_p = (
        f"This is level {level_idx} of an ongoing, potentially never-ending "
        "curriculum of indoor environments for an embodied agent.\n"
        f"Previous environment: {prev_prompt!r}\n"
        f"Agent reached the goal last time: {prev_success}\n"
        "Propose ONE new, DIFFERENT indoor environment - a different room "
        "type and different objects than the previous one, not just a "
        "bigger or more cluttered version of the same room - with "
        "realistic everyday detail and a clear, reachable goal object or "
        "pick-and-place task. Reply with just the prompt text, no "
        "markdown, no explanation."
    )
    return _call_llm(sys_p, user_p, model=model).strip().strip('"')


def _real_vocab_hint(prompt, hssd_root):
    """Optional soft nudge toward AddAsset names that are confirmed to have
    a real HSSD mesh available — grounded in the same real dataset metadata
    as infinite_world.derive_theme (see hssd_vocab.py), but deliberately a
    HINT, not a hard constraint: infinite mode's chunks have no human
    watching the LLM's per-object choices and needed a strict "select, never
    invent" rule to kill "getting boxes only"; a one-off Scene mode prompt
    is exactly the opposite case the user asked to keep expressive — "follow
    the prompt instructions and some creativity on your end" — so this
    lists real, guaranteed-matchable names as a preference the LLM can lean
    on, while leaving it free to invent anything the vocabulary doesn't
    cover (an invented name just degrades to a primitive shape, same as
    always, rather than being rejected). Returns "" if no hssd_root was
    given or its metadata can't be read, leaving DSL_DOCS's own hand-typed
    size chart as the only guidance — identical to before this existed."""
    if not hssd_root:
        return ""
    try:
        import hssd_vocab
        vocab = hssd_vocab.load_room_vocab(hssd_root)
    except Exception:
        return ""
    if not vocab["furniture"]:
        return ""
    rooms = hssd_vocab.match_room_types(prompt, vocab, max_types=3)
    lines = []
    for room in rooms:
        names = [e[0] for e in vocab["furniture"][room][:20]]
        portables = vocab["portables"].get(room, [])[:12]
        lines.append(f"{room}: furniture={names}; small items={portables}")
    vocab_text = "\n".join(lines)
    return textwrap.dedent(f"""
        # Real object vocabulary (optional guidance, not a constraint)
        The names below are confirmed to exist in the real local HSSD 3D
        asset library, picked for the room type(s) this prompt most
        resembles. Using one of these EXACT names as (or as the root of,
        before any AddAsset auto-dedup "_2"/"_3" suffix) an asset's name
        guarantees a real retrieved mesh instead of a plain colored box.
        Prefer them where they genuinely fit what the prompt describes, but
        you are NOT limited to this list — invent whatever better matches
        the prompt's own specific/creative intent for anything not covered
        here; an invented name just falls back to a primitive shape rather
        than failing anything.
        {vocab_text}
        """)


def synthesize_scene(prompt, width=None, depth=None, use_mock=False,
                      max_retries=3, model="gpt-4o", retriever=None, hssd_root=None):
    """
    Programmer + Debugger loop. Returns (scene, program_source, log).

    retriever: optional AssetRetriever (e.g. asset_retrieval.FallbackRetriever())
    — when given, every AddAsset call transparently tries a real-mesh lookup
    keyed on the asset's own name before falling back to a primitive shape.
    None (default) skips retrieval entirely — every object is a primitive,
    same as before this existed.

    hssd_root: optional path to a local HSSD download — when given, the
    system prompt gets a real-vocabulary hint (see _real_vocab_hint) so the
    LLM prefers names guaranteed to retrieve a real mesh through `retriever`
    without losing its own creative freedom. None (default) skips this
    entirely, same as before this existed.

    width/depth: None (default) auto-estimates room size from the prompt's
    own described complexity (see _estimate_room_size) — a simple prompt
    still gets the original 800x600 every existing example relies on; an
    elaborate, many-object prompt gets more floor space. Pass explicit
    values to override the estimate entirely.
    """
    if width is None or depth is None:
        auto_width, auto_depth = _estimate_room_size(prompt)
        width = auto_width if width is None else width
        depth = auto_depth if depth is None else depth
    llm = get_llm(use_mock)
    system_prompt = (DSL_DOCS + "\n# Example: simple goal-reaching scene\n" + FEWSHOT
                      + "\n# Example: pick-and-place scene\n" + FEWSHOT_PICK_PLACE
                      + _real_vocab_hint(prompt, hssd_root))
    user_prompt = f'Prompt: "{prompt}"\n\nWrite build_scene(scene) for this.'

    log = []
    program_src = _extract_code(llm(system_prompt, user_prompt, model=model))

    for attempt in range(max_retries + 1):
        scene = Scene3D(width, depth)
        scene.retriever = retriever
        namespace = {"scene": scene}
        try:
            exec(compile(program_src, "<scene_program_3d>", "exec"), namespace)
            namespace["build_scene"](scene)
            if scene.agent is None:
                raise RuntimeError("build_scene must call scene.set_agent(...)")
            # Right-size the room to what actually got BUILT (the initial
            # width/depth above was only ever a guess from the prompt's own
            # text, before the LLM decided how many of those mentioned
            # items would become real floor furniture vs. small desk
            # clutter) — see Scene3D.fit_room_to_content's docstring. Must
            # run BEFORE resolve_layout, which then guarantees no overlap/
            # out-of-bounds at whatever size this settles on.
            scene.fit_room_to_content()
            # Auto-repair placement geometry BEFORE the ground-truth checks
            # below: an LLM routinely gets raw coordinates numerically
            # wrong (an anchor near a wall + too-large a place_relative
            # distance, or two independently-placed pieces of furniture
            # that just happen to intersect) in ways a geometry pass can
            # simply fix, rather than spending a synthesis retry on it. See
            # Scene3D.resolve_layout's docstring. Placing the agent AFTER
            # this (not before) means its free-spot search sees the
            # corrected layout, not the pre-repair one.
            scene.resolve_layout()
            scene.agent.set_pose(*scene._find_free_spot(scene.agent.half_xz))
            if scene.has_bad_overlap():
                raise RuntimeError(
                    "FURNITURE OVERLAP: two or more solid (non-portable) objects "
                    "overlap each other in the floor plan. Space them out explicitly.")
            out_of_bounds = scene.out_of_bounds_assets()
            if out_of_bounds:
                raise RuntimeError(
                    f"OUT OF BOUNDS: {out_of_bounds!r} extend outside the room's own "
                    f"walls (floor is {width}x{depth}). A relative placement call "
                    "(place_relative/place_around/etc.) likely used too large a "
                    "distance from an anchor near a wall. Use place_at with explicit "
                    "coordinates, or a smaller distance, so every object's full "
                    "footprint stays within [0, width] x [0, depth].")
            if not scene.is_reachable():
                raise RuntimeError(
                    "REACHABILITY CHECK FAILED: no clear floor-plan path from the "
                    "agent to the goal after placement. This is a ground-truth check.")
            log.append({"attempt": attempt, "ok": True})
            return scene, program_src, log
        except Exception as e:
            tb = traceback.format_exc(limit=3)
            log.append({"attempt": attempt, "ok": False, "error": str(e)})
            if attempt == max_retries:
                raise RuntimeError(
                    f"Debugger exhausted {max_retries} retries. Last error:\n{tb}")
            fix_prompt = (
                f"This program raised an error:\n\n{program_src}\n\n"
                f"Traceback:\n{tb}\n\nReturn a corrected build_scene(scene) function only. "
                "If this was a reachability failure, spread obstacles out more or "
                "move the goal so a path around them exists."
            )
            program_src = _extract_code(llm(system_prompt, fix_prompt, model=model))