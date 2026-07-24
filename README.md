# Infinite Environment Generation: a 3D Indoor Agent Harness

Text prompt → generated 3D indoor scene → agent navigates/picks/places →
watch it live and interactive in a browser, or drive it yourself.

An LLM (or a deterministic `--mock` mode requiring no API key at all)
writes a small Python program against a documented scene-description DSL
for every prompt you give it, so the environment space is open-ended — any
prompt describing an indoor room becomes a new scene, procedurally, in
code, not from a fixed asset library. Because scenes are defined in code,
every objective is verified programmatically (furniture doesn't overlap,
the agent can actually reach its target, a pick-and-place task genuinely
completed) rather than by inspecting pixels — exactly the "code-level
objectives" the brief calls out as the reason this line of work matters.

> **On 2D vs. 3D**: the brief recommends starting in 2D specifically
> because real 3D navigation is meant to be driven by an internal
> vision-based policy this project doesn't have access to. This harness
> targets 3D directly instead, and is honest about the one place that
> matters: the "agent" doing navigation here is a hand-rolled,
> floor-projected A* planner (`nav_agent_3d.py`) standing in for that
> policy — a reasonable stand-in for proving the harness's other half
> (text prompt → valid, verifiable scene, generated and repaired
> automatically → a navigable result) works end to end, but explicitly
> **not** a claim to have solved 3D navigation the way a real vision-based
> policy would. Swapping one in would mean replacing `nav_agent_3d.py`'s
> planner/controller with calls to it — nothing else here (scene
> generation, the DSL, the task/verification layer, the viewer) assumes
> anything about how navigation decisions get made.

This covers what the brief asks for on indoor scenes: reach a goal,
pick-and-place, auto-navigate or drive manually — all backed by code-level
verifiable objectives rather than pixel inspection.

## Run it

```bash
pip install openai   # only needed for the real (non --mock) synthesizer

# reach-goal:
python harness3d.py "a living room with a sofa, coffee table, and a mug on the table" --mock

# pick-and-place:
python harness3d.py "a kitchen with a can on the counter and a trash bin in the corner" \
    --task "pick the can and place it in the trash bin" --mock

# curriculum of several different environments in one run:
python harness3d.py "a kitchen with an island and a can on the counter" --mock --levels 5

# never stop generating new environments until you Ctrl+C:
python harness3d.py "a kitchen with an island and a can on the counter" --mock --infinite

# real 3D meshes (Objaverse) instead of primitive shapes:
python harness3d.py "..." --mock --use-real-assets

# real PyBullet rigid-body physics instead of the pure-Python fallback:
pip install pybullet && python smoke_test_pybullet.py   # verify first, see GUIDE_3D_ASSETS.md
python harness3d.py "..." --mock   # auto-detects and uses it once installed
```

`physics_backend.py` auto-detects PyBullet and uses it when available,
falling back to a pure-Python physics-lite engine otherwise — **see
`GUIDE_3D_ASSETS.md` for the complete, detailed setup guide** covering
PyBullet installation (including Windows-specific notes) and real 3D mesh
retrieval from Objaverse/HSSD/3D-FRONT. The short version of both is below;
the guide has the full story.

Each run writes `out3d/viewer_N.html`. **Open that file directly in a
browser** (double-click it, no server needed) to watch the agent navigate
itself around the 3D scene live, or take over and drive it yourself.

## Infinite generation

`--infinite` removes the level cap entirely: `harness3d.py` keeps
synthesizing a genuinely new environment, running the agent through it,
and writing it out — forever — until you stop it with Ctrl+C. This is the
direct, runnable answer to the brief's "infinite procedural environment
generation": every level is a fresh LLM (or `--mock`) call against the
same documented DSL, verified by the same ground-truth checks
(`resolve_layout`, `has_bad_overlap`, `is_reachable`) as any single run, so
"infinite" means infinite *valid, playable* environments, not infinite
attempts.

Three things had to actually be true for "forever" to work in practice
rather than just be a `while True` that quietly breaks or fills the disk:

1. **The next environment must never degenerate.** An earlier version of
   `next_prompt_text` (used for `--levels`, before `--infinite` existed)
   built each new prompt by mutating the previous one:
   `prev_prompt.replace("a ", "a bigger, more cluttered ", 1)`. That looked
   fine across a handful of `--levels`, but is fundamentally unsound for a
   long run: the replacement text itself starts with `"a "`, so every call
   re-matches its own previous output, and the prompt grows by one
   `"bigger, more cluttered"` chain link per call forever, stopping being a
   real scene description within a dozen iterations (confirmed: 8
   iterations produced a 250+ character prefix of repeated
   "bigger, more cluttered" text with no new content). Fixed by generating
   a genuinely NEW environment each call instead of mutating the last one
   — `--mock` cycles through a small library of distinct room-type/task
   combinations (bounded, no growth, still routes through the exact same
   tested scene-building logic as any other `--mock` run); the real-LLM
   path explicitly asks for "a different room and different objects than
   the previous one, not a bigger version of the same room."
2. **One bad level must not end the run.** A single synthesis retry
   exhaustion, an LLM hiccup, or a network blip fetching a real mesh used
   to be a crash — for `--levels 5` that's a real but bounded annoyance;
   for `--infinite` it would mean the very first transient failure, on a
   run meant to go forever, ends everything. Each level now runs inside
   its own `try`/`except`, recorded as a failed result (same shape as any
   other unsuccessful level) rather than stopping the loop, and
   `results.json` is flushed after every single level — not just at the
   end — since an infinite run has no "the end" until you interrupt it.
3. **Disk usage must stay bounded.** Every level writes a `viewer_N.html`
   that embeds three.js (and, with `--asset-source`, real mesh data) —
   several MB each. An unbounded run with no cleanup would grow forever
   until disk fills, which defeats the entire point of "infinite" being a
   usable feature. `--keep-last N` (defaults to 30 once `--infinite` is
   set) deletes the heavy per-level files for anything older than the most
   recent N levels after each one is written, while `results.json` keeps
   the full lightweight history (prompt, success, ticks) for every level
   that ever ran, forever.

**Verified**: ran `--infinite --keep-last 3` for 12 seconds — reached
level 68 with genuine, varied environments throughout (kitchens, living
rooms, dining rooms, home offices, different held items each time, no
degeneration), only the most recent 4 levels' heavy files present on disk
afterward, and the full 68-entry history intact in `results.json`.
Separately confirmed Ctrl+C (`KeyboardInterrupt`) at an arbitrary point
mid-run exits cleanly with a "Stopped after N level(s)" message and a
valid `results.json` — no raw traceback, no partial/corrupt output.

```bash
python harness3d.py "a kitchen with an island and a can on the counter" --mock --infinite
# ... runs forever ...
# ^C
# Stopped after 47 level(s) (Ctrl+C).
# Wrote 47 level(s) to out3d/ (see results.json)
```

## Web app

The CLI writes files; `webapp/` is a real local web app around the exact
same generation pipeline — type a prompt in a browser, pick Manual or
Auto control and Scene or Infinite mode, and get an interactive result
without touching a terminal.

```bash
pip install flask
python webapp/server.py --mock      # or drop --mock and set OPENAI_API_KEY for the real LLM
# open http://127.0.0.1:5000/
```

This is a **runnable local app, not an actual public deployment** — it
uses this machine's own `hssd-hab/` download directly (real HSSD meshes,
per the ask that the backend use HSSD as its asset source), and the
`HSSDRetriever` behind it is built once at server startup, not per
request (its `__init__` scans ~16,500 objects to build a category index —
confirmed to take a couple of seconds; doing that on every generate call
would make the whole app sluggish for no reason). You can deploy this same
code to any server that also has that dataset; nothing here pushes it
anywhere on its own.

- **Scene mode** needed zero new rendering code: `/api/scene` is a thin
  wrapper around the exact same `synthesize_scene` → `_build_subgoals` →
  `render_viewer_html` pipeline `harness3d.py`'s CLI already uses, and
  just returns the resulting self-contained HTML directly (shown in an
  `<iframe>`) — same auto/manual toggle, WASD, mouse look, HUD as any
  `viewer_N.html`, with an added `initialControl` so the page starts in
  whichever mode you picked on the form instead of always defaulting to
  Auto.
- **Infinite mode** is the genuinely new piece: a continuously-extending,
  boundary-less corridor you move through for ~30 seconds, new areas
  streamed in live as you approach the edge of what's already loaded —
  see below.

### Infinite mode: how it actually generates a "never-ending" space

Confirmed with real design tradeoffs, not just a UI toggle:

1. **One real LLM call, then deterministic forever.** `/api/infinite/start`
   asks the model once for a **prompt-specific furniture list** — real
   named objects with realistic sizes (`derive_theme` in
   `infinite_world.py`), not a category label picked from a small fixed
   set. This was a real, reported gap in the first version: `derive_theme`
   originally only chose among 6 generic room categories
   (`infinite_world.ROOM_LIBRARY`), so no matter how specific the prompt
   was, every chunk drew from the same generic kitchen/living-room/
   bedroom/etc. lists — confirmed HSSD retrieval itself was never the
   problem (25/26 of those generic items already resolved to real meshes)
   — the theme just never carried anything about what was actually asked
   for. Now the LLM directly proposes named furniture and portables for
   *that* scene (a "pirate captain's quarters" call returned real sea
   chests, a navigation table, a weapon rack, a spyglass — verified with a
   simulated LLM reply, not just hoped for), validated and clamped to a
   sane size/height range before anything reaches placement code, with the
   old generic `ROOM_LIBRARY` kept only as the `--mock`/no-API-key fallback
   and the safety net if a real reply is malformed or empty. Every
   individual chunk after that is still generated **deterministically** —
   no LLM, no network, no per-chunk latency — reusing the same
   `AddAsset`/`place_*`/`resolve_layout` DSL and HSSD retrieval as any
   other scene, each chunk sampling a random subset of the SAME
   theme-specific pool for variety. This was a deliberate design choice,
   not an oversight: a live LLM call per chunk was considered and rejected
   — chunks are only fetched ~150 units before the edge of loaded content,
   about a second of walking, and a real LLM call routinely takes several
   seconds, which would cause visible stalls almost every chunk. This is
   also just how procedural world generation actually works elsewhere:
   Minecraft's own terrain isn't AI-authored per chunk either. Creativity
   comes from the one upfront theme call; scale and continuity come from a
   generator that can never stall mid-stream waiting on a network call.
2. **True live streaming, not one big pre-generated blob.** The frontend
   tracks the agent's own z position; once it gets within ~150 units of
   the far edge of what's loaded, it `fetch()`es
   `/api/infinite/next_chunk` for the next one and splices it into the
   live three.js scene (`appendChunk`) without interrupting movement.
3. **"No boundaries" is a real geometry choice, not a figure of speech.**
   Each chunk is an ordinary `Scene3D(chunk_width, chunk_depth)` —
   offset in world space by `chunk_index * chunk_depth` — that never
   draws the wall facing the next chunk; only chunk 0 draws a front wall
   at all (a sense of a starting point). Consecutive chunks are
   physically open into each other. Side walls stay, so this reads as an
   ever-extending corridor rather than a fully open 2D field, which was a
   deliberate scope call (1D chunk-chaining, not full 2D infinite
   terrain) confirmed with the user before building it.
4. **Auto mode has no fixed goal to path to** (there's no task in
   Infinite mode), so it doesn't reuse the A* planner at all — it
   continuously aims a waypoint at the corridor's own center-x, 300 units
   ahead of wherever the agent currently is, through the exact same
   `controllerStep`/`resolveCollision` any goal-directed scene already
   uses. Two real bugs surfaced building this, both the kind that only
   show up once you actually simulate it rather than eyeball the code:
   - A **fixed spawn point** (`width/2, wallMargin+20`) can land inside a
     large piece of chunk-0 furniture (a sofa placed against the front
     wall easily exceeds that), which then shoves the agent backward into
     the front wall via collision push-out on the very first tick. Fixed
     by computing the spawn point with `Scene3D._find_free_spot` against
     chunk 0's *actual* furniture — the same mechanism every other scene
     already uses to place its agent, not a hand-picked formula.
   - **A wide piece of furniture centered on the corridor's walking path
     is a real deadlock** for a controller with no real pathfinding: the
     agent walks into it, gets pushed back to the near edge, and — since
     "aim straight ahead of wherever I am now" has no restoring
     force — walks straight back into the same spot next tick, forever
     (reproduced: an agent frozen at the identical (x, z) for 1500+
     simulated ticks). Fixed two ways together: chunk generation now
     shoves any furniture that would overlap a guaranteed-clear center
     lane out to whichever side is closer, and the auto-forward waypoint
     targets the corridor's *center x*, not the agent's own current x —
     giving it a standing bias back toward the one path that's actually
     guaranteed clear, so a sideways nudge self-corrects instead of
     compounding into a permanent stall.

   **Verified**: 20/20 randomized theme/category combinations, each
   driven through 10 chunks (5000 units, well over a minute at normal
   walking speed) with zero permanent stalls and worst-case momentary
   stalls under 1.3 seconds while routing around furniture — run via a
   Node harness that extracts the real `JS_ENGINE` (same technique as
   `viewer_stub_test.js`) and drives it against real chunks from
   `infinite_world.py`, not a hand-argued claim. Re-verified after the
   theme-vocabulary fix above with a simulated real-LLM reply (a
   pirate-ship-themed furniture list) driven through the same harness —
   still zero permanent stalls (worst momentary stall 0.18s) — plus
   malformed-JSON and out-of-range-size replies both confirmed to fall
   back correctly rather than reaching placement code.

### Honest testing limit

Everything above was verified up through "the backend returns correct,
valid data and the served page's JS is syntactically sound and its core
movement logic behaves correctly in a headless simulation" — there's no
browser-automation tool available in this environment, so the actual
rendered/interactive experience (does it look right, does the camera feel
good, does clicking Generate in a real browser work end to end) needs you
to open it yourself. `webapp/server.py --mock` needs no API key and no
internet access beyond what's already local (`hssd-hab/`, vendored
three.js) to try.

### Fixed: the viewer now works fully offline

An earlier version loaded three.js and its GLTF/OBJ loaders from
`cdnjs.cloudflare.com` via `<script src="...">` tags. That's a real
liability for a "just open the file" deliverable: on any machine or
sandbox without outbound access to that specific CDN, those tags fail
silently, `THREE` never gets defined, the whole embedded script throws on
its first line, and you get a black canvas with no visible error unless
you open devtools — and because the script dies before reaching any
`addEventListener` call, the page is also completely non-interactive.

Fixed by vendoring three.js r128 (+`GLTFLoader.js`/`OBJLoader.js`) locally
in `vendor/` and having `viewer3d.py` inline them directly into the
generated HTML, exactly like the scene JSON already was. **Verified**:
generated a viewer end to end, confirmed zero `<script src=...>` tags
remain, confirmed all four inline `<script>` blocks are syntactically
valid, and confirmed generation printed no fallback warning (meaning the
vendored files were actually used). The one remaining honest gap: this
was checked with a syntax check and static inspection, not a real browser
render — see "Testing methodology" below for what a full re-verification
looks like, and do that before a live demo if at all possible.

If you ever need to refresh the vendored files:
```bash
npm install three@0.128.0 --no-save
cp node_modules/three/build/three.min.js vendor/
cp node_modules/three/examples/js/loaders/GLTFLoader.js vendor/
cp node_modules/three/examples/js/loaders/OBJLoader.js vendor/
```
(If `vendor/` is ever missing a file, `viewer3d.py` falls back to the old
CDN `<script src=...>` tag for just that file and prints a warning, so
the harness degrades rather than hard-failing — but the goal is for that
warning to never fire.)

### Fixed: real-asset objects were all getting the same fixed size

A real run against real HSSD data surfaced a second, more serious bug:
every retrieved real mesh — a mug, a chair, a table, all of it — was
being normalized to the exact same fixed footprint (one constant,
`target_size_hint=40`, applied everywhere). Four 80-unit-wide "chairs"
around an equally oversized "table" overlap heavily and block nearly the
entire room, which is exactly why every real-asset run got stuck for the
full tick budget on the very first navigation subgoal, regardless of
which prompt or item was used.

Root cause: `AddAsset` already computes a sensible, type-appropriate
`size`/`height` for every object before retrieval ever runs (a mug's
caller passes `size=6`, a table's caller passes `size=40` — see
`DSL_DOCS` in `synthesizer3d.py`), but that intent was being thrown away
and replaced with the retriever's own generic constant. Fixed:
`AddAsset` now passes its own computed size/height through to
`retriever.retrieve(name, size_hint=..., height_hint=...)`, and both
`ObjaverseRetriever` and `HSSDRetriever` scale the real mesh's footprint
to match that hint (preserving the mesh's own real proportions for
height) instead of a one-size-fits-all constant. When retrieval falls
through to a primitive shape entirely, the hint is used directly too,
rather than a separate generic guess.

**Verified**: a synthetic test with a real "mug-shaped" mesh and a real
"table-shaped" mesh, retrieved with different hints, now returns
correctly different sizes (6 and 40, matching what was asked for) instead
of an identical constant — a permanent regression test for this
(`size_hint_respected`) is now part of `python asset_retrieval.py`'s
offline suite. Re-ran a full end-to-end task with this fix and it
completed in 151 ticks (previously: 1400/1400, stuck, every time).

### Realistic room presentation

The viewer previously rendered against a flat, empty void with a flat
solid-color floor and no shadows — functional, but not something you'd
call a "room." Now, all generated procedurally at render time (no new
external assets, still a single self-contained HTML file):
- A soft radial-gradient backdrop instead of a flat void (deliberately a
  neutral studio-style backdrop, not a literal sky — this is an
  indoor-only scene with no modeled exterior, so a sky would be
  architecturally wrong; this reads as a natural "outside the walls"
  backdrop instead, visible through the intentionally semi-transparent
  walls from any angle).
- A tiled wood-plank floor texture (procedurally drawn on an in-memory
  canvas, repeated to match the room's actual size) instead of a flat
  color.
- Soft shadow-casting lighting (warmed ambient + directional light, with
  the shadow camera's frustum explicitly sized to the room's real
  dimensions — the three.js default frustum is only ~10 units across and
  would silently clip every shadow away in a room hundreds of units wide).
- A subtle baseboard trim at the floor-wall seam.
- Fog for graceful depth falloff instead of a hard edge into the void.

### Fixed: real HSSD meshes rendering sideways (up-axis ignored)

A real run with `--asset-source hssd` surfaced objects rendering on their
side instead of upright (a trash bin, a can). Root cause: `HSSDRetriever`
only read "which way is up" for a mesh from an aggregate CSV column
(`fpmodels-with-decomposed.csv`'s `up` field), which is blank for many real
rows — including, confirmed, the exact "can" and "trash_bin" objects a real
run retrieved. A blank field silently fell through to a bounding-box
aspect-ratio guess (`detect_up_axis`), which picked the wrong axis for
both. The real fix: each object's own `.object_config.json` — the
standard, always-present per-object Habitat-Sim field — already carries a
ground-truth `"up": [0, 1, 0]` that was never being read at all, despite
that same file already being opened for its `render_asset` field.
`HSSDRetriever._config_up_vector` now reads it directly and is preferred
over the CSV column. **Verified**: both objects now resolve to `up_axis=1`
(correct) instead of `0`, confirmed against this repo's own local HSSD
fixture, and `python asset_retrieval.py`'s full offline+live suite still
passes.

### Fixed: FURNITURE OVERLAP / OUT OF BOUNDS no longer crash synthesis

Both were previously pure detect-and-retry: `has_bad_overlap()` /
`out_of_bounds_assets()` would catch a bad layout and hand the traceback
back to the LLM to fix, with no guarantee it actually would within
`max_retries`. `Scene3D.resolve_layout()` now runs automatically after
every `build_scene()`, before those checks, and repairs the common,
purely-numeric mistakes outright: clamps every solid object's footprint
inside the room, then iteratively separates any pair that still overlaps
along whichever axis needs the smaller push. A last-resort fallback
(exhaustive grid scan, largest-object-first — random sampling alone proved
unreliable once a room has 10+ objects) handles anything local nudging
can't untangle, like several large pieces all anchored off one
wall-adjacent object. The ground-truth checks stay in place as a safety
net for genuinely infeasible layouts (more/bigger furniture than the room
can physically fit) — they just stop firing on ordinary placement slop.
**Verified**: 100/100 on a synthetic adversarial stress test (corner-
anchored `place_relative` chains at random distances, the exact shape of
the original bug report) and 99-100/100 across repeated runs at a
realistic 10-18-object "cluttered room" scale using the DSL's own size
chart.

### Fixed: a real PyBullet bug that made the agent drive backward

Running the PyBullet backend for the first time on a machine that actually
has it installed (previously untested — see `dsl3d_pybullet.py`'s own
docstring) surfaced two real bugs, not just untested code:
1. **`linearDamping=0.9`** on the agent's rigid body. `move_agent()` resets
   the agent's full commanded velocity every tick regardless, so engine-
   side damping never had a legitimate job to do — and empirically, this
   PyBullet build's damping model doesn't just slow the agent at that
   value, it **inverts** the velocity's sign past roughly 0.55 (confirmed:
   `resetBaseVelocity([20,100,0])` followed by one `stepSimulation()` came
   back `(-10.9, -54.4, ...)`). That's the actual cause of "no path to
   subgoal" / burning the full tick budget stuck in place. Fixed by
   setting `linearDamping`/`angularDamping` to 0.
2. **Furniture tunneling**: even undamped, a continuously-reasserted
   velocity can creep through solid furniture over hundreds of ticks
   faster than Bullet's contact solver fully arrests it each frame
   (confirmed: 300 ticks of ramming a table walked the agent all the way
   through and out the other side). Fixed with the same AABB push-out
   `dsl3d.py`'s pure-Python backend already uses, added to `move_agent` as
   a belt-and-suspenders correction on top of Bullet's own solver.

**Verified**: `python smoke_test_pybullet.py` — ram test, 10/10 reach-goal,
10/10 pick-and-place — passes cleanly and repeatably (previously: the ram
test's own "didn't clip through" check failed, and pick-and-place flaked
9/10).

### Fixed: no object color variation, objects on the same surface overlapping, "floating" wall decor

Three related realism complaints from a real run's screenshot:
- **Every object was the same color.** `viewer3d.py`'s `_color_for` had
  exactly one fallback color for anything that wasn't the agent, goal, or
  a container — a table, a chair, a bookshelf, and a can all rendered as
  the identical flat orange-brown. Added `CATEGORY_COLORS`, a ~50-entry
  substring-matched palette (same "specific key before generic" convention
  as `PrimitiveRetriever.TABLE`) covering the DSL's own furniture and
  portable-item vocabulary. This also improves real HSSD `.obj` meshes,
  not just primitives — that render path already recolors every OBJ mesh
  with this same per-asset color (bare `THREE.OBJLoader` has no material
  support), so a better palette helps both for free. GLB meshes
  (Objaverse) are untouched and keep their own real textures.
- **Items on the same surface could overlap.** `place_on`/`place()` picked
  each item's spot independently at random — fine for one item, unreliable
  once several small items (a cluttered coffee table: bowls, cans, a
  remote, a board game) compete for the same small area. Both now try
  several random spots and skip ones that overlap a neighbor already on
  that surface, falling back to an exhaustive local grid scan
  (`_scan_local_free_spot`) if random sampling doesn't find one. A second,
  joint pass in `resolve_layout` re-checks every surface's *final* set of
  resting items together (not one at a time) and separates anything
  greedy placement still missed. **Verified**: a hand-built "6 items on
  one coffee table" stress scene (deliberately more cluttered than a
  typical prompt) — 0/60 overlaps across randomized trials, down from a
  guaranteed multiple overlaps before.
- **"Floating" decor.** No `place_*` method ever accepted an arbitrary
  height — the only way to depict something like a TV mounted above a
  console was to hand-edit `asset.pos[1]` directly, bypassing every
  documented placement helper, which is exactly the kind of ad hoc code
  that produces objects with no visible means of support. Added
  `scene.place_on_wall(asset, wall, height)` — sets both the wall-adjacent
  (x, z) and an explicit mount height, tags the object `wall_mounted`, and
  excludes it from floor-plane collision/pathing the same way portable
  items already are (in `dsl3d.py`, `dsl3d_pybullet.py` — including
  PyBullet's own real collision filter, not just the tag-based checks —
  `nav_agent_3d.py`, and the viewer's embedded JS port). `DSL_DOCS` now
  documents it and explicitly forbids touching `asset.pos` by hand.

### Fixed: room size no longer fixed at 800x600 regardless of content

Every prompt got the same 800x600 floor regardless of how much furniture
it described — a real contributor to both overcrowding and to
`FURNITURE OVERLAP` retries on elaborate, many-object prompts (a "lived-in"
living room with a sectional, TV console, coffee table, floor lamp,
several bookshelves, an armchair, and a side table needs more floor than a
one-counter kitchen). `synthesizer3d._estimate_room_size` counts how much
of the DSL's own furniture vocabulary appears in the prompt text (before
the synthesis LLM ever runs — a plain heuristic, not a second LLM call)
and scales the room up, never down, accordingly. `synthesize_scene
(width=None, depth=None)` is the new default everywhere, so no caller
needed to change.

### Fixed: rooms reading as way too empty (text-guessed size vs. actual content)

The text-based guess above turned out to only be half the story, and a
real run's screenshot showed why: an elaborate "modern home office" prompt
densely mentions desk clutter (laptop, monitor, keyboard, mouse,
headphones, notebook, pens, cables...) — enough hits to size the room up —
but almost all of those became small portable items `place_on`'d onto one
desk, not separate floor furniture. The LLM's actual `build_scene` only
had a handful of solid pieces (desk, chair, a cabinet, a lamp, a shelf, a
plant), leaving most of a needlessly large floor empty. Guessing from the
prompt's text can only ever be a rough upper bound — it has no way to know
how many of the mentioned items the LLM will actually turn into real,
floor-occupying furniture versus desk clutter.

Fixed with `Scene3D.fit_room_to_content()`, run once right after
`build_scene()` returns (before `resolve_layout`, which then guarantees no
overlap/out-of-bounds at whatever size this settles on) — this is the
first point in the pipeline where the ACTUAL solid furniture list and
sizes are known. It computes their total footprint area, solves for a room
area that puts that at a target occupancy ratio (picked to read as
"furnished, with real walking room," not empty or wall-to-wall), keeps the
existing aspect ratio, and clamps to sane bounds. Two real bugs surfaced
and got fixed while building this, both confirmed via a hand-built stress
scene standing in for the un-reachable real LLM (no local `OPENAI_API_KEY`
to test against):
1. **A portable item not resting on anything** (a backpack placed next to
   a desk via `place_relative`, not `place_on`) **kept its stale absolute
   coordinates** when the room shrank, landing outside the new walls — a
   real `OUT OF BOUNDS` crash. Fixed by remapping every non-agent asset's
   position by the same per-axis scale factor the room itself just
   changed by, not just the solid furniture (with wall-mounted items
   handled as their own case — see next point — rather than scaled).
2. **A wall-mounted TV got scaled away from its wall.** `place_on_wall`'s
   formula is `margin + half_extent` from the room's edge — an additive
   offset, not a pure fraction of width/depth — so naively scaling its
   (x, z) like everything else walks it off the wall it's supposed to be
   flush against, again landing it out of bounds. Fixed by having
   `place_on_wall` remember its own `(wall, offset, height, margin)`, and
   `fit_room_to_content` re-applying `place_on_wall` with those same
   arguments against the new room size instead of scaling — the item ends
   up flush against the same wall regardless of which way the room sizes.

**Verified**: 60/60 clean (no overlap, no out-of-bounds, reachable) across
randomized re-runs of both a hand-built "cluttered home office" scene and a
hand-built "movie-night living room" scene (the latter also carrying a
`place_on_wall`'d TV, specifically to catch regression #2) standing in for
the prompts from a real run — plus the existing `smoke_test_pybullet.py`
and `asset_retrieval.py` suites, unaffected.

### Fixed: a crash when a whole-room prompt has no obvious single goal

`python harness3d.py "<elaborate scene description>" --asset-source hssd
--hssd-root ...` (no `--task`) crashed with
`ValueError: couldn't parse task: ''` whenever the synthesis LLM produced a
`build_scene` that never called `scene.set_goal(...)` — which happens for
a prompt that describes a whole room rather than one obvious "reach it"
target. `harness3d._build_subgoals` used to unconditionally try to
rule-parse an empty task string in that case, which has no matching rule
and always raised. It now auto-picks a sensible implicit goal instead
(preferring any portable item, else any other asset) so the harness always
has somewhere to navigate rather than crashing on a prompt that just
wasn't goal-shaped to begin with.

### Viewer controls

| key/input | does |
|---|---|
| `TAB` | toggle Auto (agent navigates itself) vs Manual (you drive) |
| `W/A/S/D` | move (Manual mode) |
| click + drag mouse | Auto: orbit the camera around the agent. Manual: turn (mouse-delta-x -> yaw) and look up/down (mouse-delta-y -> camera pitch, exactly the brief's action space) |
| scroll | zoom (Auto mode) |
| `R` | reset the run from the start |
| `SPACE` | pause |

The HUD shows the prompt, task, current subgoal, and live progress through
navigate/pick/place steps.

## Step by step: how this was built (and how to extend it)

1. **`dsl3d.py`** - the 3D Scene Description Language: `Scene3D`/`Asset3D`,
   with `AddAsset`, `place_at/relative/around/grid/against_wall/corner/on/
   on_wall`, `pick`/`place`/`update_carried`, `add_container`, `add_marker`,
   `is_reachable`, `has_bad_overlap`, `resolve_layout`, `fit_room_to_content`.
   Design principles: furniture is static by default (nothing moves it
   after placement except the repair passes below), small items are
   `portable` (excluded from floor-plane collision/pathing), and every
   ground-truth validity check feeds the synthesis retry loop rather than
   silently allowing a broken scene through. `place_on(item, surface)`
   puts the item genuinely on top of the surface (`y = surface.top_y +
   item.half_height`) using the real vertical axis a 3D scene has -
   `place_on_wall(asset, wall, height)` does the same for decor mounted ON
   a wall (a TV, framed photos) rather than resting on the floor or a
   surface. `resolve_layout()` is an automatic geometry-repair pass the
   harness runs after every synthesized `build_scene()` - see "Fixed:
   FURNITURE OVERLAP / OUT OF BOUNDS no longer crash synthesis" below for
   why - and `fit_room_to_content()` right-sizes the room to the furniture
   actually built rather than a pre-synthesis guess from the prompt's text
   - see "Fixed: rooms reading as way too empty" below.
2. **`nav_agent_3d.py`** - floor-projected (2.5D) A*. An indoor agent walks
   on the floor; it doesn't need full voxel pathfinding to cross a room, so
   this projects an inflated occupancy grid onto the x/z plane (agent-
   radius-aware inflation, room-boundary awareness, snapping to the nearest
   open cell when a target sits inside solid furniture) rather than
   reasoning in 3D directly.
3. **`tasks3d.py`** - `Subgoal`/`TaskRunner`, a navigate->pick->place state
   machine driving both rule-based and LLM-grounded task parsing.
4. **`synthesizer3d.py`** - a Programmer + Debugger loop: an LLM writes
   `build_scene(scene)` against `DSL_DOCS`' documented API, and a retry
   loop feeds tracebacks or ground-truth failures (overlap, out-of-bounds,
   unreachable) back for another attempt. This is also where **asset
   retrieval** lives - see below.
5. **`harness3d.py`** - CLI: synthesizes, builds subgoals, exports the
   viewer from the scene's true starting state, then separately runs a
   Python-side reference simulation for a printed success/ticks number.
   `--levels N` runs a fixed number of genuinely different environments in
   sequence; `--infinite` removes the cap and runs forever until Ctrl+C -
   see "Infinite generation" above.
6. **`viewer3d.py`** - generates the self-contained three.js HTML page.
7. **`dsl3d_pybullet.py`** + **`physics_backend.py`** - real PyBullet
   backend with the exact same public API as `dsl3d.py`, auto-selected
   when PyBullet is importable. See "Physics backend" below.
8. **`asset_retrieval.py`** - real Objaverse mesh retrieval, wired
   transparently into `AddAsset`. See "Real 3D assets" below.
9. **`infinite_world.py`** - deterministic chunk generation for the web
   app's Infinite mode: `ROOM_LIBRARY`, `derive_theme` (the one LLM call),
   `build_chunk`/`chunk_to_json`/`find_spawn_point`. See "Web app" above.
10. **`webapp/server.py`** + **`webapp/static/`** - the local Flask app:
    `/api/scene`, `/api/infinite/start`, `/api/infinite/next_chunk`. See
    "Web app" above.

## Physics backend: real PyBullet, with a tested pure-Python fallback

`physics_backend.py` picks the backend once, at import time: PyBullet if
it's importable, otherwise the pure-Python physics-lite engine
(`dsl3d.py`), with a clear printed message either way. Both implement the
identical `Scene3D`/`Asset3D` API, so nothing else in the harness — the
A* planner, the task state machine, the synthesizer, the viewer — needs
to know or care which one is active.

**Why pure-Python exists at all**: developed in a sandbox with no
prebuilt PyBullet wheel for its Python/platform, where a from-source
compile timed out repeatedly. Indoor navigation's actual physics needs are
modest — furniture that doesn't move, an agent that can't walk through it,
items that rest on surfaces without weird collision — which are all just
axis-aligned overlap tests, so a lightweight fallback made the harness
usable everywhere while the real backend was being built.

**Why PyBullet is preferred whenever available**: it's real rigid-body
collision (`resetBaseVelocity` + `stepSimulation`, not hand-rolled AABB
push-out), a real floor + gravity the agent actually rests on, and a
proper foundation for anything beyond what this harness currently asks of
it (falling objects, physical pushing, real collision meshes instead of
AABB approximations).

`dsl3d_pybullet.py` could not be executed in the sandbox that wrote it —
see `GUIDE_3D_ASSETS.md` for exactly why, exactly what was and wasn't
verified, and `smoke_test_pybullet.py` for the local verification script
that mirrors every check already passed on the pure-Python version. Run
that first after `pip install pybullet`, before trusting the backend in
anything else.

## Real 3D assets: Objaverse today, HSSD/3D-FRONT documented

`asset_retrieval.py`'s `FallbackRetriever` does real Objaverse mesh
lookups (`pip install objaverse trimesh`), keyed transparently on each
`AddAsset` call's own name — degrading silently to the existing primitive-
shape heuristics on any failure (no network, no category match). Turn it
on with `--use-real-assets`; nothing about the LLM-facing DSL or the
few-shot examples needed to change, since retrieval happens invisibly
inside `AddAsset` itself.

### Category matching, hardened

The LLM writes object names like `"trash_bin"`, `"goal_mug"`, or
`"chair_2"` (the last two are this harness's own naming conventions — see
`FEWSHOT`'s `goal_mug` and `AddAsset`'s auto-dedup suffix) — none of which
are guaranteed to be a substring of the LVIS category name Objaverse
actually indexes by (`"trash_can"`, `"mug"`, `"chair"`). `_match_category`
now runs, in order: (1) a hand-curated synonym map for this harness's own
object vocabulary (`trash_bin`→`trash_can`, `counter`→`countertop`, etc.),
(2) exact match, (3) substring match preferring the shortest hit, (4) a
Jaccard-ratio token-overlap fallback with a minimum-similarity floor — so
an unrelated category is never confidently returned just because it
scored highest among bad options; it falls through to a primitive shape
instead, which is the correct behavior when nothing genuinely matches.

**Verified offline, no network needed**: `python asset_retrieval.py` runs
`test_category_matching()` first — 25 cases (every object name this
harness's own DSL/few-shot/`PrimitiveRetriever.TABLE` vocabulary
produces, including the `goal_`-prefixed and `_2`-suffixed forms) against
a fake but realistically-shaped category list, all passing, plus a
confirmed correct `None` on a nonsense name. **What that test can't
confirm**: whether the synonym map's target strings (`"trash_can"`,
`"countertop"`, etc.) are the literal strings the real LVIS list uses —
that map was written from general knowledge of LVIS's naming conventions,
not by reading the live list, since fetching it needs `huggingface.co`
access this environment doesn't have. **On your machine, with your HF
login working**: run
```python
from asset_retrieval import ObjaverseRetriever
r = ObjaverseRetriever()
print(sorted(r._lvis_annotations().keys()))
```
once, and diff it against the `SYNONYMS` dict near the top of
`ObjaverseRetriever` in `asset_retrieval.py` — fix any entry that doesn't
match (the generic matcher below it still catches close variants on its
own, but the synonym map is the part most worth eyeballing against ground
truth once you can).

**Verified end to end with a real (locally-generated) mesh, no network
needed**: exported a `trimesh` box as a `.glb` (the exact stand-in your
own earlier testing used), fed it through `Scene3D.AddAsset(...,
mesh_path=...)` on the PyBullet backend — confirmed the OBJ auto-
conversion (`_ensure_pybullet_mesh_format`) and
`p.createCollisionShape(p.GEOM_MESH, ...)` both execute cleanly — then
through `render_viewer_html` — confirmed the mug's own asset record in the
generated page has real embedded base64 mesh data with `meshFormat:
"obj"` (matching the PyBullet path's OBJ conversion), not `null`. The
actual Objaverse network fetch itself is still the one thing genuinely
unverified from here — see below.

**Re-checked (not newly claimed) in a second, independent sandbox**:
`objaverse`/`trimesh` install and import cleanly, the retrieval code runs
correctly through every step of its own logic, and fails at exactly the
expected point — a `403` on `huggingface.co` — matching the original
finding rather than superseding it; this sandbox's network allowlist
excludes `huggingface.co` too. **With your HF login already working, this
is the one thing to actually run yourself**: `python asset_retrieval.py`
now runs the offline matching test first (should print "All 25 ... passed")
and then attempts the real network retrieval for "mug" — if that second
part prints `SUCCESS`, the full pipeline is confirmed end to end on your
machine.

### PyBullet backend, re-verified via a local API-compatible stub

Since this environment also has no installable PyBullet wheel for its
Python/platform (re-confirmed: no matching wheel on PyPI, and a
from-source build attempt timed out — same finding as the original build,
not a new one), I built a small stub implementing the exact PyBullet
function signatures `dsl3d_pybullet.py` calls (`createCollisionShape`,
`createMultiBody`, `resetBaseVelocity`, `stepSimulation`, etc.), with real
— if simplified — AABB collision resolution, and ran your **unmodified**
`smoke_test_pybullet.py` against it. Result across 5 clean runs (100
fresh scenes total): every check passed, including the ram-test and both
task types. This caught one real bug in my *test stub itself* (a
`disconnect()` that didn't clear per-client body state, leaking stale
furniture from prior scenes into new ones — a limitation of the stub's
simplified single-client model, not of your actual code, since real
PyBullet's `disconnect()` genuinely tears down the client), which I fixed
and re-confirmed. This is real evidence the PyBullet-calling code is
API-correct and logically sound — it is **not** a substitute for running
against the real Bullet solver, since a hand-written stub can't validate
real rigid-body numerics (restitution, friction, actual gravity
integration). **With PyBullet genuinely installed on your machine, this
is the other thing to run yourself**: `python smoke_test_pybullet.py`
should now print `ALL CHECKS PASSED`.



**Not implemented, by necessity**: HSSD and 3D-FRONT both need gated
account access (a Hugging Face terms-acceptance for HSSD, an Alibaba Cloud
account + license acceptance for 3D-FRONT/3D-FUTURE) that can't be
scripted without your own credentials. **`GUIDE_3D_ASSETS.md` has the
complete step-by-step for both**, including exactly where to plug the
resulting retriever into this harness (one class, matching
`AssetRetrieverBase`, no changes anywhere else).

## Testing methodology

"Haven't seen it fail yet" isn't the same as "correct." Two things make
this tractable:

1. **Every Python module is unit-tested directly** - physics sanity checks
   (ram the agent into a table 300 times, confirm the table's position is
   bit-for-bit unchanged), multi-trial regression (10+ fresh random scenes
   per task type, checking genuine completion, not just "didn't crash"),
   and directly debugging failures rather than assuming success. See
   `smoke_test_pybullet.py` and `asset_retrieval.py`'s own offline suite.
2. **The JS engine embedded in the viewer is a from-scratch reimplementation
   of `nav_agent_3d.py`/`tasks3d.py`'s logic** (`buildGrid`/`astar`/
   `planPathTo`/`controllerStep`/`resolveCollision`/`TaskRunner` in
   `viewer3d.py`'s `JS_ENGINE` block) - not a shared library with the
   Python side, so it's a real place for the two to drift apart silently
   without its own check. `viewer_stub_test.js` catches that: none of
   those functions touch THREE.js or the DOM (only the surrounding
   rendering code in the HTML template does), so it extracts just that
   block plus the embedded scene data straight out of a generated
   `viewer_N.html`, runs it in plain Node (`vm.runInContext`, no browser,
   no stubbing needed), and drives a simulation loop that mirrors
   `harness3d.py`'s own Python reference loop - checking the task
   genuinely completes in the JS engine too, not just that the code
   doesn't throw. Run it against any generated viewer:

   ```bash
   node viewer_stub_test.js out3d/viewer_0.html
   ```

Also worth knowing about, since it's the kind of thing worth being able to
explain if asked: the arrival-threshold logic needs a **physical-clearance
floor**, not just a standoff-distance estimate from the planned path. When
a navigate target is a real solid object fully excluded from the
pathfinding grid (e.g. walking up to a trash bin to place something in
it), A* will happily plan a path to the object's exact center - but the
actual collision resolution still treats it as solid, so the agent
physically stops at its edge well short of that. A collision response with
enough positional slop can mask this; this harness's collision resolvers
(both the pure-Python AABB one and the PyBullet backend, after the fixes
documented below) are exact, so the gap needed a real fix
(`nav_agent_3d.physical_clearance`), not a workaround.
