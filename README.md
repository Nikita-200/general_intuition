# Infinite Environment Generation: a 3D Indoor Agent Harness

Text prompt → generated 3D indoor scene → agent navigates/picks/places →
watch it live and interactive in a browser, or drive it yourself.

An LLM (or a deterministic `--mock` mode requiring no API key at all)
writes a small Python program against a documented scene-description DSL
for every prompt you give it, so the environment space is open-ended — any
prompt describing an indoor room becomes a new scene, procedurally, in
code, not from a fixed asset library. 

The "agent" doing navigation here is a hand-rolled, floor-projected A* planner (`nav_agent_3d.py`) standing in for a vision-based policy. 
(text prompt → verifiable scene → a navigable result)

This covers what the brief asks for on indoor scenes: reach a goal,
pick-and-place, auto-navigate or drive manually — all backed by code-level
verifiable objectives rather than pixel inspection.

## Run it

```bash
# create a virtual env
python -m venv .venv
.venv\Scripts\Activate.ps1

# install dependencies
python -m pip install openai pybullet objaverse trimesh   # only needed for the real (non --mock) synthesizer

# download the objects/ directory from HSSD dataset
python download_hssd.py --out ./hssd-hab  

# real PyBullet rigid-body physics instead of the pure-Python fallback:
pip install pybullet && python smoke_test_pybullet.py   # verify first, see GUIDE_3D_ASSETS.md
python harness3d.py "..." --mock   # auto-detects and uses it once installed
python inspect_hssd.py --root /path/to/your/hssd-hab

# pick-and-place with LLM:
python harness3d.py "a kitchen with a can on the counter and a trash bin in the corner" --task "pick the can and place it in the trash bin" --mock --asset-source hssd --hssd-root ./hssd-hab

# pick-and-place mock:
python harness3d.py "a kitchen with a can on the counter and a trash bin in the corner" --task "pick the can and place it in the trash bin" --mock

# never stop generating new environments until you Ctrl+C:
python harness3d.py "a kitchen with an island and a can on the counter" --mock --infinite

# real 3D meshes (Objaverse) instead of primitive shapes:
python harness3d.py "..." --mock --use-real-assets
```

`physics_backend.py` auto-detects PyBullet and uses it when available,
falling back to a pure-Python physics-lite engine otherwise.

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
- **Infinite mode** is a continuously-extending,
  boundary-less corridor you move through for ~30 seconds, new areas
  streamed in live as you approach the edge of what's already loaded —
  see below.

### Infinite mode: how it actually generates a "never-ending" space

Confirmed with real design tradeoffs, not just a UI toggle:

1. **One real LLM call, then deterministic forever.** `/api/infinite/start`
   asks the model once for a **prompt-specific furniture list** — real
   named objects with realistic sizes (`derive_theme` in
   `infinite_world.py`). Now the LLM directly proposes named furniture and portables for
   *that* scene, validated and clamped to a
   sane size/height range before anything reaches placement code. Every
   individual chunk after that is still generated **deterministically** —
   no LLM, no network, no per-chunk latency — reusing the same
   `AddAsset`/`place_*`/`resolve_layout` DSL and HSSD retrieval as any
   other scene, each chunk sampling a random subset of the SAME
   theme-specific pool for variety. This was a deliberate design choice,
   not an oversight: a live LLM call per chunk was considered and rejected
   — chunks are only fetched ~150 units before the edge of loaded content,
   about a second of walking, and a real LLM call routinely takes several
   seconds, which would cause visible stalls almost every chunk. 
2. **True live streaming, not one big pre-generated blob.** The frontend
   tracks the agent's own z position; once it gets within ~150 units of
   the far edge of what's loaded, it `fetch()`es
   `/api/infinite/next_chunk` for the next one and splices it into the
   live three.js scene (`appendChunk`) without interrupting movement.
3. **"No boundaries" is a real geometry choice.**
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
   uses. 

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
   harness runs after every synthesized `build_scene()` and `fit_room_to_content()` right-sizes the room to the furniture
   actually built rather than a pre-synthesis guess from the prompt's text
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


**Not implemented, by necessity**: HSSD and 3D-FRONT both need gated
account access (a Hugging Face terms-acceptance for HSSD, an Alibaba Cloud
account + license acceptance for 3D-FRONT/3D-FUTURE) that can't be
scripted without your own credentials. **`GUIDE_3D_ASSETS.md` has the
complete step-by-step for both**, including exactly where to plug the
resulting retriever into this harness (one class, matching
`AssetRetrieverBase`, no changes anywhere else).

## Testing methodology

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
## References

- [InteriorAgent](https://openreview.net/pdf?id=ypBfokcXvA)
- [HSSD Dataset](https://huggingface.co/datasets/hssd/hssd-hab) 
- [Minecraft Open-Ended World Generation](https://github.com/mindcraft-bots/mindcraft)
