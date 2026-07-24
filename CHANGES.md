# What was actually wrong, and what changed

Verified each of these by regenerating a scene, inspecting the generated
HTML/JSON directly, and (where relevant) checking the math by hand in
Node — not just re-reading the code. Details below.

## 1. "No background, all white" — root cause found: silent CDN fallback

`viewer3d.py` inlines three.js from `vendor/three.min.js` so the generated
page works completely offline. The vendor file you had was named
`three_min.js` (underscore) — one character off from what the loader
looked for (`three.min.js`, dot). That mismatch was silent: it just prints
a console warning and falls back to a `<script src="https://cdnjs...">`
tag. Anywhere without live internet access to that exact CDN (a sandboxed
browser, an offline demo, a locked-down network), `THREE` never gets
defined, the whole embedded script throws on its very first line, and
you get a blank page with no visible error — which looks exactly like
"no background, all white, nothing rendered."

**Fixed two ways:**
- Renamed/placed the vendor files correctly (`vendor/three.min.js`,
  `vendor/OBJLoader.js`, `vendor/GLTFLoader.js`) so they're actually found.
- Made `viewer3d.py`'s loader also try the underscore-name variant
  automatically, so this specific mismatch can't silently recur.
- **Verified**: regenerated both demo viewers and confirmed zero
  `cdnjs.cloudflare.com` references remain in the output HTML.

Also added, now that rendering was confirmed working:
- Walls now use the room's real ceiling height (was hardcoded to a squat
  70 units regardless of the actual ~260-unit ceiling) plus a faint
  translucent ceiling plane, so the room reads as a real enclosed space
  from any camera angle instead of a knee-high fence around an open floor.
- (The soft backdrop, wood-floor texture, and shadows were already coded —
  they just never had a chance to render while the script was failing to
  load at all.)

## 2. WASD — A and D were swapped relative to what the camera shows

`manualStep()` computed the strafe ("right") direction using `yaw + 90°`.
For this project's own forward vector (`sin(yaw), cos(yaw)`) and
three.js's actual camera basis (checked directly against three.js's
lookAt-matrix construction), the true "right" direction is `yaw - 90°`,
not `+90°`. The old code had D moving toward what the camera renders as
the *left* side of the screen, and A toward the *right* — confirmed by
computing both vectors by hand for several yaw values in `viewer3d.py`;
they were exact negatives of each other.

**Fixed**: changed `yaw + Math.PI/2` to `yaw - Math.PI/2` in the strafe
calculation. W/A/S/D are now forward/left/backward/right exactly as seen
on screen, verified numerically against the camera's own right vector.

## 3. Object sizes — replaced vague guidance with a real, comparative chart

The only actual sizing guidance previously given to the scene-generating
LLM was a loose parenthetical ("a table ~40-50 tall, a counter ~45, a
chair ~45") that didn't cover most furniture types and wasn't internally
consistent (a "table" at 42 tall was barely taller than the "chairs"
around it, a kitchen "counter" was only 45 tall — below waist height on
a 170cm-tall agent).

**Fixed**: `synthesizer3d.py`'s `DSL_DOCS` now ships a full (size, height)
chart in centimeters, anchored to the agent's own 170cm height, covering
~30 common object categories (tables, seating, storage, appliances,
lighting, etc.) with an explicit rule to interpolate from the nearest
category for anything not listed, and to prioritize *comparative*
correctness (a chair must read as shorter than the table beside it, etc.)
over any single absolute number. The two hardcoded few-shot examples
(`FEWSHOT`, `FEWSHOT_PICK_PLACE`) were rewritten to match — e.g. the
dining table went from 42-tall (barely above its own chairs) to a correct
75, the kitchen counter from 45 to 90. `PrimitiveRetriever.TABLE` (used
for the `--asset-source` real-mesh fallback path, in both
`synthesizer3d.py` and `asset_retrieval.py`) was expanded from 10 entries
to ~30 and reordered so specific multi-word keys (`"coffee table"`) are
matched before the generic single-word keys they contain (`"table"`) —
the old order meant a couple of the original entries were silently dead
code. `add_container`'s default size dropped from a same-footprint-as-a-
small-table 45/40 to a correctly bin-sized 22/35.

**Verified**: regenerated both demo scenes and printed every asset's
actual y-range from the exported scene JSON — table 0–75cm with chairs
0–45cm around it, counter-height objects now clearly taller than seating,
mug resting exactly on the table's new top surface (75–85cm).

## 4. Floating objects — one real bug found (the agent's own mesh)

Checked this by tracing the actual y-values written to `scene_N.json`:
every piece of furniture was already correctly grounded (asset center
y = half-height, meaning bottom = 0) — `AddAsset`, `place_on`, and the
move-and-slide collision in `dsl3d.py` were all already correct here, and
this was confirmed directly from real generated scene data, not assumed.

The one genuine bug was in the **agent's own visual mesh** in
`viewer3d.py`. The group holding the agent's body/head/nose is placed at
world `y = a.y`, which — same as every other object — is already the
vertical *center* of the agent's bounding cylinder. But the body/head/nose
child offsets were written as if that same point were the *floor*, and
then stacked upward from there, double-counting the offset. Net effect:
the agent's visible body floated with its feet at half its own height
above the ground (for a 170-tall agent, floating ~85 units up) — the most
visually obvious instance of "floating," even though the underlying data
was fine.

**Fixed**: recomputed the body/head/nose local offsets so the body's
bottom lands exactly at the group's local `y = -half_h`, which combined
with the group's world position lands it at world `y = 0`. **Verified
numerically** (not just re-read): computed the resulting world-space
bottom of the body in Node using the exact same formula now in the code —
`0.0000`, i.e. touching the floor exactly.

Also added, as a preventative fix for real-mesh mode (`--asset-source
objaverse/hssd`): a `groundMeshOnBoundingBox()` helper that recenters any
loaded OBJ/GLB mesh on its own bounding box before positioning it, since a
real mesh file can be authored with any origin convention (centered,
floor-resting, offset pivot, etc.) — previously nothing corrected for
that, so a real mesh could float or clip into the floor depending on how
its source file happened to be built.

## 5. No overlap collisions

This was already enforced structurally and wasn't broken:
`synthesize_scene`'s retry loop rejects any generated scene where
`has_bad_overlap()` finds two solid objects overlapping (checked in the
`dsl3d.py`/`dsl3d_pybullet.py` `Scene3D.has_bad_overlap`), and the agent's
own movement is resolved against solid AABBs every tick in both
`move_agent` (Python reference sim) and `resolveCollision` (the browser
viewer). No change needed here beyond re-verifying it still holds with
the new, larger furniture sizes above — reran both demo scenes end-to-end
and both completed successfully with the new proportions.

## Everything not mentioned above

Was reviewed and left alone — the A* pathfinding, the pick/place task
state machine, the PyBullet backend, and the Objaverse/HSSD retrieval
pipeline were all working as designed and unrelated to the five issues
raised.

---

## Round 2: Manual mode couldn't see furniture or pick/place objects

Verified both of these the same way as round 1 — reproduced the exact
symptom first, then found and fixed the real cause, then re-verified with
an actual headless-Chromium render and a scripted playthrough (not just
re-reading the code).

### 6. Manual/first-person camera was floating near the ceiling

`agent.y` is the agent's vertical CENTER (half its height above the
floor) — same convention every object uses. The Manual-mode camera code
added ANOTHER eye-height offset on top of that already-centered value
(`agent.y + eyeHeight`), which for a 170cm-tall agent put the first-person
camera at world height ~246 — a few units short of the room's own
260-unit ceiling. Looking roughly horizontal from almost the ceiling, the
camera was looking mostly over the top of everything: furniture (max
~180 tall) sat far below its sightline, and even the agent's own body
(topping out around 155) was below where the camera was looking. Auto
mode's orbit camera didn't have this bug and was never affected, which is
why the furniture was visible for the exact demo files handed over
previously (confirmed with a real render — see below), but not in Manual
mode, which is apparently how it was actually being tested.

**Fixed**: `eyeHeight` is now an absolute world height (`agent.half_h * 1.8`,
~90% of full height), set directly as the camera's world Y — not added on
top of `agent.y`. **Verified**: rendered the actual generated HTML in a
real headless Chromium (not just a syntax check), teleported the agent
next to the coffee table in Manual mode, screenshotted the first-person
view, and confirmed the table's exact color is now clearly present in the
frame (camera Y came out to 153, matching the agent's real eye level —
previously this would have been ~246).

### 7. Manual mode could never trigger pick/place

`arrivalThreshold` (how close counts as "arrived," including the extra
clearance needed for solid objects like counters/bins) was only ever
recalculated inside `replan()`, and `replan()` was only ever called from
the `auto`-mode branch of the tick loop. In Manual mode it stayed at
whatever it last was — the default `30` if Auto was never engaged — which
is smaller than the *physical minimum approach distance* to most solid
targets (e.g. a counter's own half-width alone is 45). So walking up to
the can on the counter, the collision resolver would stop you at the
counter's edge — genuinely as close as physics allows — while the
"close enough" check still demanded getting within 30 units of the
counter's *center*, which is inside the counter and physically
unreachable. Pick/place could never fire, no matter how close you got.

**Fixed**: the subgoal-change check that triggers `replan()` now runs
unconditionally at the top of the tick loop, regardless of control mode —
Auto mode still additionally uses the waypoints it computes; Manual mode
now also gets a correctly recomputed threshold the moment each new
subgoal becomes current. **Verified** by scripting an actual playthrough
against the real generated HTML in headless Chromium: pressed real Tab
(not just flipping the JS variable — that matters, see below), walked the
agent step-by-step up to the can (not teleporting through the counter,
which would trigger an unrelated collision-resolution edge case), and
confirmed `pick` fired (`carried: 'can'`) and, continuing to the trash
bin, `place` fired too and the task's own `isComplete` flag went `true` —
a full, real manual pick-and-place completing end to end.

One related detail worth knowing: switching modes with **TAB** already
force-resets the replan state (`lastIdx = -1`), so it correctly
recomputes the threshold for wherever you are when you switch — this was
already correct and didn't need a change, but it's why the fix above only
had to move where `replan()` gets called, not touch the TAB handler.

---

## Round 3: objects vanishing entirely with real mesh retrieval (HSSD/Objaverse)

This one only shows up when `--asset-source hssd` (or `objaverse`) is
actually in use — with no asset-source flag, every object is a primitive
box/cylinder by design, and that's not a bug (see the note at the end of
this section on telling the two situations apart).

### The bug, reproduced and fixed without needing your actual HSSD download

In `viewer3d.py`'s browser-side mesh loader, once a real OBJ/GLB mesh
successfully **parsed**, the code removed the working primitive shape and
added the loaded mesh **unconditionally** — with no check that the parse
actually produced anything visible. A file can parse without throwing an
error and still contain zero real geometry: an unsupported extension
(Draco/Basis compression, common in optimized real-world asset
pipelines), a reference to an external texture/buffer file that isn't
embedded (this harness only embeds a single mesh file, not a whole
folder), or a genuinely empty/degenerate export. In every one of those
cases, the old code still deleted the visible primitive and replaced it
with something invisible — the object doesn't downgrade to a plain box,
it disappears completely.

**Verified by direct reproduction**, since I don't have access to your
real HSSD download or category-matching results:
- Built a synthetic scene (`Scene3D` + a fake retriever) with two
  objects: one pointing at a real, valid `.glb` (a trimesh-exported box),
  one pointing at a **minimal, technically-valid but mesh-less** `.glb`
  (a glTF with a scene graph and a node, but no mesh attached — simulating
  exactly the "parses fine, nothing to show" failure mode).
- Rendered the generated viewer HTML in a real headless Chromium and
  queried the actual THREE.js scene graph directly (bounding box size,
  mesh count) rather than reading pixels.
- **Before the fix**: the valid box correctly showed as a 40×80×40 mesh —
  but the mesh-less object also got swapped, becoming an empty `Group`
  with **0 meshes and a `[0,0,0]` bounding box** — i.e. genuinely nothing
  rendered, matching the first screenshot exactly (task logic completes
  correctly, since positions/collision never depended on the visual mesh,
  but visually the object is just gone).
- **After the fix**: the loader now checks (`hasVisibleGeometry()`) that
  the parsed result actually contains real vertex data before touching
  the primitive at all. Re-ran the identical test: the valid box still
  swaps in correctly, and the mesh-less object now correctly **keeps its
  primitive box** instead of vanishing, with a console warning explaining
  why (`"...parsed but has no visible geometry...— keeping primitive
  shape."`) so it's diagnosable instead of silent.

This means: going forward, an object should **never fully disappear**
regardless of what the real dataset's mesh files contain — worst case,
you get the (correctly-sized, correctly-positioned) primitive box/cylinder
back, with a warning in the browser console naming which object and why.

### The other half — "just cuboids, nothing from hssd-hab"

This is the primitive fallback working as designed *when retrieval or
loading doesn't succeed* — the question is which of these is actually
happening for you, since the failure mode and fix are different for each:

1. **No `--asset-source` flag was actually passed.** Then `scene.retriever`
   is `None` and every object is intentionally a primitive — this is the
   default, zero-dependency mode, not a bug. Image 2's scene (a fully
   custom, real-LLM-authored prompt) only shows one subgoal
   (`navigate coffee_mug`) and plain boxes, which is consistent with a
   plain run with no asset-source flag at all.
2. **`--asset-source hssd --hssd-root ...` was passed, but your actual
   download's file layout/category data doesn't match what
   `HSSDRetriever` expects.** I can't test this myself (no route to
   `huggingface.co`, no local copy of the real dataset), but the parsing
   logic itself is verified: `python asset_retrieval.py` runs 42 offline
   checks (category-matching + a synthetic HSSD fixture built with real
   `trimesh`-exported meshes on disk) — all 42 pass. That confirms the
   *logic* is sound against a fixture matching HSSD's documented layout;
   it can't confirm your specific download matches that layout. Run
   `python inspect_hssd.py --root /path/to/your/hssd-hab` and paste the
   output back — it dumps your actual `metadata/`/`semantics/` file
   shapes directly, which is exactly what would let me fix the parser
   against your real data instead of guessing.
3. **Retrieval succeeds and finds a real mesh, but the mesh itself uses a
   feature this project's vendored three.js r128 `GLTFLoader` can't
   decode** (Draco compression is the most common real-world case). Round
   3's fix above guarantees this degrades to a primitive instead of
   vanishing, but getting the *actual* real mesh to render would need a
   `DRACOLoader` wired in (needs its own decoder files) — worth doing if
   this turns out to be the actual cause; tell me what
   `inspect_hssd.py`/the browser console warnings show and I'll add it.

---

## Round 4: fixed against your REAL HSSD download's actual file layout

You ran `inspect_hssd.py` against your real download and shared the
output — this is the first point where I actually had real data to check
the parser against, instead of a synthetic fixture built from
documentation. It surfaced three concrete gaps, all fixed and all
re-verified with a NEW offline test built directly from your real file
shapes (`test_hssd_retriever_real_layout()` in `asset_retrieval.py`, 4/4
passing) — not just the older synthetic fixture, which was already
passing and wouldn't have caught these:

1. **`hssd_obj_semantics_condensed.csv`'s id column is literally called
   `"Object Hash"`**, not `"id"`/`"uid"`/etc. The old id-column detector
   only accepted a small fixed set of *exact* names, so this file's real
   id column was invisible to it. Fixed: also recognizes `"hash"` /
   `"object hash"` (space or underscore, case-insensitive) as an id
   column.
2. **`metadata/objects.json`'s real per-object entries have keys
   `id, type, scene_counts, name, tags`** — there is no `"category"` or
   `"class"` key at all, which is the only thing the old JSON-metadata
   parser looked for. This file has 16,081 entries keyed by real object
   hash UIDs (confirmed from your output) — probably the single largest
   potential source of category coverage in your whole download — and the
   old code got literally zero entries from it. Fixed: now falls back to
   the object's own `"tags"` field (first tag) when `category`/`class`
   aren't present.
3. **`semantics/hssd-hab_semantic_lexicon.json`'s real shape is
   `{"classes": [{"id": ..., "name": ...}]}`** — a flat category
   *vocabulary* (406 of them), with no per-category list of which objects
   belong to it. None of the schemas `_parse_semantic_lexicon` already
   handled expected this — a taxonomy with no membership data at all. This
   turned out to already be handled *safely* (it correctly returns `{}`
   with a clear diagnostic instead of inventing wrong mappings), which the
   new test explicitly checks for, but it's worth knowing this file alone
   will never resolve any objects for you — `fpmodels-with-decomposed.csv`
   and (now) `objects.json`'s tags are what actually carry usable
   per-object category data in your download.

Also fixed a real gap in the SYNONYM map itself: `synthesizer3d.py`'s
size-chart rework (Round 1) added a lot of furniture vocabulary — fridge,
stove, sink, dishwasher, cabinet, wardrobe, tv_stand, lamp, plant, stool,
armchair, and more — that `asset_retrieval.py`'s `CategoryMatcher.SYNONYMS`
was never updated to match. Real taxonomies use their own canonical terms
(WordNet's is `"refrigerator"`, never `"fridge"`) that plain token
overlap won't bridge on its own, so every one of those newly-supported
object types would have silently gotten zero synonym help. Added
synonym entries for all of them.

**`inspect_hssd.py` itself had a gap that mattered here**: it only showed
CSV column previews for files under `metadata/`, never `semantics/` — so
your real `semantics/objects.csv` (4.4MB, almost certainly a real
per-object category file sitting right next to the semantic lexicon)
never got shown in your output at all. Fixed: both directories now use
the same preview logic, and it now uses `csv.DictReader` instead of
raw line-splitting (a plain preview mis-renders `hssd_obj_semantics_
condensed.csv`'s header, which wraps across multiple lines inside quotes
— `DictReader` parses that correctly and reports the real column names).

**What I still can't confirm without your data**: whether
`semantics/objects.csv` — the file `inspect_hssd.py` couldn't previously
show you — actually contains a clean id+category mapping, and how much
real coverage `fpmodels-with-decomposed.csv` + `objects.json`'s tags give
you in practice. Re-run the updated `inspect_hssd.py` (now shows that
file's real columns) and/or just re-try your actual scene generation with
`--asset-source hssd` — between the `Object Hash` fix, the `tags`
fallback, and the expanded synonym map, real meshes should now resolve for
substantially more objects than before.



