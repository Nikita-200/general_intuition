# Real physics and real 3D assets: complete setup guide

This covers the two things the primitive-shapes/pure-Python version
deliberately scoped out: a real physics engine (PyBullet) and real 3D
meshes (Objaverse / HSSD / 3D-FRONT). Both are now real, working code, not
stubs — this document is how to actually turn them on.

## Part 1: PyBullet (real rigid-body physics)

### Why this wasn't already on by default

The harness now auto-detects PyBullet (`physics_backend.py`) and uses it
whenever it's importable, falling back to the pure-Python engine
otherwise. It's not force-required because the harness that built this was
developed in a sandbox where PyBullet genuinely cannot be installed (no
prebuilt wheel for that Python/platform, and compiling from source there
timed out) — the fallback exists so the harness still runs everywhere,
not because pure-Python is the preferred choice.

**Re-checked in a second, independent sandbox before submission**: same
result — no prebuilt wheel available for that environment's Python
3.12/Linux combination either (`pip download pybullet --only-binary=:all:`
found nothing), and a from-source install attempt was aborted after
timing out, matching the original finding rather than a new one.
`smoke_test_pybullet.py` genuinely has not been run end-to-end anywhere
yet — that remains the single highest-value thing to actually do on a
machine with a supported Python version (3.6–3.11 on Linux, or Mac) before
relying on the PyBullet backend for anything real.

### Installing it

```bash
pip install pybullet
```

**What to expect, by platform:**
- **Linux, Python 3.6-3.11**: prebuilt wheel, installs in seconds.
- **Windows, any Python version**: **no prebuilt wheel is published on
  PyPI** (checked directly against PyPI's package index while building
  this - every recent pybullet release has zero `win` wheels). `pip
  install` will fall back to compiling from source. This needs the
  Microsoft C++ Build Tools (the "Desktop development with C++" workload
  in the Visual Studio Installer). With that installed, the build
  typically succeeds but takes several minutes. If you'd rather avoid the
  source build entirely, running the harness under **WSL2** (Windows
  Subsystem for Linux) gets you the prebuilt Linux wheel instead - for a
  Python 3.10 install (which is what your `pygame` output showed you're
  running), that's an instant, zero-compile install.
- **Mac (Intel or Apple Silicon)**: prebuilt wheels are published for most
  recent Python versions.

### Verifying it actually works

**Do this before anything else.** The PyBullet backend (`dsl3d_pybullet.py`)
could not be executed in the sandbox that wrote it - it's built carefully
against documented PyBullet APIs and mirrors the already-tested
pure-Python backend's logic move for move, but "should work" and "does
work" are different claims. Run:

```bash
python smoke_test_pybullet.py
```

This runs the exact same checks already validated on the pure-Python
version: rams the agent into a table 300 times and confirms the table's
position doesn't move by even a fraction of a unit, confirms the agent
itself does move under velocity control, and runs 10 fresh scenes each for
reach-goal and pick-and-place, checking genuine task completion. If
anything fails, please don't route around it - that's exactly the kind of
bug worth reporting/fixing before relying on the backend elsewhere.

### Using it

Nothing else changes. `python harness3d.py "..." --mock` automatically
picks PyBullet if it's importable:

```
[physics_backend] ...
```

tells you which backend loaded. Force one explicitly with an env var if
you ever need to compare them directly:

```bash
HARNESS3D_BACKEND=pybullet python harness3d.py "..." --mock
HARNESS3D_BACKEND=pure python harness3d.py "..." --mock
```

### Watching it in PyBullet's own window

The browser viewer (`viewer_N.html`) re-implements navigation/collision in
JS so it can run standalone with no Python process behind it - genuinely
useful for sharing a result, but it's a re-implementation, not the
physics engine itself running live. For that, use:

```bash
python gui_demo_pybullet.py "a kitchen with a can on the counter and a trash bin in the corner" \
    --task "pick the can and place it in the trash bin" --mock
```

This opens PyBullet's actual OpenGL window and runs the real simulation in
real time - left-drag to orbit, scroll to zoom (PyBullet's built-in camera
controls). This is the most direct answer to "isn't it supposed to be a
physics engine": this window *is* the physics engine's own renderer.

## Part 2: Real 3D meshes

### Objaverse - works today, no account needed

```bash
pip install objaverse trimesh
```

`asset_retrieval.py`'s `ObjaverseRetriever` fuzzy-matches an object name
(e.g. "mug", "trash_bin") against Objaverse's LVIS category annotations,
deterministically picks one object in that category, downloads its `.glb`,
and computes real `(size, height, scale)` from the mesh's own bounding
box - no manual measurement needed.

**Verify it on your machine:**
```bash
python asset_retrieval.py
```
Expected output ends with `SUCCESS - Objaverse retrieval is working end to
end on this machine.` In the sandbox that wrote this code, this fails at
exactly one specific point - a `403 Forbidden` fetching
`https://huggingface.co/datasets/allenai/objaverse/...` (confirmed by
reading `objaverse`'s own source: both the LVIS annotation index and every
mesh file are hosted on `huggingface.co`, which that sandbox's network
allowlist doesn't include) - with everything before that line executing
correctly. On a machine with normal internet access, that fetch should
just succeed.

**Turn it on in the harness:**
```bash
python harness3d.py "a kitchen with a can on the counter and a trash bin in the corner" \
    --task "pick the can and place it in the trash bin" --use-real-assets
```

What actually happens: `synthesize_scene(..., retriever=FallbackRetriever())`
attaches the retriever to the scene, and every `scene.AddAsset(name, ...)`
call transparently tries `retriever.retrieve(name)` first - using the
LLM-generated scene program's own object names ("can", "trash_bin",
"counter") as the lookup key, with zero changes needed to the LLM-facing
DSL docs or few-shot examples. On any failure (no network, no category
match, a corrupt download), it silently falls back to the primitive-shape
heuristics that already exist, so `--use-real-assets` never crashes a run
even with no internet at all - verified directly, since that's exactly
what happened in the sandbox that built this (see the transcript: every
asset falls back to a primitive, and the task still completes).

The viewer (`viewer_N.html`) picks up real meshes automatically too: when
an asset has a retrieved `mesh_path`, `render_viewer_html` base64-embeds
the `.glb` directly into the page (so it's still a single file, no server
needed) and the JS side loads it via `THREE.GLTFLoader`, keeping the
primitive shape visible as an instant placeholder until the real mesh
finishes loading. Verified end to end with an offline-generated test mesh
(a trimesh-exported box standing in for a real Objaverse download) -
correct base64 embedding, correct loader wiring, valid page.

### HSSD (Habitat Synthetic Scenes Dataset) — implemented, offline-tested

HSSD is a better fit than raw Objaverse specifically for *indoor* objects:
curated (not the general Objaverse firehose), consistent real-world scale,
and - notably for the PyBullet backend - many objects ship with
**precomputed convex collision decompositions**, meaning good
dynamics-ready collision shapes for free instead of needing to run V-HACD
yourself.

**This is implemented now** — `HSSDRetriever` in `asset_retrieval.py`,
`download_hssd.py` for the download step, `inspect_hssd.py` for diagnosing
your specific download if matching still isn't working, and
`--asset-source hssd` / `--asset-source both` wired into `harness3d.py`.

> **Correction, based on a real run against the actual dataset**: the
> first version of this guide (and of `HSSDRetriever`) guessed at HSSD's
> file layout without being able to fetch it. A real run surfaced the
> actual structure and two real bugs, both now fixed:
> 1. `download_hssd.py` only pulled `objects/*` by default — the
>    `metadata/` and `semantics/` directories `HSSDRetriever` actually
>    needs for category matching were never downloaded, so every object
>    silently fell back to a primitive shape (not a retrieval bug — there
>    was nothing to retrieve from). Fixed: it now fetches `objects/*`,
>    `metadata/*`, and `semantics/*` by default.
> 2. The guessed metadata filenames (root-level `metadata.csv`, etc.)
>    don't exist in the real layout, which is actually:
>    ```
>    objects/<hash-bucket>/<uid>.object_config.json + <uid>.glb (+ .collider.glb, ...)
>    metadata/*.csv, *.json
>    semantics/hssd-hab_semantic_lexicon.json
>    ```
>    `HSSDRetriever` now targets these real locations, with a flexible
>    reader for the semantic lexicon (schema unconfirmed without network
>    access, so it tries several plausible shapes — see
>    `_parse_semantic_lexicon`'s docstring in `asset_retrieval.py`) rather
>    than assuming one. The directory-name-based fallback tier is now OFF
>    by default: the real `objects/*/*.glb` folders are hash buckets (the
>    standard Habitat-Sim on-disk convention), not category names, so that
>    tier was previously producing ~233 meaningless "categories" that could
>    never semantically match anything — confirmed directly from a real
>    run's output.

1. Request access at https://huggingface.co/datasets/hssd/hssd-hab (free,
   requires accepting their terms; approval is usually near-instant).
2. `pip install huggingface_hub`, then `huggingface-cli login` with a
   token from your HF account settings.
3. `python download_hssd.py --out ./hssd-hab` — downloads `objects/*`,
   `metadata/*`, and `semantics/*` by default; pass `--full-scenes` for
   the prebuilt room layouts too. **If you already ran an older version of
   this script and only have `objects/`**, just re-run it with the same
   `--out` — `huggingface_hub` skips files you already have and only
   pulls the newly-added `metadata`/`semantics` patterns. This calls
   `huggingface_hub.snapshot_download` directly; run it on your machine
   with your HF login, not from a sandbox without `huggingface.co` access.
4. `HSSDRetriever(hssd_root="./hssd-hab")` walks
   `<hssd_root>/objects/**/*.object_config.json` (the standard
   Habitat-Sim per-object config format — each references its own mesh via
   a `render_asset` field) and builds a name→category index from, in
   order: `semantics/hssd-hab_semantic_lexicon.json`, then any
   `metadata/*.csv`/`*.json`, then each object's own embedded
   `category`-like field. All resolved per-object and merged (not "pick
   one tier for everything"), which matters because real metadata is
   often partial across sources — testing this the naive way (whole-index
   tier selection) was actually where I caught a real bug during
   development, twice: once for partial-CSV-coverage merging, and again
   for the semantic lexicon's own record schema (a "categories:
   [{name, handles}]" wrapper — the first parsing attempt mismatched
   "categories" itself as a category name; fixed and covered by a
   regression test now). Whichever tier resolves a given object, its
   actual size/height is always computed from **its own mesh's real
   bounding box** via `trimesh` — never trusted from metadata — so a weak
   category tier only affects whether a match is *found*, never whether a
   found match's dimensions are correct. **If matching still doesn't work
   on your download**: run `python inspect_hssd.py --root ./hssd-hab` — it
   dumps the actual structure of your `metadata/`/`semantics/` files so a
   schema mismatch is immediately visible instead of a silent miss, and
   the fix is almost always a small addition to
   `HSSDRetriever._parse_semantic_lexicon`.
5. Already wired into the CLI — no manual code edits needed:
   ```bash
   python harness3d.py "a kitchen with a can on the counter and a trash bin in the corner" \
       --task "pick the can and place it in the trash bin" --mock \
       --asset-source hssd --hssd-root ./hssd-hab

   # or try HSSD first, falling back to Objaverse, then a primitive shape:
   python harness3d.py "..." --mock --asset-source both --hssd-root ./hssd-hab
   ```
   `--use-real-assets` still works as a backward-compatible alias for
   `--asset-source objaverse`. Nothing about `AddAsset`, the PyBullet
   backend's mesh loading, or the viewer's `GLTFLoader`/`OBJLoader`
   selection needed to change — same `AssetRetrieverBase` contract as
   `ObjaverseRetriever`.

**Verified offline, no network needed**: `python asset_retrieval.py` runs
`test_hssd_retriever()` — builds a synthetic directory tree matching the
REAL confirmed layout (hash-bucket object folders, `metadata/*.csv`, a
`semantics/hssd-hab_semantic_lexicon.json` using the
`categories: [{name, handles}]` shape), a metadata CSV covering only one
of three objects, and 3 additional lexicon schema variants plus one
deliberately-unrecognizable shape — 12 cases total, all passing, including
the `goal_`/`chair_2`-style name mangling, a query with no match correctly
returning `None`, and a missing `hssd_root` path failing gracefully.
Also ran the full harness end-to-end against this fixture and confirmed
the exported scene's assets carry the fixture's real mesh paths, not
primitives.

**What's still genuinely unverified from here**: whether your specific
download's `semantics/hssd-hab_semantic_lexicon.json` uses one of the
schemas `_parse_semantic_lexicon` already handles, or whether real
downloaded meshes actually render correctly in a browser. **With your HF
login and downloaded `objects/` already in place**: re-run
`python download_hssd.py --out ./hssd-hab` once to pull the
`metadata/`/`semantics/` files this now also needs (it won't re-download
`objects/` you already have), then
`python harness3d.py "..." --mock --asset-source hssd --hssd-root ./hssd-hab`
and check the console output at startup. If it says something like
"3 via metadata/lexicon" for most of your objects, matching is working.
If it says "zero categories resolved" or "couldn't parse a handle
mapping", run `python inspect_hssd.py --root ./hssd-hab` and use its
output to extend `HSSDRetriever._parse_semantic_lexicon` with whatever
shape your file actually has — that's the one remaining thing that needs
your real data to get exactly right.

### 3D-FRONT (+ 3D-FUTURE) - gated, full room layouts

3D-FRONT is what InteriorAgent-style systems are usually built against for
*whole-room* layouts (not just individual objects): ~18k professionally
designed rooms with furniture already laid out, each object linked to a
textured CAD model in the companion 3D-FUTURE dataset. This is a heavier
integration than Objaverse/HSSD because it's really a *second, alternative
scene-generation path* (import a real designer's room layout wholesale)
rather than just an asset retriever plugged into the existing
LLM-generates-a-program flow - worth knowing that distinction before
starting.

1. Request the dataset at https://tianchi.aliyun.com/dataset/65347 (3D-FRONT)
   and https://tianchi.aliyun.com/dataset/65387 (3D-FUTURE) - both need a
   (free) Alibaba Cloud account and accepting the license; approval can
   take a few days, unlike HSSD's near-instant HF gate.
2. 3D-FRONT ships room layouts as JSON: each room lists furniture
   instances by a `jid` (linking to a 3D-FUTURE model), plus its position,
   rotation, and scale already solved by a human designer.
3. 3D-FUTURE ships the actual textured meshes (`.obj` + texture), each
   with a category label and a normalized bounding box in its metadata.
4. Two integration options:
   - **As a retriever** (matches this harness's existing flow): index
     3D-FUTURE by category the same way as HSSD above, mostly useful if
     you just want better-quality individual meshes while keeping this
     harness's own LLM-driven layout logic.
   - **As a layout source** (bigger change, more faithful to what 3D-FRONT
     is actually good for): write a loader that reads a 3D-FRONT room JSON
     directly into a `Scene3D` - for each furniture instance, call
     `scene.AddAsset(...)` with the JSON's own position/rotation instead of
     going through `place_at`/`place_around`/etc., skipping the LLM
     scene-generation step for that room entirely (useful for "start from
     a real designed room, then let the agent complete tasks in it" rather
     than "generate a room from a text prompt").

## What's verified vs. documented

| | verified without needing your machine | needs your own machine to verify |
|---|---|---|
| PyBullet backend logic | ran the actual, unmodified `smoke_test_pybullet.py` against a hand-written PyBullet API stub (real signatures, simplified but real AABB collision) — 5 clean runs, 100 fresh scenes, all passing; also confirmed no prebuilt wheel exists here either (same finding as before, not new) | `smoke_test_pybullet.py` against the REAL Bullet solver (do this first) |
| Category matching (LLM name → dataset category) | 25 offline test cases against a realistic fake taxonomy, including `goal_`/`_2`-suffix name mangling and a correct-`None` case for no-match | whether the hand-curated `SYNONYMS` map's target strings match the real live LVIS/HSSD category names |
| Objaverse retriever | runs correctly up to the exact network call, confirmed via traceback (re-confirmed in a second sandbox) | the actual mesh download + viewer rendering |
| HSSD retriever | **implemented** (was previously "documented, not implemented"): `HSSDRetriever` + `download_hssd.py` + `--asset-source hssd/both` CLI wiring. 8 offline test cases against a synthetic fixture (real meshes, real object_config.json files, partial metadata coverage merged across 3 discovery tiers), plus a full harness run confirmed to actually pick up the fixture's mesh | `download_hssd.py` against the real dataset; whether real HSSD's metadata schema matches the guessed filenames/fields (tier 3 directory-fallback still works even if not) |
| trimesh bbox computation | fully verified — offline geometry processing, exercised by both retrievers' tests | - |
| Real-mesh → PyBullet path | verified end to end: real (locally-generated) mesh through `_ensure_pybullet_mesh_format`'s trimesh conversion, `p.createCollisionShape(GEOM_MESH, ...)`, and viewer embedding with correct format detection | rendering a *real* downloaded mesh in an actual browser |
| Full pipeline with `--asset-source` | verified: HSSD-fixture run completes the task and the exported scene JSON carries the real mesh path; graceful fallback to primitives confirmed when a source fails | the actual mesh-visible-in-browser experience with real downloaded assets |
| 3D-FRONT | documented, not implemented (gated access, and a bigger integration — see the section above) | writing the retriever/layout-importer + testing against real downloaded data |
