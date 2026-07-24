"""
viewer3d.py — renders a self-contained, interactive three.js HTML page.

The JS engine embedded here (grid A*, nearest-open-cell snap, move-and-
slide collision, proportional controller) is a direct, tested port of
nav_agent_3d.py — see nav_test.js in this directory, which runs the exact
same algorithms under node and was cross-checked against the Python
version on an identical scene (same 25-waypoint path, byte for byte). This
file adds only what's genuinely new for a browser: rendering, camera/input,
and a small TaskRunner port for the pick/navigate/place state machine.

No build step, no server, no internet required at open-time: three.js r128
(plus its GLTFLoader/OBJLoader) is vendored locally in vendor/ and inlined
directly into the page as a <script> block, exactly like the scene JSON
already was. An earlier version loaded these from cdnjs.cloudflare.com
instead — that broke silently (blank/black canvas, no interactivity, no
visible error unless you opened devtools) on any machine or sandbox
without outbound access to that CDN, which is a real risk for a one-shot
"open this file and it should just work" deliverable. Vendoring removes
that dependency entirely: this file works completely offline now.

To refresh the vendored files later (new three.js version, etc.):
    npm install three@0.128.0 --no-save
    cp node_modules/three/build/three.min.js vendor/
    cp node_modules/three/examples/js/loaders/GLTFLoader.js vendor/
    cp node_modules/three/examples/js/loaders/OBJLoader.js vendor/
"""

import json
import os

_VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")

_CDN_FALLBACK = {
    "three.min.js": "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js",
    "GLTFLoader.js": "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/examples/js/loaders/GLTFLoader.js",
    "OBJLoader.js": "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/examples/js/loaders/OBJLoader.js",
}


def _load_vendor_script(filename):
    """Returns (inline_js_or_None, script_tag_html). Prefers the vendored
    local copy (works fully offline); falls back to a CDN <script src=...>
    tag with a printed warning only if the vendor file is genuinely
    missing, so the harness still runs somewhere rather than hard-failing."""
    path = os.path.join(_VENDOR_DIR, filename)
    if not os.path.isfile(path):
        # Tolerate the underscore/dot naming mismatch that has bitten this
        # project before (three_min.js vs the expected three.min.js) —
        # a missing vendor file silently forces a CDN <script src=...>
        # fallback, which fails with NO visible error in any browser
        # without outbound access to cdnjs.cloudflare.com (offline demo,
        # sandboxed iframe, locked-down network). THREE never gets
        # defined, the entire inline script throws on its first line, and
        # the page is a blank, non-interactive canvas — exactly the "no
        # background / all white / nothing renders" failure mode. Since
        # that's a much worse failure than a slightly-off filename, try
        # the underscore variant before giving up.
        alt = os.path.join(_VENDOR_DIR, filename.replace(".", "_", 1))
        if os.path.isfile(alt):
            path = alt
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f"<script>\n{f.read()}\n</script>"
    print(f"[viewer3d] WARNING: vendor/{filename} not found — falling back to a CDN "
          f"<script> tag. The generated viewer will require internet access to "
          f"{_CDN_FALLBACK[filename]} to render. Run the npm/cp steps in this "
          f"module's docstring to vendor it locally and remove that dependency.")
    return f'<script src="{_CDN_FALLBACK[filename]}"></script>'


JS_ENGINE = r"""
const CELL = 20;

function buildGrid(assets, width, depth, exclude, inflate) {
  const cols = Math.floor(width / CELL), rows = Math.floor(depth / CELL);
  const blocked = Array.from({ length: rows }, () => new Array(cols).fill(false));
  for (const name in assets) {
    if (exclude.has(name)) continue;
    const o = assets[name];
    // portable/wall_mounted assets never block floor-plane pathing (this
    // grid has no real height axis) — matches nav_agent_3d.py's
    // plan_path_to, which unions ALL portable/wall_mounted names into its
    // own exclude set before building the grid, not just the current
    // subgoal's target. This call site previously only excluded the
    // agent and the current target by name, so any OTHER portable item
    // in the scene (a second small object not currently being navigated
    // to) was silently treated as a solid obstacle here — a real
    // Python/JS parity gap, fixed by checking the tag directly instead of
    // depending on every caller's exclude set to enumerate every
    // non-blocking asset by name.
    if (o.tags.includes("portable") || o.tags.includes("wall_mounted")) continue;
    const half = o.half_xz;
    const r0 = Math.floor((o.z - half - inflate) / CELL), r1 = Math.floor((o.z + half + inflate) / CELL);
    const c0 = Math.floor((o.x - half - inflate) / CELL), c1 = Math.floor((o.x + half + inflate) / CELL);
    for (let r = Math.max(r0, 0); r <= Math.min(r1, rows - 1); r++)
      for (let c = Math.max(c0, 0); c <= Math.min(c1, cols - 1); c++)
        blocked[r][c] = true;
  }
  return { blocked, cols, rows };
}

function blockBoundary(blocked, cols, rows, width, depth, margin) {
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      const x = (c + 0.5) * CELL, z = (r + 0.5) * CELL;
      if (x < margin || x > width - margin || z < margin || z > depth - margin) blocked[r][c] = true;
    }
}

function nearestOpenCell(blocked, cols, rows, rc) {
  if (!blocked[rc[0]][rc[1]]) return rc;
  const seen = new Set([rc.join(",")]);
  const q = [rc];
  while (q.length) {
    const [r, c] = q.shift();
    for (const [dr, dc] of [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]) {
      const nr = r + dr, nc = c + dc;
      const k = nr + "," + nc;
      if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && !seen.has(k)) {
        if (!blocked[nr][nc]) return [nr, nc];
        seen.add(k);
        q.push([nr, nc]);
      }
    }
  }
  return rc;
}

function astar(blocked, cols, rows, start, goal) {
  const h = (p) => Math.abs(p[0] - goal[0]) + Math.abs(p[1] - goal[1]);
  const key = (p) => p[0] + "," + p[1];
  const open = [[h(start), 0, start, null]];
  const came = new Map();
  const best = new Map([[key(start), 0]]);
  while (open.length) {
    open.sort((a, b) => a[0] - b[0]);
    const [, g, cur, parent] = open.shift();
    const ck = key(cur);
    if (came.has(ck)) continue;
    came.set(ck, parent);
    if (cur[0] === goal[0] && cur[1] === goal[1]) {
      const path = [];
      let c = cur;
      while (c) { path.push(c); c = came.get(key(c)); }
      return path.reverse();
    }
    for (const [dr, dc] of [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]) {
      const nr = cur[0] + dr, nc = cur[1] + dc;
      if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && !blocked[nr][nc]) {
        const ng = g + Math.hypot(dr, dc);
        const nk = nr + "," + nc;
        if (ng < (best.has(nk) ? best.get(nk) : 1e9)) {
          best.set(nk, ng);
          open.push([ng + h([nr, nc]), ng, [nr, nc], cur]);
        }
      }
    }
  }
  return null;
}

function planPathTo(state, agentXZ, targetXZ, exclude, agentHalf) {
  const inflate = agentHalf + 4;
  const { blocked, cols, rows } = buildGrid(state.assets, state.width, state.depth, exclude, inflate);
  blockBoundary(blocked, cols, rows, state.width, state.depth, state.wallMargin - 4);
  let start = [Math.floor(agentXZ[1] / CELL), Math.floor(agentXZ[0] / CELL)];
  let goal = [Math.floor(targetXZ[1] / CELL), Math.floor(targetXZ[0] / CELL)];
  start = [Math.min(Math.max(start[0], 0), rows - 1), Math.min(Math.max(start[1], 0), cols - 1)];
  goal = [Math.min(Math.max(goal[0], 0), rows - 1), Math.min(Math.max(goal[1], 0), cols - 1)];
  goal = nearestOpenCell(blocked, cols, rows, goal);
  blocked[start[0]][start[1]] = false;
  blocked[goal[0]][goal[1]] = false;
  const gridPath = astar(blocked, cols, rows, start, goal);
  if (!gridPath) return null;
  return gridPath.map(([r, c]) => [(c + 0.5) * CELL, (r + 0.5) * CELL]);
}

function standoffDistance(waypoints, targetXZ) {
  if (!waypoints || !waypoints.length) return 0;
  const [x, z] = waypoints[waypoints.length - 1];
  return Math.hypot(x - targetXZ[0], z - targetXZ[1]);
}

function physicalClearance(state, agent, targetName, margin) {
  const a = state.assets[targetName];
  if (!a || a.tags.includes("portable") || a.tags.includes("wall_mounted")) return 0;
  return a.half_xz + agent.half_xz + margin;
}

function resolveCollision(state, agent, exclude) {
  const ar = agent.half_xz;
  for (const name in state.assets) {
    if (exclude.has(name)) continue;
    const a = state.assets[name];
    if (a.tags.includes("portable") || a.tags.includes("wall_mounted")) continue;
    const minx = a.x - a.half_xz - ar - 2, maxx = a.x + a.half_xz + ar + 2;
    const minz = a.z - a.half_xz - ar - 2, maxz = a.z + a.half_xz + ar + 2;
    if (agent.x > minx && agent.x < maxx && agent.z > minz && agent.z < maxz) {
      const penLeft = agent.x - minx, penRight = maxx - agent.x;
      const penNear = agent.z - minz, penFar = maxz - agent.z;
      const m = Math.min(penLeft, penRight, penNear, penFar);
      if (m === penLeft) agent.x = minx;
      else if (m === penRight) agent.x = maxx;
      else if (m === penNear) agent.z = minz;
      else agent.z = maxz;
    }
  }
  agent.x = Math.max(ar + 2, Math.min(state.width - ar - 2, agent.x));
  agent.z = Math.max(ar + 2, Math.min(state.depth - ar - 2, agent.z));
}

function controllerStep(state, agent, waypoints, wpIdx, exclude, speed, dt) {
  if (wpIdx >= waypoints.length) return { action: "noop", wpIdx, done: true };
  let [tx, tz] = waypoints[wpIdx];
  let dx = tx - agent.x, dz = tz - agent.z;
  let dist = Math.hypot(dx, dz);
  if (dist < 14) {
    wpIdx += 1;
    if (wpIdx >= waypoints.length) return { action: "noop", wpIdx, done: true };
    [tx, tz] = waypoints[wpIdx];
    dx = tx - agent.x; dz = tz - agent.z;
    dist = Math.hypot(dx, dz);
  }
  const desired = Math.atan2(dx, dz);
  let diff = ((desired - agent.yaw + Math.PI) % (2 * Math.PI)) - Math.PI;
  if (diff < -Math.PI) diff += 2 * Math.PI;
  agent.yaw = agent.yaw + Math.max(-0.28, Math.min(0.28, diff * 0.28));
  const alignment = Math.max(0, Math.cos(diff));
  const remaining = (wpIdx === waypoints.length - 1) ? dist : 1e9;
  const approach = Math.min(1, remaining / 60);
  const v = speed * alignment * approach * dt;
  agent.x += v * Math.sin(agent.yaw);
  agent.z += v * Math.cos(agent.yaw);
  resolveCollision(state, agent, exclude);
  const action = alignment > 0.6 ? "forward" : (diff > 0 ? "turn_left" : "turn_right");
  return { action, wpIdx, done: false };
}

class TaskRunner {
  constructor(subgoals) { this.subgoals = subgoals; this.idx = 0; }
  get current() { return this.idx < this.subgoals.length ? this.subgoals[this.idx] : null; }
  get isComplete() { return this.idx >= this.subgoals.length; }
  targetPos(state) {
    const sg = this.current;
    if (!sg) return null;
    if (state.markers[sg.target]) return [state.markers[sg.target][0], state.markers[sg.target][2]];
    const a = state.assets[sg.target];
    return [a.x, a.z];
  }
  tryAdvance(state, agent, threshold) {
    const sg = this.current;
    if (!sg) return false;
    const [tx, tz] = this.targetPos(state);
    if (Math.hypot(agent.x - tx, agent.z - tz) > threshold) return false;
    if (sg.kind === "pick") {
      state.carried = sg.target;
      state.assets[sg.target].tags = state.assets[sg.target].tags.filter(t => t !== "placed").concat(["carried"]);
    } else if (sg.kind === "place") {
      const item = state.assets[sg.target], container = state.assets[sg.container];
      item.x = container.x + (Math.random() - 0.5) * container.half_xz;
      item.z = container.z + (Math.random() - 0.5) * container.half_xz;
      item.y = container.y + container.half_h + item.half_h;
      item.tags = item.tags.filter(t => t !== "carried").concat(["placed"]);
      state.carried = null;
    }
    sg.done = true;
    this.idx += 1;
    return true;
  }
}
"""


COLORS = {
    "wall": 0x3c3c3c, "floor": 0xe8e4da, "agent": 0x1e90ff, "goal": 0x22c85a,
    "obstacle": 0xc85a3c, "container": 0x966e3c, "carried": 0xffbe3c,
    "placed": 0x78aa5a, "marker_a": 0x2882c8, "marker_b": 0xc83c8c,
}

# Realistic per-category colors, keyed by substring match against the
# object's own name — same "most-specific-key-first" convention
# synthesizer3d.py's PrimitiveRetriever.TABLE already uses (e.g.
# "coffee_table" must be checked before the generic "table", or the
# generic entry always wins first and the specific one is dead code; same
# reasoning for "armchair"/"chair", "bookshelf"/"shelf", "candle"/"can",
# etc.). Without this, EVERY piece of furniture and every portable item
# rendered as the exact same flat color (COLORS["obstacle"]) regardless of
# what it actually was — confirmed against a real run's screenshot: a
# wooden table, a trash bin, and a can all reading as identical
# orange-brown. This also improves real HSSD .obj meshes, not just
# primitives: the OBJ loading path below (bare THREE.OBJLoader, no MTL
# support, so a real mesh's own referenced material/texture never actually
# loads) already recolors every OBJ mesh using this same asset's `color`,
# so a richer, more accurate category palette benefits both render paths
# for free. GLB meshes (Objaverse) are NOT touched — they keep their own
# real embedded PBR materials/textures, which are strictly better than a
# flat color when present.
CATEGORY_COLORS = [
    ("coffee_table", 0x8a6a48), ("coffee table", 0x8a6a48),
    ("side_table", 0x8a6a48), ("side table", 0x8a6a48),
    ("end_table", 0x8a6a48), ("end table", 0x8a6a48),
    ("nightstand", 0x8a6a48),
    ("dining_table", 0x9c7a4f), ("dining table", 0x9c7a4f),
    ("countertop", 0xdcd6c8), ("counter", 0xdcd6c8), ("island", 0xdcd6c8),
    ("table", 0x9c7a4f),
    ("desk", 0x8a6a48),
    ("tv_stand", 0x3a2e22), ("tv stand", 0x3a2e22),
    ("media_console", 0x3a2e22), ("media console", 0x3a2e22),
    ("television", 0x181818), ("tv", 0x181818),
    ("armchair", 0x54728f),
    ("loveseat", 0x4a6584), ("sectional", 0x4a6584),
    ("sofa", 0x4a6584), ("couch", 0x4a6584),
    ("stool", 0x6b4a35), ("ottoman", 0x7a5c46),
    ("chair", 0x6b4a35),
    ("bed", 0xeae3d3),
    ("bookshelf", 0x6b4a35), ("bookcase", 0x6b4a35),
    ("shelf", 0x6b4a35),
    ("wardrobe", 0x5c4632), ("closet", 0x5c4632),
    ("dresser", 0x6b4a35), ("sideboard", 0x6b4a35),
    ("cabinet", 0xdcd6c8),
    ("refrigerator", 0xd6d8da), ("fridge", 0xd6d8da),
    ("dishwasher", 0xc7cace), ("washer", 0xc7cace),
    ("stove", 0x2e2e30), ("oven", 0x2e2e30),
    ("sink", 0xc7cace),
    ("lamp", 0xe8d9a0),
    ("plant", 0x4c7a4a),
    ("rug", 0x8a4040),
    ("curtain", 0xb5a48a),
    ("trash", 0x5a6b52), ("basket", 0x8a6a48), ("bin", 0x5a6b52),
    ("candle", 0xd8c9a0),
    ("mug", 0xf2ede0), ("cup", 0xf2ede0), ("bowl", 0xece6d8),
    ("can", 0xb0313a),
    ("book", 0x8a3c3c), ("magazine", 0xb0313a),
    ("apple", 0xb0313a),
    ("remote", 0x2a2a2a),
    ("key", 0xb8b0a0),
    ("ball", 0xd0602c),
    ("laptop", 0x9a9a9a),
    ("backpack", 0x4a5a3a),
    ("slipper", 0xa0654a),
    ("photo", 0x7a6a58),
    ("pillow", 0xc9b79a),
    ("blanket", 0xb08a6a),
    ("board_game", 0x8a5a3a), ("board game", 0x8a5a3a),
    ("soda", 0xc0392b),
]


def _category_color(name):
    key = name.lower().replace("_", " ")
    for substr, color in CATEGORY_COLORS:
        if substr.replace("_", " ") in key:
            return color
    return None


def _color_for(name, asset, agent_name, goal_name):
    if name == agent_name:
        return COLORS["agent"]
    if name == goal_name:
        return COLORS["goal"]
    tags = asset.get("tags", [])
    if "carried" in tags:
        return COLORS["carried"]
    if "placed" in tags:
        return COLORS["placed"]
    category = _category_color(name)
    if category is not None:
        return category
    if "container" in tags:
        return COLORS["container"]
    return COLORS["obstacle"]


def assets_to_json(state, agent_name=None, goal_name=None):
    """
    Builds the per-asset JS-facing dict (color, embedded mesh data, etc.)
    from a `Scene3D.to_state()`-shaped dict. Factored out of
    `render_viewer_html` so `infinite_world.py`'s chunk export can produce
    assets in the EXACT same shape/coloring/mesh-embedding as the static
    single-scene exporter — one source of truth instead of a second copy
    that could quietly drift from this one.
    """
    assets_js = {}
    for name, a in state.items():
        mesh_b64, mesh_format = None, None
        mesh_path = a.get("mesh_path")
        if mesh_path and os.path.isfile(mesh_path):
            try:
                from asset_retrieval import mesh_to_base64
                mesh_format = os.path.splitext(mesh_path)[1].lstrip(".").lower()
                if mesh_format not in ("obj", "glb", "gltf"):
                    raise ValueError(f"unsupported mesh format {mesh_format!r}")
                mesh_b64 = mesh_to_base64(mesh_path)
            except Exception as e:
                print(f"[viewer3d] couldn't embed mesh for {name!r} ({e}); "
                      f"using a primitive shape instead.")
                mesh_b64, mesh_format = None, None
        assets_js[name] = dict(
            kind=a["kind"], x=a["x"], y=a["y"], z=a["z"], yaw=a["yaw"],
            half_xz=a["half_xz"], half_h=a["half_h"], tags=a["tags"],
            color=_color_for(name, a, agent_name, goal_name),
            meshB64=mesh_b64, meshFormat=mesh_format, meshScale=a.get("mesh_scale", 1.0),
            meshUpAxis=a.get("mesh_up_axis", 1), meshUpSign=a.get("mesh_up_sign", 1),
        )
    return assets_js


def render_viewer_html(scene, subgoals, trajectory, result, out_path, initial_control="auto"):
    state = scene.to_state()
    agent_name = scene.agent.name
    goal_name = scene.goal.name if scene.goal else None
    assets_js = assets_to_json(state, agent_name, goal_name)

    data = dict(
        width=scene.width, depth=scene.depth, wallMargin=scene.wall_margin,
        ceiling=scene.ceiling, assets=assets_js,
        markers={k: list(v) for k, v in scene.markers.items()},
        agentName=agent_name, goalName=goal_name,
        subgoals=[dict(kind=s.kind, target=s.target, container=s.container, done=False)
                  for s in subgoals],
        prompt=result.get("prompt", ""), task=result.get("task") or "(navigate to goal)",
        referenceResult=dict(success=result.get("success"), ticks=result.get("ticks")),
        initialControl=initial_control if initial_control in ("auto", "manual") else "auto",
    )
    data_json = json.dumps(data)

    html = (HTML_TEMPLATE
            .replace("__SCENE_DATA__", data_json)
            .replace("__JS_ENGINE__", JS_ENGINE)
            .replace("__THREE_SCRIPT_TAG__", _load_vendor_script("three.min.js"))
            .replace("__GLTFLOADER_SCRIPT_TAG__", _load_vendor_script("GLTFLoader.js"))
            .replace("__OBJLOADER_SCRIPT_TAG__", _load_vendor_script("OBJLoader.js"))
            .replace("__JS_MESH_BUILDER__", JS_MESH_BUILDER))
    # encoding="utf-8" explicitly — without it, Python uses the platform's
    # default locale encoding to WRITE this file, which on Windows is
    # typically cp1252, not UTF-8. This template's own code comments
    # contain real em-dashes and other non-ASCII characters; cp1252 can't
    # represent all of them, and the ones it silently mis-encodes instead
    # produce a file that later fails to re-open as UTF-8 (confirmed: a
    # real crash reading a generated viewer back in webapp/server.py,
    # `UnicodeDecodeError: 'utf-8' codec can't decode byte 0x97`). The
    # CLI path never re-reads its own output so this went unnoticed there,
    # but the file itself was always subtly corrupted on Windows.
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


# Per-asset mesh construction: primitive box/cylinder/agent shapes, with a
# real Objaverse/HSSD mesh (OBJ or GLB, base64-embedded) swapped in once
# parsed if one was retrieved — see asset_retrieval.py. Factored out of
# HTML_TEMPLATE into its own constant (rather than left inline) so
# webapp/'s live "infinite corridor" page can call the exact same
# `buildMesh` for chunks streamed in after the page has already loaded,
# instead of forking a second copy of this logic that could silently drift
# from this one — this is precisely the up-axis/coloring/real-mesh-loading
# code that's already been the source of real, hard-to-spot bugs once.
JS_MESH_BUILDER = r"""
const gltfLoader = new THREE.GLTFLoader();
const objLoader = new THREE.OBJLoader();

function base64ToArrayBuffer(b64) {
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return buf.buffer;
}

// Real meshes (Objaverse/HSSD/3D-FRONT) are authored with all kinds of
// origin conventions — centered, resting at y=0, offset from some export
// pivot, etc. The primitive shape it replaces is always grounded correctly
// (mesh.position.y = a.y = half_h, geometry centered, bottom at world 0),
// so re-parent the real mesh's own children under a wrapper offset by its
// bounding-box center — that makes the wrapper's origin the mesh's true
// visual center, which can then be positioned exactly like a primitive.
// Returns false if the object has no actual visible geometry (empty/
// degenerate bounding box) — the caller MUST treat that as a failed load,
// not a successful one with nothing to reposition.
function groundMeshOnBoundingBox(object, a) {
  // `center` comes out of Box3().setFromObject in object's PARENT frame —
  // i.e. AFTER object's own scale and rotation are applied (object.position
  // is (0,0,0) at this point, so matrixWorld = rotation * scale). The old
  // code below subtracted this already-transformed `center` from each
  // CHILD's pre-transform local position (or, for a bare Mesh, translated
  // the geometry directly in local space) — mixing two different
  // coordinate spaces. With scale=1 and no rotation those spaces happened
  // to coincide, so this looked fine; once real meshes got non-uniform
  // per-axis scale and up-axis rotation (see fitted_scale_and_dims), the
  // mismatch became large and directional — this is what threw a
  // correctly-scaled, correctly-rotated mesh up onto the wall instead of
  // centering it on the floor.
  //
  // The fix: shift the OBJECT's own position by -center, not its
  // children. object.position lives in the PARENT's frame — the same
  // frame `center` is already expressed in — so this is a straight,
  // correct cancellation regardless of whatever scale/rotation `object`
  // itself carries, and works identically whether `object` is a Group
  // with children or a bare Mesh.
  const box = new THREE.Box3().setFromObject(object);
  if (!isFinite(box.min.y) || !isFinite(box.max.y)) return false;  // empty/degenerate mesh
  const center = box.getCenter(new THREE.Vector3());
  object.position.sub(center);
  return true;
}

// A gltf/obj file can "parse" without throwing and still produce nothing
// worth looking at — an unsupported extension (Draco/Basis compression,
// an external-resource reference our single-file embed can't resolve, a
// degenerate export) can silently yield a Group with zero real Mesh
// children. Swapping the working primitive out for that is strictly worse
// than keeping the primitive — this is what actually makes an object
// disappear from the scene entirely despite generation/retrieval having
// "succeeded." Only ever swap in a load that demonstrably has geometry.
function hasVisibleGeometry(object) {
  let found = false;
  object.traverse((child) => {
    if (child.isMesh && child.geometry) {
      const pos = child.geometry.attributes && child.geometry.attributes.position;
      if (pos && pos.count > 0) found = true;
    }
  });
  return found;
}

// Real meshes now carry a per-axis (non-uniform) scale [sx, sy, sz] —
// asset_retrieval.py forces the mesh to exactly fill the DSL's own
// (size, height) box regardless of the raw mesh's native proportions or
// up-axis convention (see CHANGES.md Round 5), so each axis can need a
// different factor. Still accepts a plain number for backward
// compatibility with any mesh_path set some other way.
function applyMeshScale(object, meshScale) {
  if (Array.isArray(meshScale)) {
    object.scale.set(meshScale[0], meshScale[1], meshScale[2]);
  } else {
    object.scale.setScalar(meshScale || 1.0);
  }
}

// meshScale alone only fixes the MAGNITUDE of whichever local axis is
// really "up" — it says nothing about DIRECTION or SIGN. Without this, a
// correctly-scaled mesh whose real up-axis is a different local axis (or
// the NEGATIVE direction of the right one) still renders sideways or
// upside-down, which is what made a correctly-sized counter mesh look
// like a flat rug on the floor, and other objects render upside-down once
// that was fixed but sign wasn't yet accounted for. Rotates the mesh's OWN
// rotation (not the wrapper's — see buildMesh, which keeps this on a
// child object so the per-tick yaw sync on the parent never clobbers it)
// so local signed axis (upAxis, upSign) points along three.js's world-up,
// Y. Hand-derived per (axis, sign) rather than computed generically —
// only 6 fixed cases exist (3 axes x 2 signs), verified numerically
// rather than just derived on paper.
const UP_AXIS_SIGN_TO_Y_ROTATION = {
  "0,1":  [[0, 0, 1],  Math.PI / 2],
  "0,-1": [[0, 0, 1], -Math.PI / 2],
  "1,1":  null,                        // already +Y, identity
  "1,-1": [[0, 0, 1],  Math.PI],
  "2,1":  [[1, 0, 0], -Math.PI / 2],
  "2,-1": [[1, 0, 0],  Math.PI / 2],
};
function applyUpAxisCorrection(object, upAxis, upSign) {
  const entry = UP_AXIS_SIGN_TO_Y_ROTATION[`${upAxis},${upSign}`];
  if (!entry) return;  // already Y-up, no rotation needed
  const [axis, angle] = entry;
  object.quaternion.setFromAxisAngle(new THREE.Vector3(axis[0], axis[1], axis[2]), angle);
}

const meshes = {};
function buildMesh(name, a) {
  let geo, mat, mesh;
  const color = a.color;
  if (name === SCENE_DATA.agentName) {
    const group = new THREE.Group();
    // group.position (set below, shared with every other object) is a.y —
    // the CENTER height of the agent's full a.half_h*2 tall bounding
    // cylinder, exactly like every box/cylinder asset. So local y=0 here
    // IS ground-truth center; the body's bottom must sit at local
    // y = -a.half_h to actually touch the floor at world y=0.
    const bodyH = a.half_h * 1.6;
    const bodyLocalY = -a.half_h + bodyH / 2;
    const body = new THREE.Mesh(
      new THREE.CylinderGeometry(a.half_xz, a.half_xz, bodyH, 16),
      new THREE.MeshLambertMaterial({ color })
    );
    body.position.y = bodyLocalY;
    body.castShadow = true;
    const headR = a.half_xz * 0.7;
    const head = new THREE.Mesh(new THREE.SphereGeometry(headR, 12, 12),
      new THREE.MeshLambertMaterial({ color }));
    head.position.y = bodyLocalY + bodyH / 2 + headR * 0.85;  // resting on top of the body
    head.castShadow = true;
    const nose = new THREE.Mesh(new THREE.ConeGeometry(a.half_xz * 0.35, a.half_xz * 0.9, 8),
      new THREE.MeshLambertMaterial({ color: 0xffffff }));
    nose.rotation.x = Math.PI / 2;
    nose.position.set(0, bodyLocalY, a.half_xz * 1.1);  // chest height, facing forward
    nose.castShadow = true;
    group.add(body, head, nose);
    mesh = group;
  } else if (a.kind === "cylinder") {
    geo = new THREE.CylinderGeometry(a.half_xz, a.half_xz, a.half_h * 2, 16);
    mat = new THREE.MeshLambertMaterial({ color, transparent: a.tags.includes("container"), opacity: 0.85 });
    mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
  } else {
    geo = new THREE.BoxGeometry(a.half_xz * 2, a.half_h * 2, a.half_xz * 2);
    mat = new THREE.MeshLambertMaterial({ color, transparent: a.tags.includes("container"), opacity: 0.85 });
    mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    if (a.tags.includes("container")) {
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: 0x3c2c14 }));
      mesh.add(edges);
    }
  }
  mesh.position.set(a.x, a.y, a.z);
  mesh.rotation.y = a.yaw;
  scene3.add(mesh);
  meshes[name] = mesh;

  // Real mesh (Objaverse/HSSD/3D-FRONT, via asset_retrieval.py) available?
  // Swap it in once ready — the primitive shape above stays visible in the
  // meantime, so the scene is never blank while meshes load. Format is
  // whatever the Python side actually produced (OBJ, from the PyBullet-
  // compatibility conversion, or GLB) — using the wrong loader for the
  // actual bytes is exactly what caused a black screen before this fix,
  // for every mesh-bearing object at once.
  if (a.meshB64 && name !== SCENE_DATA.agentName) {
    try {
      if (a.meshFormat === "obj") {
        const text = atob(a.meshB64);  // OBJ is plain text — no ArrayBuffer needed
        const real = objLoader.parse(text);  // synchronous, no callback
        applyMeshScale(real, a.meshScale);
        applyUpAxisCorrection(real, a.meshUpAxis, a.meshUpSign);
        real.traverse((child) => {
          if (child.isMesh) {
            child.material = new THREE.MeshLambertMaterial({ color: a.color });
            child.castShadow = true;
            child.receiveShadow = true;
          }
        });
        if (!hasVisibleGeometry(real)) {
          console.warn(`OBJ mesh for ${name} parsed but has no visible geometry — keeping primitive shape.`);
        } else if (!groundMeshOnBoundingBox(real, a)) {
          console.warn(`OBJ mesh for ${name} has a degenerate bounding box — keeping primitive shape.`);
        } else {
          // Wrap in a plain Group so the per-tick yaw sync (syncMeshes,
          // which sets meshes[name].rotation.y every frame) rotates the
          // WRAPPER, not `real` itself — setting .rotation.y directly on
          // an object that already has a non-identity up-axis-correction
          // quaternion would silently overwrite that correction on the
          // very next tick.
          const wrapper = new THREE.Group();
          wrapper.add(real);
          const oldMesh = meshes[name];
          wrapper.position.copy(oldMesh.position);
          wrapper.rotation.copy(oldMesh.rotation);
          scene3.remove(oldMesh);
          scene3.add(wrapper);
          meshes[name] = wrapper;
        }
      } else {
        const buf = base64ToArrayBuffer(a.meshB64);
        gltfLoader.parse(buf, "", (gltf) => {
          const real = gltf.scene;
          applyMeshScale(real, a.meshScale);
          applyUpAxisCorrection(real, a.meshUpAxis, a.meshUpSign);
          real.traverse((child) => {
            if (child.isMesh) {
              // Same reasoning as the OBJ branch above: the Python side
              // strips each GLB's own materials/textures before embedding
              // it (see asset_retrieval._strip_glb_textures) specifically
              // BECAUSE this viewer always wants its own semantic color,
              // never the source mesh's original look — so every real
              // mesh, OBJ or GLB alike, gets recolored the same way here.
              child.material = new THREE.MeshLambertMaterial({ color: a.color });
              child.castShadow = true;
              child.receiveShadow = true;
            }
          });
          if (!hasVisibleGeometry(real)) {
            console.warn(`GLB mesh for ${name} parsed but has no visible geometry (unsupported extension, external resource, or empty export?) — keeping primitive shape.`);
          } else if (!groundMeshOnBoundingBox(real, a)) {
            console.warn(`GLB mesh for ${name} has a degenerate bounding box — keeping primitive shape.`);
          } else {
            // See the OBJ branch above for why this is wrapped in a Group
            // rather than positioning/rotating `real` directly.
            const wrapper = new THREE.Group();
            wrapper.add(real);
            const oldMesh = meshes[name];
            wrapper.position.copy(oldMesh.position);
            wrapper.rotation.copy(oldMesh.rotation);
            scene3.remove(oldMesh);
            scene3.add(wrapper);
            meshes[name] = wrapper;
          }
        }, (err) => {
          console.warn(`GLB mesh load failed for ${name}, keeping primitive shape`, err);
        });
      }
    } catch (err) {
      console.warn(`mesh load failed for ${name}, keeping primitive shape`, err);
    }
  }
}
"""


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>3D Indoor Agent Viewer</title>
__THREE_SCRIPT_TAG__
__GLTFLOADER_SCRIPT_TAG__
__OBJLOADER_SCRIPT_TAG__
<style>
  html, body { margin: 0; padding: 0; overflow: hidden; background: #1a1a1a; font-family: -apple-system, Arial, sans-serif; }
  #hud {
    position: absolute; top: 10px; left: 10px; color: #eee; background: rgba(20,20,20,0.72);
    padding: 10px 14px; border-radius: 8px; font-size: 13px; line-height: 1.55; max-width: 420px;
    pointer-events: none; z-index: 5;
  }
  #hud b { color: #7ec8ff; }
  #hud .ok { color: #6ee87a; }
  #hud .subgoal-done { color: #6ee87a; }
  #hud .subgoal-cur { color: #ffd479; font-weight: bold; }
  #crosshair {
    position: absolute; top: 50%; left: 50%; width: 6px; height: 6px; margin: -3px 0 0 -3px;
    border-radius: 50%; background: rgba(255,255,255,0.8); display: none; z-index: 5;
  }
  #hint { position: absolute; bottom: 10px; left: 10px; color: #aaa; font-size: 12px; background: rgba(20,20,20,0.6); padding: 6px 10px; border-radius: 6px; z-index: 5;}
</style>
</head>
<body>
<div id="hud"></div>
<div id="crosshair"></div>
<div id="hint">TAB: auto/manual &nbsp; WASD: move (manual) &nbsp; drag mouse: look &nbsp; R: reset run &nbsp; SPACE: pause</div>
<script>
const SCENE_DATA = __SCENE_DATA__;
__JS_ENGINE__

const scene3 = new THREE.Scene();

// --- Procedural backdrop: a soft radial-gradient "void" instead of a flat
// color, generated on an in-memory canvas (no external image file, no CDN
// dependency, keeps the single-HTML-file promise). This is deliberately a
// neutral studio-style backdrop, not a literal sky — this is an indoor-
// only scene (no roof/exterior modeled), so a sky would be architecturally
// wrong; a soft vignette reads as a natural, tasteful "outside the walls"
// backdrop instead of a jarring flat void, especially since the walls
// themselves are semi-transparent by design (see addWall below) so this
// backdrop is visible right through them from any camera angle.
function makeBackdropTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 512;
  const ctx = c.getContext('2d');
  const grad = ctx.createRadialGradient(256, 200, 40, 256, 256, 380);
  grad.addColorStop(0, '#3d4249');
  grad.addColorStop(0.55, '#24272d');
  grad.addColorStop(1, '#111317');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 512, 512);
  return new THREE.CanvasTexture(c);
}
scene3.background = makeBackdropTexture();

const camera = new THREE.PerspectiveCamera(65, window.innerWidth / window.innerHeight, 1, 3000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

// Warm-neutral ambient + a soft directional "sun" that actually casts
// shadows now, for real depth cues instead of flat, shadowless lighting.
scene3.add(new THREE.AmbientLight(0xfff2e0, 0.55));
const sun = new THREE.DirectionalLight(0xfff6e8, 0.75);
sun.position.set(200, 400, 150);
sun.castShadow = true;
sun.shadow.mapSize.set(1024, 1024);
scene3.add(sun);

const W = SCENE_DATA.width, D = SCENE_DATA.depth;
// The default shadow-camera frustum is only ~10 units across — far too
// small for a room that's typically hundreds of units wide, which would
// silently clip every shadow away. Size it to the room's actual footprint
// instead, with headroom for furniture height.
const shadowSpan = Math.max(W, D) * 0.75;
sun.shadow.camera.left = -shadowSpan;
sun.shadow.camera.right = shadowSpan;
sun.shadow.camera.top = shadowSpan;
sun.shadow.camera.bottom = -shadowSpan;
sun.shadow.camera.near = 50;
sun.shadow.camera.far = 1200;
sun.target.position.set(W / 2, 0, D / 2);
scene3.add(sun.target);

scene3.fog = new THREE.Fog(0x24272d, Math.max(W, D) * 0.85, Math.max(W, D) * 2.5);

// --- Procedural wood-plank floor texture, tiled to the room's actual
// size (so plank scale stays consistent regardless of room dimensions),
// replacing the old flat solid-color floor.
function makeFloorTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#c9b896';
  ctx.fillRect(0, 0, 256, 256);
  ctx.strokeStyle = 'rgba(120,95,60,0.35)';
  ctx.lineWidth = 2;
  for (let y = 0; y <= 256; y += 32) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(256, y); ctx.stroke();
  }
  for (let i = 0; i < 70; i++) {
    ctx.strokeStyle = `rgba(90,70,45,${(0.05 + Math.random() * 0.08).toFixed(3)})`;
    const y = Math.random() * 256;
    ctx.beginPath();
    ctx.moveTo(Math.random() * 40, y);
    ctx.lineTo(256 - Math.random() * 40, y + (Math.random() * 6 - 3));
    ctx.stroke();
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(Math.max(1, Math.round(W / 90)), Math.max(1, Math.round(D / 90)));
  return tex;
}

const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(W, D),
  new THREE.MeshStandardMaterial({ map: makeFloorTexture(), roughness: 0.85, metalness: 0.02, side: THREE.DoubleSide })
);
floor.rotation.x = -Math.PI / 2;
floor.position.set(W / 2, 0, D / 2);
floor.receiveShadow = true;
scene3.add(floor);

const grid = new THREE.GridHelper(Math.max(W, D), Math.max(W, D) / 40, 0xbbbbbb, 0xcccccc);
grid.position.set(W / 2, 0.5, D / 2);
scene3.add(grid);

const CEILING_H = SCENE_DATA.ceiling || 260;
function addWall(x, z, w, d) {
  const wall = new THREE.Mesh(
    new THREE.BoxGeometry(w, CEILING_H, d),
    new THREE.MeshStandardMaterial({ color: 0xede6da, roughness: 0.9, transparent: true, opacity: 0.35 })
  );
  wall.position.set(x, CEILING_H / 2, z);
  scene3.add(wall);
  // A thin, slightly darker baseboard strip along the floor-wall seam —
  // a cheap detail that reads as "furnished interior" rather than a bare
  // primitive box.
  const baseboard = new THREE.Mesh(
    new THREE.BoxGeometry(w, 6, d),
    new THREE.MeshStandardMaterial({ color: 0xd8cfc0, roughness: 0.8, transparent: true, opacity: 0.5 })
  );
  baseboard.position.set(x, 3, z);
  scene3.add(baseboard);
}
addWall(W / 2, 2, W, 4);
addWall(W / 2, D - 2, W, 4);
addWall(2, D / 2, 4, D);
addWall(W - 2, D / 2, 4, D);

// Faint translucent ceiling plane so the room reads as fully enclosed
// from orbit/overhead angles, without ever blocking the camera's view in
// (opacity is deliberately very low — this is a cue, not a solid lid).
const ceilingPlane = new THREE.Mesh(
  new THREE.PlaneGeometry(W, D),
  new THREE.MeshStandardMaterial({ color: 0xf2efe6, roughness: 1.0, transparent: true, opacity: 0.12, side: THREE.DoubleSide })
);
ceilingPlane.rotation.x = Math.PI / 2;
ceilingPlane.position.set(W / 2, CEILING_H, D / 2);
scene3.add(ceilingPlane);

__JS_MESH_BUILDER__
for (const name in SCENE_DATA.assets) buildMesh(name, SCENE_DATA.assets[name]);

for (const mname in SCENE_DATA.markers) {
  const [mx, my, mz] = SCENE_DATA.markers[mname];
  const ring = new THREE.Mesh(new THREE.RingGeometry(8, 10, 24),
    new THREE.MeshBasicMaterial({ color: mname.endsWith("_b") ? 0xc83c8c : 0x2882c8, side: THREE.DoubleSide }));
  ring.rotation.x = -Math.PI / 2;
  ring.position.set(mx, 0.6, mz);
  scene3.add(ring);
}

const state = { width: W, depth: D, wallMargin: SCENE_DATA.wallMargin, assets: {}, markers: SCENE_DATA.markers, carried: null };
for (const name in SCENE_DATA.assets) {
  const a = SCENE_DATA.assets[name];
  state.assets[name] = { x: a.x, y: a.y, z: a.z, yaw: a.yaw, half_xz: a.half_xz, half_h: a.half_h, tags: a.tags.slice() };
}
const agentInit = SCENE_DATA.assets[SCENE_DATA.agentName];
let agent = { x: agentInit.x, y: agentInit.y, z: agentInit.z, yaw: agentInit.yaw, half_xz: agentInit.half_xz, half_h: agentInit.half_h };

let subgoals = SCENE_DATA.subgoals.map(s => ({ ...s }));
let runner = new TaskRunner(subgoals);
let waypoints = null, wpIdx = 0, lastIdx = -1, arrivalThreshold = 30;
let control = (SCENE_DATA.initialControl === "manual") ? "manual" : "auto";
let paused = false;
let ticks = 0;
let stuckHistory = [];

function resetRun() {
  agent = { x: agentInit.x, y: agentInit.y, z: agentInit.z, yaw: agentInit.yaw, half_xz: agentInit.half_xz, half_h: agentInit.half_h };
  for (const name in SCENE_DATA.assets) {
    const a = SCENE_DATA.assets[name];
    state.assets[name] = { x: a.x, y: a.y, z: a.z, yaw: a.yaw, half_xz: a.half_xz, half_h: a.half_h, tags: a.tags.slice() };
  }
  state.carried = null;
  subgoals = SCENE_DATA.subgoals.map(s => ({ ...s }));
  runner = new TaskRunner(subgoals);
  waypoints = null; wpIdx = 0; lastIdx = -1; ticks = 0; stuckHistory = [];
}

function replan() {
  const sg = runner.current;
  if (!sg) { waypoints = null; return; }
  const target = runner.targetPos(state);
  const exclude = new Set([SCENE_DATA.agentName, sg.target]);
  if (sg.container) exclude.add(sg.container);
  waypoints = planPathTo(state, [agent.x, agent.z], target, exclude, agent.half_xz);
  wpIdx = 0; lastIdx = runner.idx;
  arrivalThreshold = waypoints ? Math.max(
    30,
    standoffDistance(waypoints, target) + 24,
    physicalClearance(state, agent, sg.target, 20)
  ) : 30;
  stuckHistory = [];
}

function isStuck() {
  stuckHistory.push([agent.x, agent.z]);
  if (stuckHistory.length > 45) stuckHistory.shift();
  if (stuckHistory.length < 45) return false;
  const [x0, z0] = stuckHistory[0], [x1, z1] = stuckHistory[stuckHistory.length - 1];
  const stuck = Math.hypot(x1 - x0, z1 - z0) < 6;
  if (stuck) stuckHistory = [];
  return stuck;
}

const keys = {};
window.addEventListener("keydown", (e) => {
  keys[e.code] = true;
  if (e.code === "Tab") { e.preventDefault(); control = control === "auto" ? "manual" : "auto"; lastIdx = -1; }
  if (e.code === "Space") { e.preventDefault(); paused = !paused; }
  if (e.code === "KeyR") { resetRun(); }
});
window.addEventListener("keyup", (e) => { keys[e.code] = false; });

let dragging = false, lastMouseX = 0, lastMouseY = 0;
let orbitYaw = Math.PI, orbitPitch = 0.5, orbitRadius = 260;
let lookPitchDelta = 0;
renderer.domElement.addEventListener("mousedown", (e) => { dragging = true; lastMouseX = e.clientX; lastMouseY = e.clientY; });
window.addEventListener("mouseup", () => { dragging = false; });
window.addEventListener("mousemove", (e) => {
  if (!dragging) return;
  const dx = e.clientX - lastMouseX, dy = e.clientY - lastMouseY;
  lastMouseX = e.clientX; lastMouseY = e.clientY;
  if (control === "manual") {
    agent.yaw += dx * 0.005;
    lookPitchDelta = Math.max(-1.0, Math.min(1.0, lookPitchDelta - dy * 0.003));
  } else {
    orbitYaw -= dx * 0.006;
    orbitPitch = Math.max(0.15, Math.min(1.3, orbitPitch - dy * 0.004));
  }
});
renderer.domElement.addEventListener("wheel", (e) => {
  orbitRadius = Math.max(80, Math.min(700, orbitRadius + e.deltaY * 0.3));
});
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function manualStep(dt) {
  const speed = 170;
  let fx = Math.sin(agent.yaw), fz = Math.cos(agent.yaw);
  // Right-hand vector for this project's forward=(sin(yaw),cos(yaw)) and
  // up=+Y convention is (-cos(yaw), sin(yaw)) — i.e. yaw MINUS 90deg, not
  // plus. Using +90deg (as an earlier version did) points "right" at what
  // the camera actually renders as the LEFT side, so D strafed left and A
  // strafed right on screen. Verified against three.js's own lookAt basis
  // (xaxis = cross(up, zaxis)) to make sure this matches what the camera
  // shows, not just what feels right on paper.
  let sx = Math.sin(agent.yaw - Math.PI / 2), sz = Math.cos(agent.yaw - Math.PI / 2);
  let vx = 0, vz = 0;
  if (keys["KeyW"]) { vx += fx; vz += fz; }
  if (keys["KeyS"]) { vx -= fx; vz -= fz; }
  if (keys["KeyD"]) { vx += sx; vz += sz; }
  if (keys["KeyA"]) { vx -= sx; vz -= sz; }
  const norm = Math.hypot(vx, vz);
  if (norm > 0) {
    agent.x += (vx / norm) * speed * dt;
    agent.z += (vz / norm) * speed * dt;
  }
  resolveCollision(state, agent, new Set([SCENE_DATA.agentName]));
  return norm > 0 ? "manual-move" : "manual-idle";
}

const hud = document.getElementById("hud");
const clock = new THREE.Clock();

function updateCarried() {
  if (!state.carried) return;
  const item = state.assets[state.carried];
  const fwd = 26;
  item.x = agent.x + fwd * Math.sin(agent.yaw);
  item.z = agent.z + fwd * Math.cos(agent.yaw);
  item.y = 110;
}

function syncMeshes() {
  for (const name in state.assets) {
    const a = state.assets[name];
    const m = meshes[name];
    if (!m) continue;
    m.position.set(a.x, a.y, a.z);
    m.rotation.y = a.yaw;
  }
  const am = meshes[SCENE_DATA.agentName];
  am.position.set(agent.x, agent.y, agent.z);
  am.rotation.y = agent.yaw;
}

function updateCamera() {
  if (control === "manual") {
    // eyeHeight is an ABSOLUTE world height (measured from the floor,
    // y=0) — NOT added on top of agent.y, which is already the agent's
    // vertical center (half_h above the floor), same as every other
    // object. Adding eyeHeight to agent.y put the camera almost at the
    // room's ceiling, looking mostly past everything.
    const eyeHeight = agent.half_h * 1.8;  // ~90% of the agent's full height
    camera.position.set(agent.x, eyeHeight, agent.z);
    const lookX = agent.x + Math.sin(agent.yaw) * 50;
    const lookZ = agent.z + Math.cos(agent.yaw) * 50;
    const lookY = eyeHeight + lookPitchDelta * 60;
    camera.lookAt(lookX, lookY, lookZ);
  } else {
    const cx = agent.x + orbitRadius * Math.sin(orbitYaw) * Math.cos(orbitPitch);
    const cz = agent.z + orbitRadius * Math.cos(orbitYaw) * Math.cos(orbitPitch);
    const cy = agent.y + orbitRadius * Math.sin(orbitPitch) + 40;
    camera.position.set(cx, cy, cz);
    camera.lookAt(agent.x, agent.y + 30, agent.z);
  }
}

function updateHud(action) {
  const lines = [];
  lines.push(`<b>${SCENE_DATA.prompt || "(procedural scene)"}</b>`);
  lines.push(`task: ${SCENE_DATA.task}`);
  lines.push(`control: <b>${control.toUpperCase()}</b> (TAB to switch) &nbsp; action: ${action}`);
  lines.push(`ticks: ${ticks} ${runner.isComplete ? '<span class="ok">— ALL SUBGOALS COMPLETE</span>' : ''}`);
  lines.push("");
  subgoals.forEach((sg, i) => {
    const cls = sg.done ? "subgoal-done" : (i === runner.idx ? "subgoal-cur" : "");
    const extra = sg.container ? ` -&gt; ${sg.container}` : "";
    lines.push(`<span class="${cls}">${sg.done ? "check" : (i === runner.idx ? "-&gt;" : "-")} ${sg.kind} ${sg.target}${extra}</span>`);
  });
  hud.innerHTML = lines.join("<br>");
  document.getElementById("crosshair").style.display = control === "manual" ? "block" : "none";
}

function tick(dt) {
  let action = "noop";
  if (!paused && !runner.isComplete) {
    // This must run regardless of control mode: it's what recomputes
    // arrivalThreshold (including the physical-clearance floor) for the
    // CURRENT subgoal. Auto mode also uses the waypoints replan() computes;
    // manual mode ignores waypoints but still needs the correct threshold.
    if (runner.idx !== lastIdx) replan();
    if (control === "auto") {
      if (waypoints) {
        const excludeSet = new Set([SCENE_DATA.agentName]);
        if (runner.current) excludeSet.add(runner.current.target);
        const r = controllerStep(state, agent, waypoints, wpIdx, excludeSet, 140, dt);
        wpIdx = r.wpIdx; action = r.action;
        if (isStuck()) replan();
      } else {
        action = "no-path";
      }
    } else {
      action = manualStep(dt);
    }
    updateCarried();
    ticks += 1;
    runner.tryAdvance(state, agent, arrivalThreshold || 30);
  }
  syncMeshes();
  updateCamera();
  updateHud(action);
}

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 1 / 30);
  tick(dt);
  renderer.render(scene3, camera);
}
animate();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Infinite-mode shell (webapp/'s "infinite corridor" page).
#
# Unlike render_viewer_html, this HTML doesn't embed any scene data at
# all — the shell is IDENTICAL for every infinite-mode run, so it's built
# once (webapp/server.py does this at startup) and served as a normal
# static file thereafter. index.html navigates here with just `?prompt=
# ...&control=...` in the URL (short text, no size concern), and this
# page's own script calls /api/infinite/start itself on load, keeping the
# JSON response in memory as a normal JS variable. An earlier version
# instead had index.html call /api/infinite/start FIRST and hand the
# result off via sessionStorage — confirmed broken once chunks started
# embedding real HSSD meshes instead of primitive shapes: a real run's
# response ran to tens of MB, and sessionStorage's ~5-10MB per-origin quota
# threw "Setting the value of 'infiniteStart' exceeded the quota" on
# essentially every real attempt. Every later chunk still arrives via a
# live fetch() to /api/infinite/next_chunk as the agent approaches the edge
# of what's already loaded — that path was never affected, since each
# individual chunk response alone (not the whole session) stays well under
# any browser's actual per-request limits. Reuses JS_ENGINE (resolveCollision/controllerStep
# — the auto-forward controller below is just controllerStep fed a
# waypoint synthesized 300 units ahead of the agent every frame, re-issued
# continuously, rather than a real destination — never actually "arrives",
# so the agent just keeps walking forward, sliding around furniture via
# the exact same collision code any other scene uses) and JS_MESH_BUILDER
# (buildMesh et al.) verbatim — only the presentation/streaming glue below
# is new.
INFINITE_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Infinite mode</title>
<style>
  html, body { margin: 0; padding: 0; overflow: hidden; background: #111317; }
  #hud {
    position: absolute; top: 10px; left: 10px; color: #eee; background: rgba(20,20,20,0.72);
    font: 14px/1.5 -apple-system, Segoe UI, Roboto, sans-serif; padding: 10px 14px;
    border-radius: 8px; z-index: 5; max-width: 480px;
  }
  #hud b { color: #7ec8ff; }
  #hint {
    position: absolute; bottom: 10px; left: 10px; color: #aaa; font-size: 12px;
    background: rgba(20,20,20,0.6); padding: 6px 10px; border-radius: 6px; z-index: 5;
  }
</style>
</head>
<body>
<div id="hud">Deriving a theme and generating the first stretch of corridor&hellip;</div>
<div id="hint">TAB: auto/manual &nbsp; WASD: move (manual) &nbsp; drag mouse: look/orbit &nbsp; scroll: zoom (auto)</div>
__THREE_SCRIPT_TAG__
__GLTFLOADER_SCRIPT_TAG__
__OBJLOADER_SCRIPT_TAG__
<script>
__JS_ENGINE__
__JS_STREAMING_DRIVER__
</script>
</body>
</html>
"""

JS_STREAMING_DRIVER = r"""
// buildMesh (from JS_MESH_BUILDER, inlined separately below this block —
// see render_infinite_shell_html) special-cases the one asset named
// SCENE_DATA.agentName to build the peg-man body/head/nose group instead
// of a plain box/cylinder. Infinite-mode chunks never contain an asset
// literally named "agent" (every chunk asset name carries a chunk-index
// suffix — see infinite_world.build_chunk), so this sentinel is safe and
// lets the agent's own mesh go through that exact same special-cased
// construction rather than a second copy of the peg-man geometry.
const SCENE_DATA = { agentName: "agent" };

// scene3 MUST be declared here at top level, not inside main() below —
// buildMesh (from JS_MESH_BUILDER, concatenated in just above this block)
// calls `scene3.add(...)` directly, and JS closures only see variables
// from the scope a function was DEFINED in, not wherever it's called
// from. A scene3 declared inside main()'s own function body would be
// completely invisible to buildMesh (a ReferenceError the instant the
// first mesh tried to render), even though main() runs first at page
// load. camera/renderer/lighting don't strictly need to live out here
// too — nothing outside main() touches them — but starting them
// immediately means the canvas is attached and rendering (an empty dark
// room) right away instead of staying blank for however long the
// /api/infinite/start fetch inside main() takes.
const scene3 = new THREE.Scene();
scene3.background = new THREE.Color(0x1c1f24);
scene3.fog = new THREE.Fog(0x1c1f24, 500, 1400);

const camera = new THREE.PerspectiveCamera(65, window.innerWidth / window.innerHeight, 1, 3000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

scene3.add(new THREE.AmbientLight(0xfff2e0, 0.55));
const sun = new THREE.DirectionalLight(0xfff6e8, 0.75);
sun.position.set(200, 400, 150);
sun.castShadow = true;
sun.shadow.mapSize.set(1024, 1024);
sun.shadow.camera.left = -400; sun.shadow.camera.right = 400;
sun.shadow.camera.top = 400; sun.shadow.camera.bottom = -400;
sun.shadow.camera.near = 50; sun.shadow.camera.far = 1200;
scene3.add(sun);
scene3.add(sun.target);

async function main() {

const params = new URLSearchParams(window.location.search);
const prompt = params.get("prompt") || "";
const control0 = params.get("control") || "auto";
if (!prompt) {
  document.body.innerHTML =
    '<div style="color:#eee;padding:40px;font:16px sans-serif">' +
    "No prompt given &mdash; go back and press Generate again." +
    "</div>";
  return;
}

let START_DATA;
try {
  const resp = await fetch("/api/infinite/start", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, control: control0 }),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.error || `server returned ${resp.status}`);
  }
  START_DATA = await resp.json();
} catch (e) {
  document.body.innerHTML =
    '<div style="color:#eee;padding:40px;font:16px sans-serif">' +
    "Couldn't start infinite mode: " + e.message +
    "<br><br>Go back and press Generate again." +
    "</div>";
  return;
}

const CEILING_H = 260;
const CHUNK_WIDTH = START_DATA.chunkWidth, CHUNK_DEPTH = START_DATA.chunkDepth;

function addWall(x, z, w, d) {
  const wall = new THREE.Mesh(
    new THREE.BoxGeometry(w, CEILING_H, d),
    new THREE.MeshStandardMaterial({ color: 0xede6da, roughness: 0.9, transparent: true, opacity: 0.35 })
  );
  wall.position.set(x, CEILING_H / 2, z);
  wall.receiveShadow = true;
  scene3.add(wall);
}

// Each chunk gets its own floor segment and only its LEFT/RIGHT walls —
// no chunk ever draws the wall facing the next one (that seam is left
// open on purpose: consecutive chunks read as physically continuous, not
// a chain of separately-walled rooms), and only chunk 0 draws a front
// wall at all, for a sense of a starting point. This is what makes the
// corridor read as "no boundaries" rather than bounded rooms end to end.
function addChunkToScene(chunk) {
  const zStart = chunk.zOffset;
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(chunk.width, chunk.depth),
    new THREE.MeshStandardMaterial({ color: 0xc9b896, roughness: 0.88, metalness: 0.02 })
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(chunk.width / 2, 0, zStart + chunk.depth / 2);
  floor.receiveShadow = true;
  scene3.add(floor);

  const t = 5;
  addWall(t / 2, zStart + chunk.depth / 2, t, chunk.depth);
  addWall(chunk.width - t / 2, zStart + chunk.depth / 2, t, chunk.depth);
  if (chunk.drawFrontWall) addWall(chunk.width / 2, zStart + t / 2, chunk.width, t);

  for (const name in chunk.assets) {
    const a = chunk.assets[name];
    state.assets[name] = { x: a.x, y: a.y, z: a.z, yaw: a.yaw, half_xz: a.half_xz, half_h: a.half_h, tags: a.tags.slice() };
    buildMesh(name, a);
  }
  state.depth = Math.max(state.depth, zStart + chunk.depth);
  sun.target.position.set(chunk.width / 2, 0, zStart + chunk.depth / 2);
}

const state = {
  width: CHUNK_WIDTH, depth: CHUNK_DEPTH, wallMargin: START_DATA.wallMargin,
  assets: {}, markers: {}, carried: null,
};
const agentInit = START_DATA.agent;
const agent = { x: agentInit.x, y: agentInit.y, z: agentInit.z, yaw: agentInit.yaw,
                half_xz: agentInit.half_xz, half_h: agentInit.half_h };
buildMesh("agent", { kind: "cylinder", x: agent.x, y: agent.y, z: agent.z, yaw: agent.yaw,
                     half_xz: agent.half_xz, half_h: agent.half_h, tags: [], color: 0x1e90ff });

for (const chunk of START_DATA.chunks) addChunkToScene(chunk);
let chunksLoaded = START_DATA.chunks.length;
let chunkFetchInFlight = false;

function maybeFetchNextChunk() {
  if (chunkFetchInFlight || ended) return;
  const loadedFarEdge = chunksLoaded * CHUNK_DEPTH;
  if (agent.z < loadedFarEdge - 150) return;
  chunkFetchInFlight = true;
  const nextIndex = chunksLoaded;
  fetch("/api/infinite/next_chunk", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ themeId: START_DATA.themeId, index: nextIndex }),
  }).then((r) => r.json()).then((chunk) => {
    if (!chunk.error) { addChunkToScene(chunk); chunksLoaded += 1; }
    chunkFetchInFlight = false;
  }).catch(() => { chunkFetchInFlight = false; });
}

let control = (START_DATA.control === "manual") ? "manual" : "auto";
let orbitYaw = Math.PI, orbitPitch = 0.5, orbitRadius = 220;
let lookPitchDelta = 0;
const keys = {};
window.addEventListener("keydown", (e) => {
  keys[e.code] = true;
  if (e.code === "Tab") { e.preventDefault(); control = control === "auto" ? "manual" : "auto"; }
});
window.addEventListener("keyup", (e) => { keys[e.code] = false; });

let dragging = false, lastMouseX = 0, lastMouseY = 0;
renderer.domElement.addEventListener("mousedown", (e) => { dragging = true; lastMouseX = e.clientX; lastMouseY = e.clientY; });
window.addEventListener("mouseup", () => { dragging = false; });
window.addEventListener("mousemove", (e) => {
  if (!dragging) return;
  const dx = e.clientX - lastMouseX, dy = e.clientY - lastMouseY;
  lastMouseX = e.clientX; lastMouseY = e.clientY;
  if (control === "manual") {
    agent.yaw += dx * 0.005;
    lookPitchDelta = Math.max(-1.0, Math.min(1.0, lookPitchDelta - dy * 0.003));
  } else {
    orbitYaw -= dx * 0.006;
    orbitPitch = Math.max(0.15, Math.min(1.3, orbitPitch - dy * 0.004));
  }
});
renderer.domElement.addEventListener("wheel", (e) => {
  orbitRadius = Math.max(80, Math.min(500, orbitRadius + e.deltaY * 0.3));
});
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Continuously aims a waypoint 300 units further down the corridor than
// wherever the agent currently is, re-issued every single tick — it never
// gets close enough to "arrive" (see controllerStep's dist<14 check), so
// the agent just keeps walking forward, turning smoothly to face however
// resolveCollision's AABB push-out just redirected it (sliding around
// furniture exactly like any other scene's collision), rather than
// needing a real A*-planned path to a fixed destination the way a
// goal-directed task does.
function autoForwardTick(dt) {
  // Aims at the CORRIDOR'S CENTER x, not "wherever the agent currently
  // is" — confirmed as a real deadlock otherwise: a collision push-out
  // can nudge the agent sideways off-center, and aiming straight ahead
  // from wherever it drifted to has no restoring force, so it can walk
  // straight into a LATER piece of furniture that isn't in the (otherwise
  // guaranteed-clear — see infinite_world.build_chunk's lane-clearing
  // step) center lane at all, and get stuck there instead. Steering back
  // toward center every tick gives the agent a standing bias toward the
  // one path that's actually guaranteed obstacle-free, so a sideways
  // nudge self-corrects instead of compounding.
  const waypoint = [CHUNK_WIDTH / 2, agent.z + 300];
  controllerStep(state, agent, [waypoint], 0, new Set(), 140, dt);
}

function manualStep(dt) {
  const speed = 170;
  const fx = Math.sin(agent.yaw), fz = Math.cos(agent.yaw);
  const sx = Math.sin(agent.yaw - Math.PI / 2), sz = Math.cos(agent.yaw - Math.PI / 2);
  let vx = 0, vz = 0;
  if (keys["KeyW"]) { vx += fx; vz += fz; }
  if (keys["KeyS"]) { vx -= fx; vz -= fz; }
  if (keys["KeyD"]) { vx += sx; vz += sz; }
  if (keys["KeyA"]) { vx -= sx; vz -= sz; }
  const norm = Math.hypot(vx, vz);
  if (norm > 0) {
    agent.x += (vx / norm) * speed * dt;
    agent.z += (vz / norm) * speed * dt;
  }
  resolveCollision(state, agent, new Set());
}

function updateCamera() {
  if (control === "manual") {
    const eyeHeight = agent.half_h * 1.8;
    camera.position.set(agent.x, eyeHeight, agent.z);
    const lookX = agent.x + Math.sin(agent.yaw) * 50;
    const lookZ = agent.z + Math.cos(agent.yaw) * 50;
    camera.lookAt(lookX, eyeHeight + lookPitchDelta * 60, lookZ);
  } else {
    const cx = agent.x + orbitRadius * Math.sin(orbitYaw) * Math.cos(orbitPitch);
    const cz = agent.z + orbitRadius * Math.cos(orbitYaw) * Math.cos(orbitPitch);
    const cy = agent.y + orbitRadius * Math.sin(orbitPitch) + 40;
    camera.position.set(cx, cy, cz);
    camera.lookAt(agent.x, agent.y + 30, agent.z);
  }
}

const RUN_SECONDS = 30;
const startTime = performance.now();
let ended = false;
const hud = document.getElementById("hud");
function updateHud() {
  const elapsed = (performance.now() - startTime) / 1000;
  const remaining = Math.max(0, RUN_SECONDS - elapsed);
  if (remaining <= 0) ended = true;
  hud.innerHTML =
    `<b>Infinite mode</b> &mdash; ${START_DATA.style}<br>` +
    `control: <b>${control.toUpperCase()}</b> (TAB to switch)<br>` +
    (ended
      ? `<span style="color:#ffd479">Infinite run complete &mdash; reload to try again.</span>`
      : `time left: ${remaining.toFixed(1)}s &nbsp; chunks loaded: ${chunksLoaded}`);
}

const meshes_agent_sync = () => {
  meshes["agent"].position.set(agent.x, agent.y, agent.z);
  meshes["agent"].rotation.y = agent.yaw;
};

const clock = new THREE.Clock();
function tick(dt) {
  if (!ended) {
    if (control === "auto") autoForwardTick(dt); else manualStep(dt);
    maybeFetchNextChunk();
  }
  meshes_agent_sync();
  updateCamera();
  updateHud();
}
function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 1 / 30);
  tick(dt);
  renderer.render(scene3, camera);
}
animate();

}
main();
"""


def render_infinite_shell_html(out_path):
    """
    Generates webapp/static/infinite.html — see INFINITE_HTML_TEMPLATE's
    module comment for why this shell has no per-request data embedded
    (unlike render_viewer_html) and can just be built once.
    """
    html = (INFINITE_HTML_TEMPLATE
            .replace("__THREE_SCRIPT_TAG__", _load_vendor_script("three.min.js"))
            .replace("__GLTFLOADER_SCRIPT_TAG__", _load_vendor_script("GLTFLoader.js"))
            .replace("__OBJLOADER_SCRIPT_TAG__", _load_vendor_script("OBJLoader.js"))
            .replace("__JS_ENGINE__", JS_ENGINE + "\n" + JS_MESH_BUILDER)
            .replace("__JS_STREAMING_DRIVER__", JS_STREAMING_DRIVER))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path