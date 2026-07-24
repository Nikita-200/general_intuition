"""
dsl3d_pybullet.py — a REAL PyBullet-backed implementation of Scene3D/Asset3D.

Same public API as dsl3d.py, on purpose: AddAsset, place_at/relative/around/
grid/against_wall/corner/on, add_container, add_marker, set_agent, set_goal,
pick/place/update_carried, move_agent, has_bad_overlap, is_reachable,
to_state — every one of these has the identical signature and semantics as
the pure-Python version. nav_agent_3d.py, tasks3d.py, synthesizer3d.py, and
viewer3d.py talk to Scene3D/Asset3D only through this interface and were
never touched to make this backend possible — that's the point of having
refactored collision handling behind Scene3D.move_agent() first.

What's actually real here, that wasn't in the pure-Python version:
  - Furniture immovability is now a physics-engine guarantee (mass=0
    rigid bodies), not custom AABB-overlap code.
  - Agent-vs-furniture collision is resolved by Bullet's actual rigid-body
    solver (resetBaseVelocity + stepSimulation), not a hand-written
    move-and-slide.
  - settle() steps real physics.
  - AddAsset accepts an optional mesh_path (OBJ, or GLB on newer PyBullet
    builds) to use a real retrieved mesh instead of a primitive box/
    cylinder — see asset_retrieval.py.

Coordinate convention: PyBullet's native world is Z-up. Everywhere ELSE in
this harness (dsl3d.py, nav_agent_3d.py, the viewer) uses Y-up (x=right,
y=up, z=depth), matching three.js's default and what a graphics-facing
reader expects. Rather than fight that convention throughout the rest of
the codebase, this module keeps PyBullet in its own natural Z-up frame
internally and translates at exactly one boundary: _to_pb()/_from_pb().
Nothing outside this file needs to know PyBullet's axes exist.

IMPORTANT — testing honesty: this file could not be executed in the
sandbox that built it (no PyBullet wheel for that sandbox's Python/
platform; see the README for why, and why your machine is a different
story). It's written carefully against well-established PyBullet APIs and
mirrors the ALREADY-TESTED pure-Python backend's logic move for move, but
it has not been run. `smoke_test_pybullet.py` in this directory runs the
exact same checks that were used to validate the pure-Python version (ram
a table 300 times and confirm it never moves, run 10 fresh scenes per task
type end to end) — run that first thing after installing PyBullet, before
trusting this in anything else.
"""

import math
import os
import random

import pybullet as p
import pybullet_data


def _frange(start, stop, step):
    """See dsl3d.py's _frange — identical, kept in sync since both
    backends' Scene3D.resolve_layout use it for the same grid-scan
    fallback."""
    if stop <= start:
        return [start]
    n = int((stop - start) / step)
    return [start + i * step for i in range(n + 1)] + [stop]


def _scan_local_free_spot(cx, cz, half, item_half, avoid, step):
    """See dsl3d.py's _scan_local_free_spot — identical algorithm, used by
    place_on/place() when random sampling doesn't find a clear spot on a
    crowded surface."""
    if half <= 0:
        return (cx, cz)
    xs = _frange(cx - half, cx + half, step)
    zs = _frange(cz - half, cz + half, step)
    random.shuffle(xs)
    random.shuffle(zs)
    best, best_gap = (cx, cz), -1e18
    for z in zs:
        for x in xs:
            gap = min(
                (math.hypot(x - a.pos[0], z - a.pos[2]) - item_half - a.half_xz for a in avoid),
                default=1e9,
            )
            if gap >= 0:
                return (x, z)
            if gap > best_gap:
                best, best_gap = (x, z), gap
    return best


def _to_pb(x, y, z):
    """our (x, y=up, z=depth) -> pybullet's native (x, y, z=up)"""
    return (x, z, y)


def _from_pb(px, py, pz):
    """pybullet's native (x, y, z=up) -> our (x, y=up, z=depth)"""
    return (px, pz, py)


_MESH_CONVERT_CACHE = {}

# Rotation (as a PyBullet [x, y, z, w] quaternion) that brings a mesh's
# LOCAL signed axis (axis 0/1/2 with a sign, asset_retrieval.py's
# fitted_scale_and_dims convention) onto PyBullet's native world-up axis,
# Z. mesh_scale alone only corrects the MAGNITUDE of that axis (see
# fitted_scale_and_dims's docstring) — without also rotating it here, a
# correctly-scaled mesh whose real "up" is a different local axis (or the
# NEGATIVE direction of the right one) still renders sideways or upside-
# down, which is what made a correctly-sized counter mesh look like a flat
# rug on the floor, and other objects render upside-down once that was
# fixed but sign wasn't yet accounted for.
#
# Hand-derived per (axis, sign) rather than computed generically — there
# are only 6 fixed cases (3 axes x 2 signs), so a lookup table is simpler
# and easier to verify than general quaternion construction:
#   (2, +1): already +Z — identity.
#   (0, +1): -90 deg about Y brings local +X onto +Z.
#   (0, -1): +90 deg about Y brings local -X onto +Z.
#   (1, +1): +90 deg about X brings local +Y onto +Z.
#   (1, -1): -90 deg about X brings local -Y onto +Z.
#   (2, -1): 180 deg about X brings local -Z onto +Z.
_UP_AXIS_SIGN_TO_PYBULLET_Z_QUAT = {
    (2, 1): (0.0, 0.0, 0.0, 1.0),
    (0, 1): (0.0, -0.7071067811865476, 0.0, 0.7071067811865476),
    (0, -1): (0.0, 0.7071067811865476, 0.0, 0.7071067811865476),
    (1, 1): (0.7071067811865476, 0.0, 0.0, 0.7071067811865476),
    (1, -1): (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476),
    (2, -1): (1.0, 0.0, 0.0, 0.0),
}


def _orientation_for_up_axis(up_axis, up_sign=1):
    return list(_UP_AXIS_SIGN_TO_PYBULLET_Z_QUAT.get((up_axis, up_sign), (0.0, 0.0, 0.0, 1.0)))




def _ensure_pybullet_mesh_format(mesh_path):
    """
    PyBullet's GEOM_MESH loader (createCollisionShape/createVisualShape)
    only accepts OBJ or STL — it does NOT accept GLB/GLTF, even though the
    error message ("invalid mesh filename extension") makes that easy to
    miss at a glance. Objaverse (and most real mesh sources) hand you a
    .glb. Since `trimesh` is already a dependency for asset_retrieval.py's
    bounding-box computation, use it here too to convert once and cache the
    result next to the source file, rather than requiring a separate
    preprocessing step.
    """
    ext = os.path.splitext(mesh_path)[1].lower()
    if ext in (".obj", ".stl"):
        return mesh_path
    if mesh_path in _MESH_CONVERT_CACHE:
        return _MESH_CONVERT_CACHE[mesh_path]
    obj_path = os.path.splitext(mesh_path)[0] + ".converted.obj"
    if not os.path.isfile(obj_path):
        import trimesh
        mesh = trimesh.load(mesh_path, force="mesh")
        mesh.export(obj_path)
    _MESH_CONVERT_CACHE[mesh_path] = obj_path
    return obj_path


class Asset3D:
    def __init__(self, name, kind, body_id, half_xz, half_h, static=True,
                 portable=False, tags=None, mesh_path=None, mesh_scale=1.0,
                 mesh_up_axis=1, mesh_up_sign=1):
        self.name = name
        self.kind = kind
        self.body_id = body_id
        self._half_xz = half_xz
        self._half_h = half_h
        self.static = static
        self.portable = portable
        self.tags = list(tags or [])
        if portable and "portable" not in self.tags:
            self.tags.append("portable")
        self.resting_on = None
        self._yaw = 0.0
        self.mesh_path = mesh_path
        self.mesh_scale = mesh_scale
        self.mesh_up_axis = mesh_up_axis  # see dsl3d.py's Asset3D for what this means
        self.mesh_up_sign = mesh_up_sign  # see dsl3d.py's Asset3D for what this means

    @property
    def pos(self):
        ppos, _ = p.getBasePositionAndOrientation(self.body_id)
        x, y, z = _from_pb(*ppos)
        return [x, y, z]

    @property
    def yaw(self):
        return self._yaw

    @yaw.setter
    def yaw(self, value):
        # Plain tracking value ONLY — no pybullet call here. This used to
        # call self._teleport(), which does a full
        # resetBasePositionAndOrientation + explicit velocity zero, on
        # EVERY simulation tick (nav_agent_3d.py's steering controller sets
        # agent.yaw every tick, right before calling move_agent to set real
        # translational velocity). That meant the agent's pose was being
        # forcibly reset, and its velocity explicitly zeroed, once per
        # frame by a completely separate code path from the one actually
        # trying to move it — two uncoordinated authorities fighting over
        # the same body every tick. This was the real reason the agent
        # never made sustained progress even after the linearVelocity
        # axis-swap fix (that fix was also correct, just moot underneath
        # this). move_agent() below is now the only place that applies
        # orientation to the physics body, as one coherent part of its own
        # per-tick update.
        self._yaw = value

    @property
    def half_xz(self):
        return self._half_xz

    @property
    def half_height(self):
        return self._half_h

    @property
    def top_y(self):
        return self.pos[1] + self._half_h

    def _teleport(self, x, y, z, yaw):
        """Directly set position+orientation, bypassing physics. Used for
        scene authoring (placement calls) — NOT used for agent movement
        during simulation, which goes through Scene3D.move_agent()'s
        velocity-based real collision instead."""
        px, py, pz = _to_pb(x, y, z)
        quat = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(self.body_id, [px, py, pz], quat)
        p.resetBaseVelocity(self.body_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        self._yaw = yaw

    def set_pose(self, x=None, z=None, y=None, yaw=None):
        cx, cy, cz = self.pos
        nx = x if x is not None else cx
        ny = y if y is not None else cy
        nz = z if z is not None else cz
        nyaw = yaw if yaw is not None else self._yaw
        self._teleport(nx, ny, nz, nyaw)
        return self

    def get_aabb_xz(self):
        x, _, z = self.pos
        h = self._half_xz
        return (x - h, x + h, z - h, z + h)

    def distance_to(self, other):
        ax, ay, az = self.pos
        bx, by, bz = other.pos
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)

    def xz_distance_to(self, other):
        ax, _, az = self.pos
        bx, _, bz = other.pos
        return math.hypot(ax - bx, az - bz)

    def is_overlapping_xz(self, other, margin=0.0):
        a, b = self.get_aabb_xz(), other.get_aabb_xz()
        return not (a[1] + margin < b[0] or b[1] + margin < a[0] or
                    a[3] + margin < b[2] or b[3] + margin < a[2])

    def contains_xz(self, other):
        a = self.get_aabb_xz()
        ox, _, oz = other.pos
        return a[0] <= ox <= a[1] and a[2] <= oz <= a[3]


DIRECTIONS = {
    "front": (0, 1), "back": (0, -1), "left": (-1, 0), "right": (1, 0),
    "front_left": (-0.7, 0.7), "front_right": (0.7, 0.7),
    "back_left": (-0.7, -0.7), "back_right": (0.7, -0.7),
}


class Scene3D:
    """Real-physics Scene3D. See module docstring for the coordinate
    convention and what's genuinely different from dsl3d.py's version."""

    def __init__(self, width=800, depth=600, ceiling=260, wall_margin=20, gui=False):
        self.width, self.depth, self.ceiling = width, depth, ceiling
        self.wall_margin = wall_margin
        self.assets = {}
        self.agent = None
        self.goal = None
        self.carried = None
        self.markers = {}
        self.retriever = None  # set by synthesize_scene(retriever=...); see asset_retrieval.py

        self._client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setPhysicsEngineParameter(fixedTimeStep=1 / 60, numSolverIterations=10)
        # Real floor: the agent (the one dynamic body in the scene) actually
        # rests on this via gravity + collision, rather than being held at a
        # fixed height by us.
        floor_shape = p.createCollisionShape(p.GEOM_PLANE)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=floor_shape,
                           basePosition=[width / 2, depth / 2, 0])
        self._wall_body_ids = []
        self._build_walls()

    def _build_walls(self):
        # Body ids tracked (self._wall_body_ids) so fit_room_to_content can
        # tear these down and rebuild at a new size — a real rigid body's
        # position is fixed at creation, unlike self.width/self.depth being
        # plain numbers, so resizing the room means actually replacing
        # these, not just changing two attributes.
        t = 5
        segs = [
            (self.width / 2, 0, self.width, t),   # front (z=0)
            (self.width / 2, self.depth, self.width, t),  # back
            (0, self.depth / 2, t, self.depth),   # left
            (self.width, self.depth / 2, t, self.depth),  # right
        ]
        for x, z, sx, sz in segs:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx / 2, sz / 2, self.ceiling / 2])
            px, py, pz = _to_pb(x, self.ceiling / 2, z)
            body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                         basePosition=[px, py, pz])
            self._wall_body_ids.append(body_id)

    # ---------- placement support ----------
    def _find_free_spot(self, half_xz, tries=25):
        m = self.wall_margin
        best, best_score = None, -1
        for _ in range(tries):
            x = random.uniform(half_xz + m, self.width - half_xz - m)
            z = random.uniform(half_xz + m, self.depth - half_xz - m)
            min_gap = min(
                (math.hypot(x - a.pos[0], z - a.pos[2]) - half_xz - a.half_xz
                 for a in self.assets.values()),
                default=1e9,
            )
            if min_gap > 0:
                return (x, z)
            if min_gap > best_score:
                best, best_score = (x, z), min_gap
        return best or (self.width / 2, self.depth / 2)

    def _scan_free_spot(self, half_xz, avoid, step=12):
        """See dsl3d.py's Scene3D._scan_free_spot — identical algorithm,
        used by resolve_layout's fallback."""
        m = self.wall_margin
        x0, x1 = half_xz + m, self.width - half_xz - m
        z0, z1 = half_xz + m, self.depth - half_xz - m
        if x1 <= x0 or z1 <= z0:
            return (self.width / 2, self.depth / 2)
        xs = list(_frange(x0, x1, step))
        zs = list(_frange(z0, z1, step))
        random.shuffle(xs)
        random.shuffle(zs)
        best, best_gap = None, -1e18
        for z in zs:
            for x in xs:
                gap = min(
                    (math.hypot(x - a.pos[0], z - a.pos[2]) - half_xz - a.half_xz
                     for a in avoid),
                    default=1e9,
                )
                if gap >= 0:
                    return (x, z)
                if gap > best_gap:
                    best, best_gap = (x, z), gap
        return best or (self.width / 2, self.depth / 2)

    # ---------- object registration ----------
    def AddAsset(self, name, kind="box", size=30, height=None, dynamic=False,
                 portable=False, tags=None, mesh_path=None, mesh_scale=1.0,
                 mesh_up_axis=1, mesh_up_sign=1):
        """
        Same signature as dsl3d.py's AddAsset, plus:
          mesh_path: path to a real mesh. PyBullet's own mesh loader only
            accepts OBJ/STL — GLB (what Objaverse hands you) is NOT
            supported directly (confirmed: PyBullet raises "invalid mesh
            filename extension" for .glb), so this auto-converts via
            trimesh and caches the result next to the source file. `size`/
            `height` are still used for the floor-plane footprint
            (pathfinding inflation, overlap checks), so pass them as the
            mesh's real-world bounding box even when mesh_path is set.
          mesh_scale: per-axis (sx, sy, sz) scale applied to the mesh file
            (also accepts a single float for backward compatibility) —
            typically computed from the mesh's raw bounding box vs. the
            target `size`/`height`; see asset_retrieval.py.
          mesh_up_axis/mesh_up_sign: which of the mesh's own local axes is
            really "up", and which DIRECTION along it — used to orient the
            body so that signed axis points along PyBullet's native
            world-Z, since mesh_scale alone only fixes magnitude, not
            direction or sign (see the module-level comment above
            _orientation_for_up_axis).
        """
        h = height if height is not None else size * 1.4
        half_h = h / 2
        if mesh_path is None and self.retriever is not None and name != "agent":
            try:
                # See dsl3d.py's AddAsset for why this passes size/h as
                # hints rather than letting the retriever invent its own
                # fixed footprint for every object regardless of type.
                result = self.retriever.retrieve(name, size_hint=size, height_hint=h)
                if result and result.get("mesh_path"):
                    size = result.get("size", size)
                    h = result.get("height", h)
                    half_h = h / 2
                    mesh_path = result["mesh_path"]
                    mesh_scale = result.get("mesh_scale", 1.0)
                    mesh_up_axis = result.get("mesh_up_axis", 1)
                    mesh_up_sign = result.get("mesh_up_sign", 1)
            except Exception as e:
                print(f"[AddAsset] retrieval failed for {name!r}: {e}")
        x, z = self._find_free_spot(size)

        base_orn = [0.0, 0.0, 0.0, 1.0]  # identity; only real meshes with a non-Z up-axis need rotating
        if mesh_path:
            try:
                mesh_path = _ensure_pybullet_mesh_format(mesh_path)
                # mesh_scale is now a non-uniform (sx, sy, sz) 3-tuple from
                # asset_retrieval.py's fitted_scale_and_dims (needed so the
                # real mesh is forced into the DSL's exact size/height box
                # regardless of its native proportions/up-axis — see
                # CHANGES.md Round 5) — but also still accepts a plain float
                # for backward compatibility with any mesh_path set some
                # other way. The old `[mesh_scale] * 3` here assumed a
                # float and, once mesh_scale became a tuple, was building
                # `[(sx,sy,sz), (sx,sy,sz), (sx,sy,sz)]` — a list of three
                # TUPLES instead of a flat [sx, sy, sz] vector. PyBullet's
                # C++ binding can't parse that and throws from inside
                # createCollisionShape, with no indication of why — this is
                # the actual cause of every "mesh load failed" warning.
                if isinstance(mesh_scale, (int, float)):
                    mesh_scale_vec = [mesh_scale, mesh_scale, mesh_scale]
                else:
                    mesh_scale_vec = list(mesh_scale)
                col_shape = p.createCollisionShape(p.GEOM_MESH, fileName=mesh_path,
                                                    meshScale=mesh_scale_vec)
                vis_shape = p.createVisualShape(p.GEOM_MESH, fileName=mesh_path,
                                                 meshScale=mesh_scale_vec)
                # mesh_scale fixes the MAGNITUDE of whichever local axis is
                # really "up" but says nothing about its DIRECTION or SIGN —
                # without this, a correctly-scaled mesh whose real up-axis
                # isn't already +Z renders sideways or upside-down, which is
                # exactly what made a correctly-sized counter mesh look
                # like a flat rug on the floor (wrong axis) and other real
                # meshes render upside-down (right axis, wrong sign).
                base_orn = _orientation_for_up_axis(mesh_up_axis, mesh_up_sign)
            except Exception as e:
                # PyBullet's mesh loader is picky (some OBJ exports still
                # fail it — non-manifold geometry, multiple materials,
                # etc.); rather than aborting the whole scene, fall back to
                # a primitive shape for this one object and keep going.
                print(f"[AddAsset] mesh load failed for {name!r} ({e}); "
                      f"using a primitive {kind} instead.")
                mesh_path = None
                base_orn = [0.0, 0.0, 0.0, 1.0]

        if not mesh_path:
            if kind == "cylinder":
                col_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=size, height=half_h * 2)
                vis_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=size, length=half_h * 2,
                                                 rgbaColor=[0.75, 0.35, 0.25, 1])
            else:
                col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size, size, half_h])
                vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[size, size, half_h],
                                                 rgbaColor=[0.75, 0.35, 0.25, 1])

        mass = 1.0 if dynamic else 0.0  # mass=0 is PyBullet's native "static/immovable" —
        # this is the actual fix for "furniture shouldn't move", enforced by the physics
        # engine itself now, not application code.
        px, py, pz = _to_pb(x, half_h, z)
        body_id = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col_shape,
                                     baseVisualShapeIndex=vis_shape, basePosition=[px, py, pz],
                                     baseOrientation=base_orn)
        # linearDamping/angularDamping deliberately 0, not a "small friction"
        # value: move_agent() calls resetBaseVelocity() every single tick,
        # fully overriding whatever velocity damping would otherwise decay —
        # our controller IS the only thing that ever sets this body's
        # velocity, so engine-side damping has no legitimate job to do here,
        # only room to cause harm. And it did: confirmed empirically (this
        # was the first time this file actually got to run against real
        # PyBullet — see the module docstring) that at the previous 0.9
        # value, this build's damping isn't the mild per-step decay the
        # number suggests — resetBaseVelocity([20,100,0]) followed by ONE
        # stepSimulation() came back as (-10.9, -54.4, ...): sign-INVERTED
        # and roughly halved, not just reduced. That's not friction, it's
        # the agent driving backward relative to every command it's given —
        # confirmed as the actual cause of "no path to subgoal" / running
        # out the full tick budget stuck in place. Swept 0.0-0.9 and found
        # the flip starts around ~0.55; only 0.0 reproduced the expected,
        # undamped displacement for a commanded velocity, so that's what's
        # actually used, rather than picking another value that merely
        # moves the cliff edge.
        p.changeDynamics(body_id, -1, lateralFriction=0.8, restitution=0.0,
                          linearDamping=0.0, angularDamping=0.0)

        if portable:
            # No first-class "sensor" flag on a PyBullet MultiBody, but a
            # zeroed collision filter group/mask achieves the same effect:
            # the body still exists and renders, but generates no collision
            # response with anything — exactly what an item resting on/in
            # furniture needs.
            p.setCollisionFilterGroupMask(body_id, -1, collisionFilterGroup=0, collisionFilterMask=0)

        asset = Asset3D(name, kind, body_id, size, half_h, static=not dynamic,
                         portable=portable, tags=tags, mesh_path=mesh_path, mesh_scale=mesh_scale,
                         mesh_up_axis=mesh_up_axis, mesh_up_sign=mesh_up_sign)
        uniq = name
        i = 2
        while uniq in self.assets:
            uniq = f"{name}_{i}"
            i += 1
        self.assets[uniq] = asset
        return asset

    # ---------- placement primitives (identical logic to dsl3d.py) ----------
    def place_at(self, asset, x, z, yaw=0.0):
        asset.set_pose(x=x, z=z, yaw=yaw)
        return asset

    def place_relative(self, asset, anchor, direction, distance=60):
        dx, dz = DIRECTIONS[direction]
        ax, _, az = anchor.pos
        asset.set_pose(x=ax + dx * distance, z=az + dz * distance)
        return asset

    def place_around(self, assets, anchor, radius=80, start_angle=0.0):
        ax, _, az = anchor.pos
        n = len(assets)
        for i, a in enumerate(assets):
            ang = start_angle + 2 * math.pi * i / n
            a.set_pose(x=ax + radius * math.cos(ang), z=az + radius * math.sin(ang),
                       yaw=ang + math.pi)
        return assets

    def place_against_wall(self, asset, wall, offset=None, margin=20):
        half = asset.half_xz
        off = offset if offset is not None else 0
        if wall == "back":
            x, z = self.width / 2 + off, self.depth - margin - half
        elif wall == "front":
            x, z = self.width / 2 + off, margin + half
        elif wall == "left":
            x, z = margin + half, self.depth / 2 + off
        elif wall == "right":
            x, z = self.width - margin - half, self.depth / 2 + off
        else:
            raise ValueError("wall must be front/back/left/right")
        asset.set_pose(x=x, z=z)
        return asset

    def place_corner(self, asset, corner, margin=20):
        half = asset.half_xz
        x = margin + half if "left" in corner else self.width - margin - half
        z = margin + half if "front" in corner else self.depth - margin - half
        asset.set_pose(x=x, z=z)
        return asset

    def place_grid(self, assets, rows, cols, spacing=70, origin=(100, 100)):
        ox, oz = origin
        for i, a in enumerate(assets):
            r, c = divmod(i, cols)
            a.set_pose(x=ox + c * spacing, z=oz + r * spacing)
        return assets

    def place_on(self, item, surface, jitter=0.6, tries=25):
        """See dsl3d.py's Scene3D.place_on for the overlap-avoidance
        rationale — identical algorithm, kept in sync."""
        half = surface.half_xz * jitter
        sx, _, sz = surface.pos
        neighbors = [a for a in self.assets.values() if a.resting_on is surface and a is not item]
        xz = None
        for _ in range(tries):
            x = sx + random.uniform(-half, half)
            z = sz + random.uniform(-half, half)
            gap = min(
                (math.hypot(x - a.pos[0], z - a.pos[2]) - item.half_xz - a.half_xz
                 for a in neighbors),
                default=1e9,
            )
            if gap > 0:
                xz = (x, z)
                break
        if xz is None:
            xz = _scan_local_free_spot(sx, sz, half, item.half_xz, neighbors,
                                        step=max(2.0, item.half_xz))
        x, z = xz
        item.set_pose(x=x, z=z, y=surface.top_y + item.half_height)
        item.resting_on = surface
        return item

    def place_on_wall(self, asset, wall, height, offset=None, margin=4):
        """See dsl3d.py's Scene3D.place_on_wall for the full rationale —
        identical semantics, kept in sync."""
        half = asset.half_xz
        off = offset if offset is not None else 0
        if wall == "back":
            x, z = self.width / 2 + off, self.depth - margin - half
        elif wall == "front":
            x, z = self.width / 2 + off, margin + half
        elif wall == "left":
            x, z = margin + half, self.depth / 2 + off
        elif wall == "right":
            x, z = self.width - margin - half, self.depth / 2 + off
        else:
            raise ValueError("wall must be front/back/left/right")
        asset.set_pose(x=x, z=z, y=height)
        if "wall_mounted" not in asset.tags:
            asset.tags.append("wall_mounted")
            # AddAsset already fixed this body's collision filter at
            # creation time (only "portable" gets zeroed there, since
            # "wall_mounted" is applied here, later). Real PyBullet
            # collision is genuinely 3D — unlike the tag-based checks in
            # has_bad_overlap/resolve_layout/move_agent's AABB backstop, a
            # rigid-body solver doesn't know or care about our tags, only
            # about actual overlapping volumes — and a wall-mounted
            # object's height often still falls inside the agent's own
            # 0-to-170-tall collision cylinder. Zero the filter here too so
            # Bullet's real solver also treats it as non-colliding, not
            # just our own bookkeeping.
            p.setCollisionFilterGroupMask(asset.body_id, -1,
                                           collisionFilterGroup=0, collisionFilterMask=0)
        # See dsl3d.py's place_on_wall for why this is remembered —
        # fit_room_to_content re-snaps against it after a resize.
        asset._wall_mount = (wall, off, height, margin)
        return asset

    def add_container(self, name, size=22, height=35, **kw):
        kw.setdefault("tags", [])
        kw["tags"] = list(kw["tags"]) + ["container"]
        return self.AddAsset(name, kind="box", size=size, height=height, dynamic=False, **kw)

    def add_marker(self, name, x, z, y=0.0):
        uniq = name
        i = 2
        while uniq in self.markers:
            uniq = f"{name}_{i}"
            i += 1
        self.markers[uniq] = (x, y, z)
        return uniq

    # ---------- roles ----------
    def set_agent(self, asset):
        if asset.static:
            # Force dynamic — same reasoning as the pure-Python backend's
            # set_agent fix: the agent must move under control regardless
            # of how it was created. In PyBullet terms, give it real mass
            # (it was created with baseMass=0) via changeDynamics.
            p.changeDynamics(asset.body_id, -1, mass=5.0)
            asset.static = False
        self.agent = asset
        return asset

    def set_goal(self, asset):
        self.goal = asset
        return asset

    # ---------- agent movement: THE actual PyBullet payoff ----------
    def move_agent(self, vx, vz, dt):
        """
        Real rigid-body collision, not hand-rolled AABB code: apply the
        desired yaw (set as a plain value by the steering controller just
        before this call — see the yaw property above) as one coherent
        orientation update, then set velocity and step. Bullet's own
        solver prevents interpenetration with static (mass=0) furniture.
        Any unwanted tip-over from collision impulses is corrected after
        the step by zeroing out roll/pitch (a standard "upright character
        controller" trick) while preserving whatever yaw physics actually
        arrived at.
        """
        agent = self.agent
        pos, _ = p.getBasePositionAndOrientation(agent.body_id)
        p.resetBasePositionAndOrientation(agent.body_id, pos,
                                           p.getQuaternionFromEuler([0, 0, agent._yaw]))

        pvx, pvy, pvz = _to_pb(vx, 0, vz)  # our vertical velocity is always 0; gravity handles that
        cur_lin, cur_ang = p.getBaseVelocity(agent.body_id)
        p.resetBaseVelocity(agent.body_id, linearVelocity=[pvx, pvy, cur_lin[2]],
                             angularVelocity=[0, 0, 0])
        p.stepSimulation()
        # re-level: kill any roll/pitch picked up from a collision impulse
        pos, orn = p.getBasePositionAndOrientation(agent.body_id)
        euler = list(p.getEulerFromQuaternion(orn))
        euler[0] = 0.0
        euler[1] = 0.0
        p.resetBasePositionAndOrientation(agent.body_id, pos, p.getQuaternionFromEuler(euler))
        agent._yaw = euler[2]

        # Belt-and-suspenders AABB push-out against solid furniture, same
        # logic dsl3d.py's pure-Python move_agent uses for its own
        # collision. Confirmed necessary, not just defensive: Bullet's
        # contact solver DOES push back on a single step of interpenetration,
        # but move_agent reasserts the agent's full commanded velocity via
        # resetBaseVelocity every tick regardless of how much of that push-
        # back survived — so a small net creep per tick, invisible in any
        # one step, accumulates over hundreds of ticks into the agent
        # plowing straight through solid furniture (confirmed with
        # smoke_test_pybullet.py's ram test: 300 ticks of ramming a table
        # walked the agent from one side of it to 189 units out the other
        # side, table included, before this fix). Real rigid-body collision
        # is still what does the per-step work (this only mops up whatever
        # penetration made it through), so this stays a correction, not a
        # replacement for Bullet's own solver.
        ar = agent.half_xz
        ax, _, az = agent.pos
        for a in self.assets.values():
            if a is agent or "portable" in a.tags or "wall_mounted" in a.tags:
                continue
            minx, maxx, minz, maxz = a.get_aabb_xz()
            minx -= ar; maxx += ar
            minz -= ar; maxz += ar
            if minx < ax < maxx and minz < az < maxz:
                pen_left, pen_right = ax - minx, maxx - ax
                pen_near, pen_far = az - minz, maxz - az
                m = min(pen_left, pen_right, pen_near, pen_far)
                if m == pen_left:
                    ax = minx
                elif m == pen_right:
                    ax = maxx
                elif m == pen_near:
                    az = minz
                else:
                    az = maxz
        if (ax, az) != (agent.pos[0], agent.pos[2]):
            agent.set_pose(x=ax, z=az)

        # clamp to room bounds (belt-and-suspenders alongside the wall colliders)
        x, y, z = agent.pos
        m = agent.half_xz + 2
        nx = max(m, min(self.width - m, x))
        nz = max(m, min(self.depth - m, z))
        if (nx, nz) != (x, z):
            agent.set_pose(x=nx, z=nz)

    # ---------- manipulation ----------
    def pick(self, item):
        if self.carried is not None:
            raise RuntimeError(f"already carrying {self.carried.name}")
        item.tags = [t for t in item.tags if t != "placed"] + ["carried"]
        self.carried = item
        return item

    def update_carried(self, forward_offset=26, height=110):
        if self.carried is None or self.agent is None:
            return
        ax, _, az = self.agent.pos
        yaw = self.agent.yaw
        x = ax + forward_offset * math.sin(yaw)
        z = az + forward_offset * math.cos(yaw)
        self.carried.set_pose(x=x, z=z, y=height)

    def place(self, container=None):
        if self.carried is None:
            raise RuntimeError("nothing is being carried")
        item = self.carried
        item.tags = [t for t in item.tags if t != "carried"]
        if container is not None:
            half = container.half_xz * 0.5
            cx, _, cz = container.pos
            neighbors = [a for a in self.assets.values()
                         if a.resting_on is container and a is not item]
            xz = None
            for _ in range(25):
                x = cx + random.uniform(-half, half)
                z = cz + random.uniform(-half, half)
                gap = min(
                    (math.hypot(x - a.pos[0], z - a.pos[2]) - item.half_xz - a.half_xz
                     for a in neighbors),
                    default=1e9,
                )
                if gap > 0:
                    xz = (x, z)
                    break
            if xz is None:
                xz = _scan_local_free_spot(cx, cz, half, item.half_xz, neighbors,
                                            step=max(2.0, item.half_xz))
            x, z = xz
            item.set_pose(x=x, z=z, y=container.top_y + item.half_height)
            item.resting_on = container
            item.tags += ["placed", f"placed_in:{container.name}"]
        self.carried = None
        return item

    # ---------- validity checks ----------
    def has_bad_overlap(self, margin=-3):
        solids = [a for a in self.assets.values()
                  if a is not self.agent and "portable" not in a.tags
                  and "wall_mounted" not in a.tags]
        for i, a in enumerate(solids):
            for b in solids[i + 1:]:
                if a.is_overlapping_xz(b, margin=margin):
                    return True
        return False

    def out_of_bounds_assets(self):
        """See dsl3d.py's Scene3D.out_of_bounds_assets — identical check,
        kept in sync since both backends feed the same synthesize_scene
        retry loop."""
        bad = []
        for name, a in self.assets.items():
            if a is self.agent:
                continue
            x0, x1, z0, z1 = a.get_aabb_xz()
            if x0 < 0 or z0 < 0 or x1 > self.width or z1 > self.depth:
                bad.append(name)
        return bad

    def _resize_room(self, new_width, new_depth):
        """PyBullet backend: unlike dsl3d.py, the walls are real rigid
        bodies fixed at creation time — resizing means actually tearing
        them down and rebuilding at the new dimensions, not just updating
        two numbers. The floor is a GEOM_PLANE (mathematically infinite in
        its own plane), so it needs no rebuild regardless of width/depth."""
        for body_id in self._wall_body_ids:
            p.removeBody(body_id)
        self._wall_body_ids = []
        self.width, self.depth = new_width, new_depth
        self._build_walls()

    def fit_room_to_content(self, target_occupancy=0.12, min_width=400, min_depth=300,
                             max_width=1300, max_depth=1000):
        """See dsl3d.py's Scene3D.fit_room_to_content for the full
        rationale — identical algorithm, kept in sync since both backends
        feed the same synthesize_scene pipeline (fit_room_to_content then
        resolve_layout, right after build_scene() returns)."""
        solids = [a for a in self.assets.values()
                  if a is not self.agent and "portable" not in a.tags
                  and "wall_mounted" not in a.tags]
        if not solids:
            return
        total_area = sum((2 * a.half_xz) ** 2 for a in solids)
        if total_area <= 0:
            return
        old_width, old_depth = self.width, self.depth
        target_area = total_area / target_occupancy
        scale = math.sqrt(target_area / (old_width * old_depth))
        new_width = min(max(old_width * scale, min_width), max_width)
        new_depth = min(max(old_depth * scale, min_depth), max_depth)
        if abs(new_width - old_width) < 1 and abs(new_depth - old_depth) < 1:
            return

        self._resize_room(new_width, new_depth)
        # Scale EVERY non-agent asset uniformly (not just solids, re-gluing
        # surface items by delta) — see dsl3d.py's Scene3D.fit_room_to_content
        # docstring for why: a portable that isn't resting on anything (a
        # backpack on the floor) has no delta to inherit and kept stale
        # coordinates outside the new walls under the old approach.
        kx, kz = new_width / old_width, new_depth / old_depth
        for a in self.assets.values():
            if a is self.agent:
                continue
            wall_mount = getattr(a, "_wall_mount", None)
            if wall_mount is not None:
                wall, off, height, margin = wall_mount
                self.place_on_wall(a, wall, height, offset=off, margin=margin)
            else:
                a.set_pose(x=a.pos[0] * kx, z=a.pos[2] * kz)

    def resolve_layout(self, iterations=300, gap=4.0):
        """See dsl3d.py's Scene3D.resolve_layout for the full rationale —
        identical algorithm, kept in sync since both backends feed the
        same synthesize_scene retry loop and expose the same public API
        (a.pos, a.half_xz, a.get_aabb_xz(), a.set_pose, a.resting_on)."""
        solids = [a for a in self.assets.values()
                  if a is not self.agent and "portable" not in a.tags
                  and "wall_mounted" not in a.tags]
        if not solids:
            return
        start = {id(a): (a.pos[0], a.pos[2]) for a in solids}

        def clamp(a):
            hx = a.half_xz
            mx = min(self.wall_margin, max(0.0, self.width / 2 - hx))
            mz = min(self.wall_margin, max(0.0, self.depth / 2 - hx))
            x = min(max(a.pos[0], hx + mx), self.width - hx - mx)
            z = min(max(a.pos[2], hx + mz), self.depth - hx - mz)
            if x != a.pos[0] or z != a.pos[2]:
                a.set_pose(x=x, z=z)

        for a in solids:
            clamp(a)

        for _ in range(iterations):
            moved = False
            for i, a in enumerate(solids):
                for b in solids[i + 1:]:
                    if not a.is_overlapping_xz(b, margin=gap):
                        continue
                    ax0, ax1, az0, az1 = a.get_aabb_xz()
                    bx0, bx1, bz0, bz1 = b.get_aabb_xz()
                    push_x = (min(ax1, bx1) - max(ax0, bx0)) + gap
                    push_z = (min(az1, bz1) - max(az0, bz0)) + gap
                    dx, dz = a.pos[0] - b.pos[0], a.pos[2] - b.pos[2]
                    if push_x < push_z:
                        sign = 1.0 if dx > 0 else (-1.0 if dx < 0 else (1.0 if i % 2 == 0 else -1.0))
                        half = push_x / 2 + 0.05
                        a.set_pose(x=a.pos[0] + sign * half)
                        b.set_pose(x=b.pos[0] - sign * half)
                    else:
                        sign = 1.0 if dz > 0 else (-1.0 if dz < 0 else (1.0 if i % 2 == 0 else -1.0))
                        half = push_z / 2 + 0.05
                        a.set_pose(z=a.pos[2] + sign * half)
                        b.set_pose(z=b.pos[2] - sign * half)
                    moved = True
            for a in solids:
                clamp(a)
            if not moved:
                break

        # Last-resort fallback — see dsl3d.py's Scene3D.resolve_layout for
        # the full rationale (a crowded, realistic-scale room — 10+
        # objects, some large — that local pairwise nudging doesn't fully
        # untangle within a bounded pass count; random sampling alone
        # wasn't reliable enough at that scale either, hence the
        # exhaustive grid scan and largest-first ordering).
        for _ in range(4):
            remaining = [a for a in solids
                         if any(a is not b and a.is_overlapping_xz(b, margin=gap) for b in solids)]
            if not remaining:
                break
            remaining.sort(key=lambda a: a.half_xz, reverse=True)
            for a in remaining:
                avoid = [b for b in solids if b is not a]
                x, z = self._scan_free_spot(a.half_xz, avoid)
                a.set_pose(x=x, z=z)

        for a in solids:
            sx, sz = start[id(a)]
            dx, dz = a.pos[0] - sx, a.pos[2] - sz
            if dx == 0 and dz == 0:
                continue
            for item in self.assets.values():
                if item.resting_on is a:
                    item.set_pose(x=item.pos[0] + dx, z=item.pos[2] + dz)

        # Per-surface clutter declutter — see dsl3d.py's Scene3D.resolve_layout
        # for the full rationale (place_on's own greedy per-item overlap
        # avoidance still leaves rare residual overlaps once a surface has
        # many small items; this looks at each surface's final set jointly).
        by_surface = {}
        for a in self.assets.values():
            if a.resting_on is not None:
                by_surface.setdefault(id(a.resting_on), []).append(a)
        for items in by_surface.values():
            if len(items) < 2:
                continue
            for _ in range(60):
                moved = False
                for i, a in enumerate(items):
                    for b in items[i + 1:]:
                        if not a.is_overlapping_xz(b, margin=0.5):
                            continue
                        ax0, ax1, az0, az1 = a.get_aabb_xz()
                        bx0, bx1, bz0, bz1 = b.get_aabb_xz()
                        push_x = (min(ax1, bx1) - max(ax0, bx0)) + 0.5
                        push_z = (min(az1, bz1) - max(az0, bz0)) + 0.5
                        dx, dz = a.pos[0] - b.pos[0], a.pos[2] - b.pos[2]
                        if push_x < push_z:
                            sign = 1.0 if dx > 0 else (-1.0 if dx < 0 else (1.0 if i % 2 == 0 else -1.0))
                            half = push_x / 2 + 0.05
                            a.set_pose(x=a.pos[0] + sign * half)
                            b.set_pose(x=b.pos[0] - sign * half)
                        else:
                            sign = 1.0 if dz > 0 else (-1.0 if dz < 0 else (1.0 if i % 2 == 0 else -1.0))
                            half = push_z / 2 + 0.05
                            a.set_pose(z=a.pos[2] + sign * half)
                            b.set_pose(z=b.pos[2] - sign * half)
                        moved = True
                if not moved:
                    break

    def is_reachable(self, target_pos=None, exclude_names=()):
        import nav_agent_3d
        if target_pos is None:
            if self.goal is None:
                # See dsl3d.py's Scene3D.is_reachable for why this can't just
                # `return True` — pick-and-place scenes have no single
                # scene.goal, so this used to skip validation entirely.
                return self._all_named_targets_reachable()
            target_pos = self.goal.pos
        exclude = set(exclude_names) | ({self.agent.name} if self.agent else set())
        if self.goal is not None and target_pos is self.goal.pos:
            exclude.add(self.goal.name)
        return nav_agent_3d.plan_path_to(self, target_pos, exclude=exclude) is not None

    def _all_named_targets_reachable(self):
        import nav_agent_3d
        if self.agent is None:
            return True
        for name, a in self.assets.items():
            if a is self.agent:
                continue
            exclude = {self.agent.name, name}
            if nav_agent_3d.plan_path_to(self, tuple(a.pos), exclude=exclude) is None:
                return False
        return True

    def settle(self, steps=120):
        """Real physics settling: steps the simulation. With furniture at
        mass=0 this mostly only matters for any (rare) dynamic objects —
        matches the pure-Python backend's scope, which relies on
        has_bad_overlap() as the real gate for static-furniture spacing."""
        for _ in range(steps):
            p.stepSimulation()

    def disconnect(self):
        p.disconnect(self._client)

    # ---------- export ----------
    def to_state(self):
        out = {}
        for name, a in self.assets.items():
            x, y, z = a.pos
            out[name] = dict(kind=a.kind, x=x, y=y, z=z, yaw=a.yaw,
                              half_xz=a.half_xz, half_h=a.half_height,
                              tags=list(a.tags), mesh_path=a.mesh_path, mesh_scale=a.mesh_scale,
                              mesh_up_axis=a.mesh_up_axis, mesh_up_sign=a.mesh_up_sign)
        return out