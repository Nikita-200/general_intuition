"""
dsl3d.py — the 3D Scene Description Language.

Deliberate engineering choice: NOT pybullet. This sandbox can't build it in
reasonable time (no prebuilt wheel for this platform), and — more
importantly — we don't actually need full rigid-body dynamics for indoor
navigation. Everything the 2D harness needed physics for was: (1) furniture
that doesn't move, (2) an agent that can't walk through furniture, (3) small
items that can rest on/in furniture without weird collision behavior. All
three are just axis-aligned box/cylinder overlap tests. This module
implements exactly that, in pure Python, with zero build risk and zero
install friction for you.

Coordinate system: x = right, z = forward/depth, y = up. The floor is y=0.
An object's `pos.y` is its CENTER height, so something resting on the floor
has pos.y == its own half-height — the same "resting" logic 2D had for x/y,
just with a real vertical axis added instead of pretending everything is
flat.

API mirrors dsl.py's Scene/Asset almost method-for-method on purpose: the
same mental model that was already debugged in 2D (static-by-default
furniture, portable=sensor-like items, place_on, pick/place/carry,
is_reachable, has_bad_overlap) carries over directly. What's NEW and
3D-specific: place_on now puts an item genuinely ON TOP of its surface
(y = surface's top + item's half-height) instead of 2D's "sensor flag so
the overlap doesn't matter" hack — the vertical axis makes that hack
unnecessary. Read that as this port doing something more honestly correct
than the 2D version, not just a translation of it.
"""

import math
import random


def _frange(start, stop, step):
    """Inclusive-ish float range (stop included if reachable) — used by
    Scene3D._scan_free_spot's grid search. Guarantees at least [start] so a
    degenerate (start >= stop) range still yields one candidate."""
    if stop <= start:
        return [start]
    n = int((stop - start) / step)
    return [start + i * step for i in range(n + 1)] + [stop]


def _scan_local_free_spot(cx, cz, half, item_half, avoid, step):
    """
    Exhaustive grid scan within [cx-half, cx+half] x [cz-half, cz+half] —
    used by place_on/place() when random sampling doesn't find a clear
    spot for one more small item on an already-cluttered surface (a coffee
    table with several items already resting on it). Same idea as
    Scene3D._scan_free_spot's room-level version, applied to a single
    surface's local footprint instead of the whole room: confirmed
    necessary, not just defensive — a coffee table with 6 small items
    already placed via random sampling alone left a real, confirmed
    overlap for the 7th (popcorn bowl vs. a goal marker, TV remote vs. a
    board game), because by that point most of the jittered area is
    already within some existing item's exclusion radius and a bounded
    number of random guesses has real odds of missing what open space is
    left.
    """
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


class Asset3D:
    def __init__(self, name, kind, pos, size, yaw=0.0, static=True,
                 portable=False, tags=None, mesh_path=None, mesh_scale=1.0,
                 mesh_up_axis=1, mesh_up_sign=1):
        self.name = name
        self.kind = kind          # "box" or "cylinder"
        self.pos = list(pos)      # [x, y, z] — mutable, unlike 2D's pymunk Vec2d
        self.size = size          # box: (hx, hy, hz) half-extents; cylinder: (radius, half_height)
        self.yaw = yaw            # radians around y axis — COSMETIC ONLY (see note below)
        self.static = static
        self.portable = portable
        self.tags = list(tags or [])
        if portable and "portable" not in self.tags:
            self.tags.append("portable")
        self.resting_on = None
        self.mesh_path = mesh_path    # optional real mesh (.glb/.obj) for the VIEWER only —
        self.mesh_scale = mesh_scale  # collision here always uses the AABB, real mesh or not
        # Which of the mesh's OWN local axes (0=X, 1=Y, 2=Z) is really "up",
        # per asset_retrieval.py's fitted_scale_and_dims — the viewer needs
        # this to ROTATE the mesh so that axis points along world Y, not
        # just scale it (mesh_scale alone corrects magnitude, not
        # direction; see fitted_scale_and_dims's docstring for why a
        # correctly-scaled-but-unrotated mesh can still render sideways,
        # flattened onto the floor). Default 1 (Y) matches "no rotation
        # needed" for the ordinary primitive-shape case.
        self.mesh_up_axis = mesh_up_axis
        # Which DIRECTION (+1/-1) along mesh_up_axis is really up — a
        # bounding-box-only guess can identify the axis but never the
        # sign (extents are always positive), so a mesh whose true "up" is
        # the negative direction still renders upside-down without this.
        # Real per-object data (HSSD's own `up` column) resolves this when
        # available; otherwise it defaults to +1.
        self.mesh_up_sign = mesh_up_sign

    # ---- footprint helpers (x/z plane — what navigation and overlap care about) ----
    @property
    def half_xz(self):
        """Half-extent in the floor plane (x/z) — the 3D analog of 2D's
        half_extent, used for pathfinding inflation and overlap checks.
        NOTE: yaw is ignored for collision (axis-aligned boxes only, same
        simplification 2D effectively made) — ok for furniture that's
        basically rectangular and not dramatically non-square; it means a
        45-degree-rotated long table would have a slightly larger effective
        footprint than strictly necessary, which only makes pathing more
        conservative, never wrong in the unsafe direction."""
        if self.kind == "cylinder":
            return self.size[0]
        return max(self.size[0], self.size[2])

    @property
    def half_height(self):
        return self.size[1] if self.kind == "box" else self.size[1]

    @property
    def top_y(self):
        """The y coordinate of this object's top surface — where something
        placed "on" it (place_on) should rest."""
        return self.pos[1] + self.half_height

    def get_aabb_xz(self):
        hx = self.half_xz
        return (self.pos[0] - hx, self.pos[0] + hx, self.pos[2] - hx, self.pos[2] + hx)

    def distance_to(self, other):
        dx, dy, dz = (self.pos[0] - other.pos[0], self.pos[1] - other.pos[1],
                      self.pos[2] - other.pos[2])
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def xz_distance_to(self, other):
        dx, dz = self.pos[0] - other.pos[0], self.pos[2] - other.pos[2]
        return math.hypot(dx, dz)

    def is_overlapping_xz(self, other, margin=0.0):
        a, b = self.get_aabb_xz(), other.get_aabb_xz()
        return not (a[1] + margin < b[0] or b[1] + margin < a[0] or
                    a[3] + margin < b[2] or b[3] + margin < a[2])

    def contains_xz(self, other):
        a = self.get_aabb_xz()
        return a[0] <= other.pos[0] <= a[1] and a[2] <= other.pos[2] <= a[3]

    def set_pose(self, x=None, z=None, y=None, yaw=None):
        if x is not None:
            self.pos[0] = x
        if z is not None:
            self.pos[2] = z
        if y is not None:
            self.pos[1] = y
        if yaw is not None:
            self.yaw = yaw
        return self


DIRECTIONS = {  # same 8-direction vocabulary as 2D, now on the x/z floor plane
    "front": (0, 1), "back": (0, -1), "left": (-1, 0), "right": (1, 0),
    "front_left": (-0.7, 0.7), "front_right": (0.7, 0.7),
    "back_left": (-0.7, -0.7), "back_right": (0.7, -0.7),
}


class Scene3D:
    """Root object exposed to LLM-generated 3D scene programs."""

    def __init__(self, width=800, depth=600, ceiling=260, wall_margin=20):
        self.width, self.depth, self.ceiling = width, depth, ceiling
        self.wall_margin = wall_margin
        self.assets = {}
        self.agent = None
        self.goal = None
        self.carried = None
        self.markers = {}
        self.retriever = None  # set by synthesize_scene(retriever=...); see asset_retrieval.py   # name -> (x, y, z)

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
        """
        Exhaustive grid-scan free-spot search — used by resolve_layout's
        fallback, where `_find_free_spot`'s random sampling isn't reliable
        enough: confirmed against a cluttered-room stress test (10-18
        objects, some large — a realistic "lived-in living room" prompt's
        actual scale) that random sampling alone left ~35% of trials still
        overlapping after every pass. A fine deterministic scan finds any
        gap that genuinely exists at this resolution, rather than hoping a
        bounded number of random guesses lands in it. `avoid` is passed in
        explicitly (rather than reading self.assets.values()) so the
        caller can exclude the object being placed itself. Scan origin and
        direction are randomized per call so multiple fallback placements
        in the same pass don't all pile into the same corner.
        """
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
        Register a new object. See dsl3d_pybullet.py's AddAsset for the
        full mesh_path/mesh_scale docs — same meaning here, except this
        backend's collision still always uses the AABB (mesh_path only
        affects what the viewer renders, not collision, in this backend).
        """
        h = height if height is not None else size * 1.4
        half_h = h / 2
        if mesh_path is None and self.retriever is not None and name != "agent":
            try:
                # Pass the ALREADY-COMPUTED size/h as hints — these are the
                # DSL author's own intended real-world-plausible dimensions
                # for this specific object (a mug's caller passes size=6,
                # a table's caller passes size=40), not a generic constant.
                # Without this, every retrieved real mesh — mug, chair, or
                # table alike — got normalized to the SAME fixed footprint,
                # which is why a real run produced four 80-unit-wide
                # "chairs" overlapping an equally oversized "table" and
                # blocking all navigation. Passing the hint through lets
                # the retriever preserve the real mesh's own proportions
                # while scaling its overall size to match what was asked
                # for, keeping every object plausibly sized relative to
                # the room instead of uniform.
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
        asset = Asset3D(name, kind, (x, half_h, z), (size, half_h, size),
                         static=not dynamic, portable=portable, tags=tags,
                         mesh_path=mesh_path, mesh_scale=mesh_scale,
                         mesh_up_axis=mesh_up_axis, mesh_up_sign=mesh_up_sign)
        uniq = name
        i = 2
        while uniq in self.assets:
            uniq = f"{name}_{i}"
            i += 1
        self.assets[uniq] = asset
        return asset

    # ---------- placement primitives (x/z plane, mirrors dsl.py) ----------
    def place_at(self, asset, x, z, yaw=0.0):
        asset.set_pose(x=x, z=z, yaw=yaw)
        return asset

    def place_relative(self, asset, anchor, direction, distance=60):
        dx, dz = DIRECTIONS[direction]
        asset.set_pose(x=anchor.pos[0] + dx * distance, z=anchor.pos[2] + dz * distance)
        return asset

    def place_around(self, assets, anchor, radius=80, start_angle=0.0):
        n = len(assets)
        for i, a in enumerate(assets):
            ang = start_angle + 2 * math.pi * i / n
            a.set_pose(x=anchor.pos[0] + radius * math.cos(ang),
                       z=anchor.pos[2] + radius * math.sin(ang),
                       yaw=ang + math.pi)  # face inward, toward the anchor
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
        """
        Place `item` genuinely ON TOP of `surface` — item.pos.y becomes
        surface.top_y + item.half_height (resting on its top face), with
        (x, z) landing somewhere within `jitter` of the surface's footprint
        center. This is the 3D-correct version of "the can is on the
        counter": no overlap, no sensor-flag workaround needed, because
        there's a real vertical axis now.

        Tries several random spots and keeps the first that doesn't
        overlap any OTHER item already resting on this same surface. If
        random sampling doesn't find one — a real, confirmed case once a
        surface already has ~5+ small items on it, since most of the
        jittered area is then within some existing item's exclusion
        radius and a bounded number of random guesses has real odds of
        missing what open space is left — falls back to an exhaustive
        local grid scan (_scan_local_free_spot) before giving up. Without
        this, a coffee table with five items place_on'd onto it one after
        another (bowls, cans, a remote, magazines, a board game — a real
        "cluttered table" prompt) picked each item's spot fully
        independently, with real odds of two small items landing on top
        of each other — confirmed as the actual cause of a user-reported
        "objects penetrating each other" on tabletop clutter.
        """
        half = surface.half_xz * jitter
        neighbors = [a for a in self.assets.values() if a.resting_on is surface and a is not item]
        xz = None
        for _ in range(tries):
            x = surface.pos[0] + random.uniform(-half, half)
            z = surface.pos[2] + random.uniform(-half, half)
            gap = min(
                (math.hypot(x - a.pos[0], z - a.pos[2]) - item.half_xz - a.half_xz
                 for a in neighbors),
                default=1e9,
            )
            if gap > 0:
                xz = (x, z)
                break
        if xz is None:
            xz = _scan_local_free_spot(surface.pos[0], surface.pos[2], half,
                                        item.half_xz, neighbors, step=max(2.0, item.half_xz))
        x, z = xz
        item.set_pose(x=x, z=z, y=surface.top_y + item.half_height)
        item.resting_on = surface
        return item

    def place_on_wall(self, asset, wall, height, offset=None, margin=4):
        """
        For decor that's conceptually MOUNTED ON a wall rather than
        resting on the floor or another surface — a TV, framed photos,
        wall art, a wall clock — none of which place_at's floor-level
        default or place_on's "resting on furniture" model actually fits.
        Without a real primitive for this, the only way to depict "TV
        mounted above the console" was to improvise — most plausibly by
        setting asset.pos[1] directly, bypassing every documented
        placement helper (this module exposes `pos` as a plain mutable
        list specifically so the DSL's own place_* methods can move
        objects, not so generated code should reach in and hand-edit
        coordinates) — which is exactly the kind of ad hoc code that
        produces objects looking like they're floating with no visible
        means of support. This sets the wall-adjacent (x, z) position
        like place_against_wall AND an explicit y (the mount height off
        the floor), and tags the object "wall_mounted" so it's excluded
        from floor-plane collision/pathing the same way portable items
        already are — a wall-mounted TV shouldn't block the agent from
        walking underneath it.
        """
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
        # Remembered so fit_room_to_content can re-snap this object flush
        # against the SAME wall after a resize, instead of uniformly
        # scaling its (x, z) like every other asset — that naive scaling
        # doesn't preserve "flush against the wall" (this formula is
        # margin + half from the edge, an ADDITIVE offset, not a pure
        # multiple of width/depth), and was confirmed to walk a wall-
        # mounted TV's position away from the wall and out of the room's
        # new, smaller bounds after a resize.
        asset._wall_mount = (wall, off, height, margin)
        return asset

    def add_container(self, name, size=22, height=35, **kw):
        kw.setdefault("tags", [])
        kw["tags"] = list(kw["tags"]) + ["container"]
        return self.AddAsset(name, kind="box", size=size, height=height,
                              dynamic=False, **kw)

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
        asset.static = False  # force dynamic, same reasoning as 2D's set_agent fix
        self.agent = asset
        return asset

    def set_goal(self, asset):
        self.goal = asset
        return asset

    # ---------- agent movement (the physics-backend boundary) ----------
    # nav_agent_3d.py (the pathfinding/steering layer) NEVER touches
    # agent.pos directly — it only ever calls move_agent(vx, vz, dt) and
    # lets the active Scene implementation decide how movement actually
    # gets realized. This is the pure-Python backend: a straightforward
    # move-and-slide against solid (non-portable) AABBs, done by hand.
    # dsl3d_pybullet.py implements the SAME method using real rigid-body
    # collision (resetBaseVelocity + stepSimulation) instead — nothing in
    # nav_agent_3d.py, tasks3d.py, synthesizer3d.py, or the viewer needs to
    # know or care which one is active.
    def move_agent(self, vx, vz, dt, margin=2.0):
        agent = self.agent
        agent.pos[0] += vx * dt
        agent.pos[2] += vz * dt
        ar = agent.half_xz
        for a in self.assets.values():
            # wall_mounted excluded for the same reason portable is: this
            # collision model is floor-projected (2.5D, x/z only, no real
            # height check), so an object mounted high on a wall (a TV,
            # framed photos) must not block floor-level movement the way
            # ordinary furniture correctly does.
            if a is agent or "portable" in a.tags or "wall_mounted" in a.tags:
                continue
            minx, maxx, minz, maxz = a.get_aabb_xz()
            minx -= ar + margin; maxx += ar + margin
            minz -= ar + margin; maxz += ar + margin
            ax, az = agent.pos[0], agent.pos[2]
            if minx < ax < maxx and minz < az < maxz:
                pen_left, pen_right = ax - minx, maxx - ax
                pen_near, pen_far = az - minz, maxz - az
                m = min(pen_left, pen_right, pen_near, pen_far)
                if m == pen_left:
                    agent.pos[0] = minx
                elif m == pen_right:
                    agent.pos[0] = maxx
                elif m == pen_near:
                    agent.pos[2] = minz
                else:
                    agent.pos[2] = maxz
        m = ar + margin
        agent.pos[0] = max(m, min(self.width - m, agent.pos[0]))
        agent.pos[2] = max(m, min(self.depth - m, agent.pos[2]))

    # ---------- manipulation ----------
    def pick(self, item):
        if self.carried is not None:
            raise RuntimeError(f"already carrying {self.carried.name}")
        item.tags = [t for t in item.tags if t != "placed"] + ["carried"]
        self.carried = item
        return item

    def update_carried(self, forward_offset=26, height=110):
        """Glue the carried item in front of the agent at roughly chest
        height, each tick."""
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
            neighbors = [a for a in self.assets.values()
                         if a.resting_on is container and a is not item]
            xz = None
            for _ in range(25):
                x = container.pos[0] + random.uniform(-half, half)
                z = container.pos[2] + random.uniform(-half, half)
                gap = min(
                    (math.hypot(x - a.pos[0], z - a.pos[2]) - item.half_xz - a.half_xz
                     for a in neighbors),
                    default=1e9,
                )
                if gap > 0:
                    xz = (x, z)
                    break
            if xz is None:
                xz = _scan_local_free_spot(container.pos[0], container.pos[2], half,
                                            item.half_xz, neighbors, step=max(2.0, item.half_xz))
            x, z = xz
            item.set_pose(x=x, z=z, y=container.top_y + item.half_height)
            item.resting_on = container
            item.tags += ["placed", f"placed_in:{container.name}"]
        self.carried = None
        return item

    # ---------- validity checks (ground truth, feed the synthesis retry loop) ----------
    def has_bad_overlap(self, margin=-3):
        """Same purpose as 2D: furniture is static, so overlaps don't
        resolve themselves — this is a real authoring error to catch."""
        solids = [a for a in self.assets.values()
                  if a is not self.agent and "portable" not in a.tags
                  and "wall_mounted" not in a.tags]
        for i, a in enumerate(solids):
            for b in solids[i + 1:]:
                if a.is_overlapping_xz(b, margin=margin):
                    return True
        return False

    def out_of_bounds_assets(self):
        """
        Nothing previously checked that a placed object's footprint
        actually stays inside the room at all — has_bad_overlap() only
        catches objects overlapping EACH OTHER, is_reachable() only checks
        paths. An LLM-authored build_scene (e.g. a place_relative() call
        with too large a distance from an anchor near a wall) could place
        something entirely outside [0, width] x [0, depth] and nothing in
        the retry loop would ever notice — confirmed against a real run: a
        floor lamp rendered standing outside the room's floor plane
        entirely. Returns a list of asset names whose footprint AABB
        extends past the room's own walls (using wall_margin as the same
        boundary nav_agent_3d/the placement helpers already respect).
        """
        bad = []
        for name, a in self.assets.items():
            if a is self.agent:
                continue
            x0, x1, z0, z1 = a.get_aabb_xz()
            if x0 < 0 or z0 < 0 or x1 > self.width or z1 > self.depth:
                bad.append(name)
        return bad

    def _resize_room(self, new_width, new_depth):
        """Pure-Python backend: the room boundary is just two numbers
        (move_agent/_find_free_spot/etc. all read self.width/self.depth
        directly, there's no separate physical wall object to rebuild —
        contrast dsl3d_pybullet.py's version, which has to actually tear
        down and recreate real wall bodies)."""
        self.width = new_width
        self.depth = new_depth

    def fit_room_to_content(self, target_occupancy=0.12, min_width=400, min_depth=300,
                             max_width=1300, max_depth=1000):
        """
        Right-size the room to the furniture that actually got BUILT, run
        once right after build_scene() returns (before resolve_layout,
        which then guarantees no overlap/out-of-bounds at whatever size
        this settles on). synthesizer3d._estimate_room_size already picks
        an initial width/depth before the synthesis LLM even runs, so its
        own place_against_wall/place_corner calls have a reasonable canvas
        — but it can only guess from the PROMPT'S TEXT, not from how many
        of the mentioned items the LLM actually decided to instantiate as
        real floor-occupying furniture. Confirmed against a real run: an
        elaborate "home office" prompt densely mentioning desk clutter
        (laptop, mouse, pens, cables, sticky notes...) inflated the text
        guess to a big room, but almost all of those became small portable
        items place_on'd onto the desk, not separate floor furniture — the
        LLM's actual build_scene only had a handful of solid pieces (desk,
        chair, a cabinet, a lamp, a plant), leaving most of a needlessly
        large floor empty.

        Computes the total footprint AREA of solid (non-portable, non-
        wall_mounted) furniture, solves for the room area that puts that at
        `target_occupancy` of the total floor (a plain area ratio, not a
        real packing solver — picked to read as "furnished, with real
        walking room," not empty or wall-to-wall), keeps the room's
        existing width:depth aspect ratio, and clamps to sane min/max
        bounds so one tiny or one huge object can't produce a degenerate
        room either direction.

        EVERY non-agent asset's (x, z) — solid furniture, portables resting
        on a surface, AND portables that aren't (a backpack on the floor,
        placed via place_relative rather than place_on) — is then remapped
        by the same per-axis scale factor the room itself just changed by.
        Scaling only the solids and re-gluing surface items by an unscaled
        delta was tried first and had a real bug: any portable NOT resting
        on something (nothing marks it as needing to move at all) kept its
        old absolute coordinates, which land outside the room's new walls
        the moment the room shrinks — confirmed as an actual `OUT OF
        BOUNDS` crash on a backpack placed next to a desk. Scaling
        everything uniformly has no such blind spot.
        """
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
        kx, kz = new_width / old_width, new_depth / old_depth
        for a in self.assets.values():
            if a is self.agent:
                continue
            wall_mount = getattr(a, "_wall_mount", None)
            if wall_mount is not None:
                # Re-snap flush against the SAME wall at the new room size
                # rather than uniformly scaling — see place_on_wall's
                # comment for why naive scaling doesn't preserve "flush
                # against the wall" for this additive placement formula.
                wall, off, height, margin = wall_mount
                self.place_on_wall(a, wall, height, offset=off, margin=margin)
            else:
                a.set_pose(x=a.pos[0] * kx, z=a.pos[2] * kz)

    def resolve_layout(self, iterations=300, gap=4.0):
        """
        Deterministic geometry repair, run after build_scene finishes and
        before the ground-truth checks below. An LLM-authored program
        routinely gets raw coordinates wrong in two specific, purely
        numeric ways that better prompting alone doesn't reliably prevent:
        (1) a place_relative/place_around/place_at call whose distance
        from an anchor already near a wall pushes the new object's
        footprint past [0, width] x [0, depth] (OUT OF BOUNDS), and (2)
        two independently-placed pieces of static furniture whose
        footprints simply happen to intersect (FURNITURE OVERLAP). Both
        are geometry problems with a geometry solution: clamp every solid
        object's footprint fully inside the room, then iteratively
        separate any pair still overlapping along whichever axis needs
        the smaller push, re-clamping after each pass. This does NOT
        replace has_bad_overlap()/out_of_bounds_assets() below — an
        extreme case (more/bigger furniture than the room can physically
        fit) can still legitimately fail them and fall through to the
        synthesis retry loop — it just means those checks stop firing on
        the common case of an LLM's numbers being slightly off, instead of
        spending an expensive retry on something a few lines of geometry
        can fix outright.

        Portable items placed via place_on/place() carry a `resting_on`
        reference to their surface. When a surface moves during
        separation, every item resting on it is re-glued by the SAME
        delta, so "the can on the counter" stays on the counter's new
        position instead of floating over empty floor.
        """
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

        # Last-resort fallback: a handful of large objects all anchored
        # near the same corner (place_around/place_relative chains sharing
        # one wall-adjacent anchor), or just a genuinely cluttered room (10+
        # objects, some large — confirmed against a realistic "lived-in
        # living room" stress test at this scale) can mutually overlap in a
        # way local pairwise nudging doesn't fully untangle within a
        # bounded number of passes. Rather than raise the iteration cap
        # indefinitely chasing a pairwise-nudge algorithm that isn't
        # guaranteed to converge, anything still overlapping after the
        # bounded pass gets moved with _scan_free_spot — an exhaustive grid
        # scan, not random sampling (random sampling alone was tried first
        # and confirmed insufficient at this scale: it left ~35% of a
        # 10-18-object stress test still overlapping). Largest-first order
        # is a standard bin-packing heuristic: placing big objects while
        # the most open space is still available, and letting small ones
        # fill the remaining gaps, converges far more often than an
        # arbitrary order.
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

        # Portable items sharing one surface can still end up overlapping
        # EACH OTHER even with place_on's own overlap-avoidance: that's a
        # greedy, one-item-at-a-time placement, and confirmed (a coffee
        # table with 6+ small items — bowls, cans, a remote, a board game)
        # to still leave one residual overlap even after place_on's own
        # random-then-grid-scan search, since a later item can only avoid
        # what's already there, not re-arrange it. This pass looks at every
        # surface's FINAL set of resting items jointly and pushes apart any
        # pair that still overlaps — the same pairwise nudge already used
        # for solid furniture above, just scoped to one surface's clutter
        # instead of the whole room (no room-bounds clamping needed here;
        # items only need to stay clear of each other, not of a wall).
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
                # Pick-and-place scenes (FEWSHOT_PICK_PLACE's own convention)
                # never call set_goal — this used to just `return True` here,
                # which meant the generation-time retry loop validated NOTHING
                # for these scenes: an unreachable "can" or "trash_bin" could
                # pass straight through and only fail later, mid-episode (a
                # real, confirmed case: "no path to subgoal target 'can'"
                # after 420 ticks, on a scene that passed this check). Since
                # the actual task targets aren't known yet at this point in
                # synthesize_scene (subgoals are built afterward), validate

                # reachability to every solid, named object instead — any of
                # them is a plausible task target, and this is the only
                # ground-truth check available before the episode runs.
                return self._all_named_targets_reachable()
            target_pos = self.goal.pos
        exclude = set(exclude_names) | ({self.agent.name} if self.agent else set())
        if target_pos is None:
            return True
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

    # ---------- export (engine-agnostic; this is what the viewer consumes) ----------
    def to_state(self):
        out = {}
        for name, a in self.assets.items():
            out[name] = dict(kind=a.kind, x=a.pos[0], y=a.pos[1], z=a.pos[2],
                              yaw=a.yaw, half_xz=a.half_xz, half_h=a.half_height,
                              tags=list(a.tags), mesh_path=a.mesh_path, mesh_scale=a.mesh_scale,
                              mesh_up_axis=a.mesh_up_axis, mesh_up_sign=a.mesh_up_sign)
        return out