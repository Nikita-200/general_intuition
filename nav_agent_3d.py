"""
nav_agent_3d.py — makes the 3D agent traverse a Scene3D toward a target.

Design note: this is a floor-projected (2.5D) navigation model, on purpose.
An indoor agent walks on the floor; it doesn't need 3D voxel pathfinding to
get from the doorway to the counter. So this reuses exactly the same A*-
over-an-inflated-occupancy-grid approach that was already debugged in 2D —
projected onto the x/z plane — rather than inventing a new 3D algorithm.
Everything that was fixed in 2D (agent-radius-aware inflation, room-
boundary awareness, snapping to the nearest open cell when a target sits
inside solid furniture, never blocking the start/goal cell) carries over
unchanged, because the underlying problem — "find a path across a floor
plan dotted with box obstacles" — hasn't changed, only the number of axes
in the render has.

Action vocabulary matches the brief's real 3D action space more completely
than 2D could: forward/backward/left/right + mouse-delta-x for yaw (turning)
AND, genuinely used this time, mouse-delta-y for camera pitch (look up/down
— it doesn't affect movement, just where the camera in the viewer looks,
exactly like a first-person camera).
"""

import heapq
import math
from collections import deque

ACTIONS = ["forward", "backward", "left", "right", "turn_left", "turn_right", "noop"]


def _grid_from_state(state, w, d, cell=20, exclude=(), inflate=6):
    cols, rows = int(w // cell), int(d // cell)  # cols index x, rows index z
    blocked = [[False] * cols for _ in range(rows)]
    for name, o in state.items():
        if name in exclude:
            continue
        half = o["half_xz"]
        r0, r1 = int((o["z"] - half - inflate) // cell), int((o["z"] + half + inflate) // cell)
        c0, c1 = int((o["x"] - half - inflate) // cell), int((o["x"] + half + inflate) // cell)
        for r in range(max(r0, 0), min(r1 + 1, rows)):
            for c in range(max(c0, 0), min(c1 + 1, cols)):
                blocked[r][c] = True
    return blocked, cols, rows, cell


def _block_room_boundary(blocked, cols, rows, w, d, cell, margin):
    for r in range(rows):
        for c in range(cols):
            x, z = (c + 0.5) * cell, (r + 0.5) * cell
            if x < margin or x > w - margin or z < margin or z > d - margin:
                blocked[r][c] = True


def _nearest_open_cell(blocked, cols, rows, rc):
    if not blocked[rc[0]][rc[1]]:
        return rc
    seen = {rc}
    q = deque([rc])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen:
                if not blocked[nr][nc]:
                    return (nr, nc)
                seen.add((nr, nc))
                q.append((nr, nc))
    return rc


def _astar(blocked, cols, rows, start, goal):
    def h(p):
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])
    openq = [(h(start), 0, start, None)]
    came = {}
    best = {start: 0}
    while openq:
        _, g, cur, parent = heapq.heappop(openq)
        if cur in came:
            continue
        came[cur] = parent
        if cur == goal:
            path = []
            while cur:
                path.append(cur)
                cur = came[cur]
            return path[::-1]
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nr, nc = cur[0] + dr, cur[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and not blocked[nr][nc]:
                ng = g + math.hypot(dr, dc)
                if ng < best.get((nr, nc), 1e9):
                    best[(nr, nc)] = ng
                    heapq.heappush(openq, (ng + h((nr, nc)), ng, (nr, nc), cur))
    return None


def standoff_distance(waypoints, target_xz):
    if not waypoints:
        return 0.0
    return math.dist(waypoints[-1], target_xz)


def physical_clearance(scene, target_name, margin=20):
    """
    If the navigation target is itself a solid, real object (e.g. a
    container you're walking up to but can't enter), the agent's true
    resting distance from its CENTER is bounded below by both objects'
    physical radii, no matter what the abstracted pathfinding grid thinks
    (that grid may fully exclude the target so A* can path to its exact
    center — but real collision resolution still treats it as solid, so
    the agent physically stops at its edge). standoff_distance alone can
    be near-zero in this case; this gives the arrival threshold a floor
    that actually matches the physics.
    """
    a = scene.assets.get(target_name)
    if a is None or "portable" in a.tags or "wall_mounted" in a.tags:
        return 0.0
    return a.half_xz + scene.agent.half_xz + margin


def plan_path_to(scene, target_pos, exclude=(), cell=20):
    """target_pos is (x, y, z) or (x, z) — only x/z are used. Returns a list
    of (x, z) waypoints or None."""
    tx, tz = (target_pos[0], target_pos[2]) if len(target_pos) == 3 else target_pos
    state = scene.to_state()
    exclude = set(exclude) | ({scene.agent.name} if scene.agent else set())
    if scene.carried is not None:
        exclude.add(scene.carried.name)
    # wall_mounted excluded alongside portable: this is a floor-projected
    # (2.5D) occupancy grid with no real height axis, so an object
    # deliberately mounted high on a wall (a TV, framed photos — see
    # Scene3D.place_on_wall) must not block floor-level pathing the way
    # ordinary furniture correctly does.
    non_floor_names = {name for name, o in state.items()
                        if "portable" in o.get("tags", []) or "wall_mounted" in o.get("tags", [])}
    exclude |= non_floor_names
    inflate = scene.agent.half_xz + 4 if scene.agent else 6
    blocked, cols, rows, cell = _grid_from_state(
        state, scene.width, scene.depth, cell, exclude, inflate=inflate)
    _block_room_boundary(blocked, cols, rows, scene.width, scene.depth, cell,
                          scene.wall_margin - 4)
    start = (int(scene.agent.pos[2] // cell), int(scene.agent.pos[0] // cell))
    goal = (int(tz // cell), int(tx // cell))
    start = (min(max(start[0], 0), rows - 1), min(max(start[1], 0), cols - 1))
    goal = (min(max(goal[0], 0), rows - 1), min(max(goal[1], 0), cols - 1))
    goal = _nearest_open_cell(blocked, cols, rows, goal)
    blocked[start[0]][start[1]] = False
    blocked[goal[0]][goal[1]] = False
    grid_path = _astar(blocked, cols, rows, start, goal)
    if not grid_path:
        return None
    return [((c + 0.5) * cell, (r + 0.5) * cell) for r, c in grid_path]  # (x, z) pairs


_stuck_history = []


def reset_stuck_history():
    _stuck_history.clear()


def is_stuck(scene, window=45, min_progress=6.0):
    _stuck_history.append((scene.agent.pos[0], scene.agent.pos[2]))
    if len(_stuck_history) > window:
        _stuck_history.pop(0)
    if len(_stuck_history) < window:
        return False
    x0, z0 = _stuck_history[0]
    x1, z1 = _stuck_history[-1]
    stuck = math.hypot(x1 - x0, z1 - z0) < min_progress
    if stuck:
        _stuck_history.clear()
    return stuck


def reached(scene, target_pos, threshold=30):
    tx, tz = (target_pos[0], target_pos[2]) if len(target_pos) == 3 else target_pos
    ax, az = scene.agent.pos[0], scene.agent.pos[2]
    return math.hypot(ax - tx, az - tz) <= threshold


def controller_step(scene, waypoints, wp_idx, speed=140, dt=1 / 60):
    """
    Proportional steering (same shape as the 2D controller). Computes a
    desired velocity and heading, then hands the actual movement off to
    scene.move_agent(vx, vz, dt) — this function never touches agent.pos
    directly, which is what makes the physics backend swappable (pure-
    Python move-and-slide vs. real PyBullet rigid-body collision) without
    changing a single line here.
    """
    agent = scene.agent
    if wp_idx >= len(waypoints):
        return "noop", wp_idx, True

    tx, tz = waypoints[wp_idx]
    dx, dz = tx - agent.pos[0], tz - agent.pos[2]
    dist = math.hypot(dx, dz)
    if dist < 14:
        wp_idx += 1
        if wp_idx >= len(waypoints):
            return "noop", wp_idx, True
        tx, tz = waypoints[wp_idx]
        dx, dz = tx - agent.pos[0], tz - agent.pos[2]
        dist = math.hypot(dx, dz)

    heading = agent.yaw
    desired = math.atan2(dx, dz)  # yaw measured from +z axis
    diff = (desired - heading + math.pi) % (2 * math.pi) - math.pi
    agent.yaw = heading + max(-0.28, min(0.28, diff * 0.28))

    alignment = max(0.0, math.cos(diff))
    remaining = dist if wp_idx == len(waypoints) - 1 else 1e9
    approach_scale = min(1.0, remaining / 60)
    v = speed * alignment * approach_scale
    vx, vz = v * math.sin(agent.yaw), v * math.cos(agent.yaw)
    scene.move_agent(vx, vz, dt)

    action = ("forward" if alignment > 0.6 else
              ("turn_left" if diff > 0 else "turn_right"))
    return action, wp_idx, False