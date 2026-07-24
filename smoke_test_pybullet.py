"""
smoke_test_pybullet.py — run this FIRST after `pip install pybullet`,
before using the PyBullet backend for anything else.

This mirrors, check for check, the tests already run (and passed) against
the pure-Python backend during development:
  1. Ram the agent into a table 300 times — table position must not move
     by even a fraction of a unit (mass=0 rigid body).
  2. Confirm the agent (a real dynamic rigid body) DOES move under
     velocity control.
  3. 10 fresh reach-goal scenes, end to end, checking genuine task
     completion (not just "didn't crash").
  4. 10 fresh pick-and-place scenes, same.

Usage:
    pip install pybullet
    python smoke_test_pybullet.py

If PyBullet was never actually run against this code before you see this
(true as of when this was written — see dsl3d_pybullet.py's docstring),
this script is the thing that turns "should work" into "does work" on your
machine. If anything here fails, that's real signal to fix before relying
on the PyBullet backend for anything else — please don't skip this step.
"""

import math
import os
import sys

os.environ["HARNESS3D_BACKEND"] = "pybullet"

try:
    import pybullet  # noqa: F401
except ImportError:
    print("PyBullet isn't installed. Run: pip install pybullet")
    print("(No Windows wheels are published on PyPI as of this writing — pip will "
          "compile from source, which needs Visual C++ Build Tools. See README.md "
          "for details, including the WSL alternative if the source build is "
          "troublesome.)")
    sys.exit(1)

from physics_backend import Scene3D, BACKEND
import nav_agent_3d as nav

assert BACKEND == "pybullet", f"expected pybullet backend, got {BACKEND!r}"
print(f"backend confirmed: {BACKEND}\n")

failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(name)


print("=== 1-2. Ram test: furniture must be immovable, agent must move ===")
scene = Scene3D(800, 600)
agent = scene.AddAsset("agent", kind="cylinder", size=15, height=170)
scene.set_agent(agent)
table = scene.AddAsset("table", kind="box", size=40, height=42)
scene.place_at(table, 400, 300)
agent.set_pose(x=200, z=300, yaw=math.pi / 2)

before_table = list(table.pos)
before_agent = list(agent.pos)
for _ in range(300):
    scene.move_agent(300, 0, 1 / 60)
after_table = list(table.pos)
after_agent = list(agent.pos)

table_moved = math.dist(before_table, after_table)
agent_moved = math.dist(before_agent[::2], after_agent[::2])
check("table did not move under 300 collisions", table_moved < 0.5,
      f"moved {table_moved:.3f} units")
check("agent DID move under velocity control", agent_moved > 50,
      f"moved {agent_moved:.1f} units")
check("agent was physically stopped by the table (didn't clip through)",
      after_agent[0] < table.pos[0] - table.half_xz + 5,
      f"agent x={after_agent[0]:.1f}, table left edge={table.pos[0] - table.half_xz:.1f}")
scene.disconnect()

print("\n=== 3. Reach-goal: 10 fresh scenes ===")
from synthesizer3d import synthesize_scene
from tasks3d import parse_task, TaskRunner, Subgoal

reach_fails = 0
for i in range(10):
    scene, src, log = synthesize_scene(
        "a living room with a sofa, coffee table, and a mug on the table", use_mock=True)
    subgoals = [Subgoal("navigate", scene.goal.name)]
    runner = TaskRunner(subgoals)
    nav.reset_stuck_history()
    waypoints, wp_idx, last_idx = None, 0, None
    threshold = 30
    start_pos = list(scene.agent.pos)
    for t in range(800):
        if runner.is_complete:
            break
        if runner.idx != last_idx:
            target_pos = runner.current_target_pos(scene)
            waypoints = nav.plan_path_to(scene, target_pos, exclude={runner.current.target})
            wp_idx, last_idx = 0, runner.idx
            if waypoints is None:
                break
            threshold = max(30, nav.standoff_distance(waypoints, (target_pos[0], target_pos[2])) + 24,
                             nav.physical_clearance(scene, runner.current.target))
        action, wp_idx, _ = nav.controller_step(scene, waypoints, wp_idx)
        if nav.is_stuck(scene):
            waypoints = nav.plan_path_to(scene, target_pos, exclude={runner.current.target})
            wp_idx = 0
            if waypoints is None:
                break
            threshold = max(30, nav.standoff_distance(waypoints, (target_pos[0], target_pos[2])) + 24,
                             nav.physical_clearance(scene, runner.current.target))
        runner.try_advance(scene, arrival_threshold=threshold)
    if not runner.is_complete:
        reach_fails += 1
        if reach_fails == 1:  # diagnostic for the FIRST failure only, so output stays readable
            import math as _math
            end_pos = list(scene.agent.pos)
            moved = _math.dist(start_pos[::2], end_pos[::2])  # x/z only
            target_pos = runner.current_target_pos(scene)
            remaining = _math.hypot(end_pos[0] - target_pos[0], end_pos[2] - target_pos[2])
            print(f"    [diagnostic] trial {i}: agent moved {moved:.1f} units in {t} ticks "
                  f"(start x/z={start_pos[0]:.0f}/{start_pos[2]:.0f}, "
                  f"end x/z={end_pos[0]:.0f}/{end_pos[2]:.0f}); "
                  f"{remaining:.1f} units from target, threshold was {threshold:.1f}; "
                  f"waypoints found: {waypoints is not None}")
    scene.disconnect()
check("reach-goal: 10/10 fresh scenes completed", reach_fails == 0,
      f"{10 - reach_fails}/10 passed")

print("\n=== 4. Pick-and-place: 10 fresh scenes ===")
pp_fails = 0
for i in range(10):
    scene, src, log = synthesize_scene(
        "a kitchen with a can on the counter and a trash bin in the corner", use_mock=True)
    subgoals = parse_task("pick the can and place it in the trash bin", scene, use_mock=True)
    runner = TaskRunner(subgoals)
    nav.reset_stuck_history()
    waypoints, wp_idx, last_idx = None, 0, None
    threshold = 30
    for t in range(1000):
        if runner.is_complete:
            break
        if runner.idx != last_idx:
            target_pos = runner.current_target_pos(scene)
            exclude = {runner.current.target}
            if runner.current.container:
                exclude.add(runner.current.container)
            waypoints = nav.plan_path_to(scene, target_pos, exclude=exclude)
            wp_idx, last_idx = 0, runner.idx
            if waypoints is None:
                break
            threshold = max(30, nav.standoff_distance(waypoints, (target_pos[0], target_pos[2])) + 24,
                             nav.physical_clearance(scene, runner.current.target))
        action, wp_idx, _ = nav.controller_step(scene, waypoints, wp_idx)
        if scene.carried is not None:
            scene.update_carried()
        if nav.is_stuck(scene):
            waypoints = nav.plan_path_to(scene, target_pos, exclude=exclude)
            wp_idx = 0
            if waypoints is None:
                break
            threshold = max(30, nav.standoff_distance(waypoints, (target_pos[0], target_pos[2])) + 24,
                             nav.physical_clearance(scene, runner.current.target))
        runner.try_advance(scene, arrival_threshold=threshold)
    if not runner.is_complete:
        pp_fails += 1
    scene.disconnect()
check("pick-and-place: 10/10 fresh scenes completed", pp_fails == 0,
      f"{10 - pp_fails}/10 passed")

print(f"\n{'=' * 50}")
if failures:
    print(f"{len(failures)} check(s) FAILED: {failures}")
    print("Please fix these before relying on the PyBullet backend elsewhere.")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED — PyBullet backend verified on this machine.")