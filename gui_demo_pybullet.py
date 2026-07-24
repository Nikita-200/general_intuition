"""
gui_demo_pybullet.py — watch the agent navigate in PyBullet's OWN window,
with real-time rigid-body physics, not the browser viewer's re-implemented
JS engine.

    python gui_demo_pybullet.py "a kitchen with a can on the counter and a trash bin in the corner" \
        --task "pick the can and place it in the trash bin" --mock

Requires PyBullet (see README). Opens a real OpenGL window: left-drag to
orbit, scroll to zoom, Ctrl+drag to pan (PyBullet's built-in camera
controls). This is the most direct answer to "isn't it supposed to be a
physics engine" — this window IS the physics engine's own renderer,
running the same rigid-body simulation `smoke_test_pybullet.py` validates,
live, in real time, not a re-implementation.
"""

import argparse
import time

import pybullet as p

from physics_backend import Scene3D, BACKEND
from tasks3d import parse_task, TaskRunner, Subgoal
import nav_agent_3d as nav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--task", default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--model", default="gpt-4o")
    args = ap.parse_args()

    if BACKEND != "pybullet":
        raise SystemExit(
            "PyBullet isn't installed/importable — this demo needs the real "
            "backend. `pip install pybullet` (see README), then re-run."
        )

    scene = Scene3D(800, 600, gui=True)  # gui=True: real PyBullet window, not headless
    p.resetDebugVisualizerCamera(cameraDistance=500, cameraYaw=45, cameraPitch=-35,
                                  cameraTargetPosition=[scene.width / 2, 40, scene.depth / 2])

    import synthesizer3d
    llm = synthesizer3d.get_llm(args.mock)
    system_prompt = (synthesizer3d.DSL_DOCS + "\n# Example\n" + synthesizer3d.FEWSHOT
                      + "\n# Example\n" + synthesizer3d.FEWSHOT_PICK_PLACE)
    user_prompt = f'Prompt: "{args.prompt}"\n\nWrite build_scene(scene) for this.'
    program_src = synthesizer3d._extract_code(llm(system_prompt, user_prompt, model=args.model))
    namespace = {"scene": scene}
    exec(compile(program_src, "<scene_program_3d>", "exec"), namespace)
    namespace["build_scene"](scene)
    scene.agent.set_pose(*scene._find_free_spot(scene.agent.half_xz))

    subgoals = (parse_task(args.task, scene, use_mock=args.mock, model=args.model)
                if args.task else [Subgoal("navigate", scene.goal.name)])
    runner = TaskRunner(subgoals)
    nav.reset_stuck_history()

    print(f"Task: {args.task or '(navigate to goal)'}")
    print("Real PyBullet window open — drag to orbit, scroll to zoom.")

    waypoints, wp_idx, last_idx, threshold = None, 0, None, 30
    while True:
        if not runner.is_complete:
            if runner.idx != last_idx:
                target_pos = runner.current_target_pos(scene)
                exclude = {runner.current.target}
                if runner.current.container:
                    exclude.add(runner.current.container)
                waypoints = nav.plan_path_to(scene, target_pos, exclude=exclude)
                wp_idx, last_idx = 0, runner.idx
                if waypoints:
                    threshold = max(30, nav.standoff_distance(waypoints, (target_pos[0], target_pos[2])) + 24,
                                     nav.physical_clearance(scene, runner.current.target))
            if waypoints:
                action, wp_idx, _ = nav.controller_step(scene, waypoints, wp_idx)
                if scene.carried is not None:
                    scene.update_carried()
                if nav.is_stuck(scene):
                    new_wp = nav.plan_path_to(scene, target_pos, exclude=exclude)
                    if new_wp is not None:
                        waypoints, wp_idx = new_wp, 0
                runner.try_advance(scene, arrival_threshold=threshold)
        else:
            print("Task complete. Window stays open — close it to exit.")
        time.sleep(1 / 60)


if __name__ == "__main__":
    main()