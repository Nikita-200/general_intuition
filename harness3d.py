"""
harness3d.py — the 3D agent harness CLI.

    python harness3d.py "a living room with a sofa, coffee table, and a mug on the table" --mock
    python harness3d.py "a kitchen with a can on the counter and a trash bin in the corner" \
        --task "pick the can and place it in the trash bin" --mock

Each run:
  1. Synthesizes a Scene3D from the text prompt (Programmer+Debugger loop —
     furniture-overlap and reachability are ground-truth gates inside it).
  2. Builds subgoals (navigate/pick/place) from --task, or an implicit
     single "navigate to scene.goal" subgoal if --task is omitted — same
     TaskRunner drives both, matching the 2D harness's unified design.
  3. Runs the episode in Python (A* + move-and-slide collision), recording
     a full per-tick trajectory (agent position/yaw, carried-item state).
  4. Writes out/scene_N.json (raw state) and out/viewer_N.html — a SELF-
     CONTAINED interactive three.js page with the scene and the recorded
     trajectory embedded directly (no server, no fetch, just open the
     file). The viewer has its own JS ports of the same A* planner and
     move-and-slide collision used here, so besides replaying the recorded
     run, it can also re-plan and drive the agent live in the browser
     (Auto mode) or take WASD+mouse control yourself (Manual mode) —
     genuinely interactive, not just an animation.
"""

import argparse
import json
import os

from synthesizer3d import synthesize_scene, next_prompt_text
import nav_agent_3d as nav
from tasks3d import parse_task, TaskRunner, Subgoal
from viewer3d import render_viewer_html


def _build_subgoals(scene, task_text, use_mock, model):
    if task_text:
        return parse_task(task_text, scene, use_mock=use_mock, model=model)
    if scene.goal is not None:
        return [Subgoal("navigate", scene.goal.name)]
    # No explicit --task AND the synthesized scene never called
    # scene.set_goal(...) — confirmed to happen for real (non-mock) LLM
    # runs given a pure scene-description prompt with no single obvious
    # "reach it" object (a whole-room description like "a kitchen after
    # breakfast with cabinets, an island, a sink..." has no one clear
    # target). This used to unconditionally call
    # parse_task("", scene, use_mock=True), which is guaranteed to raise
    # ValueError("couldn't parse task: ''") — _rule_based_parse has no rule
    # that matches an empty string and no scene.goal to fall back on,
    # crashing the whole run instead of just... navigating somewhere.
    # Auto-pick an implicit goal instead: prefer a portable item (most
    # "reach it"-like, matching what a real goal object usually is), else
    # any other non-agent asset, so the harness always has SOME destination
    # rather than crashing on a prompt that just wasn't goal-shaped.
    fallback = next((a for a in scene.assets.values()
                      if a is not scene.agent and "portable" in a.tags), None)
    fallback = fallback or next((a for a in scene.assets.values() if a is not scene.agent), None)
    if fallback is not None:
        return [Subgoal("navigate", fallback.name)]
    raise RuntimeError(
        "synthesized scene has no assets besides the agent — nothing to navigate to")


def _build_retriever(use_real_assets, asset_source, hssd_root):
    """
    asset_source: None (no real assets), "objaverse", "hssd", or "both"
      (HSSD tried first — it's curated for indoor objects and ships better
      collision geometry — falling through to Objaverse, then a primitive
      shape, on any miss). --use-real-assets is kept as a backward-
      compatible alias for asset_source="objaverse".
    hssd_root: required when asset_source is "hssd" or "both" — the
      directory download_hssd.py wrote to.
    """
    if asset_source is None and use_real_assets:
        asset_source = "objaverse"
    if asset_source is None:
        return None

    from asset_retrieval import ObjaverseRetriever, HSSDRetriever, TieredRetriever
    sources = []
    if asset_source in ("hssd", "both"):
        if not hssd_root:
            raise SystemExit("--asset-source hssd (or both) requires --hssd-root "
                              "/path/to/downloaded/hssd-hab (see download_hssd.py)")
        sources.append(HSSDRetriever(hssd_root=hssd_root))
    if asset_source in ("objaverse", "both"):
        sources.append(ObjaverseRetriever())
    if not sources:
        raise SystemExit(f"unknown --asset-source {asset_source!r} "
                          f"(expected: objaverse, hssd, or both)")
    return TieredRetriever(sources)


def run_episode(prompt, task_text, out_dir, idx, use_mock, model="gpt-4o",
                 max_ticks=1400, use_real_assets=False, asset_source=None, hssd_root=None):
    os.makedirs(out_dir, exist_ok=True)
    retriever = _build_retriever(use_real_assets, asset_source, hssd_root)
    scene, program_src, log = synthesize_scene(prompt, use_mock=use_mock, model=model,
                                                 retriever=retriever, hssd_root=hssd_root)
    subgoals = _build_subgoals(scene, task_text, use_mock, model)

    # Export the viewer from this PRE-simulation state — fresh subgoals, the
    # agent at its true starting position. The browser runs its own live
    # simulation from here; it does NOT replay the Python run below. That
    # Python run exists only to produce a reference success/ticks number for
    # the CLI and results.json, and must not be allowed to mutate the scene
    # before the viewer captures it (it did, in an earlier version of this
    # function — the viewer would open already showing a completed task).
    viewer_path = os.path.join(out_dir, f"viewer_{idx}.html")
    render_viewer_html(scene, subgoals, [], dict(prompt=prompt, task=task_text), viewer_path)

    runner = TaskRunner(subgoals)
    nav.reset_stuck_history()

    trajectory = []
    waypoints, wp_idx, last_idx = None, 0, None
    arrival_threshold, target_pos, exclude = 30, None, set()
    t, no_path = 0, False

    for t in range(max_ticks):
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
                no_path = True
                break
            arrival_threshold = max(
                30,
                nav.standoff_distance(waypoints, (target_pos[0], target_pos[2])) + 24,
                nav.physical_clearance(scene, runner.current.target))
        action, wp_idx, _ = nav.controller_step(scene, waypoints, wp_idx)
        if scene.carried is not None:
            scene.update_carried()
        if nav.is_stuck(scene):
            waypoints = nav.plan_path_to(scene, target_pos, exclude=exclude)
            wp_idx = 0
            if waypoints is None:
                no_path = True
                break
            arrival_threshold = max(
                30,
                nav.standoff_distance(waypoints, (target_pos[0], target_pos[2])) + 24,
                nav.physical_clearance(scene, runner.current.target))
        runner.try_advance(scene, arrival_threshold=arrival_threshold)
        trajectory.append(dict(
            x=scene.agent.pos[0], y=scene.agent.pos[1], z=scene.agent.pos[2],
            yaw=scene.agent.yaw, carried=scene.carried.name if scene.carried else None,
            subgoal_idx=runner.idx,
        ))

    result = dict(prompt=prompt, task=task_text, success=runner.is_complete,
                  ticks=t + 1, subgoals=runner.describe(), synth_attempts=len(log))
    if no_path:
        result["reason"] = f"no path to subgoal target {runner.current.target!r}"

    state_path = os.path.join(out_dir, f"scene_{idx}.json")
    with open(state_path, "w") as f:
        json.dump(dict(scene=scene.to_state(),
                        markers={k: list(v) for k, v in scene.markers.items()},
                        agent=scene.agent.name, goal=scene.goal.name if scene.goal else None,
                        width=scene.width, depth=scene.depth,
                        subgoals=[dict(kind=s.kind, target=s.target, container=s.container)
                                  for s in subgoals],
                        trajectory=trajectory, result=result), f)

    # encoding="utf-8" explicit — program_src is LLM-authored text and can
    # contain non-ASCII characters (e.g. an em-dash in a generated
    # comment); without it, Python writes using the platform's default
    # locale encoding, which is cp1252 on Windows and can silently corrupt
    # bytes that later fail to re-read as UTF-8 (see viewer3d.py's
    # render_viewer_html for the confirmed version of this exact bug).
    with open(os.path.join(out_dir, f"level_{idx}.py"), "w", encoding="utf-8") as f:
        f.write(program_src)
    return result, scene, viewer_path


def _prune_old_levels(out_dir, current_idx, keep_last):
    """
    Delete the generated files for exactly the one level that just fell
    outside the most-recent-`keep_last` window. An --infinite run
    generates one scene_N.json + level_N.py + viewer_N.html per level
    forever; viewer_N.html alone embeds three.js and (with --asset-source)
    real mesh data, easily several MB each, so a truly unbounded run with
    no cleanup would grow without limit until disk fills — the literal
    opposite of "infinite" being a usable feature rather than a footgun.
    Only ever touches files this same naming scheme produced, at an index
    this run already passed, so it can't reach into unrelated output.
    """
    cutoff = current_idx - keep_last
    if cutoff < 0:
        return
    for fname in (f"scene_{cutoff}.json", f"level_{cutoff}.py", f"viewer_{cutoff}.html"):
        path = os.path.join(out_dir, fname)
        if os.path.isfile(path):
            os.remove(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--task", default=None,
                     help='optional, e.g. "pick the can and place it in the trash bin"')
    ap.add_argument("--levels", type=int, default=1,
                     help="how many environments to generate and run, in sequence "
                          "(each one a genuinely different room/task than the last — "
                          "see next_prompt_text). Ignored if --infinite is set.")
    ap.add_argument("--infinite", action="store_true",
                     help="keep generating and running new environments forever, "
                          "until you stop it with Ctrl+C — see README's 'Infinite "
                          "generation' section. Overrides --levels.")
    ap.add_argument("--keep-last", type=int, default=None,
                     help="delete generated files for all levels older than the most "
                          "recent N (disk safety for long/--infinite runs, which would "
                          "otherwise accumulate one multi-MB viewer_N.html per level "
                          "forever). Defaults to 30 when --infinite is set, and to "
                          "'keep everything' for a normal bounded --levels run.")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--out", default="out3d")
    ap.add_argument("--use-real-assets", action="store_true",
                     help="alias for --asset-source objaverse (kept for backward "
                          "compatibility). Needs internet access and "
                          "`pip install objaverse trimesh`.")
    ap.add_argument("--asset-source", choices=["objaverse", "hssd", "both"], default=None,
                     help="try real meshes for named objects, falling back to "
                          "primitives on any miss/failure. 'hssd' needs --hssd-root "
                          "(see download_hssd.py); 'both' tries hssd first, then "
                          "objaverse.")
    ap.add_argument("--hssd-root", default=None,
                     help="path to a local HSSD download (see download_hssd.py); "
                          "required when --asset-source is hssd or both.")
    args = ap.parse_args()

    keep_last = args.keep_last
    if keep_last is None and args.infinite:
        keep_last = 30

    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "results.json")

    prompt = args.prompt
    results = []
    i = 0
    try:
        while args.infinite or i < args.levels:
            # --task is only meaningful for the level it was actually
            # written against (its wording names specific objects, e.g.
            # "the can" — those literal words). Every level after the
            # first is a freshly auto-generated, genuinely different
            # environment (see next_prompt_text), which may not even have
            # a "can" in it — confirmed as a real crash: level 3 of a
            # --levels run generated a room with a "key" on the counter,
            # but the CLI's fixed --task string still said "pick the can",
            # and _rule_based_parse correctly failed to find any asset
            # named after that word. Reusing the literal task text past
            # level 0 was never sound; only apply it there and let later
            # levels fall back to their own implicit goal.
            task_text = args.task if i == 0 else None
            task_label = task_text or "(navigate to scene.goal)"
            print(f"[level {i}] env: {prompt!r}  task: {task_label!r}")
            try:
                result, scene, viewer_path = run_episode(
                    prompt, task_text, args.out, i, args.mock, model=args.model,
                    use_real_assets=args.use_real_assets, asset_source=args.asset_source,
                    hssd_root=args.hssd_root)
                print(f"[level {i}] result: {result}")
                print(f"[level {i}] viewer: {viewer_path}  (open directly in a browser)")
            except Exception as e:
                # A single bad synthesis/episode (an LLM hiccup, an
                # exhausted retry loop, a network blip fetching a real
                # mesh) must not end an otherwise long-running or infinite
                # generation loop — that would make "infinite" a lie the
                # first time anything went wrong. Record it as a failed
                # level, same shape as any other unsuccessful result, and
                # move on to the next one.
                print(f"[level {i}] FAILED: {e}")
                result = dict(prompt=prompt, task=task_text, success=False, error=str(e))
            results.append(result)
            # Flushed every level, not just at the end — an --infinite run
            # has no "the end" until you stop it, so results so far need
            # to be real output at any point in time, not just on a clean exit.
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            if keep_last is not None:
                _prune_old_levels(args.out, i, keep_last)
            i += 1
            if args.infinite or i < args.levels:
                prompt = next_prompt_text(prompt, result.get("success", False), args.mock,
                                           model=args.model, level_idx=i)
    except KeyboardInterrupt:
        print(f"\nStopped after {len(results)} level(s) (Ctrl+C).")

    print(f"\nWrote {len(results)} level(s) to {args.out}/ (see results.json)")


if __name__ == "__main__":
    main()