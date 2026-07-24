"""
webapp/server.py — local Flask web app for the 3D harness.

    python webapp/server.py

Serves a form page (prompt, task, control mode, run mode) at `/`, and two
generation paths:
  - "Scene": a thin wrapper around the exact same synthesize_scene ->
    _build_subgoals -> render_viewer_html pipeline harness3d.py's CLI
    already uses (see harness3d.run_episode), returning the resulting
    self-contained HTML directly. No new rendering code needed for this
    path at all.
  - "Infinite": a continuously-extending, boundary-less corridor (see
    infinite_world.py) — one real LLM call derives a theme from the
    prompt, then the frontend streams new deterministic chunks live as the
    agent approaches the edge of what's already loaded.

Both paths use ONE shared HSSDRetriever built at startup — its
`__init__` scans the whole local HSSD download (~16,500 objects) to build
a category index, confirmed to take a couple of seconds; rebuilding that
per HTTP request would make every generate call sluggish for no reason.
Uses the real LLM by default when OPENAI_API_KEY is set (matching the CLI's
own auto-detect convention in synthesizer3d.get_llm), --mock to force the
deterministic/no-network path regardless (useful for trying the whole app
with no API key).
"""

import argparse
import os
import sys
import tempfile
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request, send_from_directory

import infinite_world as iw
from asset_retrieval import HSSDRetriever, TieredRetriever
from harness3d import _build_subgoals
from synthesizer3d import synthesize_scene
from viewer3d import render_infinite_shell_html, render_viewer_html

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = Flask(__name__)
app.config["RETRIEVER"] = None
app.config["USE_MOCK"] = True
app.config["HSSD_ROOT"] = None
# In-memory theme store: theme_id -> theme dict. Fine for a single-user
# local app — no database needed, and nothing here needs to survive a
# server restart (a restarted server just can't resolve old themeIds
# anymore, handled explicitly below rather than crashing).
_themes = {}


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/scene", methods=["POST"])
def api_scene():
    body = request.get_json(force=True) or {}
    prompt = (body.get("prompt") or "").strip()
    task = (body.get("task") or "").strip() or None
    control = body.get("control") if body.get("control") in ("auto", "manual") else "auto"
    if not prompt:
        return jsonify(error="prompt is required"), 400

    use_mock = app.config["USE_MOCK"]
    try:
        scene, program_src, log = synthesize_scene(
            prompt, use_mock=use_mock, retriever=app.config["RETRIEVER"],
            hssd_root=app.config["HSSD_ROOT"])
        subgoals = _build_subgoals(scene, task, use_mock, "gpt-4o")
    except Exception as e:
        return jsonify(error=str(e)), 500

    fd, tmp_path = tempfile.mkstemp(suffix=".html")
    os.close(fd)
    try:
        render_viewer_html(scene, subgoals, [], dict(prompt=prompt, task=task),
                            tmp_path, initial_control=control)
        with open(tmp_path, encoding="utf-8") as f:
            html = f.read()
    finally:
        os.remove(tmp_path)
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/api/infinite/start", methods=["POST"])
def api_infinite_start():
    body = request.get_json(force=True) or {}
    prompt = (body.get("prompt") or "").strip()
    control = body.get("control") if body.get("control") in ("auto", "manual") else "auto"
    if not prompt:
        return jsonify(error="prompt is required"), 400

    theme = iw.derive_theme(prompt, app.config["USE_MOCK"], app.config["HSSD_ROOT"])
    theme_id = uuid.uuid4().hex
    _themes[theme_id] = theme
    try:
        scene0, category0 = iw.build_chunk(theme, 0, retriever=app.config["RETRIEVER"])
        # Spawn point computed against chunk 0's ACTUAL furniture (see
        # infinite_world.find_spawn_point) — not a fixed formula, which can
        # land inside a large piece of furniture placed against the front
        # wall and shove the agent backward on the very first tick.
        spawn_x, spawn_z = iw.find_spawn_point(scene0)
        chunk0 = iw.chunk_to_json(scene0, 0, scene0.depth, category=category0, draw_front_wall=True)
        chunk1 = iw.generate_chunk_json(theme, 1, retriever=app.config["RETRIEVER"])
    except Exception as e:
        return jsonify(error=str(e)), 500

    # The agent belongs to the persistent page, not any one chunk (chunks
    # are pure furniture content — see infinite_world.py).
    agent = dict(x=spawn_x, y=85.0, z=spawn_z, yaw=0.0, half_xz=15, half_h=85.0)
    return jsonify(themeId=theme_id, style=theme["style"], control=control,
                   chunkWidth=chunk0["width"], chunkDepth=chunk0["depth"],
                   wallMargin=chunk0["wallMargin"], agent=agent,
                   chunks=[chunk0, chunk1])


@app.route("/api/infinite/next_chunk", methods=["POST"])
def api_infinite_next_chunk():
    body = request.get_json(force=True) or {}
    theme_id = body.get("themeId")
    index = body.get("index")
    theme = _themes.get(theme_id)
    if theme is None:
        return jsonify(error="unknown themeId (server may have restarted — reload to start a new run)"), 404
    if not isinstance(index, int) or index < 0:
        return jsonify(error="index must be a non-negative integer"), 400
    try:
        chunk = iw.generate_chunk_json(theme, index, retriever=app.config["RETRIEVER"])
    except Exception as e:
        return jsonify(error=str(e)), 500
    return jsonify(chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hssd-root", default=os.path.join(ROOT_DIR, "hssd-hab"),
                     help="path to a local HSSD download (see download_hssd.py); "
                          "the backend always uses HSSD as its asset source.")
    ap.add_argument("--mock", action="store_true",
                     help="force the deterministic/no-network synthesizer even if "
                          "OPENAI_API_KEY is set (useful for trying the app with no key).")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    # Regenerated on every startup, not committed/hand-maintained — it has
    # no per-request data embedded (see render_infinite_shell_html's own
    # docstring), so there's no reason for it to ever be stale relative to
    # viewer3d.py's current JS_STREAMING_DRIVER. Confirmed as a real
    # gap during development: hand-running the generator once, then fixing
    # a bug in the driver source afterward, left the actually-served static
    # file silently out of date with no error of any kind.
    render_infinite_shell_html(os.path.join(STATIC_DIR, "infinite.html"))
    app.config["USE_MOCK"] = args.mock or "OPENAI_API_KEY" not in os.environ
    app.config["HSSD_ROOT"] = args.hssd_root
    print(f"[webapp] building the shared HSSD retriever from {args.hssd_root!r} "
          f"(once, at startup)...")
    app.config["RETRIEVER"] = TieredRetriever([HSSDRetriever(hssd_root=args.hssd_root)])
    print(f"[webapp] mode: {'MOCK (deterministic, no API key needed)' if app.config['USE_MOCK'] else 'real LLM'}")
    print(f"[webapp] open http://127.0.0.1:{args.port}/ in a browser")
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
