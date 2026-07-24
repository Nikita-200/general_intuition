"""
physics_backend.py — chooses the physics backend once, at import time.

Import Scene3D from HERE everywhere else in the harness (synthesizer3d.py,
harness3d.py), never directly from dsl3d.py or dsl3d_pybullet.py. That's
what makes "use real PyBullet when available, fall back otherwise"
transparent to the rest of the code.

    from physics_backend import Scene3D, BACKEND

PyBullet is preferred whenever it's importable, since it's real rigid-body
physics rather than the pure-Python move-and-slide stand-in. See the
README for exactly why PyBullet may or may not "just work" via pip on your
platform, and how to force one backend or the other.
"""

import os

_FORCE = os.environ.get("HARNESS3D_BACKEND", "").lower()  # "pybullet" | "pure" | ""

BACKEND = None
Scene3D = None

if _FORCE != "pure":
    try:
        import pybullet  # noqa: F401
        from dsl3d_pybullet import Scene3D
        BACKEND = "pybullet"
    except ImportError:
        pass

if Scene3D is None:
    if _FORCE == "pybullet":
        raise RuntimeError(
            "HARNESS3D_BACKEND=pybullet was set but `import pybullet` failed. "
            "Install it first: pip install pybullet (see README for platform notes)."
        )
    from dsl3d import Scene3D
    BACKEND = "pure"
    print("[physics_backend] PyBullet not found — using the pure-Python physics-lite "
          "engine. `pip install pybullet` to switch to real rigid-body physics "
          "(see README for platform-specific notes), or set HARNESS3D_BACKEND=pure "
          "to silence this message if the fallback is intentional.")