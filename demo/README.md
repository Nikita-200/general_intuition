# Demo viewers (open directly in any browser — no server, no internet needed)

- `index.html` — reach-goal: living room, walk to the mug on the coffee table
- `pick_and_place.html` — pick-and-place: kitchen, pick up the can, carry it to the trash bin

Controls: TAB toggles Auto/Manual, WASD moves in Manual mode (W forward,
A left, S back, D right), drag the mouse to look around, scroll to zoom
(Auto), R resets, SPACE pauses. Manual mode correctly picks/places when
you walk up to an object.

These two demos are primitive-shape only (no --asset-source flag). For
real-mesh (HSSD/Objaverse) retrieval fixes, see CHANGES.md Rounds 3-4 —
run `python asset_retrieval.py` for the full offline self-test suite
(46 checks) and `python inspect_hssd.py --root <your hssd-hab>` to debug
your own download's category matching.

Both files are fully self-contained (three.js is embedded directly in the
HTML) — double-click to open locally, or drop this whole folder onto any
static host (see ../DEPLOY.md) to get a shareable link.
