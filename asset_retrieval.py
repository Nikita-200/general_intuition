"""
asset_retrieval.py — real object retrieval, replacing PrimitiveRetriever.

Implements synthesizer3d.AssetRetriever against Objaverse. This is REAL,
runnable code, not a stub, verified as far as this sandbox's network
allowlist permits: `objaverse` and `trimesh` install and import cleanly,
and the LVIS-category lookup / mesh download calls are exactly the
documented objaverse API. What's confirmed NOT reachable from here:
objaverse's annotation index and mesh files are both hosted on
huggingface.co (verified directly by reading load_lvis_annotations()'s and
_load_object_paths()'s source, both hit
https://huggingface.co/datasets/allenai/objaverse/...), which is outside
this sandbox's egress allowlist. On your own machine, with normal internet
access, this should just work — see the smoke test at the bottom of this
file.

Bounding-box computation (trimesh) IS fully verified here — it's pure
offline geometry processing, no network needed.
"""

import base64
import csv
import hashlib
import json
import os
import re
import struct

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".asset_cache")


class AssetRetrieverBase:
    """Matches synthesizer3d.AssetRetriever's contract.

    size_hint/height_hint: the caller's (AddAsset's) own intended
    real-world dimensions for this specific object — pass these through
    when overriding retrieve(), and use them as the target size to scale
    a real mesh to, rather than a single fixed constant for every object
    regardless of type (see ObjaverseRetriever.retrieve's docstring for
    why that was a real, previously-shipped bug)."""
    def retrieve(self, semantic_name, size_hint=None, height_hint=None):
        raise NotImplementedError


class CategoryMatcher:
    """
    Shared name -> category matching logic, used by both ObjaverseRetriever
    (against LVIS categories) and HSSDRetriever (against whatever category
    taxonomy your HSSD download's metadata uses). Pulled out into one place
    so both retrievers run through the identical, single-tested matching
    algorithm instead of two copies that could quietly drift apart.

    Matching order: (1) a hand-curated synonym map for this harness's own
    object vocabulary, (2) exact match, (3) substring match preferring the
    shortest hit, (4) Jaccard-ratio token overlap with a minimum-similarity
    floor. See test_category_matching() at the bottom of this file for the
    offline self-test (no network needed) that exercises all four tiers.
    """

    # Best-effort synonym map from this harness's own object vocabulary
    # (see synthesizer3d.py's DSL_DOCS/FEWSHOT and the task-name grounding in
    # tasks3d.py) to common dataset category names. Real taxonomies
    # (LVIS, HSSD's own) are typically singular, underscore-joined nouns
    # (e.g. "trash_can", "coffee_table"), which often differs from the more
    # casual names an LLM naturally writes in a scene program ("bin",
    # "counter", "island"). Straight substring/token matching against the
    # raw category list handles a lot of cases already, but silently fails
    # on ones where the words genuinely differ ("trash_bin" contains no
    # substring in common with "trash_can"). This map is checked FIRST,
    # before falling back to the generic substring/token matcher below, so
    # it fixes exactly those cases.
    #
    # NOTE: this was written from general knowledge of common indoor-object
    # taxonomy naming conventions, not by inspecting either live category
    # list (LVIS is fetched from huggingface.co at runtime; HSSD's own
    # metadata schema needs your own downloaded copy to inspect — see the
    # module docstrings for why neither is reachable from every
    # environment). If a mapped category below turns out not to exist in
    # your actual taxonomy, the generic matcher underneath still runs as a
    # fallback — nothing breaks, it just falls through to a primitive shape
    # for that one object. Treat this map as a starting point to
    # verify/extend once you can inspect the real category list, not as
    # ground truth.
    SYNONYMS = {
        "trash_bin": ["trashcan", "trash_can", "ashcan", "garbage"],
        "trash": ["trashcan", "trash_can", "ashcan", "garbage"],
        "bin": ["trashcan", "trash_can", "ashcan", "garbage", "basket"],
        "garbage_bin": ["trashcan", "trash_can", "ashcan"],
        "waste_basket": ["trashcan", "trash_can", "ashcan", "basket"],
        "counter": ["countertop", "counter"],
        "kitchen_counter": ["countertop"],
        "island": ["table", "countertop"],  # no direct equivalent for "kitchen island"
        "coffee_table": ["coffee_table", "table"],
        "dining_table": ["table", "dining_table"],
        "desk": ["desk", "table"],
        "sofa": ["sofa", "couch"],
        "couch": ["sofa", "couch"],
        "bookshelf": ["bookcase", "shelf"],
        "book_shelf": ["bookcase", "shelf"],
        "shelf": ["shelf", "bookcase"],
        "dresser": ["dresser", "chest_of_drawers_(furniture)"],
        "nightstand": ["nightstand", "table"],
        "bed": ["bed"],
        "chair": ["chair"],
        "mug": ["mug", "cup"],
        "can": ["can", "tin_can", "beer_can", "soda_can"],
        "soda_can": ["soda_can", "can", "tin_can"],
        "cup": ["cup", "mug"],
        "book": ["book"],
        "ball": ["ball"],
        "key": ["key"],
        "remote": ["remote_control"],
        "apple": ["apple"],
        "fridge": ["refrigerator", "icebox", "fridge"],
        "refrigerator": ["refrigerator", "icebox", "fridge"],
        "stove": ["stove", "cooktop", "range", "oven"],
        "oven": ["oven", "stove", "range"],
        "sink": ["sink", "washbasin", "basin"],
        "dishwasher": ["dishwasher"],
        "washer": ["washer", "washing_machine", "washing_machine_(appliance)"],
        "washing_machine": ["washing_machine", "washer"],
        "cabinet": ["cabinet", "cupboard"],
        "wardrobe": ["wardrobe", "closet", "armoire"],
        "closet": ["closet", "wardrobe", "armoire"],
        "tv_stand": ["tv_stand", "television_stand", "entertainment_center", "stand"],
        "lamp": ["lamp", "floor_lamp", "table_lamp", "lamppost"],
        "floor_lamp": ["floor_lamp", "lamp"],
        "plant": ["plant", "houseplant", "pot_plant", "flowerpot"],
        "stool": ["stool", "footstool"],
        "ottoman": ["ottoman", "footstool", "hassock"],
        "armchair": ["armchair", "chair"],
        "loveseat": ["loveseat", "sofa", "couch"],
        "sideboard": ["sideboard", "buffet", "credenza"],
        "side_table": ["end_table", "table"],
        "end_table": ["end_table", "table"],
        "fruit_bowl": ["bowl", "fruit_bowl"],
        "bowl": ["bowl"],
        "dish": ["plate", "dish"],
        "plate": ["plate", "dish"],
        "mirror": ["mirror"],
    }

    @staticmethod
    def normalize(semantic_name):
        """Strips naming artifacts this harness itself introduces before a
        name ever reaches a retriever: synthesizer3d.py's few-shot goal
        objects are literally named "goal_mug"/"goal_can" (see FEWSHOT),
        AddAsset auto-deduplicates repeated names as "chair_2", "chair_3",
        etc., and infinite_world.py's build_chunk stacks a SECOND numeric
        suffix on top of that ("bookshelf_0_1" - chunk index, then
        position-in-chunk index). Neither "goal_" nor a trailing index
        (however many are stacked) is semantically part of the object's
        category, but left in, they actively hurt matching. Confirmed as a
        real bug: a single `_\\d+$` strip only removes the LAST suffix,
        leaving e.g. "bookshelf_0" for "bookshelf_0_1" — the residual
        "_0" doesn't stop a name that's already a literal category
        ("couch_0" still contains category token "couch") but silently
        breaks any name that needs the SYNONYM map's exact key lookup
        ("bookshelf_0" is not a key in SYNONYMS; only "bookshelf" is),
        which is exactly what turned bookshelf-type chunk furniture into
        plain boxes. `(?:_\\d+)+$` strips ALL stacked trailing numeric
        suffixes in one pass instead of just the innermost one."""
        name = semantic_name.lower()
        if name.startswith("goal_"):
            name = name[len("goal_"):]
        name = re.sub(r"(?:_\d+)+$", "", name)  # "chair_2" -> "chair"; "bookshelf_0_1" -> "bookshelf"
        return name.replace("_", " ").strip()

    @staticmethod
    def _strip_wordnet_suffix(s):
        """Category taxonomies sometimes use WordNet synset keys as their
        category identifier (confirmed from a real HSSD download: e.g.
        'candlestick.n.01', 'showerhead.n.01' in fpmodels-with-decomposed.csv's
        wnsynsetkey column) — the trailing '.n.01'/'.v.03'/etc. is a
        part-of-speech + sense-number suffix, not part of the word itself,
        and left in, it doesn't break matching but the underlying word
        needs to be isolated cleanly for the token-based comparisons below."""
        return re.sub(r"\.[nvasr]\.\d+$", "", s)

    @classmethod
    def _tokens(cls, s):
        s = cls._strip_wordnet_suffix(s).lower().replace("_", " ").replace(".", " ")
        return set(s.split())

    @classmethod
    def _singular_candidates(cls, s):
        """
        Best-effort English singularization of the LAST word in `s`
        (words joined by "_"), yielding alternate strings also worth
        matching against. Confirmed as a real, non-obvious matching gap:
        a real HSSD download's own `main_category` data uses the literal
        plural "shelves", which shares NO token with the taxonomy's own
        actual category "shelf" — every tier below (synonym map, exact
        match, token containment) failed on it, since these are irregular
        plurals, not just a trailing "s" (shelf/shelves, knife/knives,
        wolf/wolves), silently sending real, retrievable objects to a
        primitive-shape fallback for no reason other than singular vs.
        plural. Deliberately narrow — not a general lemmatizer, just the
        -ves -> -f/-fe pattern plus a plain trailing -es/-s strip, enough
        for the object nouns this harness's own vocabulary actually uses.
        """
        parts = s.split("_")
        last = parts[-1]
        candidates = []
        if last.endswith("ves") and len(last) > 3:
            candidates.append(last[:-3] + "f")
            candidates.append(last[:-3] + "fe")
        if last.endswith("es") and len(last) > 2:
            candidates.append(last[:-2])
        if last.endswith("s") and len(last) > 1 and not last.endswith("ss"):
            candidates.append(last[:-1])
        return ["_".join(parts[:-1] + [c]) for c in candidates]

    @classmethod
    def match(cls, semantic_name, categories):
        """categories: an iterable of real category strings from whatever
        taxonomy is in use. Returns the best-matching category string (in
        its ORIGINAL casing/form, not lowercased) or None if nothing
        scores well enough to be worth confidently returning — a wrong
        confident match is worse than a graceful fallback to a primitive
        shape."""
        categories = list(categories)
        lower_map = {c.lower(): c for c in categories}
        stripped_map = {cls._strip_wordnet_suffix(c).lower(): c for c in categories}
        name = cls.normalize(semantic_name)
        name_key = name.replace(" ", "_")
        # Tried in order (original first, so an exact plural-form category
        # match — if one genuinely exists — is still preferred over a
        # singularized guess) by every tier through exact match.
        name_key_variants = [name_key] + cls._singular_candidates(name_key)

        # 1. Synonym map.
        for variant in name_key_variants:
            for key in (variant, variant.replace("_", "")):
                for candidate in cls.SYNONYMS.get(key, ()):
                    if candidate.lower() in lower_map:
                        return lower_map[candidate.lower()]
                    if candidate.lower() in stripped_map:
                        return stripped_map[candidate.lower()]

        # 2. Exact match (case-insensitive, underscore/space-insensitive,
        # and WordNet-suffix-insensitive: "can" should exact-match a
        # category literally called "can.n.04" just as readily as one
        # literally called "can").
        for variant in name_key_variants:
            if variant in lower_map:
                return lower_map[variant]
            if variant in stripped_map:
                return stripped_map[variant]

        # 3. Token-SET containment (not raw substring!). A category string
        # is split into whole tokens (underscores/dots/spaces all count as
        # separators, and any WordNet suffix is stripped first), and a
        # match requires the query's token set to be a subset of the
        # category's tokens or vice versa. This is deliberately NOT the
        # same as checking `name in category_string`: a raw substring
        # check on real data let a 3-letter query like "can" match inside
        # unrelated words like "candlestick" or "cancel" purely by
        # character overlap, with no respect for word boundaries — found
        # from an actual run against real WordNet-style category strings.
        # Token containment can't do that: {"can"} is not a subset of
        # {"candlestick"}, so it's correctly rejected, while {"can"} IS a
        # (trivial) subset of {"can"} or {"tin", "can"}, so genuine matches
        # still work. Prefers the containing category with the FEWEST
        # extra tokens (closest match, least likely to be an unrelated
        # compound category). Also tries each singularized variant's own
        # token set (see _singular_candidates) for the same irregular-
        # plural reason as tiers 1-2 above.
        name_token_sets = [set(v.replace("_", " ").split()) for v in name_key_variants]
        best_hit, best_extra = None, None
        for name_tokens in name_token_sets:
            for c in categories:
                c_tokens = cls._tokens(c)
                if not c_tokens or not name_tokens:
                    continue
                if name_tokens <= c_tokens or c_tokens <= name_tokens:
                    extra = len(name_tokens ^ c_tokens)  # symmetric difference size
                    if best_extra is None or extra < best_extra:
                        best_hit, best_extra = c, extra
        if best_hit is not None:
            return best_hit

        # 4. Token-overlap fallback, scored as a RATIO of shared tokens over
        # the union of both sides' tokens (Jaccard-style), not a raw
        # intersection count — see the class docstring for why a raw count
        # is misleading. Require at least some meaningful overlap (>0.2).
        # Uses the ORIGINAL (non-singularized) token set — this tier's own
        # fuzzy overlap scoring already tolerates minor word variation.
        name_tokens = name_token_sets[0]
        best, best_ratio = None, 0.0
        for c in categories:
            c_tokens = cls._tokens(c)
            union = name_tokens | c_tokens
            if not union:
                continue
            ratio = len(name_tokens & c_tokens) / len(union)
            if ratio > best_ratio:
                best, best_ratio = c, ratio
        return best if best_ratio > 0.2 else None


def axis_and_sign_from_vector(up_vec):
    """
    Ground-truth version of "which axis, which direction" — given a real
    per-object up-vector (e.g. HSSD's fpmodels-with-decomposed.csv `up`
    column, like "0,-1,0"), picks the axis with the largest-magnitude
    component and returns its (axis_index, sign). This is the thing
    bounding-box aspect ratio genuinely cannot tell you: extents are always
    positive lengths, so detect_up_axis below can identify WHICH axis is
    tallest but has no way to know whether the mesh's real "up" points
    along +axis or -axis — a mesh authored with its top surface at
    negative local coordinates renders upside-down under a sign-blind
    correction. Use this whenever a real up-vector is available; fall back
    to detect_up_axis (which implicitly assumes sign=+1) otherwise.
    """
    axis = max(range(3), key=lambda i: abs(up_vec[i]))
    sign = 1 if up_vec[axis] >= 0 else -1
    return axis, sign


def detect_up_axis(mesh, size_hint, height_hint):
    """
    Picks which of the mesh's own 3 local axes (0=X, 1=Y, 2=Z) is its real
    "up" — WITHOUT assuming axis 1 (Y), which is what silently produced a
    real, confirmed case: a 2.2m-wide kitchen counter mesh whose true
    vertical axis got read as horizontal, reporting a "height" of 4cm.
    Real asset libraries (HSSD's own fpmodels-with-decomposed.csv ships
    explicit per-object `up`/`front` columns for exactly this reason)
    frequently ship meshes that aren't Y-up, and this harness's
    `.converted.obj` step doesn't appear to normalize that.

    This is a FALLBACK for when no real per-object up-vector is available
    (see axis_and_sign_from_vector, which HSSDRetriever now prefers when
    it has one) — it tries each axis as the candidate "up" and picks
    whichever produces an aspect ratio (height / footprint) closest to
    what the CALLER already expects (size_hint/height_hint — the DSL's own
    chart entry for this category, e.g. a counter's 45/90). Ties favor
    axis 1, so already-correct Y-up meshes are unaffected. Note this can
    only determine WHICH axis, never which DIRECTION along it — extents
    are always positive, so the sign this implies is always +1, which
    means an object whose real "up" is the negative direction of the
    correct axis can still render upside-down when only this heuristic is
    available. That's a real, known limitation of guessing from box shape
    alone rather than reading ground truth.

    Returns (up_axis, raw_footprint, raw_height) using THAT axis.
    """
    extents = mesh.extents
    if len(extents) != 3:
        return 1, max(extents), max(extents)

    target_aspect = height_hint / (2 * size_hint) if size_hint else 1.0
    best = None
    for axis in (1, 0, 2):  # Y-up checked first: wins any exact tie
        other = [extents[i] for i in range(3) if i != axis]
        footprint = max(other)
        height = extents[axis]
        if footprint <= 0:
            continue
        score = abs(height / footprint - target_aspect)
        if best is None or score < best[0]:
            best = (score, axis, footprint, height)
    if best is None:
        return 1, max(extents), max(extents)
    return best[1], best[2], best[3]


def fitted_scale_and_dims(mesh, size_hint, height_hint, known_up=None):
    """
    The actual fix for FOUR compounding proportion/orientation bugs found
    against real user runs (see CHANGES.md Round 5): (1) the wrong-axis-
    is-"up" bug, (2) a units bug where `target_size` — a HALF-extent
    everywhere else in this codebase — was matched directly against
    `mesh.extents`, a FULL extent, silently rendering every real-mesh
    object at exactly half its intended footprint (confirmed against a
    real run's can: reported mesh_scale 41.35 vs. the correct 82.71),
    (3) no floor/ceiling at all on how far a real mesh's OWN proportions
    could differ from the chart before an object became effectively
    invisible or comically oversized next to the agent, and (4) even once
    the right axis and magnitude are found, nothing rotated the mesh so
    that axis actually points up in the renderer (confirmed: a correctly-
    scaled counter still rendered lying flat) — and a bounding-box-only
    guess can get the axis's correct DIRECTION wrong too (confirmed: real
    objects rendering upside-down / sideways after the rotation fix,
    because aspect ratio alone can't distinguish +axis from -axis).

    Rather than patch each one, this forces the real mesh to exactly fill
    the DSL's own (size_hint, height_hint) box — the same box a primitive
    shape would get — using PER-AXIS (non-uniform) scale factors, so the
    final on-screen size is always exactly what the size chart specifies,
    no matter what the raw mesh's native proportions or axis convention
    turn out to be. The real mesh contributes shape/detail only, never
    footprint or height — those always come from the one source that's
    actually known to be correct and consistent with the agent's own
    170cm reference scale.

    known_up: an optional real per-object up-vector (e.g. HSSD's own `up`
    column) — when given, this is trusted directly for BOTH axis and
    SIGN instead of guessing from bounding-box aspect ratio, which fixes
    the upside-down/sideways cases the heuristic alone cannot.

    Returns (size, height, mesh_scale, up_axis, up_sign). mesh_scale is a
    3-tuple (sx, sy, sz) to apply as the loaded mesh's non-uniform scale.
    up_axis (0/1/2) is which of the mesh's OWN local axes the scale's
    "sy"-like factor was applied to, and up_sign (+1/-1) is which
    direction along it is really up — the caller must rotate the mesh so
    that SIGNED axis points along the renderer's actual "up" direction
    (world Y for viewer3d.py/three.js, world Z for dsl3d_pybullet.py's
    native PyBullet frame); scale alone only fixes magnitude, not
    direction, which is why a correctly-scaled mesh could still render
    lying flat, sideways, or upside-down depending on which of these two
    corrections was still missing.
    """
    if known_up is not None:
        up_axis, up_sign = axis_and_sign_from_vector(known_up)
        extents = mesh.extents
        other = [extents[i] for i in range(3) if i != up_axis]
        raw_footprint, raw_height = max(other), extents[up_axis]
    else:
        up_axis, raw_footprint, raw_height = detect_up_axis(mesh, size_hint, height_hint)
        up_sign = 1
    if raw_footprint <= 0 or raw_height <= 0:
        raise ValueError("degenerate mesh bounding box (zero-width or zero-height)")

    scale_xz = (2 * size_hint) / raw_footprint   # full target width / full raw width
    scale_y = height_hint / raw_height            # full target height / full raw height

    # A real, confirmed case (a "counter" retrieval that landed on a mesh
    # shaped nothing like a counter): scale_xz=41, scale_y=2246 — a >50x
    # disparity between axes. Forcing ANY raw mesh into the exact target
    # box (this function's whole design, see the docstring above) is fine
    # for ordinary variation between a real mesh's proportions and the
    # chart's, but a disparity this extreme means the CATEGORY match
    # itself was almost certainly wrong (a flat mat/tray/lid stretched
    # 2246x taller than its own native shape doesn't read as "a counter" —
    # it reads as a warped spike), not just "this mesh happens to be a bit
    # stubbier or leaner than average". Rejecting here — same ValueError
    # path already used for a degenerate bbox — falls through to a
    # correctly-proportioned primitive shape instead, which is strictly
    # more honest than keeping a real mesh whose own geometry has been
    # distorted past recognition.
    ratio = max(scale_xz, scale_y) / min(scale_xz, scale_y)
    if ratio > 15:
        raise ValueError(
            f"mesh proportions too different from the target size chart to fit "
            f"without grotesque distortion (axis scale ratio {ratio:.1f}x from "
            f"scale_xz={scale_xz:.2f}, scale_y={scale_y:.2f}) — likely a bad "
            f"category match rather than normal mesh variation")

    mesh_scale = [scale_xz, scale_xz, scale_xz]
    mesh_scale[up_axis] = scale_y
    return size_hint, height_hint, tuple(mesh_scale), up_axis, up_sign


class ObjaverseRetriever(AssetRetrieverBase):
    """
    Retrieves a real mesh from Objaverse by fuzzy-matching `semantic_name`
    against LVIS category names, downloading one candidate object, and
    computing its real-world footprint from the mesh's own bounding box.

    Usage:
        retriever = ObjaverseRetriever()
        mesh_path, size, height, scale, up_axis = retriever.retrieve("mug")
        scene.AddAsset("mug", kind="box", size=size, height=height,
                        portable=True, mesh_path=mesh_path, mesh_scale=scale,
                        mesh_up_axis=up_axis)

    First call downloads and caches the LVIS annotation index (a few MB).
    Each new category downloads one .glb (typically a few hundred KB to a
    few MB) into .asset_cache/, cached by uid so repeat runs don't
    re-download.
    """

    def __init__(self, target_size_hint=40.0, cache_dir=CACHE_DIR):
        self.target_size_hint = target_size_hint  # our harness's typical "size" unit for a mid-sized object
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._lvis = None
        self._uid_cache = {}  # semantic_name -> uid, so re-retrieval is stable within a run

    def _lvis_annotations(self):
        if self._lvis is None:
            import objaverse
            self._lvis = objaverse.load_lvis_annotations()
        return self._lvis

    def _match_category(self, semantic_name):
        return CategoryMatcher.match(semantic_name, self._lvis_annotations().keys())

    def retrieve(self, semantic_name, size_hint=None, height_hint=None):
        """Returns (mesh_path, size, height, mesh_scale, up_axis) or None if
        nothing reasonable was found (caller should fall back to
        PrimitiveRetriever). up_axis (0/1/2) is which of the mesh's OWN
        local axes is really "up" — the caller MUST also rotate the mesh
        to point that axis along the renderer's actual up direction, not
        just apply mesh_scale, or a correctly-scaled mesh can still render
        sideways/flattened (see fitted_scale_and_dims's docstring).

        size_hint: the CALLER's own intended real-world footprint for this
        specific object (AddAsset's own `size` argument, before retrieval
        overwrites it) — used as the target footprint to scale the real
        mesh to, so a mug and a table don't end up the same size just
        because they both came from Objaverse. Falls back to
        self.target_size_hint (one fixed constant) only when the caller
        genuinely didn't provide one, e.g. a direct `.retrieve("mug")`
        call outside AddAsset."""
        import objaverse
        import trimesh

        target_size = size_hint if size_hint is not None else self.target_size_hint

        if semantic_name in self._uid_cache:
            uid = self._uid_cache[semantic_name]
        else:
            category = self._match_category(semantic_name)
            if category is None:
                return None
            uids = self._lvis_annotations()[category]
            if not uids:
                return None
            idx = int(hashlib.md5(semantic_name.encode()).hexdigest(), 16) % len(uids)
            uid = uids[idx]
            self._uid_cache[semantic_name] = uid

        objects = objaverse.load_objects(uids=[uid], download_processes=1)
        mesh_path = objects[uid]

        mesh = trimesh.load(mesh_path, force="mesh")
        target_height = height_hint if height_hint is not None else target_size * 1.4
        try:
            size, height, scale, up_axis, up_sign = fitted_scale_and_dims(
                mesh, target_size, target_height)
        except ValueError:
            return None
        return mesh_path, size, height, scale, up_axis, up_sign


class HSSDRetriever(AssetRetrieverBase):
    """
    Retrieves a real mesh from a locally-downloaded HSSD (Habitat
    Synthetic Scenes Dataset) object library — curated, indoor-specific,
    consistent real-world scale, and (per HSSD's own docs) many objects
    ship with precomputed convex collision decompositions, meaning better
    dynamics-ready collision shapes than Objaverse's general firehose.

    Setup (full walkthrough in GUIDE_3D_ASSETS.md):
        1. Accept the dataset's terms at
           https://huggingface.co/datasets/hssd/hssd-hab (free, near-instant).
        2. `pip install huggingface_hub` and `huggingface-cli login`.
        3. `python download_hssd.py --out /path/to/hssd-hab`
        4. retriever = HSSDRetriever(hssd_root="/path/to/hssd-hab")

    What this expects on disk: `<hssd_root>/objects/**/*.object_config.json`
    — the standard Habitat-Sim per-object config format, each referencing
    its own render mesh via a "render_asset" field (a path relative to the
    config's own directory). This part is confirmed by Habitat-Sim's own
    publicly documented object-config convention, not by fetching your
    specific download (this sandbox has no route to huggingface.co either
    — same limitation as Objaverse, see asset_retrieval.py's module
    docstring).

    Category discovery (i.e. "which config(s) correspond to 'mug'?") now
    targets HSSD's REAL, confirmed layout:
        <hssd_root>/metadata/*.csv, *.json
        <hssd_root>/semantics/hssd-hab_semantic_lexicon.json
    (thank you for posting the actual tree — the first version of this
    guessed at root-level filenames like "metadata.csv" that don't exist;
    this version scans the real `metadata/` directory and specifically
    parses the semantic lexicon file). In order:
      1. `category_map` passed explicitly to the constructor — skips
         auto-discovery entirely; still the most reliable option if
         everything below still doesn't line up with your exact download.
      2. `semantics/hssd-hab_semantic_lexicon.json`, parsed with a
         deliberately flexible reader (see `_parse_semantic_lexicon`) that
         tries several plausible schemas, since I don't have network
         access to fetch and inspect the real file directly — if none of
         them match, it prints the file's actual top-level shape so you
         can tell me (or fix it yourself in one place) exactly what's
         there instead of guessing blind.
      3. Every `*.csv`/`*.json` under `metadata/`, matching columns/keys
         that look like an id+category pair.
      4. Each object_config.json's own embedded category-like field
         ("category", "class", "semantic_id", etc.), if any exist.
    Directory-name fallback (guessing a category from a config's parent
    folder name) is OFF by default now: your own listing confirms HSSD's
    `objects/*/*.glb` folders are hash buckets (the standard Habitat-Sim
    on-disk convention), not category names — the first version of this
    file didn't know that yet and was quietly building 233 meaningless
    "categories" from them. Pass `use_directory_fallback=True` only if you
    discover your own download actually organizes folders by category
    (verify before relying on it — see `inspect_hssd.py`).

    Whichever tier actually resolves a given object, its real size/height
    is always computed directly from ITS OWN mesh's bounding box via
    trimesh (never trusted from metadata) — exactly like ObjaverseRetriever
    — so a weak/missing category tier only affects whether a match is
    FOUND, never whether a found match's dimensions are correct.
    """

    def __init__(self, hssd_root, category_map=None, target_size_hint=40.0,
                 use_directory_fallback=False):
        self.hssd_root = hssd_root
        self.target_size_hint = target_size_hint
        self.use_directory_fallback = use_directory_fallback
        self._uid_cache = {}
        self._uid_to_config = None
        if not os.path.isdir(hssd_root):
            print(f"[HSSDRetriever] WARNING: {hssd_root!r} doesn't exist or isn't a "
                  f"directory — every retrieve() call will return None (falls back "
                  f"to a primitive shape). Run download_hssd.py first.")
            self._category_map = {}
            self._up_vec_map = {}
        elif category_map is not None:
            self._category_map = category_map
            self._up_vec_map = self._build_up_vector_index()
            print(f"[HSSDRetriever] using explicit category_map "
                  f"({len(category_map)} categories)")
        else:
            self._category_map = self._build_category_index()
            self._up_vec_map = self._build_up_vector_index()

    # ---------- per-object orientation (ground truth, not guessed) ----------
    def _build_up_vector_index(self):
        """
        fpmodels-with-decomposed.csv ships explicit per-object `up` (and
        `front`) columns — real ground truth for which local axis of the
        mesh is actually "up", confirmed from a real download's own
        columns (id,name,wnsynsetkey,up,front,foundIn). This harness
        already reads this exact file for CATEGORY matching
        (_build_category_index) but never used its `up` column at all —
        every object's orientation was instead GUESSED from bounding-box
        aspect ratio (fitted_scale_and_dims's detect_up_axis fallback),
        which can identify the right AXIS but has no way to tell which
        DIRECTION along it is really up. That's exactly what let a
        correctly-scaled mesh still render upside-down or on its side —
        confirmed against a real run's trash-can/mug objects. Prefer this
        real per-object vector whenever it's present; detect_up_axis
        stays as the fallback for objects/sources where it isn't (e.g.
        Objaverse, or an HSSD object this file happens not to cover).

        Returns {uid: (ux, uy, uz)}.
        """
        table = {}
        dpath = os.path.join(self.hssd_root, "metadata")
        if not os.path.isdir(dpath):
            return table
        for fn in sorted(os.listdir(dpath)):
            if not fn.endswith(".csv"):
                continue
            path = os.path.join(dpath, fn)
            try:
                with open(path, newline="") as f:
                    reader = csv.DictReader(f)
                    if not reader.fieldnames:
                        continue
                    id_col = next((c for c in reader.fieldnames
                                   if c.strip().lower() in ("id", "uid", "objectid")), None)
                    up_col = next((c for c in reader.fieldnames
                                   if c.strip().lower() == "up"), None)
                    if not id_col or not up_col:
                        continue
                    for row in reader:
                        uid, up_raw = row.get(id_col), row.get(up_col)
                        if not uid or uid in table or not up_raw:
                            continue
                        try:
                            vec = tuple(float(x) for x in up_raw.split(","))
                        except ValueError:
                            continue
                        if len(vec) == 3:
                            table[uid.strip()] = vec
            except Exception:
                continue  # a malformed/unrelated CSV shouldn't break the whole index
        return table


    def _iter_object_configs(self):
        objects_dir = os.path.join(self.hssd_root, "objects")
        if not os.path.isdir(objects_dir):
            objects_dir = self.hssd_root  # tolerate a flatter layout too
        for dirpath, _dirnames, filenames in os.walk(objects_dir):
            for fn in filenames:
                if fn.endswith(".object_config.json"):
                    yield os.path.join(dirpath, fn)

    # Real, confirmed locations first; a few legacy root-level guesses kept
    # after them at near-zero cost in case an older/different HSSD release
    # or a partial custom download puts them there instead.
    _METADATA_DIRS = ["metadata"]
    _SEMANTIC_LEXICON_CANDIDATES = [
        "semantics/hssd-hab_semantic_lexicon.json",
        "semantics/semantic_lexicon.json",
    ]
    _LEGACY_ROOT_CANDIDATES = [
        "metadata.csv", "object_categories.csv", "objects.csv",
        "semantics/objects.csv", "semantics.csv", "hssd_metadata.csv",
        "metadata.json", "semantics.json",
    ]

    def _build_category_index(self):
        configs = list(self._iter_object_configs())
        uid_to_category, sources_used = {}, []

        # Tier 2: the semantic lexicon — checked first since it's the
        # dataset's own purpose-built category source, per HSSD's docs.
        for rel in self._SEMANTIC_LEXICON_CANDIDATES:
            path = os.path.join(self.hssd_root, rel)
            if os.path.isfile(path):
                table, diag = self._parse_semantic_lexicon(path)
                if table:
                    uid_to_category.update(table)
                    sources_used.append(f"{rel} ({len(table)} entries)")
                else:
                    print(f"[HSSDRetriever] found {rel!r} but couldn't parse a "
                          f"handle->category mapping out of it. Top-level shape: "
                          f"{diag}. See inspect_hssd.py to dump more detail, or "
                          f"pass category_map= explicitly once you've looked at it.")
                break  # only try the first one that exists

        # Tier 3: any CSV/JSON under metadata/, plus the legacy root-level
        # guesses — merged in (later sources fill gaps, don't overwrite
        # entries an earlier source already resolved).
        table_files = []
        for d in self._METADATA_DIRS:
            dpath = os.path.join(self.hssd_root, d)
            if os.path.isdir(dpath):
                for fn in sorted(os.listdir(dpath)):
                    if fn.endswith((".csv", ".json")):
                        table_files.append(os.path.join(dpath, fn))
        for rel in self._LEGACY_ROOT_CANDIDATES:
            p = os.path.join(self.hssd_root, rel)
            if os.path.isfile(p):
                table_files.append(p)
        for path in table_files:
            table = self._parse_metadata_table(path)
            if table:
                new_entries = {k: v for k, v in table.items() if k not in uid_to_category}
                uid_to_category.update(new_entries)
                sources_used.append(f"{os.path.relpath(path, self.hssd_root)} "
                                     f"({len(new_entries)} new entries)")

        # Tier 4: each object's own embedded field, then (only if enabled)
        # directory-name segments — resolved per-object so partial
        # metadata coverage from tiers 2-3 doesn't hide objects that could
        # still be resolved another way.
        idx = {}
        counts = {"metadata": 0, "embedded": 0, "directory": 0}
        for cfg_path in configs:
            uid = os.path.basename(cfg_path)[: -len(".object_config.json")]
            cat = uid_to_category.get(uid)
            tier = "metadata"
            if cat is None:
                cat = self._embedded_category(cfg_path)
                tier = "embedded"
            if cat is None and self.use_directory_fallback:
                rel = os.path.relpath(cfg_path, os.path.join(self.hssd_root, "objects"))
                parts = [seg for seg in rel.replace("\\", "/").split("/")[:-1]
                         if seg and len(seg) > 1]
                cat = parts[-1].lower() if parts else None
                tier = "directory"
            if cat:
                idx.setdefault(cat, []).append(cfg_path)
                counts[tier] += 1

        if sources_used:
            print(f"[HSSDRetriever] category sources used: {', '.join(sources_used)}")
        print(f"[HSSDRetriever] category index built from {len(configs)} objects: "
              f"{counts['metadata']} via metadata/lexicon, {counts['embedded']} via "
              f"embedded config fields"
              + (f", {counts['directory']} via directory-name fallback"
                 if self.use_directory_fallback else "")
              + f" ({len(idx)} total categories)")
        if not idx:
            print(f"[HSSDRetriever] WARNING: zero categories resolved under "
                  f"{self.hssd_root!r} — every retrieve() call will return None until "
                  f"this is fixed. Checked: {self._SEMANTIC_LEXICON_CANDIDATES}, "
                  f"metadata/*.csv|json, each object_config.json's own fields. Run "
                  f"`python inspect_hssd.py --root {self.hssd_root}` to see what's "
                  f"actually on disk, or pass category_map= explicitly.")
        elif not sources_used and counts["embedded"] == 0:
            print(f"[HSSDRetriever] WARNING: no metadata/lexicon files found and no "
                  f"embedded fields matched — did you download metadata/ and "
                  f"semantics/? (download_hssd.py's default now includes them; "
                  f"re-run it, huggingface_hub won't re-download objects/ you "
                  f"already have.)")
        return idx

    @staticmethod
    def _parse_semantic_lexicon(path):
        """
        Flexible reader for semantics/hssd-hab_semantic_lexicon.json.
        I don't have network access to fetch and inspect HSSD's real file,
        so this tries several plausible schemas rather than assuming one:
          (a) a flat dict directly mapping handle/uid -> category string
          (b) a known wrapper key ("classes", "categories", etc.) holding a
              list of records — each EITHER a category record (has a
              member-list field: "handles"/"objects"/etc.) OR an object
              record (has a category-like field: "category"/"class"/etc.)
          (c) a bare top-level list of either kind of record
        Returns (table, diagnostic_string) — table is {uid: category}
        (possibly empty), diagnostic_string describes what was actually
        found on disk so a failed parse is debuggable instead of silent.

        Confirmed against a REAL download: some HSSD releases' semantic
        lexicon is `{"classes": [{"id": N, "name": "chair"}, ...]}` — i.e.
        purely a category VOCABULARY (valid category names + ids), with NO
        object-membership information at all. That's a legitimate, useful
        file (it tells you what category names exist), but it cannot map
        any object to a category by itself. An earlier version of this
        function didn't recognize that case and fell through to a generic
        heuristic that treated "classes" as if it were itself a category
        name applied to every record's own "id" field — producing
        thousands of objects incorrectly tagged with a fake category
        literally called "classes". Fixed: once a known wrapper key is
        recognized as a list of dict records, ITS result is used as-is —
        even if that result is empty — and the generic fallback below is
        never reached for it.
        """
        with open(path) as f:
            data = json.load(f)

        def top_shape(d):
            if isinstance(d, dict):
                keys = list(d.keys())[:8]
                return f"dict with {len(d)} keys, sample: {keys}"
            if isinstance(d, list):
                sample = d[0] if d else None
                return f"list of {len(d)} items, first item: {sample!r:.200s}"
            return f"{type(d).__name__}"

        ID_KEYS = ("handle", "id", "uid", "object_handle", "name", "template_name")
        MEMBER_KEYS = ("handles", "objects", "instances", "ids", "members", "children")

        if isinstance(data, dict):
            # (a) flat handle -> category string mapping — try this first,
            # it's unambiguous when it matches.
            sample_vals = list(data.values())[:20]
            if sample_vals and all(isinstance(v, str) for v in sample_vals):
                return {uid: str(cat).lower() for uid, cat in data.items()}, top_shape(data)

            # (b) a known wrapper key holding a list of records. Recognized
            # wrapper keys are handled EXCLUSIVELY by
            # _parse_semantic_lexicon_records and its result is returned
            # as-is (empty or not) — see the docstring above for why
            # falling through past this is wrong.
            for key in ("categories", "classes", "lexicon", "records", "objects"):
                if key in data and isinstance(data[key], list) and \
                        (not data[key] or isinstance(data[key][0], dict)):
                    sub_table, note = HSSDRetriever._parse_semantic_lexicon_records(data[key])
                    diag = top_shape(data)
                    if not sub_table:
                        diag += (f" — recognized {key!r} as a record list, but found no "
                                 f"object-to-category membership in it (looks like a pure "
                                 f"category vocabulary — id/name only, no member-handle "
                                 f"list per record: {note or 'no per-object category field or member list found'})")
                    return sub_table, diag

            # (c) generic "category -> [handles]" dict, only reached when
            # NO known wrapper key was found at all — for plain category
            # -> [handle-string, ...] dicts that don't use one of the
            # recognized wrapper key names.
            table = {}
            for cat, val in data.items():
                if isinstance(val, list):
                    for member in val:
                        if isinstance(member, str):
                            table[member] = str(cat).lower()
                        elif isinstance(member, dict) and not any(k in member for k in MEMBER_KEYS):
                            uid = next((member[k] for k in ID_KEYS if k in member), None)
                            if uid is not None:
                                table[uid] = str(cat).lower()
            return table, top_shape(data)

        if isinstance(data, list):
            table, _ = HSSDRetriever._parse_semantic_lexicon_records(data)
            return table, top_shape(data)

        return {}, top_shape(data)

    @staticmethod
    def _parse_semantic_lexicon_records(records):
        table = {}
        ID_KEYS = ("handle", "id", "uid", "object_handle", "template_name")
        CAT_NAME_KEYS = ("category", "class", "semantic_category", "wnsynsetkey", "name")
        OBJ_CAT_KEYS = ("category", "class", "semantic_category", "wnsynsetkey")
        MEMBER_KEYS = ("handles", "objects", "instances", "ids", "members", "children")
        skipped_no_category, skipped_no_member_or_id, total = 0, 0, 0
        for rec in records:
            if not isinstance(rec, dict):
                continue
            total += 1
            member_key = next((k for k in MEMBER_KEYS if k in rec), None)
            if member_key:
                # This record IS a category, e.g. {"name": "mug", "handles": [...]}
                # — here "name" means the CATEGORY's name, not an object id.
                cat = next((rec[k] for k in CAT_NAME_KEYS if rec.get(k) not in (None, "")), None)
                if cat is None:
                    skipped_no_category += 1
                    continue
                for member in rec[member_key]:
                    uid = member if isinstance(member, str) else \
                        (next((member.get(k) for k in ID_KEYS
                               if isinstance(member, dict) and member.get(k) not in (None, "")), None)
                         if isinstance(member, dict) else None)
                    if uid is not None:
                        table[uid] = str(cat).lower()
            else:
                # This record IS an object, e.g. {"handle": "mug_01", "category": "mug"}
                # — here "name" (if no explicit category-ish field exists)
                # would ambiguously mean the object's own name, so it's
                # deliberately NOT in OBJ_CAT_KEYS; only explicit
                # category-like keys count here. A record with neither a
                # member-list field NOR any of these category-like fields
                # (e.g. this HSSD release's real {"id": N, "name": "chair"}
                # shape, which is a category-VOCABULARY entry, not an
                # object-to-category mapping) is correctly skipped rather
                # than guessed at.
                cat = next((rec[k] for k in OBJ_CAT_KEYS if rec.get(k) not in (None, "")), None)
                if cat is None:
                    skipped_no_category += 1
                    continue
                uid = next((rec[k] for k in ID_KEYS if rec.get(k) not in (None, "")), None)
                if uid is None and rec.get("name") not in (None, ""):
                    uid = rec["name"]
                if uid is None:
                    skipped_no_member_or_id += 1
                    continue
                table[uid] = str(cat).lower()
        note = None
        if not table and total:
            note = (f"{total} records read, {skipped_no_category} had no recognizable "
                    f"category-like field ({OBJ_CAT_KEYS} — plain 'name' alone is treated "
                    f"as the object's own name, not a category, to avoid ambiguity), "
                    f"{skipped_no_member_or_id} had a category but no id")
        return table, note

    @staticmethod
    def _embedded_category(cfg_path):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except Exception:
            return None
        for key in ("category", "semantic_category", "class", "wnsynsetkey", "tag"):
            if cfg.get(key):
                return str(cfg[key]).lower()
        return None

    @staticmethod
    def _parse_metadata_table(path):
        """Returns {uid: category}, or {} if the file doesn't look like a
        recognizable id+category table."""
        table = {}
        if path.endswith(".csv"):
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    return {}
                # Real HSSD data has a column literally called "Object Hash"
                # (hssd_obj_semantics_condensed.csv) — normalize away spaces/
                # underscores so that matches "objecthash" here, instead of
                # only ever matching the handful of exact names below.
                def _norm(c):
                    return re.sub(r"[\s_]+", "", c.lower())
                id_col = next((c for c in reader.fieldnames
                               if c.lower() in ("id", "uid", "object_id", "name")
                               or _norm(c) in ("id", "uid", "objectid", "hash", "objecthash")), None)
                cat_col = next((c for c in reader.fieldnames
                                if any(k in c.lower() for k in
                                       ("categ", "class", "tag", "wnsynset"))), None)
                if not id_col or not cat_col:
                    return {}
                for row in reader:
                    uid, cat = row.get(id_col), row.get(cat_col)
                    if uid and cat:
                        table[uid.strip()] = cat.strip().lower()
        elif path.endswith(".json"):
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                for uid, meta in data.items():
                    if isinstance(meta, dict):
                        cat = meta.get("category") or meta.get("class")
                        if not cat:
                            # Real metadata/objects.json entries look like
                            # {"id":..., "type":"furniture", "scene_counts":
                            # {...}, "name":..., "tags":[...]} — no
                            # "category"/"class" key at all, so the tiers
                            # above got nothing from this file. "tags" is
                            # the closest real signal here; take the first
                            # tag as a best-effort category guess (still
                            # gated by CategoryMatcher's own confidence
                            # threshold downstream, so a bad guess here
                            # just fails to match rather than matching
                            # wrong).
                            tags = meta.get("tags")
                            if isinstance(tags, list) and tags:
                                cat = tags[0]
                            elif isinstance(tags, str) and tags:
                                cat = tags
                        if cat:
                            table[uid] = str(cat).lower()
        return table

    def _config_path_for_uid(self, uid):
        """Only needed when the category index stores bare ids (the CSV/JSON
        metadata tiers) rather than full config paths (the embedded-field/
        directory-name tiers already store full paths directly)."""
        if self._uid_to_config is None:
            self._uid_to_config = {}
            for cfg_path in self._iter_object_configs():
                base = os.path.basename(cfg_path)
                key = base[: -len(".object_config.json")]
                self._uid_to_config[key] = cfg_path
        return self._uid_to_config.get(uid)

    def _resolve_render_asset(self, cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        render_rel = cfg.get("render_asset") or cfg.get("collision_asset")
        if not render_rel:
            return None
        return os.path.normpath(os.path.join(os.path.dirname(cfg_path), render_rel))

    def _config_up_vector(self, cfg_path):
        """
        Ground-truth up-vector read directly from THIS object's own
        `.object_config.json` — the standard per-object Habitat-Sim field
        (confirmed present, e.g. `"up": [0, 1, 0]`, on every real object
        config in this harness's own local HSSD download, including the
        exact "can" and "trash_bin" objects a real run retrieved).

        This is a separate, more reliable source than
        `_build_up_vector_index`'s `fpmodels-with-decomposed.csv` column:
        confirmed against this harness's own data that BOTH the "can" and
        "trash_bin" rows in that CSV have their `up`/`front` columns
        completely BLANK, even though each object's own config file has
        the real vector right there. `_up_vec_map.get(uid)` on those blank
        rows silently returned None, which sent both objects through
        `detect_up_axis`'s bounding-box-aspect-ratio guess — and that
        guess picked the WRONG axis for both (confirmed: axis 0 instead of
        the true axis 1), which is exactly what rendered them lying on
        their side instead of upright. Preferring the config's own field
        fixes this whenever it's present, which — unlike the CSV column —
        is apparently always, at least across this harness's own fixture.
        """
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except (OSError, ValueError):
            return None
        up_raw = cfg.get("up")
        if not up_raw or len(up_raw) != 3:
            return None
        try:
            return tuple(float(x) for x in up_raw)
        except (TypeError, ValueError):
            return None

    # ---------- retrieval ----------
    def retrieve(self, semantic_name, size_hint=None, height_hint=None):
        """Returns (mesh_path, size, height, mesh_scale, up_axis) or None.
        See ObjaverseRetriever.retrieve's docstring for what up_axis means
        and why the caller must rotate, not just scale, by it.

        size_hint: see ObjaverseRetriever.retrieve's docstring — the same
        fix applies here: without it, every retrieved object (mug, chair,
        table alike) was normalized to one fixed footprint, which is what
        caused a real run's chairs and table to become the same oversized
        footprint, overlap, and block all navigation."""
        if not self._category_map:
            return None
        import trimesh

        target_size = size_hint if size_hint is not None else self.target_size_hint

        if semantic_name in self._uid_cache:
            cfg_path = self._uid_cache[semantic_name]
        else:
            category = CategoryMatcher.match(semantic_name, self._category_map.keys())
            if category is None:
                return None
            candidates = self._category_map[category]
            if not candidates:
                return None
            idx = int(hashlib.md5(semantic_name.encode()).hexdigest(), 16) % len(candidates)
            entry = candidates[idx]
            cfg_path = entry if os.path.isfile(entry) else self._config_path_for_uid(entry)
            if cfg_path is None:
                return None
            self._uid_cache[semantic_name] = cfg_path

        mesh_path = self._resolve_render_asset(cfg_path)
        if not mesh_path or not os.path.isfile(mesh_path):
            return None

        # cfg_path's own filename (stripped of ".object_config.json") IS the
        # uid, regardless of whether we just resolved it or hit the cache —
        # _iter_object_configs/_config_path_for_uid both key by exactly this,
        # so this works uniformly without needing a separate uid cache.
        uid_key = os.path.basename(cfg_path)
        if uid_key.endswith(".object_config.json"):
            uid_key = uid_key[: -len(".object_config.json")]
        # Prefer the object's OWN config field (see _config_up_vector) —
        # only fall back to the aggregate CSV index for whatever it might
        # cover that the config itself doesn't specify.
        known_up = self._config_up_vector(cfg_path) or self._up_vec_map.get(uid_key)

        mesh = trimesh.load(mesh_path, force="mesh")
        target_height = height_hint if height_hint is not None else target_size * 1.4
        try:
            size, height, scale, up_axis, up_sign = fitted_scale_and_dims(
                mesh, target_size, target_height, known_up=known_up)
        except ValueError:
            return None
        return mesh_path, size, height, scale, up_axis, up_sign


class PrimitiveRetriever(AssetRetrieverBase):
    """The always-available fallback — same table used by synthesizer3d.py
    today. Given a semantic name, returns (kind, size, height). Ignores
    size_hint/height_hint (accepted only for interface compatibility with
    AssetRetrieverBase) since TieredRetriever already prefers the caller's
    hint directly over this table when one is available — see
    TieredRetriever.retrieve."""
    # Kept in sync with synthesizer3d.PrimitiveRetriever.TABLE (cm units,
    # anchored to the agent's real 170cm height). Order matters — this is a
    # substring match, so a key that's itself a substring of a more
    # specific key (e.g. "table" inside "coffee table") must come after it.
    TABLE = {
        "dining table": (42, 75), "coffee table": (32, 40),
        "side table": (20, 55), "end table": (20, 55), "tv stand": (40, 45),
        "table": (42, 75),
        "countertop": (45, 90), "counter": (45, 90), "island": (55, 90),
        "desk": (38, 75), "armchair": (30, 75), "chair": (18, 45),
        "stool": (16, 40), "ottoman": (16, 40),
        "loveseat": (45, 75), "sofa": (60, 75), "couch": (60, 75),
        "bed": (70, 50), "bookshelf": (22, 180), "shelf": (22, 120),
        "wardrobe": (35, 200), "closet": (35, 200), "dresser": (30, 80),
        "sideboard": (35, 85), "cabinet": (25, 90), "nightstand": (20, 55),
        "refrigerator": (32, 175), "fridge": (32, 175),
        "dishwasher": (30, 85), "washer": (30, 85),
        "stove": (30, 90), "oven": (30, 90), "sink": (25, 90),
        "lamp": (10, 150), "plant": (15, 90),
        "trash": (22, 35), "basket": (22, 35), "bin": (22, 35),
    }

    def retrieve(self, semantic_name, size_hint=None, height_hint=None):
        key = semantic_name.lower().replace("_", " ")
        for k, (size, height) in self.TABLE.items():
            if k in key:
                return ("box", size, height)
        return ("box", 25, 60)


class TieredRetriever(AssetRetrieverBase):
    """Tries each retriever in `sources` in order, falling back to
    PrimitiveRetriever if all of them fail or return nothing — this is
    what you actually want to use for --asset-source hssd (HSSD first,
    since it's curated for indoor scenes and ships better collision
    geometry) as well as the original Objaverse-only path, without
    duplicating the try/except-and-fall-back logic per source."""

    def __init__(self, sources):
        self.sources = list(sources)
        self.primitive = PrimitiveRetriever()

    def retrieve(self, semantic_name, size_hint=None, height_hint=None):
        for source in self.sources:
            try:
                result = source.retrieve(semantic_name, size_hint=size_hint,
                                          height_hint=height_hint)
            except Exception as e:
                print(f"[asset_retrieval] {type(source).__name__} failed for "
                      f"{semantic_name!r} ({type(e).__name__}: {e}) — trying the "
                      f"next source.")
                continue
            if result is not None:
                mesh_path, size, height, scale, up_axis, up_sign = result
                return dict(mesh_path=mesh_path, mesh_scale=scale, size=size,
                            height=height, mesh_up_axis=up_axis, mesh_up_sign=up_sign)
        # Falling all the way through to a primitive shape: prefer the
        # caller's own size_hint/height_hint directly when given — that's
        # already the DSL's own reasoned choice for this specific object
        # (a mug's caller passes size=6, not "whatever PrimitiveRetriever's
        # generic keyword table happens to guess for the word 'mug', which
        # it doesn't even have an entry for"). Only fall back to
        # PrimitiveRetriever's own table when no hint was provided at all,
        # e.g. a bare `.retrieve("mug")` call outside AddAsset.
        if size_hint is not None and height_hint is not None:
            return dict(kind="box", size=size_hint, height=height_hint)
        kind, size, height = self.primitive.retrieve(semantic_name)
        return dict(kind=kind, size=size, height=height)


class FallbackRetriever(TieredRetriever):
    """Objaverse, falling back to primitives — kept as a named class for
    backward compatibility with existing calls/docs; equivalent to
    TieredRetriever([ObjaverseRetriever()])."""

    def __init__(self):
        super().__init__([ObjaverseRetriever()])
        self.objaverse = self.sources[0]  # preserved attribute name for compatibility


def _strip_glb_textures(data):
    """
    Given raw .glb container bytes, returns a modified .glb with its
    materials/textures/images/samplers definitions removed from the JSON
    chunk (the binary geometry chunk is left completely untouched — this
    only rewrites the JSON header, never the vertex data).

    Why: confirmed as the actual, root cause of "the infinite corridor
    renders every object as a plain box" — a real end-to-end check (base64-
    embed a batch of this HSSD download's own real .glb files, feed the
    exact same bytes through THREE.GLTFLoader in a headless Node harness,
    same as the browser would) showed 11 of 13 sampled real meshes throw
    `THREE.GLTFLoader: setKTX2Loader must be called before loading KTX2
    textures` — and GLTFLoader throws on the ENTIRE parse the instant it
    hits one KTX2 (Basis Universal compressed) texture reference with no
    KTX2Loader wired up, not just that one texture, silently discarding the
    whole mesh back to its primitive box fallback. The great majority of
    this dataset's own shipped GLBs use exactly that compression. Properly
    supporting KTX2 means vendoring a whole extra WASM Basis transcoder;
    but this harness ALREADY throws away every mesh's original material in
    favor of a flat semantic color the instant a mesh loads successfully
    (see JS_MESH_BUILDER's buildMesh, both the OBJ and GLB paths) — so the
    textures GLTFLoader is choking on were never going to be visible
    anyway. Removing the material/texture/image/sampler definitions here
    means GLTFLoader never attempts to build a texture at all: the mesh
    parses using THREE's own default material, which the viewer
    immediately overrides with the semantic color regardless.
    """
    if len(data) < 12 or data[:4] != b"glTF":
        return data
    try:
        version, _total_len = struct.unpack_from("<II", data, 4)
        chunks = []
        offset = 12
        while offset + 8 <= len(data):
            chunk_len, chunk_type = struct.unpack_from("<I4s", data, offset)
            chunk_data = data[offset + 8: offset + 8 + chunk_len]
            chunks.append((chunk_type, chunk_data))
            offset += 8 + chunk_len
        if not chunks or chunks[0][0] != b"JSON":
            return data
        gltf = json.loads(chunks[0][1].decode("utf-8"))
    except (struct.error, ValueError, UnicodeDecodeError):
        return data  # not a well-formed GLB we understand — embed as-is

    changed = False
    for key in ("materials", "textures", "images", "samplers"):
        if key in gltf:
            del gltf[key]
            changed = True
    for ext_key in ("extensionsUsed", "extensionsRequired"):
        kept = [e for e in gltf.get(ext_key, []) if "texture" not in e.lower()]
        if kept != gltf.get(ext_key, []):
            gltf[ext_key] = kept
            changed = True
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if prim.pop("material", None) is not None:
                changed = True
    if not changed:
        return data

    new_json = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    new_json += b" " * ((4 - len(new_json) % 4) % 4)  # glTF JSON chunks pad with spaces
    out = bytearray()
    out += struct.pack("<4sII", b"glTF", version, 0)  # length patched below
    out += struct.pack("<I4s", len(new_json), b"JSON")
    out += new_json
    for chunk_type, chunk_data in chunks[1:]:
        out += struct.pack("<I4s", len(chunk_data), chunk_type)
        out += chunk_data
    struct.pack_into("<I", out, 8, len(out))
    return bytes(out)


def mesh_to_base64(mesh_path):
    """Base64-encodes a mesh file's raw bytes for embedding in the
    standalone HTML viewer (zero server/CORS setup — see viewer3d.py).
    Deliberately does NOT wrap this in a `data:...;base64,` URI with a
    guessed MIME type: an earlier version did that and always claimed
    `model/gltf-binary` regardless of the file's actual format, which
    silently broke every OBJ-format mesh (produced by the PyBullet-
    compatibility GLB->OBJ conversion in dsl3d_pybullet.py) fed to
    GLTFLoader, which can't parse OBJ at all. The caller now tracks the
    real format explicitly and picks the matching THREE.js loader."""
    with open(mesh_path, "rb") as f:
        data = f.read()
    if mesh_path.lower().endswith(".glb"):
        data = _strip_glb_textures(data)
    return base64.b64encode(data).decode("ascii")


def mesh_to_data_uri(mesh_path):
    """Deprecated — kept only for backward compatibility with any external
    caller. Prefer mesh_to_base64() plus tracking the real format
    separately; see its docstring for why."""
    return f"data:model/gltf-binary;base64,{mesh_to_base64(mesh_path)}"


def test_category_matching():
    """
    Offline self-test for ObjaverseRetriever._match_category — runs against
    a small fake category list standing in for the real LVIS taxonomy
    (which needs network to fetch), so this can run anywhere, every time,
    including in CI. It exercises exactly the object names this harness's
    own DSL/few-shot examples and PrimitiveRetriever's TABLE actually
    produce (see synthesizer3d.py's FEWSHOT/FEWSHOT_PICK_PLACE and
    PrimitiveRetriever.TABLE above), plus the "goal_X" / "chair_2"-style
    name mangling AddAsset and the few-shot examples introduce.

    This is real signal about the matching LOGIC (normalization, synonym
    lookup, substring/token fallback), even though it can't confirm the
    hand-curated SYNONYMS map lines up with the actual live LVIS category
    strings — do that once by printing
    `list(ObjaverseRetriever()._lvis_annotations().keys())` on a machine
    with real network access, and adjust SYNONYMS if any entry doesn't
    exist there.
    """
    fake_lvis = {
        "table": [], "coffee_table": [], "chair": [], "sofa": [], "bed": [],
        "desk": [], "countertop": [], "bookcase": [], "dresser": [],
        "nightstand": [], "mug": [], "cup": [], "trash_can": [], "basket": [],
        "can": [], "tin_can": [], "book": [], "ball": [], "key_(metal)": [],
        "remote_control": [], "apple": [], "lamp": [], "pillow": [],
    }
    r = ObjaverseRetriever()
    r._lvis = fake_lvis  # inject directly, skip the network call

    cases = [
        ("mug", "mug"), ("goal_mug", "mug"), ("can", "can"), ("goal_can", "can"),
        ("trash_bin", "trash_can"), ("bin", "trash_can"),
        ("counter", "countertop"), ("kitchen_counter", "countertop"),
        ("table", "table"), ("coffee_table", "coffee_table"),
        ("chair", "chair"), ("chair_2", "chair"), ("chair_3", "chair"),
        ("sofa", "sofa"), ("couch", "sofa"), ("bed", "bed"), ("desk", "desk"),
        ("bookshelf", "bookcase"), ("shelf", "bookcase"),
        ("dresser", "dresser"), ("nightstand", "nightstand"),
        ("apple", "apple"), ("ball", "ball"), ("book", "book"),
    ]
    failures = []
    for query, expected in cases:
        got = r._match_category(query)
        ok = got == expected
        print(f"  [{'OK' if ok else 'FAIL'}] {query!r:20s} -> {got!r:20s} (expected {expected!r})")
        if not ok:
            failures.append((query, expected, got))

    # Also confirm the "no reasonable match" case correctly returns None
    # rather than confidently guessing something unrelated — a wrong
    # confident match is worse than a graceful fallback to a primitive.
    nonsense = r._match_category("xyzzy_nonexistent_object_qqq")
    ok = nonsense is None
    print(f"  [{'OK' if ok else 'FAIL'}] {'xyzzy_nonexistent_object_qqq':20s} -> {nonsense!r:20s} (expected None)")
    if not ok:
        failures.append(("xyzzy_nonexistent_object_qqq", None, nonsense))

    # Regression cases from an actual run against real HSSD data: WordNet
    # synset-key-style categories (e.g. "candlestick.n.01") caused a raw
    # substring match to fire on short queries — "can" matched inside
    # "candlestick" purely by character overlap, with no respect for word
    # boundaries, and returned the wrong mesh entirely. Token-set
    # containment (see CategoryMatcher.match's tier 3) fixes this; these
    # cases test it directly against category strings shaped like the
    # real data that exposed the bug.
    wordnet_style = ["candlestick.n.01", "showerhead.n.01", "can.n.01",
                      "trashcan.n.01", "canister.n.02", "ashcan.n.01"]
    wordnet_cases = [
        ("can", "can.n.01"), ("trash_bin", "trashcan.n.01"), ("bin", "trashcan.n.01"),
        ("candlestick", "candlestick.n.01"),
    ]
    for query, expected in wordnet_cases:
        got = CategoryMatcher.match(query, wordnet_style)
        ok = got == expected
        print(f"  [{'OK' if ok else 'FAIL'}] {query!r:20s} -> {got!r:20s} (expected {expected!r}) "
              f"[WordNet-style regression]")
        if not ok:
            failures.append((query, expected, got))

    print()
    total = len(cases) + 1 + len(wordnet_cases)
    if failures:
        print(f"{len(failures)}/{total} category-matching cases FAILED.")
        return False
    print(f"All {total} category-matching cases passed.")
    return True


def test_hssd_retriever(tmp_root=None):
    """
    Offline self-test for HSSDRetriever — builds a synthetic directory tree
    matching HSSD's REAL, confirmed layout (as posted by a user running
    this against their actual download):
        objects/<hash-bucket>/<uid>.object_config.json + <uid>.glb
        metadata/*.csv
        semantics/hssd-hab_semantic_lexicon.json
    with real meshes (via trimesh, no network needed) and a semantic
    lexicon using the "categories: [{name, handles}]" wrapper shape, plus
    a metadata CSV that deliberately covers only ONE of the three objects
    — testing that partial coverage across the lexicon and metadata tiers
    merges correctly, and that hash-bucket directory names (which carry no
    category information at all in real HSSD) are correctly NOT used
    unless explicitly enabled.

    This validates the retrieval LOGIC end to end against real files on
    disk. It can't confirm your actual HSSD download's semantic lexicon
    uses this exact schema — see `_parse_semantic_lexicon`'s docstring for
    the other schemas it also handles, and `inspect_hssd.py` for dumping
    your real file's actual shape if none of them match.
    """
    import shutil
    import tempfile
    import trimesh

    root = tmp_root or tempfile.mkdtemp(prefix="hssd_selftest_")
    failures = []

    def make_object(bucket, uid, extents):
        d = os.path.join(root, "objects", bucket)
        os.makedirs(d, exist_ok=True)
        trimesh.creation.box(extents=extents).export(os.path.join(d, f"{uid}.glb"))
        with open(os.path.join(d, f"{uid}.object_config.json"), "w") as f:
            json.dump({"render_asset": f"{uid}.glb"}, f)

    try:
        # Hash-bucket-style directory names (real HSSD convention) — these
        # carry NO category info, unlike an earlier fixture's category-
        # named folders, so this also tests that the (disabled-by-default)
        # directory fallback doesn't accidentally "work" here.
        make_object("aa", "mug_01", [8, 10, 8])
        make_object("bb", "chair_07", [40, 90, 40])
        make_object("cc", "trash_can_02", [30, 40, 30])

        os.makedirs(os.path.join(root, "semantics"), exist_ok=True)
        lexicon = {"categories": [
            {"name": "mug", "handles": ["mug_01"]},
            {"name": "chair", "handles": ["chair_07"]},
        ]}
        with open(os.path.join(root, "semantics", "hssd-hab_semantic_lexicon.json"), "w") as f:
            json.dump(lexicon, f)

        # trash_can_02 deliberately NOT in the lexicon — only findable via
        # metadata/*.csv. Tests merging across the two real tiers.
        os.makedirs(os.path.join(root, "metadata"), exist_ok=True)
        with open(os.path.join(root, "metadata", "objects.csv"), "w") as f:
            f.write("id,category\ntrash_can_02,trash_can\n")

        r = HSSDRetriever(hssd_root=root)
        cases = [
            ("mug", "mug_01.glb"), ("goal_mug", "mug_01.glb"),
            ("chair", "chair_07.glb"), ("chair_2", "chair_07.glb"),
            ("trash_bin", "trash_can_02.glb"), ("bin", "trash_can_02.glb"),
        ]
        for query, expected_file in cases:
            result = r.retrieve(query)
            ok = result is not None and result[0].endswith(expected_file)
            print(f"  [{'OK' if ok else 'FAIL'}] {query!r:12s} -> "
                  f"{os.path.basename(result[0]) if result else None!r:16s} "
                  f"(expected {expected_file!r})")
            if not ok:
                failures.append((query, expected_file, result))

        nonsense = r.retrieve("xyzzy_nonexistent_qqq")
        ok = nonsense is None
        print(f"  [{'OK' if ok else 'FAIL'}] {'xyzzy_nonexistent_qqq':12s} -> {nonsense!r:16s} "
              f"(expected None)")
        if not ok:
            failures.append(("xyzzy_nonexistent_qqq", None, nonsense))

        missing_root = HSSDRetriever(hssd_root=os.path.join(root, "does_not_exist"))
        ok = missing_root.retrieve("mug") is None
        print(f"  [{'OK' if ok else 'FAIL'}] missing hssd_root path handled gracefully "
              f"(no crash, returns None)")
        if not ok:
            failures.append(("missing_root", None, "crashed or returned non-None"))

        # Regression: a real run against real HSSD data showed every
        # retrieved object (mug, chair, table alike) getting the SAME
        # fixed footprint, since size_hint wasn't being passed through —
        # four oversized "chairs" ended up overlapping an equally
        # oversized "table" and blocking all navigation. This checks that
        # a small object and a large object, retrieved with DIFFERENT
        # size_hints (as AddAsset actually passes), come back with
        # correspondingly different sizes, not a shared constant.
        make_object("dd", "obj_desk_small", [0.08, 0.10, 0.08])  # a "mug"-shaped mesh
        make_object("ee", "obj_desk_large", [1.2, 0.75, 0.9])     # a "table"-shaped mesh
        with open(os.path.join(root, "metadata", "sizing_check.csv"), "w") as f:
            f.write("id,category\nobj_desk_small,sizing_check_small\n"
                    "obj_desk_large,sizing_check_large\n")
        r_sizing = HSSDRetriever(hssd_root=root)
        small_result = r_sizing.retrieve("sizing_check_small", size_hint=6, height_hint=10)
        large_result = r_sizing.retrieve("sizing_check_large", size_hint=40, height_hint=42)
        ok = (small_result is not None and large_result is not None
              and small_result[1] == 6 and large_result[1] == 40
              and small_result[1] != large_result[1])
        print(f"  [{'OK' if ok else 'FAIL'}] size_hint respected: small object size="
              f"{small_result[1] if small_result else None}, large object size="
              f"{large_result[1] if large_result else None} (expected 6 and 40, "
              f"NOT equal to each other)")
        if not ok:
            failures.append(("size_hint_respected", "6 != 40", (small_result, large_result)))

        # Lexicon schema variants — _parse_semantic_lexicon needs to handle
        # more than just the one shape used above.
        variants = [
            ("flat dict", {"mug_01": "mug", "chair_07": "chair"}),
            ("list of object records",
             [{"handle": "mug_01", "category": "mug"}, {"handle": "chair_07", "class": "chair"}]),
            ("classes/members wrapper",
             {"classes": [{"category": "mug", "members": ["mug_01"]}]}),
        ]
        for label, data in variants:
            vpath = os.path.join(root, f"_variant_{label.replace(' ', '_').replace('/', '_')}.json")
            with open(vpath, "w") as f:
                json.dump(data, f)
            table, diag = HSSDRetriever._parse_semantic_lexicon(vpath)
            ok = table.get("mug_01") == "mug"
            print(f"  [{'OK' if ok else 'FAIL'}] lexicon schema variant {label!r} -> "
                  f"{table} (expected mug_01 -> 'mug')")
            if not ok:
                failures.append((f"variant:{label}", "mug_01->mug", table))

        # Deliberately unrecognizable shape should return {} with a
        # diagnostic, not crash and not silently invent a wrong match.
        vpath = os.path.join(root, "_variant_unrecognizable.json")
        with open(vpath, "w") as f:
            json.dump({"version": "1.0", "nested": {"a": {"b": 1}}}, f)
        table, diag = HSSDRetriever._parse_semantic_lexicon(vpath)
        ok = table == {}
        print(f"  [{'OK' if ok else 'FAIL'}] unrecognizable lexicon shape -> {{}} "
              f"(diagnostic: {diag})")
        if not ok:
            failures.append(("variant:unrecognizable", {}, table))
    finally:
        if not tmp_root:
            shutil.rmtree(root, ignore_errors=True)

    total = len(cases) + 7  # 6 cases + nonsense + missing_root + size_hint check + 3 variants + unrecognizable
    print()
    if failures:
        print(f"{len(failures)}/{total} HSSDRetriever cases FAILED.")
        return False
    print(f"All {total} HSSDRetriever cases passed.")
    return True


def test_hssd_retriever_real_layout(tmp_root=None):
    """
    Offline self-test built directly from a REAL user's `inspect_hssd.py`
    output against their actual HSSD download (not a guess) — specifically:
      - metadata/fpmodels-with-decomposed.csv: id,name,wnsynsetkey,... (a
        real per-object CSV with WordNet-style categories)
      - metadata/objects.json: {uid: {id, type, scene_counts, name, tags}}
        — NO "category"/"class" key at all, only "tags"
      - metadata/hssd_obj_semantics_condensed.csv: "Object Hash" as its id
        column (not "id"/"uid")
      - semantics/hssd-hab_semantic_lexicon.json: {"classes": [{"id",
        "name"}]} — a category TAXONOMY with no per-object membership list,
        confirmed to correctly contribute nothing (not wrong data) rather
        than being treated as though it were a uid->category source
    This validates the fixes made after seeing that real structure: the
    "Object Hash" column, and the "tags" fallback for objects.json.
    """
    import shutil
    import tempfile
    import trimesh

    root = tmp_root or tempfile.mkdtemp(prefix="hssd_real_layout_selftest_")
    failures = []
    try:
        def make_object(bucket, uid, extents):
            d = os.path.join(root, "objects", bucket)
            os.makedirs(d, exist_ok=True)
            trimesh.creation.box(extents=extents).export(os.path.join(d, f"{uid}.glb"))
            with open(os.path.join(d, f"{uid}.object_config.json"), "w") as f:
                json.dump({"render_asset": f"{uid}.glb"}, f)

        # fridge_uid is ONLY resolvable via the wnsynsetkey CSV column.
        # mug_uid is ONLY resolvable via metadata/objects.json's "tags"
        # fallback (no wnsynsetkey entry for it at all) — this is the case
        # that silently returned nothing before this round's fix.
        make_object("a1", "fridge_uid_001", [70, 175, 32])
        make_object("b2", "mug_uid_002", [8, 10, 8])

        os.makedirs(os.path.join(root, "metadata"), exist_ok=True)
        with open(os.path.join(root, "metadata", "fpmodels-with-decomposed.csv"), "w", newline="") as f:
            f.write("id,name,wnsynsetkey,up,front,foundIn\n")
            f.write("fridge_uid_001,\"Stainless Steel Fridge\",refrigerator.n.01,\"0,1,0\",\"0,0,1\",kitchen\n")

        with open(os.path.join(root, "metadata", "objects.json"), "w") as f:
            json.dump({
                "mug_uid_002": {"id": "mug_uid_002", "type": "object",
                                "scene_counts": {"102343992": 1},
                                "name": "Ceramic Coffee Mug", "tags": ["mug", "kitchenware"]},
            }, f)

        # A CSV whose id column is "Object Hash", not "id"/"uid" — should
        # be recognized now but doesn't carry a real category (mirrors the
        # real file's apparent articulation-metadata content), so it's
        # expected to safely contribute nothing rather than crash.
        with open(os.path.join(root, "metadata", "hssd_obj_semantics_condensed.csv"), "w", newline="") as f:
            f.write("Object Hash,Articulated Object\nfridge_uid_001,false\n")

        os.makedirs(os.path.join(root, "semantics"), exist_ok=True)
        with open(os.path.join(root, "semantics", "hssd-hab_semantic_lexicon.json"), "w") as f:
            json.dump({"classes": [{"id": 0, "name": "refrigerator"}, {"id": 1, "name": "mug"}]}, f)

        r = HSSDRetriever(hssd_root=root)
        cases = [("fridge", "fridge_uid_001.glb"), ("refrigerator", "fridge_uid_001.glb"),
                 ("coffee_mug", "mug_uid_002.glb"), ("mug", "mug_uid_002.glb")]
        for query, expected_file in cases:
            result = r.retrieve(query)
            ok = result is not None and result[0].endswith(expected_file)
            print(f"  [{'OK' if ok else 'FAIL'}] {query!r:14s} -> "
                  f"{os.path.basename(result[0]) if result else None!r:20s} "
                  f"(expected {expected_file!r})")
            if not ok:
                failures.append((query, expected_file, result))
    finally:
        if not tmp_root:
            shutil.rmtree(root, ignore_errors=True)

    total = len(cases)
    print()
    if failures:
        print(f"{len(failures)}/{total} real-layout HSSDRetriever cases FAILED.")
        return False
    print(f"All {total} real-layout HSSDRetriever cases passed.")
    return True


if __name__ == "__main__":
    print("=== Offline self-test: category-matching logic (no network needed) ===")
    matching_ok = test_category_matching()

    print("\n=== Offline self-test: HSSDRetriever against a synthetic fixture "
          "(no network needed) ===")
    hssd_ok = test_hssd_retriever()

    print("\n=== Offline self-test: HSSDRetriever against your REAL file layout "
          "(no network needed) ===")
    hssd_real_ok = test_hssd_retriever_real_layout()

    print("\n=== Live test: actual Objaverse network retrieval (needs internet) ===")
    r = ObjaverseRetriever()
    print("Retrieving a mesh for 'mug'...")
    try:
        result = r.retrieve("mug")
    except Exception as e:
        result = None
        print(f"Network retrieval failed: {type(e).__name__}: {e}")
        print("(Expected if this machine can't reach huggingface.co — see the module "
              "docstring. On a machine with normal internet access this should just work.)")
    if result is None:
        print("No mesh retrieved this run.")
    else:
        mesh_path, size, height, scale, up_axis, up_sign = result
        print(f"mesh_path={mesh_path}")
        print(f"size={size:.1f} height={height:.1f} scale={scale} "
              f"up_axis={up_axis} up_sign={up_sign}")
        print("SUCCESS — Objaverse retrieval is working end to end on this machine.")

    if not (matching_ok and hssd_ok and hssd_real_ok):
        raise SystemExit(1)