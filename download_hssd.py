"""
download_hssd.py — one-time download helper for HSSD (Habitat Synthetic
Scenes Dataset) object assets, implementing steps 2-3 of GUIDE_3D_ASSETS.md.

Prerequisites (do these once, outside this script):
    1. Accept the dataset's terms at
       https://huggingface.co/datasets/hssd/hssd-hab (free, near-instant).
    2. pip install huggingface_hub
    3. huggingface-cli login   (paste a token from your HF account settings)

Then:
    python download_hssd.py --out ./hssd-hab

By default this downloads objects/* (the individual furniture/item meshes
+ their .object_config.json files) PLUS metadata/* and semantics/* — the
small files HSSDRetriever's category matching actually depends on. An
earlier version of this script only pulled objects/*, which meant
HSSDRetriever had nothing to match category names against at all (every
object silently fell back to a primitive shape — not a retrieval bug, just
missing input). metadata/ and semantics/ are small; there's no real reason
to skip them. Pass --full-scenes if you specifically want the much larger
prebuilt full-scene layouts too (e.g. to experiment with importing a real
designer's room layout directly — see GUIDE_3D_ASSETS.md's "As a layout
source" option for 3D-FRONT, which is the more natural fit for that, but
HSSD ships some full scenes as well).

If you already downloaded only objects/* with an older version of this
script, just re-run it with the same --out — huggingface_hub skips files
you already have and only pulls the newly-added metadata/semantics
patterns.

Once this finishes, point HSSDRetriever at the printed path:
    from asset_retrieval import HSSDRetriever
    retriever = HSSDRetriever(hssd_root="./hssd-hab")

This script could not be run end to end from the sandbox that wrote it —
see asset_retrieval.py's module docstring for why (no route to
huggingface.co from there). It's written directly against
huggingface_hub's documented snapshot_download API; run it on your own
machine, where your HF login is already working.
"""

import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="./hssd-hab",
                     help="local directory to download into (default: ./hssd-hab)")
    ap.add_argument("--full-scenes", action="store_true",
                     help="also download prebuilt scene layouts, not just object meshes "
                          "(much larger download)")
    ap.add_argument("--repo-id", default="hssd/hssd-hab",
                     help="override the HF dataset repo id (default: hssd/hssd-hab)")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub isn't installed. Run: pip install huggingface_hub")
        print("Then: huggingface-cli login   (before re-running this script)")
        sys.exit(1)

    allow_patterns = None if args.full_scenes else ["objects/*", "metadata/*", "semantics/*"]
    print(f"Downloading {args.repo_id!r} "
          f"({'all files, including full scenes' if args.full_scenes else 'objects/*, metadata/*, and semantics/* only'}) "
          f"into {args.out!r} ...")
    print("(First run downloads the object library — this can take a while depending "
          "on your connection; it's cached locally afterward.)")

    try:
        path = snapshot_download(repo_id=args.repo_id, repo_type="dataset",
                                  allow_patterns=allow_patterns, local_dir=args.out)
    except Exception as e:
        print(f"\nDownload failed: {type(e).__name__}: {e}")
        print("Common causes: terms not yet accepted at "
              "https://huggingface.co/datasets/hssd/hssd-hab, or `huggingface-cli "
              "login` wasn't run / the token has expired.")
        sys.exit(1)

    print(f"\nDownloaded to: {path}")
    print("Next:")
    print("  from asset_retrieval import HSSDRetriever")
    print(f'  retriever = HSSDRetriever(hssd_root={path!r})')
    print("  # then either pass retriever= directly to synthesize_scene(),")
    print("  # or: python harness3d.py \"...\" --mock --asset-source hssd --hssd-root "
          f"{path!r}")


if __name__ == "__main__":
    main()
