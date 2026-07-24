# Deploying this for free, from anywhere

> This page is about hosting a **pre-generated, static** `viewer_N.html` —
> no server, no live prompt input. If you want a real page where someone
> types a prompt and a scene gets generated on demand (using real HSSD
> meshes server-side), that's `webapp/` (`python webapp/server.py`) — see
> the README's "Web app" section. That one genuinely needs a running
> backend (Flask) and the local HSSD dataset, so it's a "run this
> yourself" local app rather than something to drop on a static host.

The `demo/*.html` files (and any `out3d/viewer_N.html` you generate later)
are **fully self-contained** — three.js is embedded directly in the file,
so there's no build step and no backend. That means any static file host
works, and all of these have a free tier with no credit card:

## Option A — GitHub Pages (persistent URL, best for sharing long-term)
1. Create a new GitHub repo and push this folder to it.
2. Repo Settings → Pages → Source → "Deploy from a branch" → `main` / `root`.
3. GitHub gives you a URL like `https://<you>.github.io/<repo>/demo/` —
   share that link with anyone, works on desktop and mobile.

## Option B — Netlify Drop (fastest, zero setup, ~30 seconds)
1. Go to https://app.netlify.com/drop
2. Drag the `demo/` folder onto the page.
3. You get a live public URL immediately (no login required to try it;
   sign up only if you want the URL to stay permanent).

## Option C — Vercel / Cloudflare Pages
Same idea: both accept a static folder with zero config and give a free
public URL. Any of the three is equivalent for this project since there's
no server-side code at all.

## Generating new scenes to deploy
Every run of `harness3d.py` writes a new self-contained `viewer_N.html`
into `--out`:
```bash
python harness3d.py "a bedroom with a bed, a dresser, and a lamp" --mock
```
Drop the resulting `out3d/viewer_0.html` into any of the options above
the same way.
