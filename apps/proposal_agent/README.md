---
title: ResearchAI Proposal Agent
emoji: 🚀
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
license: mit
short_description: Auto research proposal agent — mission to final delivery
---

# ResearchAI — Auto Research Proposal Agent

Turns the Stitch design mockups into a demo you can actually click through, run, and export from.
The full flow is **Start → Mission Control → Drafting → Final Delivery**.

How it runs: a pure static site — no build step, no server dependency. Opening `index.html` is all there is to it.

## First: connect the Stitch MCP

The designs live in a Stitch project, so before anything else register the Stitch MCP server with Claude Code:

```bash
claude mcp add stitch \
  --transport http \
  --header "X-Goog-Api-Key: $STITCH_API_KEY" \
  https://stitch.googleapis.com/mcp
```

`STITCH_API_KEY` is the `X-Goog-Api-Key` for `stitch.googleapis.com`; export it in your shell first so the key stays out of the repo and out of your shell history. Verify with `claude mcp list` — `stitch` should come back connected.

This is only needed to read or re-pull the mockups (see [After changing the mockups](#after-changing-the-mockups)); the site itself runs without it.

## What it does

| Page | File | Behavior |
| --- | --- | --- |
| Start | `index.html` | `Initialize New Research Mission` — name, institution, contact, education, research field, saved to localStorage as you type; `Continue` moves on to Mission Control. The profile is optional, and when filled in it names the author on Final Delivery |
| Mission Control | `mission.html` | Rounds 0–4 as a **guided wizard**, one round at a time (Foundation / Baseline / Inquiry / Metrics / Governance); move with Back / Next or jump by clicking the dots at the top, and dots light up for rounds you have filled in; `Load Example` fills in a sample task in one click; `Launch Agent` only appears on the last round and starts the agent |
| Drafting | `drafting.html` | A **Markdown file editor**: the proposal is a real `.md` source you type into, with a `Source` / `Preview` switch, `Import .md` to load a file from disk and `Save .md` to write one back (auto-saved to localStorage in between); the AI Suggestions panel on the right comes from gap analysis over the config |
| Final Delivery | `final.html` | Final-draft preview plus reference list, with export to PDF (browser print, backed by its own print stylesheet), DOCX, Markdown, and email, plus zoom |

> Heads-up on the mockups: the Stitch project now holds three screens — the Start page, a redesigned `任务配置页 - RSI-Index (Step-by-Step)` that is not wired in yet, and Final Delivery. The Mission Control and Drafting designs this site builds from were **deleted upstream**, so `build/screens/` is their only remaining copy. `sync_screens.py` reports them as missing instead of overwriting them, and the Live Analysis screen is gone from the project entirely.

The draft is one Markdown file all the way through: the first `# ` line is the title and every `## ` block is a section. Final Delivery renders that same file, so edits carry through; `New Research` clears all state and starts over.

**The file is the proposal format.** `compose()` writes a filled-in copy of `proposal_format_v2.txt` — Task Metadata, Foundation, Research Question, Reference Baseline, Evaluation, Workspace (External Resource Access + Compute Feasibility) — with each section's fields as a `| Field | Value |` table. Change that format and `compose()` in `assets/engine.js` is what has to follow it. Fields the proposal stage leaves open (Category, Tags, Expert time) are emitted as an italic `TODO` rather than invented.

## Frontend / backend

- **Frontend**: the Stitch-generated HTML (Tailwind + Material Symbols + the original design system's colors) is kept as-is; only ids, navigation, and script mount points were added.
- **"Backend"**: currently runs in the browser — `assets/engine.js` is the proposal engine, where `compose()` synthesizes the full proposal from the task config (title, abstract, the six format sections, references) and `gaps()` suggests what the config is missing.
- **To wire in a real backend**: set `RA_API_BASE` in `assets/config.js` to your service address. The frontend will `POST ${RA_API_BASE}/api/compose` (body is the task config JSON) and render the returned proposal object; the response shape matches `RA.engine.compose()`. A failed request falls back to the local engine automatically, so the demo never goes blank.

## Layout

```
index.html  mission.html  drafting.html  final.html    # build output, deploy directly
assets/
  app.js        page controllers (all interactions across the four pages + the round wizard)
  engine.js     proposal engine: compose() / gaps(), plus the Markdown layer
                (draftMarkdown / parseDraft / mdToHtml) the draft file rides on
  store.js      localStorage state (profile / mission / proposal / draft) + the sample task
  config.js     optional backend address
  app.css       animations / print styles layered on top of the Stitch styles
build/
  screens/*.html   raw Stitch design exports (read-only)
  sync_screens.py  re-pull the mockups from the Stitch MCP
  build.py         rewrite the mockups into the pages above
```

## Local preview

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

## Deploying to a Hugging Face Space

Choose the **Static** Space type (the frontmatter at the top of this README already declares it).

```bash
pip install -U huggingface_hub
hf auth login                                    # older versions: huggingface-cli login
hf repo create researchai-demo --repo-type space --space_sdk static

git init && git add . && git commit -m "ResearchAI demo"
git remote add origin https://huggingface.co/spaces/<your-username>/researchai-demo
git push -u origin main
```

Once pushed it's live, with no secrets to configure (unless you wired in a backend).

## After changing the mockups

Once you've edited the design in Stitch, re-sync and rebuild (needs the Stitch MCP connected, see [above](#first-connect-the-stitch-mcp)):

```bash
export STITCH_API_KEY=...          # the X-Goog-Api-Key for stitch.googleapis.com
python3 build/sync_screens.py      # pull build/screens/*.html
python3 build/build.py             # regenerate the four pages
```

The Start page has no nav slot in any mockup, so `add_start_nav()` in `build/build.py` clones one into every
sidebar and moves the active styling onto the link for the page being rendered (the designs each hard-code their
own highlight, and Final Delivery highlights nothing).

Every rewrite in `build.py` is guarded by an assertion: if an anchor disappears from a mockup, the build fails loudly and points at the exact spot rather than quietly emitting a page where nothing responds to clicks.

> The API key is read only from the environment and never written into the repo.
