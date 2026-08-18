#!/usr/bin/env python3
"""Turn the raw Stitch screen exports into the working ResearchAI demo site.

The Stitch designs (build/screens/*.html) are treated as read-only source. This
script rewrites them into the pages served by the Space:

    build/screens/5efd....html  ->  index.html     (Mission Control)
    build/screens/594f....html  ->  drafting.html  (Drafting & Refinement)
    build/screens/768d....html  ->  final.html     (Final Delivery)

The edits are deliberately surgical: wire the navigation, add ids/hooks the
runtime needs, and load assets/*.js. Every anchor is asserted, so a re-synced
design that moved an element fails the build loudly instead of silently
producing a dead page.

    python3 build/build.py          # rebuild the site
    python3 build/sync_screens.py   # re-pull the designs from Stitch first
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "build" / "screens"

PAGES = {
    "index.html": ("18a02818c5f341548924c8b5d8b2b0af", "start", "ResearchAI — Start"),
    "mission.html": ("5efd1b374cf44c27b5571985b983349b", "mission", "ResearchAI — Mission Control"),
    "drafting.html": ("594fd847725f44429cde3167158062b1", "drafting", "ResearchAI — Drafting"),
    "final.html": ("768df3ba9c6747c98d663989eb450714", "final", "ResearchAI — Final Delivery"),
}

# Inline avatar so the page has no dependency on expiring Google CDN links.
AVATAR = (
    "data:image/svg+xml;utf8,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 40'%3E"
    "%3Crect width='40' height='40' rx='20' fill='%231d2022'/%3E"
    "%3Ccircle cx='20' cy='16' r='6' fill='%23adc7ff'/%3E"
    "%3Cpath d='M8 36c0-6.6 5.4-12 12-12s12 5.4 12 12' fill='%234a8eff'/%3E%3C/svg%3E"
)


class BuildError(RuntimeError):
    pass


def sub_once(html: str, old: str, new: str, what: str) -> str:
    """Replace `old` exactly once, or fail the build."""
    n = html.count(old)
    if n != 1:
        raise BuildError(f"{what}: expected 1 occurrence of {old[:70]!r}, found {n}")
    return html.replace(old, new, 1)


def add_id(html: str, needle: str, tag: str, el_id: str, extra: str = "", nth: int = 0, must_contain: str = "") -> str:
    """Add an id to the innermost `tag` element that encloses `needle`.

    `nth` selects which occurrence of `needle` to anchor on (-1 = the last one,
    which is how to skip a section comment that repeats the heading text).
    `must_contain` asserts the opening tag really is the element meant — the
    difference between tagging a card and tagging the container it sits in.
    """
    starts = [m.start() for m in re.finditer(re.escape(needle), html)]
    if not starts or (nth >= 0 and len(starts) <= nth):
        raise BuildError(f"add_id({el_id}): needle {needle[:60]!r} occurs {len(starts)} times")
    pos = starts[nth]
    open_at = html.rfind("<" + tag, 0, pos)
    if open_at == -1:
        raise BuildError(f"add_id({el_id}): no enclosing <{tag}> before {needle[:60]!r}")
    close_at = html.index(">", open_at)
    if must_contain and must_contain not in html[open_at:close_at]:
        raise BuildError(f"add_id({el_id}): enclosing <{tag}> is not the expected element ({must_contain!r} missing)")
    attrs = f' id="{el_id}"' + (f" {extra}" if extra else "")
    return html[:open_at + len(tag) + 1] + attrs + html[open_at + len(tag) + 1:close_at + 1] + html[close_at + 1:]


def insert_before(html: str, anchor: str, snippet: str, what: str) -> str:
    return sub_once(html, anchor, snippet + anchor, what)


# --------------------------------------------------------------- navigation

# The flow is Mission Control -> Drafting -> Final Delivery. Nav entries that
# point at the dropped Live Analysis screen are removed from every page.
NAV_DROP = ("live analysis", "analysis")

# The Start page is the entry point. The design ships no nav slot for it, so
# add_start_nav() clones one into every sidebar — without that the page is only
# reachable by URL, which is how it went missing the first time.
NAV_TARGETS = [
    ("start", "index.html"),
    ("mission control", "mission.html"),
    ("final delivery", "final.html"),
    ("drafting", "drafting.html"),
    ("mission", "mission.html"),
    ("draft", "drafting.html"),
    ("final", "final.html"),
]

ICON_WORDS = {
    "rocket_launch", "edit_note", "description", "chat", "history", "rocket",
    "query_stats", "history_edu", "task_alt", "analytics", "add", "open_in_new",
}


def link_text(inner: str) -> str:
    text = re.sub(r"<[^>]+>", " ", inner)
    words = [w for w in re.sub(r"\s+", " ", text).strip().split(" ") if w and w not in ICON_WORDS]
    return " ".join(words).lower()


def wire_nav(html: str, page_file: str) -> str:
    def anchor(m: re.Match) -> str:
        attrs, inner = m.group(1) or "", m.group(2)
        label = link_text(inner)
        if label in NAV_DROP:
            return ""
        if "new research" in label:
            return f'<a{attrs} data-ra="new-research">{inner}</a>'.replace('href="#"', 'href="index.html"')
        for key, target in NAV_TARGETS:
            if key in label:
                new_attrs = attrs.replace('href="#"', f'href="{target}"')
                if target == page_file:
                    new_attrs += ' aria-current="page"'
                return f"<a{new_attrs}>{inner}</a>"
        return f'<a{attrs} data-ra="unimplemented">{inner}</a>'

    # `<a([^>]*)>` also matches `<aside ...>` — the tag name has to end at a
    # space or `>`, or the whole sidebar gets wired up as a nav link.
    html = re.sub(r"<a(\s[^>]*)?>(.*?)</a>", lambda m: anchor(m), html, flags=re.S)

    def button(m: re.Match) -> str:
        attrs, inner = m.group(1), m.group(2)
        if "new research" in link_text(inner):
            return f'<button{attrs} data-ra="new-research">{inner}</button>'
        return m.group(0)

    return re.sub(r"<button([^>]*)>(.*?)</button>", button, html, flags=re.S)


# ------------------------------------------------------------- start entry

def _class_of(tag: str) -> str:
    m = re.search(r'class="([^"]*)"', tag)
    return m.group(1) if m else ""


PAGE_FILES = {"index.html", "mission.html", "drafting.html", "final.html"}

# The sidebar entries across all four designs share this padding; the top-bar
# links in the Drafting mockup do not, which is how the two are told apart.
SIDEBAR_MARKER = "px-4 py-3"


def _active_from(idle_cls: str) -> str:
    """Derive an active style when the mockup highlights nothing (Final Delivery).

    Keeps the layout classes and swaps only the colour/weight tokens.
    """
    cls = idle_cls.replace("text-on-surface-variant", "bg-primary-container text-on-primary-container")
    return cls if "font-bold" in cls else cls + " font-bold"


def add_start_nav(html: str, page_file: str) -> str:
    """Give the Start page a sidebar entry and mark the current page active.

    The designs each hard-code their own active item — and Final Delivery marks
    none — so without the second half every page lights up whichever entry its
    mockup happened to highlight. Runs after wire_nav(), so hrefs are resolved.
    """
    links = list(re.finditer(r"<a(\s[^>]*)>(.*?)</a>", html, re.S))
    side = [m for m in links if SIDEBAR_MARKER in _class_of(m.group(1))]
    if not side:
        raise BuildError("add_start_nav: no sidebar nav links found")

    idle_cls = next(
        (_class_of(m.group(1)) for m in side if "bg-primary-container" not in _class_of(m.group(1))),
        "",
    )
    if not idle_cls:
        raise BuildError("add_start_nav: could not read the idle nav style")
    active_cls = next(
        (_class_of(m.group(1)) for m in side if "bg-primary-container" in _class_of(m.group(1))),
        _active_from(idle_cls),
    )

    mission = next((m for m in side if "Mission Control" in m.group(2)), None)
    if mission is None:
        raise BuildError("add_start_nav: no Mission Control link to clone")

    # Clone the Mission Control entry so the new link inherits the sidebar's
    # exact markup, then swap its icon and label.
    inner = mission.group(2).replace("rocket_launch", "person").replace("Mission Control", "Start")
    cls = active_cls if page_file == "index.html" else idle_cls
    start_link = f'<a class="{cls}" href="index.html">{inner}</a>\n'

    def restyle(m: re.Match) -> str:
        attrs, body = m.group(1), m.group(2)
        if SIDEBAR_MARKER not in _class_of(attrs):
            return m.group(0)  # top-bar links keep their own idiom
        href = (re.search(r'href="([^"]*)"', attrs) or [None, ""])[1]
        if href not in PAGE_FILES:
            return m.group(0)  # New Research, AI Assistant, History
        new = active_cls if href == page_file else idle_cls
        return "<a" + re.sub(r'class="[^"]*"', f'class="{new}"', attrs, count=1) + f">{body}</a>"

    html = re.sub(r"<a(\s[^>]*)>(.*?)</a>", restyle, html, flags=re.S)

    anchor = re.search(r'<a\s[^>]*\bhref="mission\.html"[^>]*>.*?</a>', html, re.S)
    if anchor is None:
        raise BuildError("add_start_nav: no Mission Control link to insert before")
    return html[: anchor.start()] + start_link + html[anchor.start():]


# ------------------------------------------------------------ common shell

def common(html: str, page: str, title: str) -> str:
    html = re.sub(r"<title>.*?</title>", f"<title>{title}</title>", html, count=1, flags=re.S)
    html = re.sub(r"<body([^>]*)>", lambda m: f'<body{m.group(1)} data-page="{page}">', html, count=1)
    html = re.sub(r'(<img[^>]*src=")https://lh3\.googleusercontent\.com[^"]*(")', lambda m: m.group(1) + AVATAR + m.group(2), html)
    html = html.replace(
        "</head>",
        '<link href="assets/app.css" rel="stylesheet"/>\n<link href="assets/favicon.svg" rel="icon" type="image/svg+xml"/>\n</head>',
        1,
    )
    scripts = (
        '<script src="assets/config.js"></script>\n'
        '<script src="assets/store.js"></script>\n'
        '<script src="assets/engine.js"></script>\n'
        '<script src="assets/app.js"></script>\n'
    )
    if "</body>" not in html:
        raise BuildError("no </body> in source")
    return html.replace("</body>", scripts + "</body>", 1)


# --------------------------------------------------------------- per page

# The start screen ships its own field ids; only the action button needs one.
START_FIELDS = ("full-name", "institution", "contact", "education", "research-field")


def build_start(html: str) -> str:
    for field_id in START_FIELDS:
        if f'id="{field_id}"' not in html:
            raise BuildError(f"start screen: field id {field_id!r} missing")
    html = add_id(html, "Launch Agent", "button", "start-btn", 'type="button"')
    # This page hands off to the round wizard, which is what actually launches
    # the agent; two "Launch Agent" buttons would lie about which step starts it.
    return sub_once(
        html,
        '<span class="material-symbols-outlined" data-icon="rocket_launch">rocket_launch</span>\n'
        "                        Launch Agent\n                    </button>",
        '<span class="material-symbols-outlined" data-icon="arrow_forward">arrow_forward</span>\n'
        "                        Continue\n                    </button>",
        "start continue button",
    )


# Fields the proposal format needs but the mockup never shipped a control for.
# They are injected here rather than hand-edited into mission.html, so a rebuild
# from the Stitch screens keeps them.
FIELD_CLS = (
    "w-full bg-surface-container-high border border-outline/30 rounded-lg p-3 text-on-surface "
    "font-body-md focus:border-primary focus:ring-1 focus:ring-primary transition-all"
)
LABEL_CLS = "font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider"


# The rules behind a field — what a hidden evaluation is for, what the compute
# reference is — belong next to the field being filled in, not repeated into
# every proposal the agent writes. Each one hangs off a `?` beside the label.
def help_icon(text: str) -> str:
    return (
        '<span class="relative inline-block align-middle ml-1.5">'
        '<button type="button" class="ra-help w-4 h-4 rounded-full border border-outline/50 text-on-surface-variant '
        'hover:text-primary hover:border-primary text-[10px] leading-none transition-colors" '
        'aria-label="What is this?">?</button>'
        '<span class="ra-help-body hidden absolute left-0 top-6 z-30 w-80 normal-case tracking-normal '
        'bg-surface-container-high border border-outline/30 rounded-lg p-3 shadow-lg '
        f'font-body-md text-body-md text-on-surface-variant">{text}</span>'
        "</span>"
    )


def field(field_id: str, label: str, placeholder: str, help_text: str = "") -> str:
    """One labelled single-line field, in the design's own markup."""
    return (
        '<div class="flex flex-col gap-stack-sm">\n'
        f'<label class="{LABEL_CLS}" for="{field_id}">{label}{help_icon(help_text) if help_text else ""}</label>\n'
        f'<input class="{FIELD_CLS}" id="{field_id}" placeholder="{placeholder}" type="text">\n'
        "</div>\n"
    )


HIDDEN_EVAL_HELP = (
    "Scored once after submission, never shown to the agent — no scores, no outputs, no logs. "
    "The reward above is the only signal it may iterate against."
)

RUN_BUDGET_HELP = (
    "One run = one fixed candidate, from launch to a scoreable result. "
    "Reference budget: 8 H100-equivalent GPUs and 12 hours; over either, flag for resource review."
)


def build_mission(html: str) -> str:
    # The design left the top-bar brand slot empty on this screen only.
    html, n = re.subn(
        r'(<div class="font-display-lg text-display-lg font-bold text-primary tracking-tight">)\s*<br\s*/?>\s*(</div>)',
        r"\1ResearchAI\2",
        html,
    )
    if n != 1:
        raise BuildError(f"top-bar brand slot: expected 1 match, found {n}")

    # The proposal format states the research question before the baseline, so
    # the wizard asks in that order too: the mockup's Round 1 and Round 2 cards
    # trade places, and the step indicator's labels follow them.
    marks = ("<!-- Round 1: Baseline Setup -->", "<!-- Round 2: Core Inquiry -->", "<!-- Round 3: Evaluation Metrics -->")
    for mark in marks:
        if html.count(mark) != 1:
            raise BuildError(f"round swap: {mark!r} occurs {html.count(mark)} times")
    a, b, c = (html.index(m) for m in marks)
    baseline = html[a:b].replace("Round 1: Baseline Setup", "Round 2: Baseline Setup")
    inquiry = html[b:c].replace("Round 2: Core Inquiry", "Round 1: Core Inquiry")
    html = html[:a] + inquiry + baseline + html[c:]

    chip = '<span class="text-xs text-on-surface-variant font-label-sm uppercase tracking-wide">%s</span>'
    html = sub_once(html, chip % "Baseline", chip % "\0", "step chip: Baseline")
    html = sub_once(html, chip % "Inquiry", chip % "Baseline", "step chip: Inquiry")
    html = sub_once(html, chip % "\0", chip % "Inquiry", "step chip: swap")

    # Round 3: the baseline's own score under the metrics just named. It sits
    # above the reward because the reward is usually defined as a margin over it.
    metrics_grid_end = (
        f'<input class="{FIELD_CLS}" id="eval-metrics" placeholder="e.g., Accuracy, Pass@1" type="text">\n</div>\n</div>\n'
    )
    html = sub_once(
        html,
        metrics_grid_end,
        metrics_grid_end
        + field(
            "baseline-result",
            "Baseline Result",
            "Score the repository reports for the baseline, with its source; not yet reproduced",
        ),
        "round 3 baseline result field",
    )

    # Round 3 gains the fields the proposal format asks for and the mockup never
    # shipped: the hidden final evaluation — the held-out scoring pass the agent
    # never sees, which is what keeps the reward from being gamed — and the two
    # budget fields, which belong with the metrics because what they bound is
    # how long the agent may keep evaluating, not how the baseline is set up.
    reward_input = (
        f'<input class="{FIELD_CLS}" id="reward-def" '
        'placeholder="Define the success criteria or reward function..." type="text">\n</div>\n'
    )
    html = sub_once(
        html,
        reward_input,
        reward_input
        + field(
            "hidden-eval",
            "Hidden Final Evaluation (Optional)",
            "Held-out benchmark/split and metric the agent never sees during the task",
            HIDDEN_EVAL_HELP,
        )
        + field("total-time", "Total AutoResearch Time", "e.g., 72h — this is also the agent timeout")
        + field("early-stop", "Early-Stopping Signals (Optional)", "e.g., dev-loss plateau, 1k-step proxy run"),
        "round 3 evaluation fields",
    )

    # Round 2: what "one experiment run" costs is the box these numbers go in,
    # so the planning reference hangs off that box's title.
    run_cfg_title = '<h3 class="font-label-md text-label-md text-on-surface mb-4">Baseline Running Config</h3>'
    html = sub_once(
        html,
        run_cfg_title,
        run_cfg_title.replace("</h3>", help_icon(RUN_BUDGET_HELP) + "</h3>"),
        "baseline running config help",
    )

    # Round 4: the safeguards that go with whatever permissions are granted.
    html = insert_before(
        html,
        '<div class="bg-surface-container p-4 rounded-lg border border-outline/20 mt-2">',
        field(
            "safeguards",
            "Leakage &amp; Reward-Hacking Safeguards (Optional)",
            "Concrete safeguards for the access granted above",
        ),
        "round 4 safeguards field",
    )

    # Round 4 permission checkboxes get stable ids, in document order.
    ids = iter(["perm-web", "perm-services", "perm-newdata"])
    checkbox = '<input class="w-5 h-5 rounded border-outline/30 text-primary focus:ring-primary bg-surface-container-high" type="checkbox">'
    if html.count(checkbox) != 3:
        raise BuildError(f"expected 3 permission checkboxes, found {html.count(checkbox)}")
    html = re.sub(
        re.escape(checkbox),
        lambda m: m.group(0).replace("<input ", f'<input id="{next(ids)}" ', 1),
        html,
    )

    # Benchmark lists, metric lists, a reward definition, the hidden evaluation
    # and the safeguards are all multi-line answers — the single-line inputs the
    # design shipped hide most of a realistic one, so they become textareas
    # sized like the Round 2 box.
    for field_id, rows in (
        ("benchmarks", 3),
        ("eval-metrics", 3),
        ("baseline-result", 2),
        ("reward-def", 4),
        ("hidden-eval", 3),
        ("safeguards", 3),
    ):
        m = re.search(rf'<input class="([^"]*)" id="{field_id}" placeholder="([^"]*)" type="text">', html)
        if not m:
            raise BuildError(f"field {field_id}: single-line input not found")
        cls = m.group(1).replace("rounded-lg p-3", "rounded-lg p-4") + " resize-y"
        html = sub_once(
            html,
            m.group(0),
            f'<textarea class="{cls}" id="{field_id}" placeholder="{m.group(2)}" rows="{rows}"></textarea>',
            f"textarea field {field_id}",
        )

    # Step chips light up as rounds are filled in.
    chips = re.findall(r'<div class="w-8 h-8 rounded-full[^"]*">', html)
    if len(chips) != 5:
        raise BuildError(f"expected 5 step chips, found {len(chips)}")
    counter = iter(range(5))
    html = re.sub(
        r'<div class="w-8 h-8 rounded-full',
        lambda m: f'<div data-ra-step="{next(counter)}" class="w-8 h-8 rounded-full',
        html,
    )

    # Each round becomes one step of the wizard (app.js shows one at a time).
    rounds = [
        "Round 0: Foundation",
        "Round 1: Core Inquiry",
        "Round 2: Baseline Setup",
        "Round 3: Evaluation Metrics",
        "Round 4: Capabilities &amp; Governance",
    ]
    for i, heading in enumerate(rounds):
        # nth=-1: each round's heading text also appears in the section comment
        # right before its card, and that comment sits outside the card.
        html = add_id(html, heading, "div", f"round-{i}", f'data-ra-round="{i}"', nth=-1, must_contain="glass-card")

    html = add_id(html, "Launch Agent", "button", "launch-btn")
    html = insert_before(
        html,
        '<button id="launch-btn"',
        '<button id="example-btn" class="mr-3 border border-outline/40 text-on-surface-variant hover:text-primary '
        'hover:border-primary font-label-md text-label-md py-3 px-6 rounded-lg flex items-center gap-2 transition-all '
        'duration-200 active:scale-95" type="button">'
        '<span class="material-symbols-outlined">bolt</span>Load Example</button>\n',
        "example button",
    )
    return html


# The design's toolbar shipped three decorative buttons (Undo / Redo / Format).
# The draft is edited as a Markdown file, so that slot becomes the Source /
# Preview switch instead of three controls wired to nothing.
DRAFT_TOOLBAR = """<div class="flex items-center gap-2">
<button class="p-1.5 rounded hover:bg-surface-variant text-on-surface-variant transition-colors" title="Undo">
<span class="material-symbols-outlined text-sm">undo</span>
</button>
<button class="p-1.5 rounded hover:bg-surface-variant text-on-surface-variant transition-colors" title="Redo">
<span class="material-symbols-outlined text-sm">redo</span>
</button>
<div class="w-px h-4 bg-white/10 mx-1"></div>
<button class="p-1.5 rounded hover:bg-surface-variant text-on-surface-variant transition-colors" title="Format">
<span class="material-symbols-outlined text-sm">format_align_left</span>
</button>
</div>"""

MODE_SWITCH = """<div class="flex items-center gap-1 bg-surface-container-high/70 rounded-lg p-0.5" role="tablist">
<button id="mode-source" type="button" role="tab" class="px-3 py-1 rounded-md font-label-sm text-label-sm flex items-center gap-1.5 transition-colors">
<span class="material-symbols-outlined text-sm">code</span>Source</button>
<button id="mode-preview" type="button" role="tab" class="px-3 py-1 rounded-md font-label-sm text-label-sm flex items-center gap-1.5 transition-colors">
<span class="material-symbols-outlined text-sm">visibility</span>Preview</button>
</div>"""

# File actions live under the document, where the file itself is.
FILE_BAR = """<div class="h-14 border-t border-white/10 bg-surface-container/50 flex items-center justify-between gap-3 px-4">
<div class="flex items-center gap-2">
<input accept=".md,.markdown,.txt,text/markdown,text/plain" class="hidden" id="import-file" type="file">
<button class="px-3 py-2 rounded-lg border border-outline-variant text-on-surface hover:border-primary hover:text-primary transition-colors flex items-center gap-2 font-label-md text-label-md" id="import-md" type="button">
<span class="material-symbols-outlined text-sm">upload_file</span>Import .md</button>
<button class="px-3 py-2 rounded-lg border border-outline-variant text-on-surface hover:border-primary hover:text-primary transition-colors flex items-center gap-2 font-label-md text-label-md" id="save-md" type="button">
<span class="material-symbols-outlined text-sm">save</span>Save .md</button>
</div>
<span class="font-label-sm text-label-sm text-on-surface-variant" id="draft-stats"></span>
</div>"""


def build_drafting(html: str) -> str:
    # Live Analysis was dropped from the flow: three steps, not four.
    html = sub_once(html, "Step 3 of 4: Drafting &amp; Refinement", "Step 2 of 3: Drafting &amp; Refinement", "step label")
    html = add_id(html, "Quantum Computing Integration Proposal", "h1", "draft-title")

    # Toolbar: the version chip becomes the file name, "Live Sync" becomes the
    # save indicator, and the dead buttons become the Source / Preview switch.
    html = sub_once(
        html,
        '<span class="font-label-md text-label-md text-on-surface-variant">v1.2 (Auto-Generated)</span>',
        '<span class="font-label-md text-label-md text-on-surface-variant flex items-center gap-1.5" id="draft-filename">'
        '<span class="material-symbols-outlined text-sm">description</span>proposal.md</span>',
        "file name chip",
    )
    html, n = re.subn(r"(</span>)\s*Live Sync\s*(</span>)", r'\1<span id="draft-status">Live Sync</span>\2', html)
    if n != 1:
        raise BuildError(f"draft status chip: expected 1 match, found {n}")
    html = sub_once(html, DRAFT_TOOLBAR, MODE_SWITCH, "draft toolbar")

    # Document area holds both faces of the same file: the Markdown source
    # textarea and the rendered preview.
    html = sub_once(
        html,
        '<div class="flex-1 overflow-y-auto p-8 md:p-12 document-scroll">',
        '<div id="draft-pane" class="flex-1 overflow-y-auto p-8 md:p-12 document-scroll">\n'
        '<textarea id="draft-source" class="ra-source hidden max-w-3xl mx-auto w-full block bg-transparent border-0 '
        'resize-none focus:outline-none focus:ring-0 p-0" spellcheck="false" wrap="soft"></textarea>',
        "draft pane",
    )
    html = sub_once(
        html,
        '<div class="max-w-3xl mx-auto font-body-md text-body-md text-on-surface leading-relaxed space-y-6">',
        '<div id="draft-body" class="max-w-3xl mx-auto font-body-md text-body-md text-on-surface leading-relaxed space-y-6">',
        "draft body",
    )

    # Import / Save sit inside the document panel, below the scroll area.
    html, n = re.subn(
        r"</div>(\s*)</div>(\s*)<!-- Right Panel: Refinement Panel -->",
        lambda m: "</div>" + m.group(1) + FILE_BAR + "\n</div>" + m.group(2) + "<!-- Right Panel: Refinement Panel -->",
        html,
    )
    if n != 1:
        raise BuildError(f"file action bar: expected 1 insertion point, found {n}")

    html = add_id(html, "Export Draft", "button", "export-draft")
    html = add_id(html, "Approve &amp; Finalize", "button", "approve-btn")
    html = add_id(html, '"...significant latency bottlenecks', "div", "context-quote")
    html = add_id(html, "3 New", "span", "suggestion-count")
    html = sub_once(
        html,
        '<div class="flex-1 overflow-y-auto p-4 space-y-3">',
        '<div id="suggestion-list" class="flex-1 overflow-y-auto p-4 space-y-3">',
        "suggestion list",
    )
    return html


def build_final(html: str) -> str:
    html = add_id(html, "ID: RES-8924-B", "div", "final-id")
    html = add_id(html, "Strategic Analysis of Decentralized Finance Markets in Q4", "h1", "final-title")
    html = sub_once(
        html,
        '<span class="text-outline mr-2">AUTHOR:</span>ResearchAI System',
        '<span class="text-outline mr-2">AUTHOR:</span><span id="final-author">ResearchAI System</span>',
        "author",
    )
    html = sub_once(
        html,
        '<span class="text-outline mr-2">DATE:</span>Oct 24, 2023',
        '<span class="text-outline mr-2">DATE:</span><span id="final-date">Oct 24, 2023</span>',
        "date",
    )
    html = sub_once(
        html,
        '<span class="text-outline mr-2">STATUS:</span>Approved',
        '<span class="text-outline mr-2">STATUS:</span><span id="final-status">Approved</span>',
        "status",
    )
    # The executive summary is filled from the edited Markdown, which may be more
    # than one paragraph — a <p> cannot hold those, so the slot becomes a <div>.
    html = add_id(html, "This proposal outlines a comprehensive strategy", "p", "final-summary")
    html, n = re.subn(
        r'<p id="final-summary"([^>]*)>(.*?)</p>',
        lambda m: f'<div id="final-summary"{m.group(1)}>{m.group(2)}</div>',
        html,
        flags=re.S,
    )
    if n != 1:
        raise BuildError(f"final summary slot: expected 1 match, found {n}")
    html = sub_once(
        html,
        '<div class="space-y-6 font-body-md text-body-md text-on-surface-variant leading-relaxed">',
        '<div id="final-body" class="space-y-6 font-body-md text-body-md text-on-surface-variant leading-relaxed">',
        "final body",
    )
    html = sub_once(html, '<div class="space-y-4">', '<div id="final-sources" class="space-y-4">', "sources")

    html = add_id(html, "Export as PDF", "button", "export-pdf")
    html = add_id(html, "Export as DOCX", "button", "export-docx")
    html = add_id(html, "Send to Email", "button", "export-email")
    html = add_id(html, "zoom_out", "button", "zoom-out")
    html = add_id(html, "zoom_in", "button", "zoom-in")
    html = add_id(html, "100%", "span", "zoom-label")

    # A Markdown export sits alongside the designed export buttons.
    md_button = (
        '<button id="export-md" class="flex items-center gap-2 bg-transparent border border-outline px-6 py-3 '
        'rounded-lg text-on-surface font-label-md text-label-md hover:border-primary hover:text-primary '
        'transition-all active:scale-95" type="button">'
        '<span class="material-symbols-outlined">markdown</span>Export as Markdown</button>\n'
    )
    m = re.search(r"</button>(\s*)</div>(\s*)<!-- Document Preview Container -->", html)
    if not m:
        raise BuildError("could not find the end of the export toolbar")
    html = html[: m.start()] + "</button>\n" + md_button + "</div>\n<!-- Document Preview Container -->" + html[m.end():]

    html = sub_once(
        html,
        '<div class="bg-[#1a1d21] p-margin-mobile md:p-margin-desktop h-[600px] overflow-y-auto flex flex-col items-center">',
        '<div id="doc-scroll" class="bg-[#1a1d21] p-margin-mobile md:p-margin-desktop h-[600px] overflow-y-auto flex flex-col items-center">',
        "document scroll area",
    )
    pages = html.count('class="bg-surface-container w-full max-w-[800px]')
    if pages != 2:
        raise BuildError(f"expected 2 document pages, found {pages}")
    html = html.replace(
        'class="bg-surface-container w-full max-w-[800px]',
        'class="ra-page bg-surface-container w-full max-w-[800px]',
    )
    return html


BUILDERS = {
    "start": build_start,
    "mission": build_mission,
    "drafting": build_drafting,
    "final": build_final,
}


def main() -> int:
    for out_name, (screen_id, page, title) in PAGES.items():
        src = SRC / f"{screen_id}.html"
        if not src.exists():
            print(f"missing source: {src} (run build/sync_screens.py)", file=sys.stderr)
            return 1
        html = src.read_text(encoding="utf-8")
        try:
            html = BUILDERS[page](html)
            html = wire_nav(html, out_name)
            html = add_start_nav(html, out_name)
            html = common(html, page, title)
        except BuildError as e:
            print(f"[{out_name}] {e}", file=sys.stderr)
            return 1
        (ROOT / out_name).write_text(html, encoding="utf-8")
        print(f"built {out_name:<14} from screens/{screen_id[:8]}…  ({len(html):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
