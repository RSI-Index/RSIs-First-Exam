/* ResearchAI — page controllers.
 *
 * One file, three pages. `document.body.dataset.page` selects the controller.
 * The HTML itself is the Stitch design, untouched except for the ids and hooks
 * added by build/build.py.
 */
(function () {
  const RA = window.RA;
  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  /* --------------------------------------------------------------- utils */

  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  }

  function download(filename, content, mime) {
    const blob = new Blob([content], { type: mime || 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  function slug(s) {
    return String(s || 'proposal').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 60) || 'proposal';
  }

  function toast(msg) {
    let el = $('#ra-toast');
    if (!el) {
      el = document.createElement('div');
      el.id = 'ra-toast';
      el.className =
        'fixed bottom-24 md:bottom-8 left-1/2 -translate-x-1/2 z-[100] px-5 py-3 rounded-lg bg-surface-container-high border border-primary/30 text-on-surface font-label-md text-label-md shadow-2xl transition-opacity duration-300';
      document.body.appendChild(el);
    }
    el.textContent = msg;
    el.style.opacity = '1';
    clearTimeout(el._t);
    el._t = setTimeout(() => (el.style.opacity = '0'), 2600);
  }

  /* Everything downstream of Mission Control needs a mission. */
  function requireMission() {
    const cfg = RA.store.loadMission();
    if (!cfg) {
      toast('No mission found — redirecting to Mission Control.');
      setTimeout(() => (location.href = 'mission.html'), 900);
      return null;
    }
    return cfg;
  }

  function proposalFor(cfg) {
    let p = RA.store.loadProposal();
    if (!p) {
      /* The profile is stored separately from the mission, but it is what
         names the author in the proposal's Task Metadata table. */
      p = RA.engine.compose(Object.assign({}, cfg, { profile: RA.store.loadProfile() }));
      RA.store.saveProposal(p);
    }
    return p;
  }

  /* Optional backend seam — see assets/config.js. */
  async function composeRemote(cfg) {
    const base = window.RA_API_BASE;
    if (!base) return null;
    try {
      const res = await fetch(base.replace(/\/$/, '') + '/api/compose', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(cfg),
      });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      return await res.json();
    } catch (e) {
      console.warn('[ResearchAI] backend unavailable, using local engine:', e.message);
      return null;
    }
  }

  /* ------------------------------------------------------------ chrome */

  function wireChrome() {
    $$('[data-ra="new-research"]').forEach((el) =>
      el.addEventListener('click', (e) => {
        e.preventDefault();
        RA.store.clear();
        location.href = 'index.html';
      })
    );
    /* Nav entries the demo does not implement should say so rather than dead-link. */
    $$('[data-ra="unimplemented"]').forEach((el) =>
      el.addEventListener('click', (e) => {
        e.preventDefault();
        toast('Not part of this demo — Mission → Draft → Final is the full flow.');
      })
    );
  }

  /* --------------------------------------------------------- page: start */

  /* The researcher profile. Optional — it names the proposal's author on Final
     Delivery; the mission itself is defined on the next page. */
  const PROFILE_FIELDS = {
    fullName: 'full-name',
    institution: 'institution',
    contact: 'contact',
    education: 'education',
    researchField: 'research-field',
  };

  function pageStart() {
    const saved = RA.store.loadProfile();
    Object.entries(PROFILE_FIELDS).forEach(([k, id]) => {
      const el = document.getElementById(id);
      if (el && saved && saved[k]) el.value = saved[k];
    });

    const save = () => {
      const p = {};
      Object.entries(PROFILE_FIELDS).forEach(([k, id]) => {
        const el = document.getElementById(id);
        p[k] = el ? el.value.trim() : '';
      });
      RA.store.saveProfile(p);
      return p;
    };

    /* Keep what has been typed even if the visitor navigates away. */
    Object.values(PROFILE_FIELDS).forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('change', save);
    });

    /* The profile is the first step; the round wizard is what launches the agent. */
    const start = $('#start-btn');
    if (start)
      start.addEventListener('click', (e) => {
        e.preventDefault();
        save();
        location.href = 'mission.html';
      });
  }

  /* ------------------------------------------------------- page: mission */

  const FIELDS = {
    githubUrl: 'github-url',
    commitHash: 'commit-hash',
    paperLink: 'paper-link',
    baselineName: 'baseline-name',
    trainScript: 'train-script',
    trainData: 'train-data',
    evalScript: 'eval-script',
    evalData: 'eval-data',
    estTime: 'est-time',
    gpuCount: 'gpu-count',
    gpuModel: 'gpu-model',
    researchQuestion: 'research-question',
    benchmarks: 'benchmarks',
    evalMetrics: 'eval-metrics',
    baselineResult: 'baseline-result',
    rewardDef: 'reward-def',
    hiddenEval: 'hidden-eval',
    totalTime: 'total-time',
    earlyStop: 'early-stop',
    safeguards: 'safeguards',
  };
  const CHECKS = { permWeb: 'perm-web', permServices: 'perm-services', permNewData: 'perm-newdata' };

  function readForm() {
    const cfg = {};
    Object.entries(FIELDS).forEach(([k, id]) => {
      const el = document.getElementById(id);
      cfg[k] = el ? el.value.trim() : '';
    });
    Object.entries(CHECKS).forEach(([k, id]) => {
      const el = document.getElementById(id);
      cfg[k] = el ? el.checked : false;
    });
    return cfg;
  }

  function fillForm(cfg) {
    Object.entries(FIELDS).forEach(([k, id]) => {
      const el = document.getElementById(id);
      if (el && cfg[k] != null) el.value = cfg[k];
    });
    Object.entries(CHECKS).forEach(([k, id]) => {
      const el = document.getElementById(id);
      if (el) el.checked = Boolean(cfg[k]);
    });
  }

  /* Mission Control is a five-round wizard: one round on screen at a time,
     the step chips double as its navigation, and Launch Agent (which lives
     inside the Round 4 card in the design) only surfaces on the last round. */
  const ROUND_FIELDS = [
    ['github-url', 'commit-hash', 'paper-link'],
    ['research-question'],
    ['baseline-name', 'train-script', 'train-data', 'eval-script', 'eval-data', 'est-time', 'gpu-count', 'gpu-model'],
    ['benchmarks', 'eval-metrics', 'baseline-result', 'reward-def', 'hidden-eval', 'total-time', 'early-stop'],
    ['perm-web', 'perm-services', 'perm-newdata', 'safeguards'],
  ];
  const ROUND_NAMES = ['Foundation', 'Inquiry', 'Baseline', 'Metrics', 'Governance'];

  function roundFilled(i) {
    return ROUND_FIELDS[i].some((id) => {
      const el = document.getElementById(id);
      if (!el) return false;
      return el.type === 'checkbox' ? el.checked : el.value.trim() !== '';
    });
  }

  /* The `?` beside a field label: click opens that note, clicking anywhere else
     (or Escape) closes it. One open at a time — two overlapping panels in a
     narrow card would cover the field the reader is asking about. */
  function wireHelp() {
    const panels = $$('.ra-help-body');
    if (!panels.length) return;
    const closeAll = () => panels.forEach((p) => p.classList.add('hidden'));
    $$('.ra-help').forEach((btn) => {
      const panel = btn.nextElementSibling;
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        const wasOpen = !panel.classList.contains('hidden');
        closeAll();
        if (!wasOpen) panel.classList.remove('hidden');
      });
    });
    panels.forEach((p) => p.addEventListener('click', (e) => e.stopPropagation()));
    document.addEventListener('click', closeAll);
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') closeAll();
    });
  }

  function wireRounds() {
    const rounds = $$('[data-ra-round]');
    if (!rounds.length) return () => {};
    const chips = $$('[data-ra-step]');
    const stack = rounds[0].parentElement;

    const btn = 'font-label-md text-label-md py-3 px-6 rounded-lg flex items-center gap-2 transition-all duration-200 active:scale-95';
    const footer = document.createElement('div');
    footer.className = 'flex items-center justify-between gap-4 mt-2';
    footer.innerHTML =
      `<button id="round-back" type="button" class="${btn} border border-outline/40 text-on-surface-variant hover:text-primary hover:border-primary">` +
      '<span class="material-symbols-outlined">arrow_back</span>Back</button>' +
      '<div id="round-progress" class="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider text-center"></div>' +
      '<div id="round-actions" class="flex items-center gap-3">' +
      `<button id="round-next" type="button" class="${btn} bg-primary-container text-on-primary-container font-bold hover:bg-primary hover:text-on-primary">` +
      'Next<span class="material-symbols-outlined">arrow_forward</span></button></div>';
    stack.after(footer);

    const back = $('#round-back', footer);
    const next = $('#round-next', footer);
    const progress = $('#round-progress', footer);
    /* Load Example lives in the last round's card in the design; the wizard
       needs it reachable from round 0. */
    const example = $('#example-btn');
    if (example) {
      example.classList.remove('mr-3');
      $('#round-actions', footer).prepend(example);
    }

    let cur = 0;

    function refresh() {
      rounds.forEach((el, n) => el.classList.toggle('hidden', n !== cur));
      chips.forEach((chip, i) => {
        const done = roundFilled(i);
        const here = i === cur;
        chip.className =
          'w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm cursor-pointer transition-all duration-200 ' +
          (here
            ? 'bg-primary text-on-primary ring-4 ring-primary/25 scale-110'
            : done
            ? 'bg-primary/70 text-on-primary'
            : 'bg-surface-container-high border border-outline/30 text-on-surface hover:border-primary');
        const label = chip.parentElement && chip.parentElement.querySelector('span');
        if (label)
          label.className =
            'text-xs font-label-sm uppercase tracking-wide cursor-pointer ' +
            (here ? 'text-primary font-bold' : done ? 'text-primary/70' : 'text-on-surface-variant');
      });
      back.classList.toggle('invisible', cur === 0);
      next.classList.toggle('hidden', cur === rounds.length - 1);
      progress.textContent = `Round ${cur} · ${ROUND_NAMES[cur]} — ${cur + 1} of ${rounds.length}`;
    }

    function go(i, scroll) {
      cur = Math.max(0, Math.min(rounds.length - 1, i));
      refresh();
      if (scroll !== false) window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    back.addEventListener('click', () => go(cur - 1));
    next.addEventListener('click', () => go(cur + 1));
    chips.forEach((chip, i) => {
      const target = chip.parentElement || chip;
      target.style.cursor = 'pointer';
      target.addEventListener('click', () => go(i));
    });

    document.addEventListener('input', refresh);
    document.addEventListener('change', refresh);
    refresh();
    return { refresh, go: (i) => go(i, false) };
  }

  function pageMission() {
    /* Mission Control opens empty. A previous mission stays in storage for
       Drafting and Final Delivery to read, but it is not poured back into the
       form — starting a new task should not mean clearing someone else's. */
    wireRounds();
    wireHelp();

    const example = $('#example-btn');
    if (example)
      example.addEventListener('click', (e) => {
        e.preventDefault();
        fillForm(RA.EXAMPLE);
        document.dispatchEvent(new Event('input'));
        toast('Example mission loaded — step through the rounds, then Launch.');
      });

    const launch = $('#launch-btn');
    if (launch)
      launch.addEventListener('click', async (e) => {
        e.preventDefault();
        const cfg = readForm();
        if (!cfg.researchQuestion && !cfg.baselineName && !cfg.githubUrl) {
          toast('Give the agent something to work with: repo, baseline, or research question.');
          const rq = $('#research-question');
          if (rq) rq.focus();
          return;
        }
        RA.store.saveProposal(null);
        RA.store.saveDraft(null);
        RA.store.saveMission(cfg);
        launch.disabled = true;
        launch.classList.add('opacity-60');
        const remote = await composeRemote(cfg);
        if (remote) RA.store.saveProposal(remote);
        location.href = 'drafting.html';
      });
  }

  /* ------------------------------------------------------ page: drafting */

  function suggestionCard(g) {
    const tone = { emerald: 'text-emerald-400', amber: 'text-amber-400', tertiary: 'text-tertiary' }[g.tone] || 'text-primary';
    return `
      <div class="bg-surface-container hover:bg-surface-container-high transition-colors p-4 rounded-lg border border-white/5 cursor-pointer group">
        <div class="flex justify-between items-start mb-2">
          <div class="flex items-center gap-2 ${tone}">
            <span class="material-symbols-outlined text-sm">${esc(g.icon)}</span>
            <span class="text-xs font-label-sm uppercase">${esc(g.kind)}</span>
          </div>
          <span class="bg-primary/10 text-primary text-[10px] px-1.5 py-0.5 rounded font-label-sm">${esc(g.impact)}</span>
        </div>
        <p class="text-sm text-on-surface mb-3">${esc(g.text)}</p>
        <div class="flex justify-end opacity-0 group-hover:opacity-100 transition-opacity">
          <button class="text-xs text-primary hover:text-primary-fixed-dim font-medium flex items-center gap-1" data-ra="fix">
            ${esc(g.action)} <span class="material-symbols-outlined text-[12px]">arrow_forward</span>
          </button>
        </div>
      </div>`;
  }

  /* The draft is a Markdown file. `#draft-source` is the file, `#draft-body` is
     its rendered preview, and localStorage is where the file lives between
     visits — Import/Save move it to and from disk. */
  function pageDrafting() {
    const cfg = requireMission();
    if (!cfg) return;
    const p = proposalFor(cfg);

    const src = $('#draft-source');
    const body = $('#draft-body');
    const titleEl = $('#draft-title');
    const statusEl = $('#draft-status');
    const nameEl = $('#draft-filename');
    const statsEl = $('#draft-stats');

    const saved = RA.store.loadDraft();
    const startMd = saved && typeof saved.md === 'string' ? saved.md : RA.engine.draftMarkdown(p);
    const current = () => (src ? src.value : startMd);
    const fileName = () => slug(RA.engine.parseDraft(current()).title || p.title) + '.md';

    function status(text, muted) {
      if (!statusEl) return;
      statusEl.textContent = text;
      statusEl.parentElement.classList.toggle('text-primary', !muted);
      statusEl.parentElement.classList.toggle('text-on-surface-variant', Boolean(muted));
    }

    /* The textarea grows to its content so the document panel does the
       scrolling — one scrollbar in both modes, and the same scroll position. */
    function autogrow() {
      if (!src) return;
      src.style.height = 'auto';
      src.style.height = src.scrollHeight + 'px';
    }

    function reflect() {
      const md = current();
      const doc = RA.engine.parseDraft(md);
      if (titleEl) titleEl.textContent = doc.title || p.title;
      if (nameEl && nameEl.lastChild && nameEl.lastChild.nodeType === 3) nameEl.lastChild.textContent = fileName();
      if (statsEl) {
        const words = md.trim() ? md.trim().split(/\s+/).length : 0;
        statsEl.textContent = `${md.split('\n').length} lines · ${words} words`;
      }
      const quote = $('#context-quote');
      if (quote) quote.textContent = '"' + (doc.summary || p.summary).slice(0, 180) + '…"';
    }

    function renderPreview() {
      if (body) body.innerHTML = RA.engine.mdToHtml(current(), { skipTitle: true });
    }

    /* --- source / preview switch ------------------------------------- */
    const onCls = 'bg-primary text-on-primary font-bold';
    const offCls = 'text-on-surface-variant hover:text-primary';
    const btnSource = $('#mode-source');
    const btnPreview = $('#mode-preview');
    let mode = 'source';

    function setMode(next) {
      mode = next;
      const isSrc = mode === 'source';
      if (src) src.classList.toggle('hidden', !isSrc);
      if (body) body.classList.toggle('hidden', isSrc);
      [[btnSource, isSrc], [btnPreview, !isSrc]].forEach(([b, on]) => {
        if (!b) return;
        b.className = 'px-3 py-1 rounded-md font-label-sm text-label-sm flex items-center gap-1.5 transition-colors ' + (on ? onCls : offCls);
        b.setAttribute('aria-selected', String(on));
      });
      if (isSrc) autogrow();
      else renderPreview();
    }

    if (btnSource) btnSource.addEventListener('click', () => setMode('source'));
    if (btnPreview) btnPreview.addEventListener('click', () => setMode('preview'));

    /* --- the file ---------------------------------------------------- */
    function load(md, note) {
      if (!src) return;
      src.value = md;
      RA.store.saveDraft({ md });
      reflect();
      if (mode === 'source') autogrow();
      else renderPreview();
      status('Saved');
      if (note) toast(note);
    }

    if (src) {
      src.value = startMd;
      let t;
      src.addEventListener('input', () => {
        autogrow();
        reflect();
        status('Editing…', true);
        clearTimeout(t);
        t = setTimeout(() => {
          RA.store.saveDraft({ md: src.value });
          status('Saved');
        }, 400);
      });
      /* Tab indents instead of leaving the editor. */
      src.addEventListener('keydown', (e) => {
        if (e.key !== 'Tab' || e.metaKey || e.ctrlKey) return;
        e.preventDefault();
        const a = src.selectionStart;
        const b = src.selectionEnd;
        src.value = src.value.slice(0, a) + '  ' + src.value.slice(b);
        src.selectionStart = src.selectionEnd = a + 2;
        src.dispatchEvent(new Event('input'));
      });
    }

    /* Render both faces up front so the preview never holds the design mockup. */
    renderPreview();
    setMode('source');
    reflect();
    status('Saved');

    const picker = $('#import-file');
    const importBtn = $('#import-md');
    if (importBtn && picker) {
      importBtn.addEventListener('click', () => picker.click());
      picker.addEventListener('change', () => {
        const file = picker.files && picker.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = () => load(String(reader.result), `Imported ${file.name}.`);
        reader.onerror = () => toast('Could not read that file.');
        reader.readAsText(file);
        picker.value = '';
      });
    }

    const saveBtn = $('#save-md');
    if (saveBtn)
      saveBtn.addEventListener('click', () => {
        download(fileName(), current(), 'text/markdown;charset=utf-8');
        toast('Saved ' + fileName() + ' to your downloads.');
      });

    /* Export Draft ships the same file plus the metadata and source list the
       editor does not own. */
    const exportBtn = $('#export-draft');
    if (exportBtn)
      exportBtn.addEventListener('click', () => {
        download(slug(p.title) + '-draft.md', RA.engine.toMarkdown(p, current()), 'text/markdown;charset=utf-8');
        toast('Draft exported as Markdown.');
      });

    /* --- suggestions -------------------------------------------------- */
    const sugg = $('#suggestion-list');
    if (sugg) {
      sugg.innerHTML = p.gaps.map(suggestionCard).join('\n');
      const count = $('#suggestion-count');
      if (count) count.textContent = p.gaps.length + ' New';
      $$('[data-ra="fix"]', sugg).forEach((btn) =>
        btn.addEventListener('click', () => toast('Open Mission Control to fill this in, then relaunch the agent.'))
      );
    }

    const approve = $('#approve-btn');
    if (approve)
      approve.addEventListener('click', () => {
        RA.store.saveDraft({ md: current() });
        location.href = 'final.html';
      });
  }

  /* --------------------------------------------------------- page: final */

  function sourceItem(s) {
    return `
      <div class="flex gap-4 p-4 rounded-lg hover:bg-surface-container-high transition-colors border border-transparent hover:border-white/5 group">
        <div class="mt-1"><span class="material-symbols-outlined text-outline group-hover:text-primary transition-colors text-xl">article</span></div>
        <div class="flex-1">
          <h4 class="font-body-md text-body-md text-on-surface font-medium">${esc(s.title)}</h4>
          <p class="font-label-sm text-label-sm text-on-surface-variant mt-1 break-all">${esc(s.meta)}</p>
        </div>
        ${
          s.url
            ? `<a class="self-center font-label-sm text-label-sm text-primary flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity" href="${esc(
                s.url
              )}" target="_blank" rel="noopener">View Source <span class="material-symbols-outlined text-sm">open_in_new</span></a>`
            : ''
        }
      </div>`;
  }

  function pageFinal() {
    const cfg = requireMission();
    if (!cfg) return;
    const p = proposalFor(cfg);
    /* Final Delivery renders the Markdown file the Drafting page edited. */
    const draft = RA.store.loadDraft();
    const md = draft && typeof draft.md === 'string' && draft.md.trim() ? draft.md : RA.engine.draftMarkdown(p);
    const doc = RA.engine.parseDraft(md);
    const title = doc.title || p.title;

    const set = (id, text) => {
      const el = document.getElementById(id);
      if (el) el.textContent = text;
    };
    set('final-id', 'ID: ' + p.id);
    set('final-title', title);
    set('final-date', p.date);
    /* The start page's profile, when filled in, names the author. */
    const profile = RA.store.loadProfile() || {};
    const author = [profile.fullName, profile.institution].filter(Boolean).join(' · ') || p.author;
    set('final-author', author);
    set('final-status', p.status);

    const summaryEl = $('#final-summary');
    if (summaryEl) summaryEl.innerHTML = RA.engine.mdToHtml(doc.summaryMd || p.summary, { variant: 'final' });

    const bodyEl = $('#final-body');
    if (bodyEl) bodyEl.innerHTML = RA.engine.mdToHtml(doc.bodyMd, { variant: 'final' });

    const sources = $('#final-sources');
    if (sources) sources.innerHTML = p.sources.map(sourceItem).join('\n');

    document.title = 'ResearchAI — ' + title;

    /* Printing (and "Export as PDF") renders a clone of the document pages
       only — see the @media print block in app.css. Wired to the browser's own
       print events so Cmd+P behaves the same as the button. */
    function buildPrintRoot() {
      let root = document.getElementById('ra-print-root');
      if (!root) {
        root = document.createElement('div');
        root.id = 'ra-print-root';
        document.body.appendChild(root);
      }
      const src = $('#doc-scroll');
      root.innerHTML = src ? src.innerHTML : '';
      $$('.ra-page', root).forEach((el) => (el.style.zoom = ''));
      document.body.classList.add('ra-printing');
    }
    window.addEventListener('beforeprint', buildPrintRoot);
    window.addEventListener('afterprint', () => document.body.classList.remove('ra-printing'));

    /* Exports */
    const pdf = $('#export-pdf');
    if (pdf)
      pdf.addEventListener('click', () => {
        toast('Opening the print dialog — choose "Save as PDF".');
        setTimeout(() => window.print(), 400);
      });

    const docx = $('#export-docx');
    if (docx)
      docx.addEventListener('click', () => {
        const pages = $('#doc-scroll');
        const html =
          '<html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:w="urn:schemas-microsoft-com:office:word" xmlns="http://www.w3.org/TR/REC-html40">' +
          '<head><meta charset="utf-8"><title>' + esc(title) + '</title>' +
          '<style>body{font-family:Calibri,Arial,sans-serif;line-height:1.5;color:#111}h1{font-size:20pt}h3{font-size:13pt;color:#1a4b9c}code{font-family:Consolas,monospace}</style></head><body>' +
          '<h1>' + esc(title) + '</h1>' +
          '<p><b>ID:</b> ' + esc(p.id) + ' &nbsp; <b>Author:</b> ' + esc(author) + ' &nbsp; <b>Date:</b> ' + esc(p.date) + '</p>' +
          (pages ? pages.innerHTML : '') +
          '</body></html>';
        download(slug(title) + '.doc', html, 'application/msword');
        toast('Exported as a Word document.');
      });

    const mail = $('#export-email');
    if (mail)
      mail.addEventListener('click', () => {
        const text = RA.engine.toMarkdown(p, md).slice(0, 1600);
        location.href = 'mailto:?subject=' + encodeURIComponent('[ResearchAI] ' + title) + '&body=' + encodeURIComponent(text);
      });

    const mdBtn = $('#export-md');
    if (mdBtn)
      mdBtn.addEventListener('click', () => {
        download(slug(title) + '.md', RA.engine.toMarkdown(p, md), 'text/markdown;charset=utf-8');
        toast('Exported as Markdown.');
      });

    /* Zoom */
    let zoom = 100;
    const applyZoom = () => {
      /* `zoom` (not `transform`) so the pages keep reflowing inside the scroll area. */
      $$('.ra-page').forEach((el) => (el.style.zoom = String(zoom / 100)));
      set('zoom-label', zoom + '%');
    };
    const zin = $('#zoom-in');
    const zout = $('#zoom-out');
    if (zin) zin.addEventListener('click', () => { zoom = Math.min(160, zoom + 10); applyZoom(); });
    if (zout) zout.addEventListener('click', () => { zoom = Math.max(60, zoom - 10); applyZoom(); });
  }

  /* ----------------------------------------------------------- dispatch */

  document.addEventListener('DOMContentLoaded', function () {
    wireChrome();
    const page = document.body.dataset.page;
    ({ start: pageStart, mission: pageMission, drafting: pageDrafting, final: pageFinal }[page] || function () {})();
  });
})();
