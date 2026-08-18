/* ResearchAI — proposal engine.
 *
 * Pure functions, no DOM. Everything runs in the browser: `compose()` turns a
 * mission config into the proposal document that the Drafting and Final
 * Delivery pages render, and `gaps()` reports what the config is still missing.
 *
 * Backend seam: if `window.RA_API_BASE` is set (see assets/config.js), the
 * app POSTs the mission config to `${RA_API_BASE}/api/compose` and uses the
 * returned proposal instead of the local composer. Nothing else changes.
 */
window.RA = window.RA || {};

RA.engine = (function () {
  const NA = '— not specified —';
  const val = (v, fallback) => (v && String(v).trim() ? String(v).trim() : fallback || NA);
  const has = (v) => Boolean(v && String(v).trim());

  function shortRepo(url) {
    if (!has(url)) return NA;
    const m = String(url).match(/github\.com\/([^/]+\/[^/?#]+)/i);
    return m ? m[1] : url;
  }

  /* Title = the research question, trimmed at a clause boundary so it stays
     grammatical instead of ending mid-phrase. */
  function titleFrom(cfg) {
    const raw = (cfg.researchQuestion || '').trim().replace(/\s+/g, ' ');
    if (raw) {
      const isQuestion = /\?\s*$/.test(raw) || /^(does|do|can|is|are|will|how|why|what|which|should)\b/i.test(raw);
      let t = raw.replace(/\?+\s*$/, '');
      if (t.length > 92) {
        const marks = [', ', ' without ', ' while ', ' when ', ' with ', ' under ', ' and ', ' on '];
        let cut = -1;
        marks.forEach((m) => {
          const i = t.toLowerCase().lastIndexOf(m, 92);
          if (i > 28 && i > cut) cut = i;
        });
        t = cut > 0 ? t.slice(0, cut) : t.slice(0, 89).replace(/\s\S*$/, '');
      }
      t = t.charAt(0).toUpperCase() + t.slice(1);
      return isQuestion ? t + '?' : t;
    }
    if (has(cfg.baselineName)) return `Research Proposal for ${cfg.baselineName}`;
    return 'Automated Research Proposal';
  }

  function missionId(cfg) {
    const seed = JSON.stringify(cfg);
    let h = 0;
    for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) >>> 0;
    const num = 1000 + (h % 9000);
    const letter = 'ABCDEFGH'[h % 8];
    return `RES-${num}-${letter}`;
  }

  function permissionList(cfg) {
    const on = [];
    const off = [];
    (cfg.permWeb ? on : off).push('web access');
    (cfg.permServices ? on : off).push('external services');
    (cfg.permNewData ? on : off).push('newly collected or generated data');
    return { on, off };
  }

  function computeBudget(cfg) {
    const n = parseInt(cfg.gpuCount, 10);
    const hours = parseFloat(String(cfg.estTime || '').replace(/[^0-9.]/g, ''));
    if (!n || !hours) return null;
    return { gpuHours: Math.round(n * hours), n, hours, model: val(cfg.gpuModel, 'unspecified GPUs') };
  }

  /* ---------------------------------------------------------------- gaps */
  /* Missing-field analysis. Drives the Drafting page suggestion cards. */
  function gaps(cfg) {
    const out = [];
    if (!has(cfg.commitHash)) {
      out.push({
        kind: 'Reproducibility',
        icon: 'commit',
        tone: 'amber',
        impact: 'High Impact',
        text: 'No commit hash pinned. A proposal that names a repository without a commit cannot be replayed later — pin the exact commit used for the baseline.',
        action: 'Pin Commit',
      });
    }
    if (!has(cfg.evalData) || !has(cfg.evalScript)) {
      out.push({
        kind: 'Evaluation',
        icon: 'add_chart',
        tone: 'emerald',
        impact: 'High Impact',
        text: 'The evaluation path is incomplete (missing eval script or eval data). Reviewers will ask how the reward definition is measured end to end.',
        action: 'Complete Eval Path',
      });
    }
    if (!computeBudget(cfg)) {
      out.push({
        kind: 'Compute',
        icon: 'memory',
        tone: 'amber',
        impact: 'Medium Impact',
        text: 'GPU count and estimated wall-clock time are not both set, so the proposal cannot state a compute budget in GPU-hours.',
        action: 'Estimate Budget',
      });
    }
    if (!has(cfg.rewardDef)) {
      out.push({
        kind: 'Success Criteria',
        icon: 'flag',
        tone: 'tertiary',
        impact: 'High Impact',
        text: 'No final reward definition. Without a threshold, the run has no falsifiable outcome — state the metric and the margin that counts as success.',
        action: 'Define Reward',
      });
    }
    if (!has(cfg.benchmarks)) {
      out.push({
        kind: 'Benchmarks',
        icon: 'insert_chart',
        tone: 'tertiary',
        impact: 'Medium Impact',
        text: 'No benchmarks listed. Name at least one held-out benchmark so the claim is testable outside the training distribution.',
        action: 'Add Benchmarks',
      });
    }
    if (!has(cfg.baselineResult)) {
      out.push({
        kind: 'Baseline',
        icon: 'query_stats',
        tone: 'emerald',
        impact: 'High Impact',
        text: 'No baseline result. The reward is a margin over the baseline, so without the number the repository reports there is nothing to measure that margin from.',
        action: 'Add Baseline Result',
      });
    }
    if (!has(cfg.hiddenEval)) {
      out.push({
        kind: 'Evaluation',
        icon: 'visibility_off',
        tone: 'emerald',
        impact: 'High Impact',
        text: 'No hidden final evaluation. Scoring the agent on the same signal it optimizes cannot separate a real gain from adaptive overfitting — name a held-out split it never sees.',
        action: 'Add Hidden Eval',
      });
    }
    if (!has(cfg.totalTime)) {
      out.push({
        kind: 'Compute',
        icon: 'timer',
        tone: 'amber',
        impact: 'Medium Impact',
        text: 'No total AutoResearch time. The agent timeout is this budget, so without it the run has no wall-clock bound across rounds.',
        action: 'Set Total Time',
      });
    }
    if (!has(cfg.safeguards) && (cfg.permWeb || cfg.permServices || cfg.permNewData)) {
      out.push({
        kind: 'Governance',
        icon: 'shield',
        tone: 'amber',
        impact: 'High Impact',
        text: 'Capabilities are granted but no safeguards are stated. Say concretely how leakage and reward hacking are prevented on each path that is open.',
        action: 'Add Safeguards',
      });
    }
    if (cfg.permNewData && !cfg.permWeb) {
      out.push({
        kind: 'Governance',
        icon: 'gavel',
        tone: 'amber',
        impact: 'Medium Impact',
        text: 'New data collection is permitted but web access is not. State where the new data comes from, or the governance section will read as contradictory.',
        action: 'Clarify Sourcing',
      });
    }
    if (!out.length) {
      out.push({
        kind: 'Style',
        icon: 'format_paint',
        tone: 'tertiary',
        impact: 'Low Impact',
        text: 'The configuration is complete. Consider tightening the Research Question into a single falsifiable hypothesis sentence before finalizing.',
        action: 'Auto-Revise',
      });
    }
    return out;
  }

  /* --------------------------------------------------------------- tables */
  /* The proposal is a filled-in copy of proposal_format_v2.txt, so most of a
     section is one `| Field | Value |` table. A value is written by the
     contributor, so it may contain pipes or newlines — both would tear the row
     apart, and neither is worth losing the field over. */
  const TODO = '*TODO: left blank at the proposal stage*';

  function cell(v, fallback) {
    return val(v, fallback).replace(/\s*\n\s*/g, ' ').replace(/\|/g, '\\|');
  }

  function table(rows) {
    return ['| Field | Value |', '| --- | --- |']
      .concat(rows.map(([field, value]) => `| ${field} | ${value} |`))
      .join('\n');
  }

  /* ------------------------------------------------------------- compose */
  function compose(cfg) {
    const repo = shortRepo(cfg.githubUrl);
    const budget = computeBudget(cfg);
    const perms = permissionList(cfg);
    const title = titleFrom(cfg);
    const baseline = val(cfg.baselineName, 'the baseline named in the repository');

    /* The author line is the Start page's Basic Information: `Name (email)`,
       and just the name when no contact was given. */
    const profile = cfg.profile || {};

    const summary =
      `This proposal specifies a single-variable study against ${baseline}, ` +
      (has(cfg.researchQuestion)
        ? `testing one question: “${String(cfg.researchQuestion).trim()}” `
        : 'with the research question still to be fixed. ') +
      (budget ? `The full comparison costs approximately ${budget.gpuHours} GPU-hours per arm on ${budget.n}× ${budget.model}. ` : 'The compute budget is not yet bounded. ') +
      `Success is defined as ${has(cfg.rewardDef) ? String(cfg.rewardDef).replace(/\.$/, '') : 'an improvement on the benchmarks under Evaluation, threshold to be fixed before launch'}.`;

    /* Over the planning reference, the proposal has to say so itself rather
       than leave a reviewer to multiply the numbers out. */
    const gpuN = parseInt(cfg.gpuCount, 10);
    const runHours = parseFloat(String(cfg.estTime || '').replace(/[^0-9.]/g, ''));
    const overBudget = (gpuN > 8 ? ['more than 8 GPUs'] : []).concat(runHours > 12 ? ['more than 12 hours per run'] : []);

    const sections = [
      {
        heading: 'Task Metadata',
        paragraphs: [
          table([
            ['Author', profile.fullName ? cell(profile.contact ? `${profile.fullName} (${profile.contact})` : profile.fullName) : TODO],
            ['Category', TODO],
            ['Tags', TODO],
            ['Expert time', TODO],
            ['Agent timeout', cell(cfg.totalTime)],
            ['GPU Model', cell(cfg.gpuModel)],
            ['GPU Count', cell(cfg.gpuCount)],
          ]),
        ],
      },
      {
        heading: 'Foundation',
        paragraphs: [
          table([
            ['Repository URL', cell(cfg.githubUrl)],
            ['Exact commit/tag', cell(cfg.commitHash, 'not pinned — pin the exact commit before the task is run')],
            ['Paper link (optional)', cell(cfg.paperLink, 'N/A')],
          ]),
        ],
      },
      {
        heading: 'Research Question',
        paragraphs: [
          val(
            cfg.researchQuestion,
            'Not yet stated — the experiment cannot be bounded until one independent variable is named.'
          ),
        ],
      },
      {
        heading: 'Reference Baseline',
        paragraphs: [
          table([
            ['Baseline Name', cell(cfg.baselineName)],
            ['Training Script Entry', cell(cfg.trainScript)],
            ['Training Data Link', cell(cfg.trainData)],
            ['Evaluation Script Entry', cell(cfg.evalScript)],
            ['Evaluation Data Link', cell(cfg.evalData)],
            ['Estimated Time', cell(cfg.estTime)],
            ['GPU Count', cell(cfg.gpuCount)],
            ['GPU Model', cell(cfg.gpuModel)],
          ]),
        ],
      },
      {
        heading: 'Evaluation',
        paragraphs: [
          table([
            ['Benchmarks', cell(cfg.benchmarks, 'none listed — at least one held-out benchmark is required')],
            ['Evaluation Metrics', cell(cfg.evalMetrics)],
            [
              'Baseline Result',
              cell(cfg.baselineResult, 'not reported — the baseline has no number to improve on yet'),
            ],
            ['Reward Definition', cell(cfg.rewardDef, 'not defined — without a threshold the run has no falsifiable outcome')],
            ['Hidden final evaluation (optional)', cell(cfg.hiddenEval, 'N/A')],
          ]),
        ].concat(
          /* What a hidden evaluation is for is explained in Mission Control, on
             the `?` beside the field. Only its absence is worth a line here,
             because that is a fact about this task. */
          has(cfg.hiddenEval)
            ? []
            : [
                'No hidden final evaluation is defined, so the agent is scored on the same signal it optimizes and a gain cannot be separated from adaptive overfitting. Adding a held-out pass the agent never sees is strongly recommended.',
              ]
        ),
      },
      {
        heading: 'Workspace',
        paragraphs: [
          '### External Resource Access',
          table([
            ['Web search', cfg.permWeb ? 'Enabled' : 'Disabled'],
            ['External services', cfg.permServices ? 'Allowed' : 'None'],
            ['Additional data construction / collection', cfg.permNewData ? 'Allowed' : 'Not allowed'],
            [
              'Leakage and reward-hacking safeguards (optional)',
              cell(
                cfg.safeguards,
                perms.on.length
                  ? 'none stated — every capability granted above is unguarded until concrete safeguards are named'
                  : 'N/A — the workspace is sandboxed, so there is no open path to control'
              ),
            ],
          ]),
          '### Compute Feasibility',
          table([
            ['Estimated GPU type and count per single experiment run', budget ? `${budget.n}× ${budget.model}` : cell('')],
            ['Estimated runtime per single experiment run', cell(cfg.estTime)],
            ['Estimated total AutoResearch time', cell(cfg.totalTime)],
            ['Early-stopping signals / lower-cost proxy experiments (optional)', cell(cfg.earlyStop, 'N/A')],
          ]),
          /* Same rule: the planning reference itself is on the `?` in Mission
             Control. What this proposal owes the reader is its own arithmetic
             and whether it clears the bar. */
          (budget
            ? `One experiment run is ${budget.n}× ${budget.model} for ${budget.hours} hours, about ${budget.gpuHours} GPU-hours from launch to a scoreable result. `
            : 'The per-run cost is not yet bounded: GPU count and wall-clock estimate must both be set before the task is approved. ') +
            (overBudget.length
              ? `That is ${overBudget.join(
                  ' and '
                )}, over the planning reference, so it must be flagged for resource review after baseline reproduction.`
              : 'That is within the planning reference.'),
        ],
      },
    ];

    const sources = [];
    if (has(cfg.paperLink)) sources.push({ title: 'Reference paper for the method under study', meta: cfg.paperLink, url: cfg.paperLink });
    if (has(cfg.githubUrl))
      sources.push({
        title: `Source repository — ${repo}`,
        meta: has(cfg.commitHash) ? `commit ${cfg.commitHash}` : 'commit not pinned',
        url: cfg.githubUrl,
      });
    if (has(cfg.trainData)) sources.push({ title: 'Training dataset', meta: cfg.trainData, url: cfg.trainData });
    if (has(cfg.evalData)) sources.push({ title: 'Evaluation dataset', meta: cfg.evalData, url: cfg.evalData });
    if (!sources.length) sources.push({ title: 'No sources linked', meta: 'Add a repository, paper, or dataset link in Mission Control.', url: '' });

    return {
      id: missionId(cfg),
      title,
      author: 'ResearchAI System',
      date: new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' }),
      status: 'Approved',
      summary,
      sections,
      sources,
      gaps: gaps(cfg),
    };
  }

  /* ------------------------------------------------------------ markdown */
  /* The Drafting page edits the proposal as a Markdown *file*: `draftMarkdown`
     writes the file, `parseDraft` reads it back into the document model the
     Final Delivery page renders, and `mdToHtml` is the preview renderer.
     Deliberately small — headings, paragraphs, lists, quotes, fenced code and
     the inline set the composer actually emits. No external dependency. */

  /* The editable document: a filled-in copy of proposal_format_v2.txt — the
     title, then one `## ` block per section of that format. The abstract, the
     id/author/date line and the source list are derived from the mission config
     and stay out of the file, which is the format and nothing else. */
  function draftMarkdown(p) {
    const lines = [`# ${p.title}`, ''];
    p.sections.forEach((s) => {
      lines.push(`## ${s.heading}`, '');
      s.paragraphs.forEach((para) => lines.push(para, ''));
    });
    return lines.join('\n').replace(/\n+$/, '\n');
  }

  /* Full export, including the parts the editor does not own. */
  function toMarkdown(p, md) {
    const head = [`# ${p.title}`, '', `**ID:** ${p.id}  |  **Author:** ${p.author}  |  **Date:** ${p.date}`, ''];
    const bodyMd = typeof md === 'string' && md.trim() ? md : draftMarkdown(p);
    /* The edited file carries its own `# Title`; keep the metadata line under it. */
    const body = bodyMd.replace(/^#\s+.*\n*/, '');
    const tail = ['', '## Source List', ''];
    p.sources.forEach((s) => tail.push(`- ${s.title} — ${s.meta}`));
    return head.concat(body.trim(), tail).join('\n');
  }

  function escHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  }

  /* Inline markdown. Code spans are lifted out first so the emphasis and link
     passes cannot rewrite anything inside them. */
  function mdInline(text) {
    const code = [];
    let s = escHtml(text).replace(/`([^`]+)`/g, (m, c) => {
      code.push(c);
      return '\u0000' + (code.length - 1) + '\u0000';
    });
    s = s
      .replace(/\*\*([^*]+)\*\*/g, '<strong class="text-on-surface">$1</strong>')
      .replace(/(^|[^*\w])\*([^*\n]+)\*/g, '$1<em>$2</em>')
      .replace(/\[([^\]]+)\]\((https?:[^\s)]+)\)/g, '<a class="text-primary hover:underline" href="$2" target="_blank" rel="noopener">$1</a>')
      /* Bare URLs, but not ones already inside an href we just wrote. */
      .replace(
        /(^|[\s(])(https?:\/\/[^\s<)"]+)/g,
        (m, lead, url) => {
          /* Trailing sentence punctuation is not part of the URL. */
          const trail = (url.match(/[.,;:!?]+$/) || [''])[0];
          const href = url.slice(0, url.length - trail.length);
          return `${lead}<a class="text-primary hover:underline break-all" href="${href}" target="_blank" rel="noopener">${href}</a>${trail}`;
        }
      );
    return s.replace(/\u0000(\d+)\u0000/g, (m, i) => `<code class="font-mono text-primary text-[0.9em]">${code[Number(i)]}</code>`);
  }

  /* Heading / block classes per surface: the drafting preview keeps the design's
     document styling, Final Delivery uses the print-page hierarchy. */
  const SKIN = {
    draft: {
      h1: 'font-headline-md text-headline-md text-on-background mb-4',
      h2: 'font-headline-md text-headline-md text-on-background border-b border-white/10 pb-2 mb-4 mt-8',
      h3: 'font-body-lg text-body-lg font-semibold text-primary mt-8 mb-3',
      ul: 'list-disc list-inside space-y-2 ml-4 text-on-surface-variant mt-4',
      ol: 'list-decimal list-inside space-y-2 ml-4 text-on-surface-variant mt-4',
      th: 'text-left font-label-md text-label-md text-on-surface py-2 px-3 border-b border-white/15',
      td: 'align-top py-2 px-3 border-b border-white/5 text-on-surface-variant',
    },
    final: {
      h1: 'font-headline-md text-headline-md text-on-surface mb-4',
      h2: 'font-headline-md text-headline-md text-on-surface mb-4 mt-8',
      h3: 'font-body-lg text-body-lg font-semibold text-primary mt-6 mb-3',
      ul: 'list-disc list-inside space-y-2 ml-4 mt-4',
      ol: 'list-decimal list-inside space-y-2 ml-4 mt-4',
      th: 'text-left font-label-md text-label-md text-on-surface py-2 px-3 border-b border-outline-variant',
      td: 'align-top py-2 px-3 border-b border-outline-variant/40',
    },
  };

  /* `| a | b |` → cells, with `\|` kept as a literal pipe (the composer escapes
     any pipe a contributor typed, so a value never splits its own row). */
  function rowCells(line) {
    return line
      .trim()
      .replace(/^\|/, '')
      .replace(/\|$/, '')
      .split(/(?<!\\)\|/)
      .map((c) => c.replace(/\\\|/g, '|').trim());
  }

  const isTableRow = (line) => /^\s*\|.*\|\s*$/.test(line);
  const isTableRule = (line) => /^\s*\|(\s*:?-{2,}:?\s*\|)+\s*$/.test(line);

  function mdToHtml(md, opt) {
    opt = opt || {};
    const skin = SKIN[opt.variant] || SKIN.draft;
    const lines = String(md == null ? '' : md).replace(/\r\n?/g, '\n').split('\n');
    const out = [];
    let para = [];
    let list = null; /* {tag, items} */
    let quote = [];
    let fence = null; /* array of raw lines */
    let rows = null; /* array of `| … |` lines */

    const flushPara = () => {
      if (para.length) out.push(`<p>${mdInline(para.join(' '))}</p>`);
      para = [];
    };
    const flushList = () => {
      if (list) out.push(`<${list.tag} class="${skin[list.tag]}">` + list.items.map((li) => `<li>${mdInline(li)}</li>`).join('') + `</${list.tag}>`);
      list = null;
    };
    const flushQuote = () => {
      if (quote.length)
        out.push(
          '<div class="bg-surface-variant/30 border border-outline-variant/50 rounded-lg p-4 my-6 flex gap-4 items-start">' +
            '<span class="material-symbols-outlined text-primary mt-1">format_quote</span>' +
            `<div class="text-sm italic text-on-surface-variant">${mdInline(quote.join(' '))}</div></div>`
        );
      quote = [];
    };
    /* Header row, an alignment rule, then body rows. A `|` block without the
       rule is not a table — it stays prose rather than silently losing its
       pipes. Wrapped so a wide table scrolls itself instead of the page. */
    const flushTable = () => {
      if (!rows) return;
      if (rows.length < 2 || !isTableRule(rows[1])) {
        rows.forEach((r) => para.push(r.trim()));
        rows = null;
        flushPara();
        return;
      }
      const head = rowCells(rows[0]);
      const body = rows.slice(2).map(rowCells);
      out.push(
        '<div class="overflow-x-auto my-4">' +
          '<table class="w-full border-collapse font-body-md text-body-md">' +
          '<thead><tr>' +
          head.map((c) => `<th class="${skin.th}">${mdInline(c)}</th>`).join('') +
          '</tr></thead><tbody>' +
          body
            .map(
              (cells) =>
                '<tr>' +
                head.map((_, i) => `<td class="${skin.td}">${mdInline(cells[i] || '')}</td>`).join('') +
                '</tr>'
            )
            .join('') +
          '</tbody></table></div>'
      );
      rows = null;
    };
    const flushAll = () => {
      flushTable();
      flushPara();
      flushList();
      flushQuote();
    };

    lines.forEach((raw) => {
      const line = raw.replace(/\s+$/, '');

      if (fence !== null) {
        if (/^\s*```/.test(line)) {
          out.push(
            '<pre class="bg-surface-container-lowest border border-white/10 rounded-lg p-4 my-4 overflow-x-auto"><code class="font-mono text-sm text-on-surface">' +
              escHtml(fence.join('\n')) +
              '</code></pre>'
          );
          fence = null;
        } else fence.push(raw);
        return;
      }
      if (/^\s*```/.test(line)) {
        flushAll();
        fence = [];
        return;
      }

      if (!line.trim()) {
        flushAll();
        return;
      }

      if (isTableRow(line)) {
        flushPara();
        flushList();
        flushQuote();
        if (!rows) rows = [];
        rows.push(line);
        return;
      }
      flushTable();

      const h = line.match(/^(#{1,6})\s+(.*)$/);
      if (h) {
        flushAll();
        const level = Math.min(h[1].length, 3);
        if (level === 1 && opt.skipTitle) return;
        out.push(`<h${level} class="${skin['h' + level]}">${mdInline(h[2])}</h${level}>`);
        return;
      }

      if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) {
        flushAll();
        out.push('<hr class="border-white/10 my-8">');
        return;
      }

      const q = line.match(/^\s*>\s?(.*)$/);
      if (q) {
        flushPara();
        flushList();
        quote.push(q[1]);
        return;
      }

      const bullet = line.match(/^\s*[-*+]\s+(.*)$/);
      const ordered = line.match(/^\s*\d+[.)]\s+(.*)$/);
      if (bullet || ordered) {
        flushPara();
        flushQuote();
        const tag = bullet ? 'ul' : 'ol';
        if (!list || list.tag !== tag) {
          flushList();
          list = { tag, items: [] };
        }
        list.items.push((bullet || ordered)[1]);
        return;
      }

      /* A continuation line inside a list item stays with that item. */
      if (list && /^\s{2,}\S/.test(raw)) {
        list.items[list.items.length - 1] += ' ' + line.trim();
        return;
      }

      flushList();
      flushQuote();
      para.push(line.trim());
    });

    if (fence !== null) out.push(`<pre class="bg-surface-container-lowest border border-white/10 rounded-lg p-4 my-4 overflow-x-auto"><code class="font-mono text-sm text-on-surface">${escHtml(fence.join('\n'))}</code></pre>`);
    flushAll();
    return out.join('\n');
  }

  /* Read the edited file back into the document model:
     the first `# ` line is the title, the `## Executive Summary` block is the
     summary, and every other `## ` block is a section. */
  function parseDraft(md) {
    const text = String(md == null ? '' : md).replace(/\r\n?/g, '\n');
    const lines = text.split('\n');
    let title = '';
    const blocks = [];
    let cur = null;

    lines.forEach((line) => {
      const h1 = line.match(/^#\s+(.*)$/);
      if (h1 && !title && !blocks.length) {
        title = h1[1].trim();
        return;
      }
      const h2 = line.match(/^##\s+(.*)$/);
      if (h2) {
        cur = { heading: h2[1].trim(), lines: [] };
        blocks.push(cur);
        return;
      }
      if (!cur) {
        /* Blank lines around the title belong to no block. */
        if (!line.trim()) return;
        /* Prose before the first `## ` heading — treat it as the summary. */
        cur = { heading: 'Executive Summary', lines: [], implicit: true };
        blocks.push(cur);
      }
      cur.lines.push(line);
    });

    const isSummary = (b) => /^executive summary$/i.test(b.heading);
    const summaryBlock = blocks.find(isSummary);
    const sections = blocks.filter((b) => !isSummary(b));
    const body = (b) => b.lines.join('\n').trim();

    return {
      title,
      summary: summaryBlock ? body(summaryBlock).replace(/\n{2,}/g, ' ').replace(/\n/g, ' ').trim() : '',
      summaryMd: summaryBlock ? body(summaryBlock) : '',
      bodyMd: sections.map((s) => `## ${s.heading}\n\n${body(s)}`).join('\n\n'),
      sections: sections.map((s) => ({ heading: s.heading, paragraphs: body(s).split(/\n{2,}/).filter(Boolean) })),
    };
  }

  return { compose, gaps, draftMarkdown, toMarkdown, mdToHtml, mdInline, parseDraft, titleFrom, computeBudget };
})();
