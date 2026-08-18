/* ResearchAI — shared client-side state (localStorage). */
window.RA = window.RA || {};

RA.store = (function () {
  const K_PROFILE = 'researchai.profile';
  const K_MISSION = 'researchai.mission';
  const K_PROPOSAL = 'researchai.proposal';
  const K_DRAFT = 'researchai.draft';

  function read(key) {
    try {
      const raw = localStorage.getItem(key);
      return raw ? JSON.parse(raw) : null;
    } catch (e) {
      return null;
    }
  }

  function write(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (e) {
      /* private mode / quota — the demo still works for the current page */
    }
  }

  return {
    saveProfile: (p) => write(K_PROFILE, p),
    loadProfile: () => read(K_PROFILE),
    saveMission: (cfg) => write(K_MISSION, cfg),
    loadMission: () => read(K_MISSION),
    saveProposal: (p) => write(K_PROPOSAL, p),
    loadProposal: () => read(K_PROPOSAL),
    saveDraft: (html) => write(K_DRAFT, html),
    loadDraft: () => read(K_DRAFT),
    clear() {
      [K_PROFILE, K_MISSION, K_PROPOSAL, K_DRAFT].forEach((k) => localStorage.removeItem(k));
    },
  };
})();

/* Example mission, used by the "Load Example" button on Mission Control. */
RA.EXAMPLE = {
  githubUrl: 'https://github.com/huggingface/trl',
  commitHash: 'a1b2c3d4e5f6',
  paperLink: 'https://arxiv.org/abs/2305.18290',
  baselineName: 'Qwen2.5-7B + DPO',
  trainScript: 'scripts/train_dpo.py',
  trainData: 'https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized',
  evalScript: 'scripts/eval_alpacaeval.py',
  evalData: 'https://huggingface.co/datasets/tatsu-lab/alpaca_eval',
  estTime: '18h',
  gpuCount: '8',
  gpuModel: 'H100',
  totalTime: '72h',
  earlyStop: 'Dev-split win rate after 500 steps; 1k-example proxy eval before the full AlpacaEval run.',
  researchQuestion:
    'Does length-normalized preference optimization reduce verbosity bias without hurting win rate on held-out instruction-following benchmarks?',
  benchmarks: 'AlpacaEval 2.0, MT-Bench, IFEval',
  evalMetrics: 'Length-controlled win rate, mean response length, IFEval strict accuracy',
  baselineResult:
    'AlpacaEval 2.0 length-controlled win rate 18.7% for Qwen2.5-7B + DPO, as reported in the TRL example README; not yet reproduced.',
  rewardDef:
    'Success = +2pt length-controlled win rate over the DPO baseline with mean response length not increasing by more than 5%.',
  hiddenEval:
    'A held-out 300-prompt AlpacaEval split, scored once after submission; the agent sees no score, output, or log from it during the task.',
  safeguards:
    'Web access is read-only and blocked from the evaluation datasets; generated data carries provenance records and never enters an evaluation split.',
  permWeb: true,
  permServices: false,
  permNewData: true,
};
