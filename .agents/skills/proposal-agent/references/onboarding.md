# Round 0 Orientation

Deliver this orientation once at the start of a new proposal. Use the contributor's language while preserving the meaning and the four sections.

## Design principles

- **Optimization, not reproduction.** Reproduction establishes the human baseline; the AutoResearch task gives the agent room to improve the original model-development workflow or artifact.
- **Scientific discovery, not routine search or implementation.** Each trial tests a meaningful hypothesis. Ordinary hyperparameter sweeping, a one-shot feature, a port, or a paper reproduction without a subsequent research loop is insufficient.
- **Production relevance, not a disconnected toy proxy.** The experiment should preserve the source project's causal mechanism and evaluation path so that an improvement remains informative about the real workflow.
- **A fixed, auditable scientific contract.** Keep the baseline, budget, evaluation protocol, locked variables, feedback boundary, and score explicit and consistent across candidates.

## Choose one project

The contributor must provide exactly one project through one of two routes:

1. the highest-impact project they personally coauthored; or
2. a well-known project in their field whose codebase they know especially well.

The contributor's actual role and expertise must match the proposed task area. This is an eligibility requirement, not a prestige test. If the contributor supplies multiple projects, do not rank or select among them; ask the contributor to choose one. Prefer a foundational project or one connected to a current research frontier, but frontier relevance is a preference rather than an eligibility gate.

Use the official open-source repository when available and pin an immutable commit or exact tag. A paper may explain motivation, but repository and artifact evidence must support the runnable task.

## What makes a project suitable for AutoResearch

A suitable project has:

- sufficiently open code, configurations, checkpoint or data assets, and evaluation path;
- real scientific or practical influence;
- an official path that can run end to end, or an official artifact that can be evaluated directly;
- a bounded, cost-controllable experimental lane;
- meaningful method decisions for the research agent;
- a repeated hypothesis → change → run → observe → update loop;
- a fixed baseline, budget, and independently verifiable metric;
- a final candidate artifact that a separate verifier can score directly; and
- when the full workflow is too expensive, a faithful slice that preserves the source project's causal mechanism and evaluation path.

## Compute orientation

Surface compute scale before detailed proposal work. The normal planning reference for one candidate run is at most 8 H100-equivalent GPUs and at most 12 hours. At orientation, ask only whether the task is likely to exceed either threshold significantly; collect exact estimates later. A large estimate is a resource-review flag, not a scientific rejection.

After the orientation, ask for the one selected repository URL, exact commit or tag if known, which project-selection route applies, the contributor's specific role and task-relevant expertise, a few sentences describing the initial research idea, and whether one candidate run is likely to exceed the normal compute reference significantly. Do not request exact compute details yet.
