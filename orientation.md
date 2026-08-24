# FrontierRSI AutoResearch Contributor Orientation

> Working draft for the 2026 Sep contribution cycle. This document is intended for contributor review before its contents are integrated into the proposal and review agents.

## What FrontierRSI is evaluating

FrontierRSI turns representative, fully open model-development projects into auditable research environments. A research agent starts from a human-built implementation or artifact, proposes a hypothesis, changes a candidate, runs a fixed experiment, observes declared feedback, updates its hypothesis, and repeats. The objective is to measure whether the agent can discover an improvement over a matched human baseline while preserving the experiment's scientific contract.

## Design principles

1. **Optimization, not reproduction.** The AutoResearch task must give the agent room to improve a real model-development workflow beyond the original human baseline.
2. **Scientific discovery, not mechanical search.** The task should support distinct hypotheses and teach something about why an approach helps or fails. A one-shot implementation, routine port, or ordinary hyperparameter sweep is not enough.
3. **Production relevance, not a toy result.** The selected experiment should preserve the causal mechanism and evaluation path of the source project. When the full workflow is too expensive, a faithful slice is acceptable only if improvement on that slice remains informative about the real problem.
4. **Prefer a live frontier.** Frontier timing is a prioritization signal. Prefer questions that address newly important capability in the current research cycle. A less active area receives lower priority on an active frontier.

## Bring one project

Choose exactly one project through either route:

1. **Coauthored route:** the highest-impact project that you personally coauthored; or
2. **Domain-expertise route:** a well-known project in your field whose codebase you know especially well.

In either route, explain your relationship to the project and why your experience matches the proposed research area. Contributor-domain fit is a strict eligibility gate.

## What makes a project suitable for AutoResearch

| Characteristic | What it means |
|---|---|
| Fully open | The selected lane has public, usable code, configurations, required model/checkpoint and data assets, and an evaluation path. |
| Influential | The project shaped its area through adoption, citations, community use for frontier work. |
| Timely | The research area remains active in the current research cycle. |
| Reproducible | The official experiment can run end to end. Missing critical assets or an unverifiable metric block admission. |
| Researchable | The codebase exposes meaningful scientific decisions. Different candidate changes can represent different hypotheses, not merely different parameter values. |
| Verifiable | A separate fixed evaluator can score the submitted artifact and enforce integrity boundaries without trusting self-reported results. |

## Common poor fits

- A paper reproduction with no subsequent improvement loop.
- A one-shot feature, port, refactor, or bug fix.
- A search space dominated by routine parameter sweeping.

## Compute orientation

One candidate experiment means taking one fixed idea and configuration from launch through any required training or optimization and evaluation until it produces a valid score. It should normally use no more than 8 H100-equivalent GPUs at peak and finish within 12 wall-clock hours. This reference applies to one experiment, not the full multi-experiment AutoResearch trajectory. At orientation, only determine whether a likely experiment will significantly exceed either threshold; collect exact accelerator and runtime estimates later. A large estimate is a resource-review flag, not by itself a scientific rejection.

## What to provide in Round 0

Provide:

1. the one selected project and its official repository URL;
2. the exact commit or tag, if already known;
3. whether this is your representative coauthored project or a well-known codebase in your domain expertise;
4. why the underlying research area is foundational and frontier-relevant currently;
5. one initial research question that the repository could support; and
6. whether one candidate run is likely to significantly exceed the normal compute reference.

The proposal agent should investigate repository facts and current frontier evidence itself. The contributor supplies expert judgment and explicit decisions.
