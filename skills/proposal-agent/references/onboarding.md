# Round 0 Orientation

Deliver this orientation once at the start of a new proposal. Use the contributor's language while preserving the meaning and the three sections.

## Design principles

An RSI-Index AutoResearch task studies a model-development hypothesis through repeated candidate experiments. In each trial the research agent changes a candidate, runs a fixed experiment, observes useful feedback, updates its hypothesis, and chooses the next candidate. The task should teach something about why an approach helps or fails; a one-shot feature implementation is not enough.

## Choosing a codebase

Prefer the official open-source repository that contains the starting implementation, baseline, configs, and evaluation path. Pin an immutable commit or exact tag so every claim can be checked against one repository state. A paper can explain motivation, but repository evidence must support the runnable task.

## Sample task shape

A suitable task might compare a repository's shipped training configuration with several bounded optimizer or curriculum variants under one fixed dataset, budget, evaluator, and score. The agent receives development feedback between trials and submits only its final candidate for the reserved comparison. Adding one CLI flag, porting an already specified function, or reproducing a paper once would be engineering work, not an iterative AutoResearch task.

After the orientation, ask for the repository URL, exact commit or tag if known, the contributor's relationship to or familiarity with the project, and a few sentences describing the initial research idea.
