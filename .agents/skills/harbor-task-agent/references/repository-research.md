# Repository research

Use repository evidence to translate the approved proposal into buildable inputs. This stage verifies facts; it does not reproduce the baseline or execute the project.

## Remote-first order

1. Normalize the official repository URL and reject mirrors unless the proposal explicitly uses one.
2. Resolve the requested tag/branch/ref through the remote host and record the immutable 40-character commit SHA. The proposal's final SHA remains authoritative if a moving branch advances later.
3. Inspect the tree and material files at that exact SHA through the forge API, raw immutable URLs, or remote Git object access.
4. Use a temporary, no-checkout or blob-filtered clone only when remote file APIs cannot answer a material question. Never prefer a contributor's unrelated local checkout over the approved remote ref.
5. Never run install hooks, imports, tests, training, project scripts, notebooks, or arbitrary commands from the remote repository during packaging research.

For each material claim, retain the immutable URL or repository-relative path, the SHA, and the fact it supports. Search results and default-branch pages are discovery aids, not evidence for the pinned ref.

## Facts to verify

- baseline implementation or official artifact;
- launch/benchmark/evaluation entrypoints and fixed configuration;
- dependency manifests, supported runtime/CUDA versions, and build instructions;
- data/checkpoint names, licenses, official download locations, revisions, and checksums when published;
- exact candidate-owned files and any generated files or package-install behavior;
- public tests or benchmarks that can become a safe development proxy;
- commands that can be expressed without network at Agent/Verifier runtime;
- repository-reported metric text, still labeled not yet reproduced.

Do not replace repository evidence with paper claims. A paper may explain the hypothesis but cannot establish that a path, command, artifact, or metric exists at the task ref.

## Baseline traceability record

Before Environment design, be able to state:

```text
official repository: <URL>
requested ref: <tag/branch/SHA>
resolved immutable ref: <40-character SHA>
baseline path/artifact: <repository path or immutable artifact>
entrypoint/config: <paths>
matched protocol evidence: <paths>
reported metric: <value and source, or not reported>
reproduction status: not yet reproduced
```

If the baseline exists only in an inaccessible private artifact, stop and ask for access or a contributor-provided task asset with provenance. Do not create a replacement baseline that changes the scientific comparison.

## Reproducible packaging

Prefer this order for starting material:

1. prebuilt image pinned by digest, when it is official, reviewable, and contains no hidden answer;
2. Docker build from an exact repository SHA plus pinned dependencies and checksummed assets;
3. contributor-supplied local asset only when its provenance, license, hash, and inclusion boundary are confirmed.

An immutable Git SHA alone does not pin package indexes, base images, datasets, checkpoints, submodules, Git LFS objects, or generated assets. Pin each independently where the ecosystem supports it. Do not pin apt package versions; apt repositories commonly stop serving old point versions. Pin the base image digest and clean apt lists instead.

If a build needs private credentials, do not put them in task files, build args, URLs, or image layers. Prefer an authorized prebuilt digest. If no safe delivery path exists, stop.

## Evidence changes task semantics

Ask the contributor only when evidence changes one of these decisions:

- which baseline is scientifically appropriate;
- what workload/evaluator/metric is fixed;
- which files the candidate may change;
- what data/checkpoints are allowed;
- whether runtime network or an external service is scientifically necessary;
- whether the estimated complete run can fit the available compute.

Resolve Docker syntax, installation order, source-copy layout, cache placement, and similar packaging mechanics yourself.
