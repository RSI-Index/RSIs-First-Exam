# agentdojo-attack-panel

Source Discussion: https://github.com/RSI-Index/RSI-Index-Public/discussions/16

## Stateful validation

Invoke the repository-local validator skill against the intact task package:

```text
$harbor-task-validator ./agentdojo-attack-panel
```

These execution checks remain pending and require separate authorization:

- Docker Environment preflight: NOT RUN
- GPU execution: NOT RUN
- baseline execution: NOT RUN
- evaluator execution: NOT RUN
- real-Agent execution: NOT RUN
