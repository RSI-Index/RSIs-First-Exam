# simia-curriculum-transfer

Source Discussion: https://github.com/RSI-Index/RSI-Index-Public/discussions/8

## Stateful validation

Invoke the repository-local validator skill against the intact task package:

```text
$harbor-task-validator ./simia-curriculum-transfer
```

These execution checks remain pending and require separate authorization:

- Docker Environment preflight: NOT RUN
- GPU execution: NOT RUN
- baseline execution: NOT RUN
- evaluator execution: NOT RUN
- real-Agent execution: NOT RUN
