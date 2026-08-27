# Contributing an RSI task

> [!TIP]
>
> ### From idea to validated RSI task in ≤ 30 active minutes
>
> Bring the idea—the agents handle the rest. Review, validate, and have fun! 🚀


```bash
git clone https://github.com/RSI-Index/RSI-Index-Public.git
cd RSI-Index-Public
```

1. Start with [proposal-agent](.agents/skills/proposal-agent/SKILL.md). It guides you through defining the project, baseline, evaluation, scope, and compute, then generates an **RSI task proposal**. **Review it before submitting.**
2. Post it to [GitHub Discussions](https://github.com/RSI-Index/RSI-Index-Public/discussions). Revise it based on Discussion Agent feedback until it is accepted.
3. Continue in the same Discussion: use `/task` to start task preparation, `/task <feedback>` to request changes, and `/task confirm` to approve the final version and create your private task repository. Before publication, use `/reset` to discard the current attempt and return to the accepted Proposal.
4. **GPU required:** clone the generated private repository and use its included `harbor-task-validator` skill to perform full validation.
5. Write brief feedback on `proposal-agent` and the task workflow in [GitHub Discussions](https://github.com/RSI-Index/RSI-Index-Public/discussions/7).


Thank you for contributing to RSI-Index! 🙏
