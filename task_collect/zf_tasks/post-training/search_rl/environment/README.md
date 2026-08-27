# Environment contract

The image materializes `OSU-NLP-Group/QUEST` at commit
`962e10b6b99f53efc8fd45229ed80728b9ee9a05`, initializes the pinned
LlamaFactory submodule, removes the complete upstream `evaluation/` tree, and
verifies normalized tree digest
`9238a96476fa1607fc572f8d0e62d52cdd441c851a9d51b31c5d76000a7e780c`.

The resulting `/app/project` is the candidate-editable source. Task Tools are
installed read-only at `/task-tools`, durable artifacts use `/app/output`, and
the verifier owns `/tests` and `/logs/verifier`. The portable image contains no
hidden benchmark data, model weights, training dataset, scientific evaluator,
or GPU backend. Their identities and materialization status are recorded in
`source.lock.json`.
