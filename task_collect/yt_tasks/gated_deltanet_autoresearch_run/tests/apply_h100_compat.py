#!/usr/bin/env python3
"""Apply the locked task-infrastructure patch to a pristine NVlabs checkout."""

from pathlib import Path


ROOT = Path("/opt/project")


def remove_unsupported_warps(relative: str, expected_per_warp: int) -> None:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    for warps in (8, 16, 32):
        line = f"        triton.Config({{}}, num_warps={warps}),\n"
        if text.count(line) != expected_per_warp:
            raise RuntimeError(
                f"unexpected {relative} num_warps={warps} count: {text.count(line)}"
            )
        text = text.replace(line, "")
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


remove_unsupported_warps("lit_gpt/gated_delta_rule_ops/chunk.py", 5)
remove_unsupported_warps("lit_gpt/gated_delta_rule_ops/wy_fast.py", 7)

pretrain = ROOT / "pretrain.py"
text = pretrain.read_text(encoding="utf-8")
old = """    if os.path.exists(args.out_dir):
        args.resume = True
        print('Resuming from {}'.format(args.out_dir))
    else:
        if fabric.global_rank == 0:
            os.makedirs(args.out_dir)
            target_litgpt_save_dir = os.path.join(args.out_dir, 'lit_gpt')
            target_bash_scripts_save_dir = os.path.join(args.out_dir, 'bash_scripts')
            os.makedirs(target_litgpt_save_dir)
            os.makedirs(target_bash_scripts_save_dir)
"""
new = """    resume_checkpoint = os.path.join(args.out_dir, "latest-model-ckpt.pth")
    if os.path.isfile(resume_checkpoint):
        args.resume = True
        print('Resuming from {}'.format(args.out_dir))
    else:
        args.resume = False
        if fabric.global_rank == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            target_litgpt_save_dir = os.path.join(args.out_dir, 'lit_gpt')
            target_bash_scripts_save_dir = os.path.join(args.out_dir, 'bash_scripts')
            os.makedirs(target_litgpt_save_dir, exist_ok=True)
            os.makedirs(target_bash_scripts_save_dir, exist_ok=True)
    fabric.barrier()
"""
if text.count(old) != 1:
    raise RuntimeError("unexpected pretrain output/resume block")
text = text.replace(old, new)
old_start = "    mp.set_start_method('spawn')\n"
new_start = "    mp.set_start_method('spawn', force=True)\n"
if text.count(old_start) != 1:
    raise RuntimeError("unexpected multiprocessing start-method block")
pretrain.write_text(text.replace(old_start, new_start), encoding="utf-8")
