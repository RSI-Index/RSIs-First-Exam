#!/usr/bin/env bash
set -euo pipefail

attempt_id=${TMAX_ATTEMPT_ID:-manual}
max_updates=${TMAX_MAX_UPDATES:-}
scheduler_horizon=${TMAX_SCHEDULER_HORIZON_UPDATES:-200}
project_root=${TMAX_PROJECT_ROOT:-/app/project}
output_root=${TMAX_OUTPUT_ROOT:-/app/output}
training_root=${TMAX_TRAINING_ARTIFACT_ROOT:-/app/training}
training_data=${TMAX_TRAINING_DATA_DIR:-${training_root}/data}
training_prompt=${TMAX_TRAINING_PROMPT_FILE:-${training_root}/prompt/system.txt}
training_reward=${TMAX_TRAINING_REWARD_FILE:-${training_root}/reward/reward.py}
task_data=${TMAX_TASK_DATA_DIR:-/datasets/tmax/train}

if [[ ! ${attempt_id} =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]]; then
  echo "invalid TMAX_ATTEMPT_ID: ${attempt_id}" >&2
  exit 2
fi
if [[ ! ${max_updates} =~ ^[0-9]+$ ]] || (( max_updates < 1 || max_updates > 200 )); then
  echo "TMAX_MAX_UPDATES must be an integer from 1 through 200" >&2
  exit 2
fi
if [[ ${scheduler_horizon} != 200 ]]; then
  echo "TMAX_SCHEDULER_HORIZON_UPDATES must equal 200" >&2
  exit 2
fi
training_trajectories=$((max_updates * 256))

rl_algorithm=${RSI_RL_ALGORITHM:-dppo}
rl_learning_rate=${RSI_RL_LEARNING_RATE:-1e-6}
rl_scheduler=${RSI_RL_SCHEDULER:-constant}
rl_warmup_ratio=${RSI_RL_WARMUP_RATIO:-0.0}
rl_weight_decay=${RSI_RL_WEIGHT_DECAY:-0.0}
rl_kl_coef=${RSI_RL_KL_COEF:-0.0}
rl_kl_estimator=${RSI_RL_KL_ESTIMATOR:-2}
rl_clip_lower=${RSI_RL_CLIP_LOWER:-0.2}
rl_clip_higher=${RSI_RL_CLIP_HIGHER:-0.272}
rl_num_epochs=${RSI_RL_NUM_EPOCHS:-1}
rl_num_mini_batches=${RSI_RL_NUM_MINI_BATCHES:-1}
rl_advantage_norm=${RSI_RL_ADVANTAGE_NORMALIZATION:-centered}
rl_tis_cap=${RSI_RL_TIS_CAP:-0.0}
rl_dppo_divergence=${RSI_RL_DPPO_DIVERGENCE:-tv}
rl_dppo_threshold=${RSI_RL_DPPO_THRESHOLD:-0.1}
rl_tvpo_threshold=${RSI_RL_TVPO_THRESHOLD:-0.02}
rl_tvpo_cap=${RSI_RL_TVPO_CAP:-20.0}
rl_value_loss_coef=${RSI_RL_VALUE_LOSS_COEF:-0.5}
rl_value_learning_rate=${RSI_RL_VALUE_LEARNING_RATE:-$rl_learning_rate}
rl_value_clip=${RSI_RL_VALUE_CLIP_RANGE:-0.2}
rl_gamma=${RSI_RL_GAMMA:-1.0}
rl_gae_lambda=${RSI_RL_GAE_LAMBDA:-1.0}
rl_temperature=${RSI_RL_ROLLOUT_TEMPERATURE:-1.0}
rl_unique_prompts=${RSI_RL_UNIQUE_PROMPTS:-8}
rl_group_size=${RSI_RL_GROUP_SIZE:-32}
rl_async_steps=${RSI_RL_ASYNC_STEPS:-4}
rl_max_grad_norm=${RSI_RL_MAX_GRAD_NORM:-1.0}
rl_loss_denominator=${RSI_RL_LOSS_DENOMINATOR:-token}
rl_tis_mask_lower=${RSI_RL_TIS_MASK_LOWER:-0.0}
rl_tis_mask_upper=${RSI_RL_TIS_MASK_UPPER:-0.0}
rl_sequence_tis_threshold=${RSI_RL_SEQUENCE_TIS_THRESHOLD:-0.0}
rl_active_sampling=${RSI_RL_ACTIVE_SAMPLING:-true}
rl_filter_zero_std=${RSI_RL_FILTER_ZERO_STD:-true}
rl_inflight_updates=${RSI_RL_INFLIGHT_UPDATES:-true}
rl_use_liger=${RSI_RL_USE_LIGER_LOSS:-true}
rl_use_vllm_logprobs=${RSI_RL_USE_VLLM_LOGPROBS:-true}

case "$rl_algorithm" in
  grpo) rl_loss_fn=dapo; rl_use_value_model=false ;;
  dppo) rl_loss_fn=dppo; rl_use_value_model=false ;;
  ppo) rl_loss_fn=dapo; rl_use_value_model=true ;;
  cispo) rl_loss_fn=cispo; rl_use_value_model=false ;;
  tvpo) rl_loss_fn=tvpo; rl_use_value_model=false ;;
  custom) rl_loss_fn=dapo; rl_use_value_model=false ;;
  *) echo "RSI_RL_ALGORITHM must be grpo, dppo, ppo, cispo, tvpo, or custom" >&2; exit 2 ;;
esac
if ! [[ "$rl_unique_prompts" =~ ^[1-9][0-9]*$ && "$rl_group_size" =~ ^[1-9][0-9]*$ ]] || \
   (( rl_unique_prompts * rl_group_size != 256 )); then
  echo "RSI_RL_UNIQUE_PROMPTS * RSI_RL_GROUP_SIZE must equal 256" >&2
  exit 2
fi

active_sampling_args=()
if [[ "$rl_active_sampling" == true ]]; then
  active_sampling_args+=(--active_sampling)
elif [[ "$rl_active_sampling" != false ]]; then
  echo "RSI_RL_ACTIVE_SAMPLING must be true or false" >&2
  exit 2
fi
liger_args=()
if [[ "$rl_use_liger" == true ]]; then
  liger_args+=(--use_liger_grpo_loss --liger_grpo_loss_chunk_size 8)
elif [[ "$rl_use_liger" != false ]]; then
  echo "RSI_RL_USE_LIGER_LOSS must be true or false" >&2
  exit 2
fi

attempt_root="${output_root}/attempts/${attempt_id}"
run_root="${attempt_root}/runs"
tool_config="{\"backend\":\"apptainer\",\"task_data_dir\":\"${task_data}\",\"test_timeout\":600,\"image\":\"${TMAX_SANDBOX_IMAGE:-python:3.12-slim}\"}"

training_command=(
  uv run python -u open_instruct/grpo_fast.py
  --dataset_mixer_list "$training_data" 1.0
  --dataset_mixer_list_splits train
  --max_prompt_token_length 2048
  --per_turn_max_tokens 16384
  --response_length 65536
  --pack_length 67584
  --per_device_train_batch_size 1
  --num_unique_prompts_rollout "$rl_unique_prompts"
  --num_samples_per_prompt_rollout "$rl_group_size"
  --async_steps "$rl_async_steps"
  --model_name_or_path /models/Qwen3.5-9B
  --temperature "$rl_temperature"
  --learning_rate "$rl_learning_rate"
  --total_episodes "$training_trajectories"
  --scheduler_horizon_steps "$scheduler_horizon"
  --training_reward_file "$training_reward"
  --training_artifact_root "$training_root"
  --lr_scheduler_type "$rl_scheduler"
  --warmup_ratio "$rl_warmup_ratio"
  --weight_decay "$rl_weight_decay"
  --deepspeed_stage 3
  --sequence_parallel_size 4
  --num_epochs "$rl_num_epochs"
  --num_mini_batches "$rl_num_mini_batches"
  --num_learners_per_node 8 8
  --vllm_num_engines 48
  --vllm_tensor_parallel_size 1
  --beta "$rl_kl_coef"
  --kl_estimator "$rl_kl_estimator"
  --clip_lower "$rl_clip_lower"
  --clip_higher "$rl_clip_higher"
  --use_vllm_logprobs "$rl_use_vllm_logprobs"
  --truncated_importance_sampling_ratio_cap "$rl_tis_cap"
  --tis_mask_lower "$rl_tis_mask_lower"
  --tis_mask_upper "$rl_tis_mask_upper"
  --sequence_tis_mask_log_ratio_threshold "$rl_sequence_tis_threshold"
  --seed 42
  --gradient_checkpointing
  --max_grad_norm "$rl_max_grad_norm"
  --vllm_enable_prefix_caching
  --push_to_hub false
  --save_traces
  --save_trainer_logprobs true
  --tools swerl_vanillux_sandbox
  --tool_configs "$tool_config"
  --pool_size 512
  --max_steps 64
  --verification_reward 1.0
  --tool_parser_type vllm_qwen3_xml
  --system_prompt_override_file "$training_prompt"
  "${active_sampling_args[@]}"
  --filter_zero_std_samples "$rl_filter_zero_std"
  --backend_timeout 1200
  --vllm_gdn_prefill_backend triton
  --checkpoint_state_freq 10
  --inflight_updates "$rl_inflight_updates"
  --lm_head_fp32 true
  "${liger_args[@]}"
  --advantage_normalization_type "$rl_advantage_norm"
  --loss_denominator "$rl_loss_denominator"
  --loss_fn "$rl_loss_fn"
  --dppo_divergence_type "$rl_dppo_divergence"
  --dppo_divergence_threshold "$rl_dppo_threshold"
  --tvpo_divergence_threshold "$rl_tvpo_threshold"
  --tvpo_truncation_cap "$rl_tvpo_cap"
  --use_value_model "$rl_use_value_model"
  --value_model_name_or_path /models/Qwen3.5-9B
  --value_loss_coef "$rl_value_loss_coef"
  --value_learning_rate "$rl_value_learning_rate"
  --vf_clip_range "$rl_value_clip"
  --gamma "$rl_gamma"
  --gae_lambda "$rl_gae_lambda"
  --rollouts_save_path "${attempt_root}/rollouts"
  --output_dir "$run_root"
  --exp_name "$attempt_id"
  --local_eval_every 1000000
  --save_freq 20
  --try_launch_beaker_eval_jobs_on_weka False
)

if [[ ${1:-} == --print-command ]]; then
  printf '%q ' "${training_command[@]}"
  printf '\n'
  exit 0
fi
if (( $# > 0 )); then
  echo "usage: train_tmax.sh [--print-command]" >&2
  exit 2
fi

for required in "$project_root/training/open-instruct" "$training_data" "$training_prompt" "$training_reward"; do
  [[ -e $required ]] || { echo "required training input is missing: $required" >&2; exit 2; }
done
cd "$project_root/training/open-instruct"
mkdir -p "$run_root" "$attempt_root/logs"

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1
export SWERL_SANDBOX_TIMING_LOGS=1
export SWERL_RESET_FAILURE_ZERO_REWARD=1
export SWERL_APPTAINER_OFFLINE=1
export TMAX_ATTEMPT_ID="$attempt_id"

uv run python -c 'import ray; ray.init(address="auto"); g=ray.cluster_resources().get("GPU", 0); assert g >= 64, f"TMAX 9B lane requires 64 Ray GPUs, found {g}"'
"${training_command[@]}" 2>&1 | tee "$attempt_root/logs/train.log"
