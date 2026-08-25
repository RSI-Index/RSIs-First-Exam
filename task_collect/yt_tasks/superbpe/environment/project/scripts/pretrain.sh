config=configs/alisa/OLMo2-7B-generic200k.yaml

export MAX_SEQUENCE_LENGTH=3000
export TRAIN_STEPS=107972
export TOKENIZER=om2-pts200k-t180k-mw4-colon
export MODEL_NAME=OLMo2-7B-pts200k-t180k-ctx${MAX_SEQUENCE_LENGTH}-colon

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" $(which hostname) --ip-address)
echo nodes: ${nodes}
echo head node: ${head_node} at ${head_node_ip}
srun --export=all torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=gpu \
    --start_method=fork \
    --rdzv_id=$RANDOM \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${head_node_ip}:29500 \
    scripts/train.py $config
