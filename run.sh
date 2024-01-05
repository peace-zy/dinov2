echo "Starting Training"
export PYTHONPATH=.
NUM_NODES=1
NUM_GPUS_PER_NODE=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001
TORCHRUN=/aistudio/workspace/system-default/envs/dinov2-extras/bin/torchrun
${TORCHRUN} \
    --nproc_per_node=${NUM_GPUS_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --nnodes=${NUM_NODES} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    dinov2/train/train.py \
    --config-file=dinov2/configs/train/vitb14.yaml \
    --output-dir=./checkpoints/dinov2-vitb14-bs1024-ep10
