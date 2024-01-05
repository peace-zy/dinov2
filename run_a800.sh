echo "Starting Training"
export PYTHONPATH=.
NUM_NODES=1
NUM_GPUS_PER_NODE=8
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001
LOG_NAME=`date +"%Y_%m_%d_%H_%M_%S"`
LOG_PATH="log"
SUB_NAME="dinov2-vitb14-bs1024-ep100-a800-mlp"
if [ -e ${LOG_PATH} ]; then
    echo "[\033[32m"${LOG_PATH}"\033[0m] 文件存在"

else
    echo "[\033[32m"${LOG_PATH}"\033[0m] 文件不存在"
    mkdir ${LOG_PATH}
fi
TORCHRUN=/data/nfs-ten9/nfs/zhangyan461/env/dinov2-extras/bin/torchrun
${TORCHRUN} \
    --nproc_per_node=${NUM_GPUS_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --nnodes=${NUM_NODES} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    dinov2/train/train.py \
    --config-file=dinov2/configs/train/vitb14_a800.yaml \
    --output-dir=./checkpoints/${SUB_NAME} > ${LOG_PATH}/${SUB_NAME}_${LOG_NAME}.log 2>&1
