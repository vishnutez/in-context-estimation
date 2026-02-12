#!/bin/bash
# Usage:
#   ./train.sh                            # 1 GPU, default config
#   ./train.sh configs/my_experiment.yaml  # 1 GPU, custom config
#   NGPUS=4 ./train.sh                    # 4 GPUs on this node
#   NGPUS=4 ./train.sh configs/my_experiment.yaml

CONFIG=${1:-configs/default.yaml}
NGPUS=${NGPUS:-1}

echo "Config: $CONFIG"
echo "GPUs:   $NGPUS"

if [ "$NGPUS" -gt 1 ]; then
    torchrun --nproc_per_node=$NGPUS src/train.py "$CONFIG"
else
    python src/train.py "$CONFIG"
fi
