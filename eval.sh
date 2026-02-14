#!/bin/bash
# Usage:
#   ./eval.sh configs/default.yaml --checkpoint checkpoints/<run>/checkpoint_latest.pt
#   NGPUS=4 ./eval.sh configs/default.yaml --checkpoint checkpoints/<run>/checkpoint_latest.pt
#   NGPUS=2 ./eval.sh configs/default.yaml --checkpoint checkpoints/<run>/checkpoint_latest.pt --velocities 5 15 30 60 --n_batches 50

CONFIG=${1:-configs/default.yaml}
shift  # remaining args passed to eval.py
NGPUS=${NGPUS:-1}

echo "Config: $CONFIG"
echo "GPUs:   $NGPUS"
echo "Args:   $@"

if [ "$NGPUS" -gt 1 ]; then
    torchrun --nproc_per_node=$NGPUS src/eval.py "$CONFIG" "$@"
else
    python src/eval.py "$CONFIG" "$@"
fi
