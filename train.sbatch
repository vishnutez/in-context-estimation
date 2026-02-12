#!/bin/bash
#SBATCH --job-name=in-context-estimation
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --output=logs/%x_%j.out

mkdir -p logs

echo "Job ID:   $SLURM_JOB_ID"
echo "Nodes:    $SLURM_JOB_NODELIST"
echo "GPUs:     $SLURM_JOB_GPUS"
echo "Start:    $(date)"
echo "-------------------------------------------"

# Activate environment (adjust to your setup)
# source ~/.bashrc
# conda activate in-context-estimation

# Determine number of GPUs
NGPUS=$(echo "$SLURM_JOB_GPUS" | tr ',' '\n' | wc -l)
if [ -z "$SLURM_JOB_GPUS" ]; then
    NGPUS=${SLURM_GPUS_PER_NODE:-1}
fi

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

CONFIG=${1:-configs/default.yaml}
echo "Config:   $CONFIG"

srun torchrun \
    --nproc_per_node=$NGPUS \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/train.py "$CONFIG"

echo "-------------------------------------------"
echo "End: $(date)"
