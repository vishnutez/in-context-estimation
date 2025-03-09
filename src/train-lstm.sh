#!/bin/bash
#SBATCH --job-name=ICE-T         # Job name
#SBATCH --mail-type=ALL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=bhavya_sai@tamu.edu  #Where to send mail     
#SBATCH --ntasks-per-node=8                     # Run on a 8 cpus (max) 
#SBATCH --gres=gpu:tesla:1              # Run on a single GPU (max)
#SBATCH --partition=gpu-research                 # Select GPU Partition
#SBATCH --qos=olympus-research-gpu          # Specify GPU queue
#SBATCH --time=30:00:00                 # Time limit hrs:min:sec current 5 min - 36 hour max

# use the sbatch command to submit your job to the cluster.
# sbatch train.sh

# select your singularity shell (currently cuda10.2-cudnn7-py36)
singularity shell /mnt/lab_files/ECEN403-404/containers/cuda_10.2-cudnn7-py36.sif
# source your virtual environmnet
cd /mnt/shared-scratch/Shakkottai_S/bhavya_sai/ICLDetection/CodeFiles
# execute your python job
WANDB_MODE=offline python -u train_detection.py --config conf/lstm_detection_time_invariant_snr_neg5.yaml