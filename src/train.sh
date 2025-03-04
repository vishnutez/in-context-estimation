#!/bin/bash
#SBATCH --job-name=SA-CE        # Job name
#SBATCH --mail-type=BEGIN,END,FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=vishnukunde@tamu.edu  #Where to send mail    
#SBATCH --ntasks=8                      # Run on a 8 cpus (max)
#SBATCH --gres=gpu:tesla:1              # Run on a single GPU (max)
#SBATCH --partition=gpu-research                 # Select GPU Partition
#SBATCH --qos=olympus-research-gpu          # Specify GPU queue
#SBATCH --time=36:00:00                 # Time limit hrs:min:sec current 5 min - 36 hour max

# use the sbatch command to submit your job to the cluster.
# sbatch train.sh

# select your singularity shell (currently cuda10.2-cudnn7-py36)
singularity shell /mnt/lab_files/ECEN403-404/containers/cuda_10.2-cudnn7-py36.sif
# source your virtual environmnet

cd /mnt/shared-scratch/Narayanan_K/vishnukunde/codebase/icl-detection/src

source activate in-context-learning

WANDB_MODE=offline python -u train_detection.py --config conf/detection_time_varying_process_snr_0.yaml

# WANDB_MODE=offline python -u train_detection.py --config conf/lstm_detection_time_invariant_snr_neg5.yaml

# WANDB_MODE=offline python -u train_detection.py --config conf/detection_time_invariant_snr_neg5.yaml

# WANDB_MODE=offline python -u train_detection.py --config conf/detection_time_varying_process_snr_3_16qam.yaml

# WANDB_MODE=offline python -u train_detection.py --config conf/detection_time_invariant_snr_0_qam.yaml

# WANDB_MODE=offline python -u train.py --config conf/demodulation_time_varying_snr_0.yaml

# WANDB_MODE=offline python -u train.py --config conf/demodulation_snr_neg5.yaml


# python -u DetectionTimeVariantProcess.py model_1 detection_time_variant_snr_0

# python -u DetectionTimeInvariantProcess.py model_0 detection_time_invariant_snr_neg5