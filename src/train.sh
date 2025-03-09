#!/bin/bash

WANDB_MODE=offline python -u train_detection_qam.py --config ../conf/detection-time-varying-process-snr-0.yaml
