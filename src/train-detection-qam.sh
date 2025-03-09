#!/bin/bash

WANDB_MODE=offline python -u train_detection_qam.py --config ../conf/detection-time-invariant-snr-neg2-16qam.yaml
