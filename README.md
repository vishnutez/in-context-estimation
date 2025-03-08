# In-Context Estimation using Transformers
Official implementation of the paper: "Transformers are Provably Optimal In-context Estimators for Wireless Communcations"

# Set-up
Install the dependencies

> conda env create -f environment.yml

Run the following for training:
> WANDB=offline python3 train_detection.py --config conf/<config-file-name>

where `<config-file-name>` represents the model to be trained.

The trained models are saved in `models/<task-name>/<run-id>` directory. To get the result plots with the model, run:
> python3 detection_time_variant_process.py <task-name> <run-id>

The flies are saved in `files/<task-name>.npy` and plots in `plots/<task-name>_<model-name>.png`.




