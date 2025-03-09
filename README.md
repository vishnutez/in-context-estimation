# In-Context Estimation using Transformers
Official implementation of the paper: "Transformers are Provably Optimal In-context Estimators for Wireless Communcations"

# Set-up
Clone the repository. Install the dependencies and activate the environment using conda:

```bash 
conda env create -f environment.yml
conda activate in-context-estimation
```

Enter the directory
```bash 
cd src
```

Run the following command for training:
```bash
WANDB=offline python3 train_detection.py --config conf/<config-file-name>
```

where `<config-file-name>` represents the model to be trained in `conf/` folder.

The trained models are saved in `models/<task-name>/<run-id>` directory. To get the result plots with the model, run:
```bash
python3 detection_time_variant_process.py <run-id> <task-name>
```

The files are saved in `files/<task-name>.npy` and plots in `plots/<task-name>_<model-name>.png`.




