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
WANDB=offline python3 train_detection_qam.py --config conf/<config-file-name>
```

where `<config-file-name>` represents the model to be trained in `conf/` folder.

The trained models are saved in `models/<task-name>/<run-id>` directory. To get the result plots with the model, run:
```bash
python3 eval_time_invariant_process.py <run-id> <task-name>
```
for time invariant process, and
```bash
python3 eval_time_variant_process.py <run-id> <task-name>
```
for time varying process. The results are saved as `results/<task-name>.npy` and plots as `plots/<task-name>_<model-name>.png`.

For the baselines, run:
```bash
cd traditional-methods/
python3 baselines_time_invariant.py
python3 baselines_time_variant.py
```

## Citation

```bibtex
@inproceedings{
kunde2025transformers,
title={Transformers are Provably Optimal In-context Estimators for Wireless Communications},
author={Vishnu Teja Kunde and Vicram Rajagopalan and Chandra Shekhara Kaushik Valmeekam and Krishna Narayanan and Jean-Francois Chamberland and Dileep Kalathil and Srinivas Shakkottai},
booktitle={The 28th International Conference on Artificial Intelligence and Statistics},
year={2025},
url={https://openreview.net/forum?id=nhGtq5s6GJ}
}
```


