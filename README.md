# In-Context Estimation

In-context learning for MIMO signal detection and estimation using a GPT-2 transformer. The model learns to detect or estimate transmitted symbols from noisy received signals by attending to previous (observation, label) pairs in a sequence -- no explicit channel estimation required.

## Installation

```bash
conda create -n in-context-estimation python=3.10 -y
conda activate in-context-estimation
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install scipy transformers wandb pyyaml
```

Or using `env.yaml`:

```bash
conda env create -f env.yaml
conda activate in-context-estimation
```

Adjust `cu126` to match your CUDA version. Run `wandb login` once to set up logging.

## Training

```bash
# Single GPU
./train.sh

# Multi-GPU (e.g. 4 GPUs)
NGPUS=4 ./train.sh

# Custom config
NGPUS=4 ./train.sh configs/my_experiment.yaml
```

For SLURM clusters, see `train.sbatch`:

```bash
sbatch train.sbatch                          # default config
sbatch train.sbatch configs/my_experiment.yaml
```

## Configuration

All parameters live in a single YAML file (`configs/default.yaml`):

```yaml
task: detection              # "detection" (cross-entropy) | "estimation" (MSE)

modulation:
  type: QAM                  # QAM | PSK
  signal_set_size: 16
  n_tx_antennas: 1

channel:
  type: DopplerSpread        # RayleighBlockFading | DopplerSpread | CustomBlockFading
  n_rx_antennas: 4
  fc: 2.6e9                  # DopplerSpread only
  symbol_duration: 1.0e-3    # DopplerSpread only
  velocities: [5, 15, 30]    # finite set, or use velocity_min/velocity_max for a range

snr:
  min_db: 0.0                # set min == max for fixed SNR
  max_db: 20.0

model:
  n_embd: 256
  n_layer: 6
  n_head: 4
  use_positional_embd: true

training:
  n_steps: 10_000_000
  b_size: 64
  n_points: 20
  lr: 1.0e-4
  log_every: 100
  eval_every: 500
  save_every: 500
  checkpoint_dir: checkpoints
  resume_from: null           # path to checkpoint_latest.pt to resume

wandb:
  project: in-context-estimation
  run_name: null              # auto-generated if null
```

## Project Structure

```
src/
  samplers.py   — Modulation samplers (QAM, PSK) and channel samplers
  models.py     — TransformerModel (GPT-2 backbone)
  config.py     — YAML loader and sampler builder
  train.py      — Training loop, evaluation, checkpointing
configs/
  default.yaml  — Default training config
train.sh        — Launch script (single/multi-GPU)
train.sbatch    — SLURM batch script
env.yaml        — Conda environment spec
```

## Channel Samplers

`ChannelSampler` is an abstract base class. All implementations return `(b, n_points, n_rx, n_tx)` complex64 tensors.

**Built-in channels:**

| Config `type` | Class | Description |
|---|---|---|
| `RayleighBlockFading` | `RayleighBlockFadingChannelSampler` | IID Rayleigh, constant over the sequence |
| `DopplerSpread` | `DopplerSpreadChannelSampler` | Time-varying, Clarke-Jakes Doppler spectrum |
| `CustomBlockFading` | `CustomBlockFadingChannelSampler` | Samples from a user-provided dataset |

### Using a custom channel dataset

```python
import torch
from samplers import CustomBlockFadingChannelSampler

channel_data = torch.load("my_channels.pt")  # (dataset_size, n_rx, n_tx) complex64
sampler = CustomBlockFadingChannelSampler(channel_dataset=channel_data)
```

### Subclassing `ChannelSampler`

```python
from samplers import ChannelSampler

class MyChannelSampler(ChannelSampler):
    def __init__(self, n_tx_antennas=1, n_rx_antennas=1, my_param=1.0):
        super().__init__(n_tx_antennas=n_tx_antennas, n_rx_antennas=n_rx_antennas)
        self.my_param = my_param

    def sample(self, b_size, n_points, seeds=None):
        # Return (b_size, n_points, n_rx_antennas, n_tx_antennas) complex64
        ...
```

Register it in `src/config.py` to use from YAML configs:

```python
CHANNEL_SAMPLERS["MyChannel"] = MyChannelSampler
```

```yaml
channel:
  type: MyChannel
  n_rx_antennas: 4
  my_param: 2.5
```

All keys under `channel:` (except `type`) are forwarded as kwargs to the constructor.
