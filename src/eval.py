"""Per-position evaluation by velocity.

For each velocity, generates eval batches with a fixed-velocity DopplerSpread
channel, computes per-position (k) cross-entropy loss and accuracy, saves
results as CSV and plots.

Supports distributed evaluation via torchrun.

Usage:
    python src/eval.py configs/default.yaml --checkpoint checkpoints/<run>/checkpoint_latest.pt
    torchrun --nproc_per_node=2 src/eval.py configs/default.yaml --checkpoint checkpoints/<run>/checkpoint_latest.pt
    python src/eval.py configs/default.yaml --checkpoint checkpoints/<run>/checkpoint_latest.pt --velocities 5 15 30 60
"""
import argparse
import csv
import os
import sys

import torch
import torch.nn.functional as F
import torch.distributed as dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import TransformerModel
from config import load_config, build_samplers
from train import complex_to_real, generate_batch
from samplers import DopplerSpreadChannelSampler


def setup_distributed():
    """Initialize distributed process group. Returns (rank, world_size, device)."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, world_size, device


def cleanup_distributed():
    dist.destroy_process_group()


@torch.no_grad()
def per_position_metrics(model, modulation_sampler, channel_sampler, b_size,
                         n_points, snr_min_db, snr_max_db, device, n_batches=10,
                         distributed=False):
    """Compute per-position cross-entropy and accuracy, averaged over n_batches.

    In distributed mode, each rank processes n_batches independently and results
    are averaged across all ranks.

    Returns (on all ranks):
        losses:     (n_points,) tensor — per-position cross-entropy
        accuracies: (n_points,) tensor — per-position accuracy
    """
    model.eval()
    n_tx = modulation_sampler.n_tx_antennas
    signal_set_size = modulation_sampler.signal_set_size

    total_losses = torch.zeros(n_points, device=device)
    total_correct = torch.zeros(n_points, device=device)

    for _ in range(n_batches):
        xs, signal_ids, ys, _ = generate_batch(
            modulation_sampler, channel_sampler, b_size, n_points,
            snr_min_db, snr_max_db,
        )
        xs_real = complex_to_real(xs).to(device)
        ys_real = complex_to_real(ys).to(device)

        pred = model(xs_real, ys_real)  # (b, n_points, n_dims_out)
        logits = pred.view(b_size, n_points, n_tx, signal_set_size)

        for k in range(n_points):
            loss_k = F.cross_entropy(
                logits[:, k].reshape(-1, signal_set_size),
                signal_ids[:, k].to(device).reshape(-1),
            )
            total_losses[k] += loss_k

            predicted_ids = logits[:, k].argmax(dim=-1)
            total_correct[k] += (predicted_ids == signal_ids[:, k].to(device)).float().mean()

    # Average over local batches
    total_losses /= n_batches
    total_correct /= n_batches

    # Average across ranks
    if distributed:
        dist.all_reduce(total_losses, op=dist.ReduceOp.AVG)
        dist.all_reduce(total_correct, op=dist.ReduceOp.AVG)

    return total_losses.cpu(), total_correct.cpu()


def main():
    parser = argparse.ArgumentParser(description="Per-position evaluation by velocity")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", default="results", help="Directory for CSVs and plots")
    parser.add_argument("--velocities", nargs="+", type=float, default=None,
                        help="Velocities to evaluate (default: from config)")
    parser.add_argument("--b_size", type=int, default=256, help="Batch size per GPU")
    parser.add_argument("--n_batches", type=int, default=10, help="Batches per GPU to average over")
    parser.add_argument("--n_points", type=int, default=None, help="Override n_points from config")
    args = parser.parse_args()

    # Distributed setup
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        rank, world_size, device = setup_distributed()
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    cfg = load_config(args.config)
    modulation_sampler, _ = build_samplers(cfg)

    assert cfg["task"] == "detection", "This script supports detection task only"

    tc = cfg["training"]
    mc = cfg["model"]
    sc = cfg["snr"]
    ch_cfg = cfg["channel"]

    n_points = args.n_points or tc["n_points"]
    snr_min_db = sc["min_db"]
    snr_max_db = sc["max_db"]

    # Determine velocities
    if args.velocities is not None:
        velocities = args.velocities
    elif ch_cfg.get("velocities") is not None:
        velocities = [float(v) for v in ch_cfg["velocities"]]
    elif ch_cfg.get("velocity_min") is not None and ch_cfg.get("velocity_max") is not None:
        velocities = torch.linspace(ch_cfg["velocity_min"], ch_cfg["velocity_max"], 5).tolist()
    else:
        raise ValueError("No velocities specified via --velocities or in the config")

    # Build model
    n_tx = modulation_sampler.n_tx_antennas
    n_rx = ch_cfg["n_rx_antennas"]
    signal_set_size = modulation_sampler.signal_set_size

    n_dims = max(2 * n_tx, 2 * n_rx)
    n_positions = 2 * n_points
    n_dims_out = signal_set_size * n_tx

    model = TransformerModel(
        n_dims=n_dims,
        n_positions=n_positions,
        n_embd=mc["n_embd"],
        n_layer=mc["n_layer"],
        n_head=mc["n_head"],
        n_dims_out=n_dims_out,
        use_positional_embd=mc["use_positional_embd"],
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    step = checkpoint.get("step", "unknown")
    if rank == 0:
        total_samples = args.b_size * args.n_batches * world_size
        print(f"Checkpoint step: {step}")
        print(f"Velocities:      {velocities}")
        print(f"n_points={n_points}  b_size={args.b_size}  n_batches={args.n_batches}  world_size={world_size}")
        print(f"Total samples per velocity: {total_samples}")
        print(f"SNR: [{snr_min_db}, {snr_max_db}] dB")
        print(f"Device: {device}")
        os.makedirs(args.output_dir, exist_ok=True)

    all_losses = {}
    all_accuracies = {}

    for v in velocities:
        if rank == 0:
            print(f"\n--- v = {v} m/s ---")

        # Channel sampler with this exact velocity
        ch_sampler = DopplerSpreadChannelSampler(
            n_tx_antennas=n_tx,
            n_rx_antennas=n_rx,
            fc=float(ch_cfg.get("fc", 1e9)),
            symbol_duration=float(ch_cfg.get("symbol_duration", 1e-3)),
            velocity_min=v,
            velocity_max=v,
        )

        losses, accuracies = per_position_metrics(
            model, modulation_sampler, ch_sampler,
            args.b_size, n_points, snr_min_db, snr_max_db,
            device, args.n_batches, distributed,
        )

        all_losses[v] = losses
        all_accuracies[v] = accuracies

        # Only rank 0 saves outputs
        if rank == 0:
            # Save CSV
            csv_path = os.path.join(args.output_dir, f"per_position_v={v}.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["k", "cross_entropy", "accuracy"])
                for k in range(n_points):
                    writer.writerow([k, f"{losses[k].item():.6f}", f"{accuracies[k].item():.6f}"])
            print(f"  Saved {csv_path}")

            # Per-velocity plot
            fig, ax1 = plt.subplots(figsize=(8, 5))

            ax1.plot(range(n_points), losses.numpy(), "b-o", markersize=4, label="Cross-Entropy")
            ax1.set_xlabel("k (in-context examples seen)")
            ax1.set_ylabel("Cross-Entropy Loss", color="b")
            ax1.tick_params(axis="y", labelcolor="b")

            ax2 = ax1.twinx()
            ax2.plot(range(n_points), accuracies.numpy(), "r-s", markersize=4, label="Accuracy")
            ax2.set_ylabel("Accuracy", color="r")
            ax2.tick_params(axis="y", labelcolor="r")
            ax2.set_ylim(0, 1)

            fig.suptitle(f"v = {v} m/s  |  SNR = [{snr_min_db}, {snr_max_db}] dB  |  step {step}")
            fig.tight_layout()

            plot_path = os.path.join(args.output_dir, f"per_position_v={v}.png")
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"  Saved {plot_path}")

    # Combined plot (rank 0 only)
    if rank == 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for v in velocities:
            ax1.plot(range(n_points), all_losses[v].numpy(), "-o", markersize=3, label=f"v={v}")
            ax2.plot(range(n_points), all_accuracies[v].numpy(), "-s", markersize=3, label=f"v={v}")

        ax1.set_xlabel("k (in-context examples seen)")
        ax1.set_ylabel("Cross-Entropy Loss")
        ax1.set_title("Per-Position Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("k (in-context examples seen)")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Per-Position Accuracy")
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f"SNR = [{snr_min_db}, {snr_max_db}] dB  |  step {step}")
        fig.tight_layout()

        combined_path = os.path.join(args.output_dir, "per_position_all_velocities.png")
        fig.savefig(combined_path, dpi=150)
        plt.close(fig)
        print(f"\nSaved combined plot: {combined_path}")

    if distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
