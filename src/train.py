import math
import os
import sys
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from models import TransformerModel
from config import load_config, build_samplers


def get_curriculum_n_points(step, n_points, min_n_points, curriculum_steps):
    """Linearly ramp from min_n_points to n_points over curriculum_steps."""
    if min_n_points is None or curriculum_steps is None:
        return n_points
    if step >= curriculum_steps:
        return n_points
    frac = step / curriculum_steps
    return max(min_n_points, int(min_n_points + frac * (n_points - min_n_points)))


def complex_to_real(z):
    """Convert (..., d) complex tensor to (..., 2*d) real by stacking [real, imag]."""
    return torch.cat([z.real, z.imag], dim=-1)


def generate_batch(modulation_sampler, channel_sampler, b_size, n_points, snr_min_db, snr_max_db, seeds=None):
    """Generate a batch of transmitted signals, channels, and received signals.

    Args:
        seeds: optional list of b_size integer seeds for reproducible evaluation batches.

    Returns:
        xs:         (b, n, t) complex64 — transmitted symbols
        signal_ids: (b, n, t) long      — indices into the signal set
        ys:         (b, n, r) complex64  — received signals (with noise)
        snr_db:     (b,) float           — per-element SNR in dB
    """
    # Transmitted symbols
    xs, signal_ids = modulation_sampler.sample(n_points, b_size, seeds=seeds)  # (b, n, t), (b, n, t)

    # Channel realizations
    hs = channel_sampler.sample(b_size, n_points, seeds=seeds)  # (b, n, r, t)

    # Noiseless received signal: y = H x
    ys_noiseless = torch.einsum('bnrt,bnt->bnr', hs, xs)  # (b, n, r)

    # Sample SNR per batch element (fixed if min == max, otherwise uniform)
    if seeds is not None:
        generator = torch.Generator()
        generator.manual_seed(sum(seeds))
    else:
        generator = None

    if snr_min_db == snr_max_db:
        snr_db = torch.full((b_size,), snr_min_db)
    elif seeds is not None:
        snr_db = torch.rand(b_size, generator=generator) * (snr_max_db - snr_min_db) + snr_min_db
    else:
        snr_db = torch.rand(b_size) * (snr_max_db - snr_min_db) + snr_min_db
    snr_linear = 10 ** (snr_db / 10)  # (b,)

    # Additive complex Gaussian noise: n ~ CN(0, (1/snr) * I)
    noise_std = (1.0 / torch.sqrt(2.0 * snr_linear)).view(b_size, 1, 1)  # (b, 1, 1)
    n_rx = ys_noiseless.shape[-1]
    noise = noise_std * (torch.randn(b_size, n_points, n_rx, generator=generator)
                         + 1j * torch.randn(b_size, n_points, n_rx, generator=generator))

    ys = ys_noiseless + noise  # (b, n, r)

    return xs, signal_ids, ys, snr_db


@torch.no_grad()
def evaluate(model, modulation_sampler, channel_sampler, task, b_size, n_points,
             snr_min_db, snr_max_db, device, eval_seeds):
    """Evaluate model on a fixed (seeded) dataset.

    Returns a dict of metrics: eval/loss, and eval/accuracy (for detection) or eval/mse (for estimation).
    """
    model.eval()

    n_tx = modulation_sampler.n_tx_antennas
    signal_set_size = modulation_sampler.signal_set_size

    xs, signal_ids, ys, snr_db = generate_batch(
        modulation_sampler, channel_sampler, b_size, n_points, snr_min_db, snr_max_db, seeds=eval_seeds
    )

    xs_real = complex_to_real(xs).to(device)
    ys_real = complex_to_real(ys).to(device)

    pred = model(xs_real, ys_real)

    metrics = {}
    if task == "detection":
        logits = pred.view(b_size, n_points, n_tx, signal_set_size)
        loss = F.cross_entropy(
            logits.reshape(-1, signal_set_size),
            signal_ids.to(device).reshape(-1),
        )
        predicted_ids = logits.argmax(dim=-1)  # (b, n, t)
        accuracy = (predicted_ids == signal_ids.to(device)).float().mean()
        metrics["eval/loss"] = loss.item()
        metrics["eval/accuracy"] = accuracy.item()
    else:  # estimation
        loss = F.mse_loss(pred, xs_real)
        metrics["eval/loss"] = loss.item()
        metrics["eval/mse"] = loss.item()

    model.train()
    return metrics


def save_checkpoint(model, optimizer, step, wandb_run_id, checkpoint_dir, distributed=False, max_checkpoints=None):
    """Save training checkpoint to disk, removing oldest if max_checkpoints is exceeded."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    raw_model = model.module if distributed else model
    checkpoint = {
        "step": step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "wandb_run_id": wandb_run_id,
    }
    path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
    torch.save(checkpoint, path)
    # Also save as "latest" for easy resume
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    torch.save(checkpoint, latest_path)
    print(f"  [ckpt] saved checkpoint at step {step} -> {path}")

    # Prune old checkpoints (keep checkpoint_latest.pt separate)
    if max_checkpoints is not None:
        import glob
        numbered = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_[0-9]*.pt")))
        while len(numbered) > max_checkpoints:
            oldest = numbered.pop(0)
            os.remove(oldest)
            print(f"  [ckpt] removed old checkpoint {oldest}")


def load_checkpoint(path, model, optimizer, device):
    """Load a training checkpoint. Returns the step to resume from and the wandb_run_id."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    step = checkpoint["step"]
    wandb_run_id = checkpoint.get("wandb_run_id")
    print(f"  [ckpt] resumed from {path} at step {step}")
    return step, wandb_run_id


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
    """Destroy the distributed process group."""
    dist.destroy_process_group()


def train(cfg, modulation_sampler, channel_sampler):
    """Train a TransformerModel for MIMO detection or estimation.

    Args:
        cfg:                 config dict (from load_config)
        modulation_sampler:  ModulationSampler instance
        channel_sampler:     ChannelSampler instance
    """
    # Unpack config sections
    task = cfg["task"]
    tc = cfg["training"]
    mc = cfg["model"]
    sc = cfg["snr"]
    wc = cfg["wandb"]

    n_steps = tc["n_steps"]
    b_size = tc["b_size"]
    n_points = tc["n_points"]
    lr = tc["lr"]
    log_every = tc["log_every"]
    eval_every = tc["eval_every"]
    eval_b_size = tc["eval_b_size"]
    save_every = tc["save_every"]
    max_checkpoints = tc.get("max_checkpoints")
    checkpoint_dir = tc["checkpoint_dir"]
    resume_from = tc["resume_from"]

    # Curriculum: ramp n_points from min_n_points -> n_points over curriculum_steps
    min_n_points = tc.get("min_n_points")
    curriculum_steps = tc.get("curriculum_steps")

    snr_min_db = sc["min_db"]
    snr_max_db = sc["max_db"]

    n_embd = mc["n_embd"]
    n_layer = mc["n_layer"]
    n_head = mc["n_head"]
    use_positional_embd = mc["use_positional_embd"]

    wandb_project = wc["project"]
    wandb_run_name = wc["run_name"]

    # Auto-generate run name if not specified
    if wandb_run_name is None:
        mean_snr = (snr_min_db + snr_max_db) / 2
        mod_type = cfg["modulation"]["type"]
        ch_type = cfg["channel"]["type"]
        wandb_run_name = (
            f"embd={n_embd}_nl={n_layer}_nh={n_head}"
            f"_snr_db={mean_snr:.0f}"
            f"_task={task}"
            f"_mod={mod_type.lower()}"
            f"_ch={ch_type.lower()}"
            f"_pos_embd={use_positional_embd}"
            f"_min_n_points={min_n_points}"
            f"_curriculum_steps={curriculum_steps}"
        )

    # Save checkpoints under checkpoint_dir/run_name/
    checkpoint_dir = os.path.join(checkpoint_dir, wandb_run_name)

    # Distributed setup
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        rank, world_size, device = setup_distributed()
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_tx = modulation_sampler.n_tx_antennas
    n_rx = channel_sampler.n_rx_antennas
    signal_set_size = modulation_sampler.signal_set_size

    # Model dimensions
    x_dim = 2 * n_tx
    y_dim = 2 * n_rx
    n_dims = max(x_dim, y_dim)
    n_positions = 2 * n_points  # interleaved (x, y) pairs

    if task == "detection":
        n_dims_out = signal_set_size * n_tx
    elif task == "estimation":
        n_dims_out = 2 * n_tx
    else:
        raise ValueError(f"Unknown task: {task!r}. Expected 'detection' or 'estimation'.")

    model = TransformerModel(
        n_dims=n_dims,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_dims_out=n_dims_out,
        use_positional_embd=use_positional_embd,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Resume from checkpoint (before DDP wrapping so we load into the raw model)
    start_step = 0
    wandb_run_id = None
    if resume_from is not None:
        start_step, wandb_run_id = load_checkpoint(resume_from, model, optimizer, device)

    if distributed:
        model = DDP(model, device_ids=[device])

    # Initialize wandb on rank 0 only
    if rank == 0:
        wandb_kwargs = {
            "project": wandb_project,
            "config": cfg,
        }
        if wandb_run_id is not None:
            # Resume the exact same wandb run
            wandb_kwargs["id"] = wandb_run_id
            wandb_kwargs["resume"] = "must"
            print(f"  [wandb] resuming run {wandb_run_id}")
        else:
            wandb_kwargs["name"] = wandb_run_name

        wandb.init(**wandb_kwargs)
        # Store the run id so we can save it in checkpoints
        wandb_run_id = wandb.run.id

    # eval_seeds = list(range(eval_b_size))  # fixed seeds for reproducible evaluation
    eval_seeds = None  # random evaluation batches each time

    for step in range(start_step + 1, n_steps + 1):
        model.train()

        # Curriculum: compute current n_points for training
        n_points_train = get_curriculum_n_points(step, n_points, min_n_points, curriculum_steps)

        # Each rank generates its own independent batch
        xs, signal_ids, ys, snr_db = generate_batch(
            modulation_sampler, channel_sampler, b_size, n_points_train, snr_min_db, snr_max_db
        )

        # Convert complex to stacked real
        xs_real = complex_to_real(xs).to(device)   # (b, n_points_train, 2*t)
        ys_real = complex_to_real(ys).to(device)   # (b, n_points_train, 2*r)

        # Right-pad to n_points so model input size is always fixed
        if n_points_train < n_points:
            pad_len = n_points - n_points_train
            xs_real = F.pad(xs_real, (0, 0, 0, pad_len))  # (b, n_points, 2*t)
            ys_real = F.pad(ys_real, (0, 0, 0, pad_len))  # (b, n_points, 2*r)

        # Forward pass: pred shape (b, n_points, n_dims_out)
        pred = model(xs_real, ys_real)

        # Compute loss only on the first n_points_train positions
        pred_active = pred[:, :n_points_train]
        if task == "detection":
            logits = pred_active.view(b_size, n_points_train, n_tx, signal_set_size)
            loss = F.cross_entropy(
                logits.reshape(-1, signal_set_size),
                signal_ids.to(device).reshape(-1),
            )
        else:  # estimation
            loss = F.mse_loss(pred_active, xs_real[:, :n_points_train])

        # Backward pass (DDP synchronizes gradients automatically)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log training metrics
        if rank == 0 and (step % log_every == 0 or step == 1):
            train_metrics = {"train/loss": loss.item(), "train/step": step, "train/n_points": n_points_train}
            if task == "detection":
                with torch.no_grad():
                    train_acc = (logits.argmax(dim=-1) == signal_ids.to(device)).float().mean()
                train_metrics["train/accuracy"] = train_acc.item()
            wandb.log(train_metrics, step=step)
            print(f"step {step:>6d}/{n_steps} | loss: {loss.item():.4f} | n_pts: {n_points_train}")

        # Evaluate periodically
        if rank == 0 and (step % eval_every == 0):
            eval_model = model.module if distributed else model
            with torch.no_grad():
                # Sanity: eval the SAME training batch
                # Set the model to evaluation mode
                eval_model.eval()
                pred_check = eval_model(xs_real, ys_real)
                logits_check = pred_check[:, :n_points_train].view(b_size, n_points_train, n_tx, signal_set_size)
                loss_check = F.cross_entropy(logits_check.reshape(-1, signal_set_size), signal_ids.to(device).reshape(-1))
                print(f"  [sanity] train_loss={loss.item():.4f}  same_batch_eval_loss={loss_check.item():.4f}")
                wandb.log({"sanity/train_loss": loss.item(), "sanity/eval_loss": loss_check.item()}, step=step)
                # Set the model back to training mode
                eval_model.train()
            eval_metrics = evaluate(
                eval_model, modulation_sampler, channel_sampler, task,
                eval_b_size, n_points, snr_min_db, snr_max_db, device, eval_seeds,
            )
            wandb.log(eval_metrics, step=step)
            eval_summary = " | ".join(f"{k}: {v:.4f}" for k, v in eval_metrics.items())
            print(f"  [eval] {eval_summary}")

        # Save checkpoint periodically
        if rank == 0 and (step % save_every == 0):
            save_checkpoint(model, optimizer, step, wandb_run_id, checkpoint_dir, distributed, max_checkpoints)

    # Final checkpoint
    if rank == 0:
        save_checkpoint(model, optimizer, n_steps, wandb_run_id, checkpoint_dir, distributed, max_checkpoints)
        wandb.finish()

    if distributed:
        cleanup_distributed()

    # Return the unwrapped model
    return model.module if distributed else model


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
    cfg = load_config(config_path)
    modulation_sampler, channel_sampler = build_samplers(cfg)
    train(cfg, modulation_sampler, channel_sampler)
