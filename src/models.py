import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config


class TransformerModel(nn.Module):
    def __init__(
        self,
        n_dims,
        n_positions,
        n_embd=128,
        n_layer=12,
        n_head=4,
        n_dims_out=1,
        use_positional_embd=True,
    ):
        super().__init__()
        config = GPT2Config(
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )

        self.n_positions = n_positions
        self.n_dims = n_dims
        self.n_dims_out = n_dims_out

        self._in_proj = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(config)
        self._out_proj = nn.Linear(n_embd, n_dims_out)

        # Freeze token embeddings (wte) — unused since we pass inputs_embeds directly
        self._backbone.wte.weight.requires_grad = False

        # Disable positional encodings by zeroing and freezing wpe
        if not use_positional_embd:
            self._backbone.wpe.weight.data.zero_()
            self._backbone.wpe.weight.requires_grad = False

        self.name = f"gpt2_embd={n_embd}_nl={n_layer}_nh={n_head}_pos_embd={use_positional_embd}"

    @staticmethod
    def _combine(xs, ys):
        """Interleave xs and ys into y1, x1, y2, x2, ...
        Zero-pads the smaller-dim tensor to match the larger one."""
        bsize, points, x_dim = xs.shape
        y_dim = ys.shape[-1] if ys.dim() == 3 else 1
        dim = max(x_dim, y_dim)

        # Ensure ys are 3D
        if ys.dim() == 2:
            ys = ys.unsqueeze(-1)

        # Zero-pad to common dim
        if x_dim < dim:
            xs = torch.cat(
                [xs, torch.zeros(bsize, points, dim - x_dim, device=xs.device)],
                dim=2,
            )
        if y_dim < dim:
            ys = torch.cat(
                [ys, torch.zeros(bsize, points, dim - y_dim, device=ys.device)],
                dim=2,
            )

        zs = torch.stack((ys, xs), dim=2)  # (B, N, 2, dim)
        return zs.view(bsize, 2 * points, dim)

    def forward(self, xs, ys, pred_positions=None, output_hidden_states=False):
        if pred_positions is None:
            pred_positions = torch.arange(ys.shape[1])
        else:
            pred_positions = torch.tensor(pred_positions)
            if max(pred_positions) >= ys.shape[1] or min(pred_positions) < 0:
                raise ValueError("pred_positions contain out-of-range indices")

        zs = self._combine(xs, ys)
        embeds = self._in_proj(zs)
        out = self._backbone(
            inputs_embeds=embeds,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        prediction = self._out_proj(out.last_hidden_state)

        # Return predictions at x-positions (even indices)
        pred = prediction[:, ::2, :][:, pred_positions]
        if output_hidden_states:
            return pred, out.hidden_states
        return pred
