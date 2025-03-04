"""
Evaluation function for an encoder model (i.e. InverseProblemBidirectionalTransformerModel).
In order to have a fair comparison between encoder and decoder (e.g. GPT-2) models, we
apply masking in such a way that predictions are made in a similar manner.
"""
import numpy as np
import torch

def bert_eval_with_masking(model, xs, ys):
    """
    For nmask = 0 to n_points - 1, mask out all but the first nmask x's. Compute preds for
    just the first masked x, to keep in line with previous experiments.
    x and y are used in the pre-swap sense.

    Output: model predictions.
    """
    preds_list = []
    bsize, n_points, xdim = xs.shape
    assert ys.shape[0] == bsize
    assert ys.shape[1] == n_points
    ydim = ys.shape[2]

    for idx_first_mask in range(n_points):
        # modified from train_step()
        mask_indices = np.arange(idx_first_mask, n_points)
        # Masked elements have a nonzero value in the zeroth dim, unmasked elements have a 0.
        y_mask = torch.zeros(bsize, n_points, 1, dtype=ys.dtype, device=ys.device)
        x_mask_single = torch.zeros(1, n_points, 1, dtype=xs.dtype, device=xs.device)
        x_mask_single[0, mask_indices, 0] = 1
        x_mask = torch.tile(x_mask_single, (bsize, 1, 1))
        assert x_mask.shape == (bsize, n_points, 1)

        curr_xs = torch.cat([x_mask, xs], dim=2)
        assert curr_xs.shape == (bsize, n_points, xdim + 1)

        curr_ys = torch.cat([y_mask, ys], dim=2)
        assert curr_ys.shape == (bsize, n_points, ydim + 1)

        # For fair comparison with a similar GPT2 model, fully "mask out" the future (both xs and ys) by
        # removing it before passing tensors to the model.
        curr_xs_sliced = curr_xs[:, :(idx_first_mask + 1), :]
        curr_ys_sliced = curr_ys[:, :(idx_first_mask + 1), :]
        assert curr_xs_sliced.shape == (bsize, (idx_first_mask + 1), xdim + 1)
        assert curr_ys_sliced.shape == (bsize, (idx_first_mask + 1), ydim + 1)

        curr_preds = model(curr_xs_sliced, curr_ys_sliced)

        out_dim = curr_preds.shape[2]
        assert curr_preds.shape == (bsize, (idx_first_mask + 1), out_dim)
        assert out_dim == xdim

        # Just predict on the one desired element
        pred_desired = curr_preds[:, [idx_first_mask], :]  # Keep the inner dimension with size 1
        assert pred_desired.shape == (bsize, 1, out_dim)

        preds_list.append(pred_desired)
    preds = torch.cat(preds_list, dim=1)
    return preds
