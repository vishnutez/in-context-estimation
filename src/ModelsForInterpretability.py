import torch
import torch.nn as nn
from torch.nn import LSTM, RNN, Conv1d
from transformers import GPT2Model, GPT2Config, BertConfig, BertModel
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso
import warnings
from sklearn import tree
import xgboost as xgb

from base_models import (
    TwoLayerNeuralNetwork,
    ThreeLayerNeuralNetwork,
    ParallelNetworks,
    GPT2ModelWOPosEncodings,
)

# Comment to commit

def build_model(conf):
    if conf.family == "gpt2_detection":
        model = TransformerForICD(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            pos_encode=conf.pos_encode,
            n_dims_out=conf.n_dims_out,
        )
    elif conf.family == "lstm":
        print('In LSTM model call')
        model = SeqModelForICD(n_dims=conf.n_dims,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer, 
            n_dims_out=conf.n_dims_out, 
            backbone_seq_model="lstm")  # to be consistent with the remaining models
    elif conf.family == "rnn":
        print('In RNN model call')
        model = SeqModelForICD(n_dims=conf.n_dims,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer, 
            n_dims_out=conf.n_dims_out, 
            backbone_seq_model="rnn")  # to be consistent with the remaining models
    elif conf.family == "ccnn":
        print('In CCNN model call')
        model = CausalConvModelForICD(n_dims=conf.n_dims,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer, 
            n_dims_out=conf.n_dims_out, 
            kernel_size=conf.kernel_size)
    else:
        raise NotImplementedError

    return model

# CausalConvNet

class CausalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(CausalConv1D, self).__init__()
        if "dilation" in kwargs:
            self.dilation = kwargs["dilation"]
        else:
            self.dilation = 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kwargs = kwargs
        self.convnet1d = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, **kwargs)

    def forward(self, x):
        """
        x: B, Cin, L 
        """
        left_pad = (self.kernel_size-1)* self.dilation
        x_pad = torch.nn.functional.pad(x, pad=(left_pad, 0))
        out = self.convnet1d(x_pad)
        return out
    
class CausalConvModelForICD(nn.Module):
    def __init__(self, n_dims, n_positions=100, n_embd=256, n_layer=12, n_dims_out=1, kernel_size=5):
        super(CausalConvModelForICD, self).__init__()
        self.n_positions = n_positions
        self.n_dims = n_dims
        self.n_dims_out = n_dims_out
        self.kernel_size = kernel_size  # for completeness

        self._read_in = nn.Linear(n_dims, n_embd)

        layers = []

        for _ in range(n_layer):
            layers.append(CausalConv1D(in_channels=n_embd, out_channels=n_embd, kernel_size=kernel_size))
            layers.append(nn.ReLU())

        self._backbone = nn.Sequential(*layers)
        self._read_out = nn.Linear(n_embd, n_dims_out)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = ys_b.shape
        x_size = xs_b.shape[2]

        # print('ys_shape = ', ys_b.shape)
        # print('xs shape = ', xs_b.shape)
        
        # if self.n_dims_out < dim:  # Pad y's to have the same shape as x's

        xs_b_wide = torch.cat(
            (
                xs_b.view(bsize, points, x_size),
                torch.zeros(bsize, points, dim - x_size, device=xs_b.device),
            ),
            axis=2,
        )  # pads xs to make them to be the same dims as of ys

        # else:
        #     assert self.n_dims_out == dim  # TODO also implement case where y's are larger than x's
        #     xs_b_wide = xs_b.view(bsize, points, self.n_dims_out)

        
        # print('xs_shape = ', xs_b_wide.shape)

        zs = torch.stack((ys_b, xs_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs
    
    def forward(self, xs, ys, get_hidden_states=False):
        """
        x: (B, N, n_dims)
        out: (B, N, n_dims_out)
        """
        zs = self._combine(xs_b=xs, ys_b=ys)

        out = self._read_in(zs) # (B, N, n_embd)
        if get_hidden_states:
            hidden_states = [out]
            out = torch.transpose(out, 2, 1)  # (B, N, n_embd) -> (B, n_embd, N)
            for layer in self._backbone:
                out = layer(out)
                hid = torch.transpose(out, 2, 1)  # (B, n_embd, N) -> (B, N, n_embd)
                hidden_states.append(hid)
        else:
            out = torch.transpose(out, 2, 1)  # (B, N, n_embd) -> (B, n_embd, N)
            out = self._backbone(out)
        out = torch.transpose(out, 2, 1)  # (B, n_embd, N) -> (B, N, n_embd)
        logits = self._read_out(out)
        if get_hidden_states:
            return logits[:, ::2, :], hidden_states
        else:
            return logits[:, ::2, :] # return only logit predictions on x



class SeqModelForICD(nn.Module):
    def __init__(self, n_dims, n_positions=100, n_embd=256, n_layer=12, n_dims_out=1,
    backbone_seq_model='lstm', nonlinearity_for_rnn='relu'):
        super(SeqModelForICD, self).__init__()
        self.n_positions = n_positions  # for sake of completeness
        self.n_dims = n_dims  # Dimension of the y's = 2d real dimensions
        self.n_dims_out = n_dims_out  # Dimension of the logits of x's = sig_set_size
        if backbone_seq_model == 'lstm':
            print('LSTM backbone')
            self._backbone = LSTM(input_size=n_dims, hidden_size=n_embd, num_layers=n_layer, batch_first=True)
        else:
            print('RNN backbone with relu non-linearity')
            self._backbone = RNN(input_size=n_dims, hidden_size=n_embd, num_layers=n_layer, batch_first=True, nonlinearity=nonlinearity_for_rnn)
        self._read_out = nn.Linear(n_embd, n_dims_out)  # logits

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = ys_b.shape
        x_size = xs_b.shape[2]
        xs_b_wide = torch.cat(
            (
                xs_b.view(bsize, points, x_size),
                torch.zeros(bsize, points, dim - x_size, device=xs_b.device),
            ),
            axis=2,
        )  # pads xs to make them to be the same dims as of ys
        zs = torch.stack((ys_b, xs_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs
    
    def forward(self, xs, ys):

        zs = self._combine(xs_b=xs, ys_b=ys)  # interleave y1, x1, y2, x2
        backbone_out, _ = self._backbone(zs)
        logits = self._read_out(backbone_out)  # of shape (B, N, S) S is the size of the signal set
        return logits[:, ::2, :]  # LSTM predicts the next states: _x(k+1) from y1, x1, ..., yk, xk, y(k+1)
                                  # and so take only values corresponding to _x(1), _x(2), ..., _x(k)


################### Transformer for In-context Detection (ICD) Problem #########################################

class TransformerForICD(nn.Module):
    """
    Modified from TransformerModel. Input xs and ys are swapped so that the xs are being predicted,
    and the ys are used to predict the xs. Useful for a scenario where the xs are generated iid,
    the ys are generated from the xs, and we want to learn the inverse transformation to go from ys
    to xs.

    Also allows for a variable output dimension (n_dims_out).
    """
    def __init__(
        self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, pos_encode=True, n_dims_out=1
    ):
        super(TransformerForICD, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = (
            f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}_pos_encode{pos_encode}"
        )

        self.n_positions = n_positions
        self.n_dims = n_dims  # Dimension of the y's = 2d real dimensions
        self.n_dims_out = n_dims_out  # Dimension of the logits of x's = sig_set_size
        self._read_in = nn.Linear(n_dims, n_embd)
        if pos_encode:
            self._backbone = GPT2Model(configuration)
        else:
            self._backbone = GPT2ModelWOPosEncodings(configuration)
        self._read_out = nn.Linear(n_embd, n_dims_out)

    def _combine(self, xs_b, ys_b):
        """
        Interleaves the x's and the y's into a single sequence. xs_b and ys_b should be post-swap.
        This is no longer a class method because it uses self.n_dims_out. 
        """
        bsize, points, dim = ys_b.shape
        x_size = xs_b.shape[2]
        # print('ys_shape = ', ys_b.shape)
        # print('xs shape = ', xs_b.shape)
        
        # if self.n_dims_out < dim:  # Pad y's to have the same shape as x's

        xs_b_wide = torch.cat(
            (
                xs_b.view(bsize, points, x_size),
                torch.zeros(bsize, points, dim - x_size, device=xs_b.device),
            ),
            axis=2,
        )
        # else:
        #     assert self.n_dims_out == dim  # TODO also implement case where y's are larger than x's
        #     xs_b_wide = xs_b.view(bsize, points, self.n_dims_out)

        
        # print('xs_shape = ', xs_b_wide.shape)

        zs = torch.stack((ys_b, xs_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None, get_hidden_states=False, get_embeds=False):
        
        # # Swap xs and ys before doing anything else.
        # tmp = xs
        # xs = ys
        # ys = tmp
        # # From here, we refer only to the post-swap variables.

        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(
            xs, ys
        )  # interleaved x and y in n_points dim i.e. x1, y1, x2, y2, ...
        embeds = self._read_in(zs)
        backbone_output = self._backbone(
            inputs_embeds=embeds,
            output_hidden_states=get_hidden_states,
            return_dict=True,
        )
        # if output_hidden_states=True, backbone_output.hidden_states = list of len n_layers with each element of shape (batch_size, n_points * 2, embed_dim). Each element corresponds to the output from the application of i-th layer (0th element being the input and 12th element being the output of last layer)

        # input_shape = embeds.size()[:-1]
        # temp = torch.arange(0, input_shape[-1], dtype=torch.long, device=embeds.device)
        # post_encods = self._backbone.wpe(temp).unsqueeze(0).view(-1, 256)
        # print(post_encods.shape)
        # pos_embeds = embeds + post_encods
        # test_eq = pos_embeds == backbone_output.hidden_states[0]
        # a=torch.all(test_eq)
        # import pdb
        # pdb.set_trace()
        output = backbone_output.last_hidden_state
        pred_logits = self._read_out(output)
        # NOTE: due to n_dims_out, the output will have 1 extra dimension, compared to TransformerModel. Make sure this is properly handled in calling functions.
        if get_hidden_states and get_embeds:
            print('In output_hidden_states and get_embeds')
            return pred_logits[:, ::2, :][:, inds, :], embeds, backbone_output.hidden_states
        elif get_hidden_states:
            print('In output_hidden_states')
            return pred_logits[:, ::2, :][:, inds, :], backbone_output.hidden_states
        elif get_embeds:
            print('In embeds')
            return pred_logits[:, ::2, :][:, inds, :], embeds
        else:
            print('In only pred_logits')
            return pred_logits[:, ::2, :][:, inds, :]