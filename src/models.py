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
    if conf.family == "gpt2":
        try:
            model = TransformerModel(
                n_dims=conf.n_dims,
                n_positions=conf.n_positions,
                n_embd=conf.n_embd,
                n_layer=conf.n_layer,
                n_head=conf.n_head,
                pos_encode=conf.pos_encode,
            )
        except AttributeError:
            model = TransformerModel(
                n_dims=conf.n_dims,
                n_positions=conf.n_positions,
                n_embd=conf.n_embd,
                n_layer=conf.n_layer,
                n_head=conf.n_head,
            )
    # elif conf.family == "gpt2_task_prefix":
    #     model = TaskPrefixTransformerModel(
    #         n_dims=conf.n_dims,
    #         n_positions=conf.n_positions,
    #         n_embd=conf.n_embd,
    #         n_layer=conf.n_layer,
    #         n_head=conf.n_head,
    #         pos_encode=conf.pos_encode,
    #     )
    # elif conf.family == "gpt2_inverse_problem":
    #     model = InverseProblemTransformerModel(
    #         n_dims=conf.n_dims,
    #         n_positions=conf.n_positions,
    #         n_embd=conf.n_embd,
    #         n_layer=conf.n_layer,
    #         n_head=conf.n_head,
    #         pos_encode=conf.pos_encode,
    #         n_dims_out=conf.n_dims_out,
    #     )
    # elif conf.family == "bert_inverse_problem":
    #     model = InverseProblemBidirectionalTransformerModel(
    #         n_dims=conf.n_dims,
    #         n_positions=conf.n_positions,
    #         n_embd=conf.n_embd,
    #         n_layer=conf.n_layer,
    #         n_head=conf.n_head,
    #         pos_encode=conf.pos_encode,
    #         n_dims_out=conf.n_dims_out,
    #     )
    # elif conf.family == "gpt2_estimation":
    #     model = TransformerModel(
    #             n_dims=conf.n_dims,
    #             n_positions=conf.n_positions,
    #             n_embd=conf.n_embd,
    #             n_layer=conf.n_layer,
    #             n_head=conf.n_head,
    #             pos_encode=conf.pos_encode,
    #             est=conf.estimation_task,
    #         )
        # print('In the right init of model')
    elif conf.family == "gpt2_detection":
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


# def get_relevant_baselines(task_name):
#     task_to_baselines = {
#         "linear_regression": [
#             (LeastSquaresModel, {}),
#             (NNModel, {"n_neighbors": 3}),
#             (AveragingModel, {}),
#         ],
#         "gaussian_mixture_linear_regression": [
#             (LeastSquaresModel, {}),
#             (NNModel, {"n_neighbors": 3}),
#             (AveragingModel, {}),
#         ],
#         "linear_classification": [
#             (NNModel, {"n_neighbors": 3}),
#             (AveragingModel, {}),
#         ],
#         "sparse_linear_regression": [
#             (LeastSquaresModel, {}),
#             (NNModel, {"n_neighbors": 3}),
#             (AveragingModel, {}),
#         ]
#         + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]],
#         "relu_2nn_regression": [
#             (LeastSquaresModel, {}),
#             (NNModel, {"n_neighbors": 3}),
#             (AveragingModel, {}),
#             (
#                 GDModel,
#                 {
#                     "model_class": TwoLayerNeuralNetwork,
#                     "model_class_args": {
#                         "in_size": 20,
#                         "hidden_size": 100,
#                         "out_size": 1,
#                     },
#                     "opt_alg": "adam",
#                     "batch_size": 100,
#                     "lr": 5e-3,
#                     "num_steps": 100,
#                 },
#             ),
#         ],
#         "decision_tree": [
#             (LeastSquaresModel, {}),
#             (NNModel, {"n_neighbors": 3}),
#             (DecisionTreeModel, {"max_depth": 4}),
#             (DecisionTreeModel, {"max_depth": None}),
#             (XGBoostModel, {}),
#             (AveragingModel, {}),
#         ],
#     }

#     models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
#     return models



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
    
    def forward(self, xs, ys):
        """
        x: (B, N, n_dims)
        out: (B, N, n_dims_out)
        """
        zs = self._combine(xs_b=xs, ys_b=ys)

        out = self._read_in(zs) # (B, N, n_embd)
        out = torch.transpose(out, 2, 1)  # (B, N, n_embd) -> (B, n_embd, N)
        out = self._backbone(out)
        out = torch.transpose(out, 2, 1)  # (B, n_embd, N) -> (B, N, n_embd)
        logits = self._read_out(out)
        return logits[:, ::2, :] # return only logit predictions on x



class SeqModelForICD(nn.Module):
    def __init__(self, n_dims, n_positions=100, n_embd=256, n_layer=12, n_dims_out=1,
    backbone_seq_model='lstm'):
        super(SeqModelForICD, self).__init__()
        self.n_positions = n_positions  # for sake of completeness
        self.n_dims = n_dims  # Dimension of the y's = 2d real dimensions
        self.n_dims_out = n_dims_out  # Dimension of the logits of x's = sig_set_size
        # self.non_linearity = 'relu'

        if backbone_seq_model == 'lstm':
            print('LSTM backbone')
            self._backbone = LSTM(input_size=n_dims, hidden_size=n_embd, num_layers=n_layer, batch_first=True)
        else:
            print('RNN backbone with relu non-linearity')
            self._backbone = RNN(input_size=n_dims, hidden_size=n_embd, num_layers=n_layer, batch_first=True, nonlinearity='relu')
        self._read_out = nn.Linear(n_embd, n_dims_out)  # logits

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
    
    def forward(self, xs, ys):

        zs = self._combine(xs_b=xs, ys_b=ys)  # interleave y1, x1, y2, x2

        backbone_out, _ = self._backbone(zs)
        # print('LSTM backbone out = ', backbone_out)
        logits = self._read_out(backbone_out)  # of shape (B, N, S) S is the size of the signal set

        return logits[:, ::2, :]  # LSTM predicts the next states: _x(k+1) from y1, x1, ..., yk, xk, y(k+1)
                                  # and so take only values corresponding to _x(1), _x(2), ..., _x(k)

        


class TransformerModel(nn.Module):
    def __init__(
        self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, pos_encode=True, est=False,
    ):
        super(TransformerModel, self).__init__()
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
        self.n_dims = n_dims
        self.is_estimation = est
        self._read_in = nn.Linear(n_dims, n_embd)
        if pos_encode:
            self._backbone = GPT2Model(configuration)
        else:
            self._backbone = GPT2ModelWOPosEncodings(configuration)
        if not self.is_estimation: 
            self._read_out = nn.Linear(n_embd, 1)
        else:
            self._read_out = nn.Linear(n_embd, n_dims)
            # print('Printing the right n_dims!')

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs


    def forward(self, xs, ys, inds=None, output_hidden_states=False):
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
            output_hidden_states=output_hidden_states,
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
        prediction = self._read_out(output)
        if not self.is_estimation:
            if not output_hidden_states:
                return prediction[:, ::2, 0][:, inds]  # predict only on xs
            else:
                return prediction[:, ::2, 0][:, inds], backbone_output.hidden_states
        else:
            # print('Returning the right dimensions')
            return prediction[:, 1::2, :][:, inds]  # predict for ws



class InverseProblemTransformerModel(nn.Module):
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
        super(InverseProblemTransformerModel, self).__init__()
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
        self.n_dims = n_dims  # Dimension of the x's (post-swap)
        self.n_dims_out = n_dims_out  # Dimension of the y's (post-swap)
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
        bsize, points, dim = xs_b.shape
        if self.n_dims_out < dim:  # Pad y's to have the same shape as x's
            ys_b_wide = torch.cat(
                (
                    ys_b.view(bsize, points, self.n_dims_out),
                    torch.zeros(bsize, points, dim - self.n_dims_out, device=ys_b.device),
                ),
                axis=2,
            )
        else:
            assert self.n_dims_out == dim  # TODO also implement case where y's are larger than x's
            ys_b_wide = ys_b.view(bsize, points, self.n_dims_out)
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None, output_hidden_states=False):
        # Swap xs and ys before doing anything else.
        tmp = xs
        xs = ys
        ys = tmp
        # From here, we refer only to the post-swap variables.

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
            output_hidden_states=output_hidden_states,
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
        prediction = self._read_out(output)
        # NOTE: due to n_dims_out, the output will have 1 extra dimension, compared to TransformerModel. Make sure this is properly handled in calling functions.
        if not output_hidden_states:
            return prediction[:, ::2, :][:, inds, :]
        else:
            return prediction[:, ::2, :][:, inds, :], backbone_output.hidden_states
        

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

    def forward(self, xs, ys, inds=None, output_hidden_states=False):
        
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
            output_hidden_states=output_hidden_states,
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
        if not output_hidden_states:
            return pred_logits[:, ::2, :][:, inds, :]
        else:
            return pred_logits[:, ::2, :][:, inds, :], backbone_output.hidden_states




class InverseProblemBidirectionalTransformerModel(nn.Module):
    """
    Modified from InverseProblemTransformerModel. In addition to swapping xs and ys, this uses a BERT
    model instead of a GPT2 model. This model is used for a problem similar to masked language modeling,
    where some of the inputs are masked out and must be predicted.

    NOTE: n_dims and n_dims_out should NOT include the mask dimension. This class will add 1 to those dims
    internally to account for the mask.
    """
    def __init__(
        self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, pos_encode=True, n_dims_out=1
    ):
        assert pos_encode  # For now, no way to disable positional encoding.
        super(InverseProblemBidirectionalTransformerModel, self).__init__()
        # This configuration is similar to the configuration used in the GPT2 model.
        # Some parameters will have different defaults, but the most important ones should
        # be set similarly.
        configuration = BertConfig(
            max_position_embeddings=2 * n_positions,
            hidden_size=n_embd,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            hidden_dropout_prob=0.0,  # corresponds to resid_pdrop
            attention_probs_dropout_prob=0.0,  # corresponds to attn_pdrop
            use_cache=False,
            intermediate_size=4 * n_embd,  # GPT2 sets n_inner to 4 * n_embd
            position_embedding_type='absolute',  # GPT2 also uses absolute position embeddings
            is_decoder=False,
        )
        # configuration = GPT2Config(
        #     n_positions=2 * n_positions,
        #     n_embd=n_embd,
        #     n_layer=n_layer,
        #     n_head=n_head,
        #     resid_pdrop=0.0,
        #     embd_pdrop=0.0,
        #     attn_pdrop=0.0,
        #     use_cache=False,
        # )
        self.name = (
            f"bert_embd={n_embd}_layer={n_layer}_head={n_head}_pos_encode{pos_encode}"
        )

        self.n_positions = n_positions
        self.n_dims = n_dims
        self.n_dims_with_mask = n_dims + 1  # Dimension of the x's (post-swap), including the mask dimension
        self.n_dims_out = n_dims_out
        self._read_in = nn.Linear(self.n_dims_with_mask, n_embd)
        self._backbone = BertModel(configuration, add_pooling_layer=False)
        self._read_out = nn.Linear(n_embd, self.n_dims_out)

    def _combine_and_mask(self, xs_b, ys_b):
        """
        Interleaves the x's and the y's into a single sequence. xs_b and ys_b should be post-swap.
        Also, for any xs and ys which are masked, set those vectors to 0. The zeroth element of the
        vectors should be nonzero if the vector should be masked, and 0 otherwise.
        This is no longer a class method because it uses self.n_dims_out.
        """
        bsize, points, dim = xs_b.shape
        y_mask_dim = ys_b.shape[2]
        if y_mask_dim < dim:  # Pad y's to have the same shape as x's
            ys_b_wide = torch.cat(
                (
                    ys_b.view(bsize, points, y_mask_dim),
                    torch.zeros(bsize, points, dim - y_mask_dim, device=ys_b.device),
                ),
                axis=2,
            )
        else:
            assert y_mask_dim == dim  # TODO also implement case where y's are larger than x's
            ys_b_wide = ys_b.view(bsize, points, y_mask_dim)
        # mask xs
        xs_masks = (xs_b[:, :, 0] == 0).to(xs_b.dtype)  # 1 for vectors to keep, 0 for vectors to mask
        assert xs_masks.shape == (bsize, points)
        xs_b_masked = xs_b * xs_masks.view(bsize, points, 1)
        xs_b_masked[:, :, 0] += xs_b[:, :, 0]  # Add back the mask vector so that the model can use that information

        # mask ys
        ys_masks = (ys_b_wide[:, :, 0] == 0).to(ys_b_wide.dtype)
        assert ys_masks.shape == (bsize, points)
        ys_b_masked = ys_b_wide * ys_masks.view(bsize, points, 1)
        ys_b_masked[:, :, 0] += ys_b_wide[:, :, 0]

        zs = torch.stack((xs_b_masked, ys_b_masked), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None, output_hidden_states=False):
        # Swap xs and ys before doing anything else.
        tmp = xs
        xs = ys
        ys = tmp
        # From here, we refer only to the post-swap variables.

        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine_and_mask(
            xs, ys
        )  # interleaved x and y in n_points dim i.e. x1, y1, x2, y2, ...
        embeds = self._read_in(zs)
        backbone_output = self._backbone(
            inputs_embeds=embeds,
            output_hidden_states=output_hidden_states,
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
        prediction = self._read_out(output)
        # TODO: update the output dim to accommodate x's, then also return x's here.
        if not output_hidden_states:
            return prediction[:, ::2, :][:, inds, :]
        else:
            return prediction[:, ::2, :][:, inds, :], backbone_output.hidden_states


# class TaskPrefixTransformerModel(TransformerModel):
#     def __init__(
#         self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, pos_encode=True
#     ):
#         super(TaskPrefixTransformerModel, self).__init__(
#             n_dims, n_positions, n_embd, n_layer, n_head, pos_encode
#         )
#         self.task_prefix_embeddings = nn.Embedding(2, n_embd)

#     def forward(self, xs, ys, prefix=None, inds=None, output_hidden_states=False):
#         if inds is None:
#             inds = torch.arange(ys.shape[1])
#         else:
#             inds = torch.tensor(inds)
#             if max(inds) >= ys.shape[1] or min(inds) < 0:
#                 raise ValueError("inds contain indices where xs and ys are not defined")

#         if prefix is None:
#             prefix = torch.zeros(xs.shape[0]).to(xs.device).long()
#         zs = self._combine(
#             xs, ys
#         )  # interleaved x and y in n_points dim i.e. x1, y1, x2, y2, ...
#         embeds = self._read_in(zs)
#         prefix_embeds = self.task_prefix_embeddings(prefix).unsqueeze(1)
#         embeds = torch.cat([prefix_embeds, embeds], axis=1)
#         backbone_output = self._backbone(
#             inputs_embeds=embeds,
#             output_hidden_states=output_hidden_states,
#             return_dict=True,
#         )

#         output = backbone_output.last_hidden_state
#         prediction = self._read_out(output)
#         if not output_hidden_states:
#             return prediction[:, 1::2, 0][:, inds]  # predict only on xs
#         else:
#             return prediction[:, 1::2, 0][:, inds], backbone_output.hidden_states


# class NNModel:
#     def __init__(self, n_neighbors, weights="uniform"):
#         # should we be picking k optimally
#         self.n_neighbors = n_neighbors
#         self.weights = weights
#         self.name = f"NN_n={n_neighbors}_{weights}"

#     def __call__(self, xs, ys, inds=None):
#         if inds is None:
#             inds = range(ys.shape[1])
#         else:
#             if max(inds) >= ys.shape[1] or min(inds) < 0:
#                 raise ValueError("inds contain indices where xs and ys are not defined")

#         preds = []

#         for i in inds:
#             if i == 0:
#                 preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
#                 continue
#             train_xs, train_ys = xs[:, :i], ys[:, :i]
#             test_x = xs[:, i : i + 1]
#             dist = (train_xs - test_x).square().sum(dim=2).sqrt()

#             if self.weights == "uniform":
#                 weights = torch.ones_like(dist)
#             else:
#                 weights = 1.0 / dist
#                 inf_mask = torch.isinf(weights).float()  # deal with exact match
#                 inf_row = torch.any(inf_mask, axis=1)
#                 weights[inf_row] = inf_mask[inf_row]

#             pred = []
#             k = min(i, self.n_neighbors)
#             ranks = dist.argsort()[:, :k]
#             for y, w, n in zip(train_ys, weights, ranks):
#                 y, w = y[n], w[n]
#                 pred.append((w * y).sum() / w.sum())
#             preds.append(torch.stack(pred))

#         return torch.stack(preds, dim=1)


# # xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
# class LeastSquaresModel:
#     def __init__(self, driver=None):
#         self.driver = driver
#         self.name = f"OLS_driver={driver}"

#     def __call__(self, xs, ys, inds=None, return_weights=False):
#         xs, ys = xs.cpu(), ys.cpu()
#         if inds is None:
#             inds = range(ys.shape[1])
#         else:
#             if max(inds) >= ys.shape[1] or min(inds) < 0:
#                 raise ValueError("inds contain indices where xs and ys are not defined")

#         preds = []
#         all_ws = []
#         for i in inds:
#             if i == 0:
#                 preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
#                 continue
#             train_xs, train_ys = xs[:, :i], ys[:, :i]
#             test_x = xs[:, i : i + 1]

#             ws, _, _, _ = torch.linalg.lstsq(
#                 train_xs, train_ys.unsqueeze(2), driver=self.driver
#             )
#             all_ws.append(ws)
#             pred = test_x @ ws
#             preds.append(pred[:, 0, 0])

#         if not return_weights:
#             return torch.stack(preds, dim=1)
#         else:
#             return torch.stack(preds, dim=1), all_ws


# class AveragingModel:
#     def __init__(self):
#         self.name = "averaging"

#     def __call__(self, xs, ys, inds=None):
#         if inds is None:
#             inds = range(ys.shape[1])
#         else:
#             if max(inds) >= ys.shape[1] or min(inds) < 0:
#                 raise ValueError("inds contain indices where xs and ys are not defined")

#         preds = []

#         for i in inds:
#             if i == 0:
#                 preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
#                 continue
#             train_xs, train_ys = xs[:, :i], ys[:, :i]
#             test_x = xs[:, i : i + 1]

#             train_zs = train_xs * train_ys.unsqueeze(dim=-1)
#             w_p = train_zs.mean(dim=1).unsqueeze(dim=-1)
#             pred = test_x @ w_p
#             preds.append(pred[:, 0, 0])

#         return torch.stack(preds, dim=1)


# # Lasso regression (for sparse linear regression).
# # Seems to take more time as we decrease alpha.
# class LassoModel:
#     def __init__(self, alpha, max_iter=100000):
#         # the l1 regularizer gets multiplied by alpha.
#         self.alpha = alpha
#         self.max_iter = max_iter
#         self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"

#     # inds is a list containing indices where we want the prediction.
#     # prediction made at all indices by default.
#     def __call__(self, xs, ys, inds=None):
#         xs, ys = xs.cpu(), ys.cpu()

#         if inds is None:
#             inds = range(ys.shape[1])
#         else:
#             if max(inds) >= ys.shape[1] or min(inds) < 0:
#                 raise ValueError("inds contain indices where xs and ys are not defined")

#         preds = []  # predict one for first point

#         # i: loop over num_points
#         # j: loop over bsize
#         for i in inds:
#             pred = torch.zeros_like(ys[:, 0])

#             if i > 0:
#                 pred = torch.zeros_like(ys[:, 0])
#                 for j in range(ys.shape[0]):
#                     train_xs, train_ys = xs[j, :i], ys[j, :i]

#                     # If all points till now have the same label, predict that label.

#                     clf = Lasso(
#                         alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter
#                     )

#                     # Check for convergence.
#                     with warnings.catch_warnings():
#                         warnings.filterwarnings("error")
#                         try:
#                             clf.fit(train_xs, train_ys)
#                         except Warning:
#                             print(f"lasso convergence warning at i={i}, j={j}.")
#                             raise

#                     w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

#                     test_x = xs[j, i : i + 1]
#                     y_pred = (test_x @ w_pred.float()).squeeze(1)
#                     pred[j] = y_pred[0]

#             preds.append(pred)

#         return torch.stack(preds, dim=1)


# # Gradient Descent and variants.
# # Example usage: gd_model = GDModel(NeuralNetwork, {'in_size': 50, 'hidden_size':400, 'out_size' :1}, opt_alg = 'adam', batch_size = 100, lr = 5e-3, num_steps = 200)
# class GDModel:
#     def __init__(
#         self,
#         model_class,
#         model_class_args,
#         opt_alg="sgd",
#         batch_size=1,
#         num_steps=1000,
#         lr=1e-3,
#         loss_name="squared",
#     ):
#         # model_class: torch.nn model class
#         # model_class_args: a dict containing arguments for model_class
#         # opt_alg can be 'sgd' or 'adam'
#         # verbose: whether to print the progress or not
#         # batch_size: batch size for sgd
#         self.model_class = model_class
#         self.model_class_args = model_class_args
#         self.opt_alg = opt_alg
#         self.lr = lr
#         self.batch_size = batch_size
#         self.num_steps = num_steps
#         self.loss_name = loss_name

#         self.name = f"gd_model_class={model_class}_model_class_args={model_class_args}_opt_alg={opt_alg}_lr={lr}_batch_size={batch_size}_num_steps={num_steps}_loss_name={loss_name}"

#     def __call__(self, xs, ys, inds=None, verbose=False, print_step=100):
#         # inds is a list containing indices where we want the prediction.
#         # prediction made at all indices by default.
#         # xs: bsize X npoints X ndim.
#         # ys: bsize X npoints.
#         xs, ys = xs.cuda(), ys.cuda()

#         if inds is None:
#             inds = range(ys.shape[1])
#         else:
#             if max(inds) >= ys.shape[1] or min(inds) < 0:
#                 raise ValueError("inds contain indices where xs and ys are not defined")

#         preds = []  # predict one for first point

#         # i: loop over num_points
#         for i in tqdm(inds):
#             pred = torch.zeros_like(ys[:, 0])
#             model = ParallelNetworks(
#                 ys.shape[0], self.model_class, **self.model_class_args
#             )
#             model.cuda()
#             if i > 0:
#                 pred = torch.zeros_like(ys[:, 0])

#                 train_xs, train_ys = xs[:, :i], ys[:, :i]
#                 test_xs, test_ys = xs[:, i : i + 1], ys[:, i : i + 1]

#                 if self.opt_alg == "sgd":
#                     optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
#                 elif self.opt_alg == "adam":
#                     optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
#                 else:
#                     raise NotImplementedError(f"{self.opt_alg} not implemented.")

#                 if self.loss_name == "squared":
#                     loss_criterion = nn.MSELoss()
#                 else:
#                     raise NotImplementedError(f"{self.loss_name} not implemented.")

#                 # Training loop
#                 for j in range(self.num_steps):
#                     # Prepare batch
#                     mask = torch.zeros(i).bool()
#                     perm = torch.randperm(i)
#                     mask[perm[: self.batch_size]] = True
#                     train_xs_cur, train_ys_cur = train_xs[:, mask, :], train_ys[:, mask]

#                     if verbose and j % print_step == 0:
#                         model.eval()
#                         with torch.no_grad():
#                             outputs = model(train_xs_cur)
#                             loss = loss_criterion(
#                                 outputs[:, :, 0], train_ys_cur
#                             ).detach()
#                             outputs_test = model(test_xs)
#                             test_loss = loss_criterion(
#                                 outputs_test[:, :, 0], test_ys
#                             ).detach()
#                             print(
#                                 f"ind:{i},step:{j}, train_loss:{loss.item()}, test_loss:{test_loss.item()}"
#                             )

#                     optimizer.zero_grad()

#                     model.train()
#                     outputs = model(train_xs_cur)
#                     loss = loss_criterion(outputs[:, :, 0], train_ys_cur)
#                     loss.backward()
#                     optimizer.step()

#                 model.eval()
#                 pred = model(test_xs).detach()

#                 assert pred.shape[1] == 1 and pred.shape[2] == 1
#                 pred = pred[:, 0, 0]

#             preds.append(pred)

#         return torch.stack(preds, dim=1)


# class DecisionTreeModel:
#     def __init__(self, max_depth=None):
#         self.max_depth = max_depth
#         self.name = f"decision_tree_max_depth={max_depth}"

#     # inds is a list containing indices where we want the prediction.
#     # prediction made at all indices by default.
#     def __call__(self, xs, ys, inds=None):
#         xs, ys = xs.cpu(), ys.cpu()

#         if inds is None:
#             inds = range(ys.shape[1])
#         else:
#             if max(inds) >= ys.shape[1] or min(inds) < 0:
#                 raise ValueError("inds contain indices where xs and ys are not defined")

#         preds = []

#         # i: loop over num_points
#         # j: loop over bsize
#         for i in inds:
#             pred = torch.zeros_like(ys[:, 0])

#             if i > 0:
#                 pred = torch.zeros_like(ys[:, 0])
#                 for j in range(ys.shape[0]):
#                     train_xs, train_ys = xs[j, :i], ys[j, :i]

#                     clf = tree.DecisionTreeRegressor(max_depth=self.max_depth)
#                     clf = clf.fit(train_xs, train_ys)
#                     test_x = xs[j, i : i + 1]
#                     y_pred = clf.predict(test_x)
#                     pred[j] = y_pred[0]

#             preds.append(pred)

#         return torch.stack(preds, dim=1)


# class XGBoostModel:
#     def __init__(self):
#         self.name = "xgboost"

#     # inds is a list containing indices where we want the prediction.
#     # prediction made at all indices by default.
#     def __call__(self, xs, ys, inds=None):
#         xs, ys = xs.cpu(), ys.cpu()

#         if inds is None:
#             inds = range(ys.shape[1])
#         else:
#             if max(inds) >= ys.shape[1] or min(inds) < 0:
#                 raise ValueError("inds contain indices where xs and ys are not defined")

#         preds = []

#         # i: loop over num_points
#         # j: loop over bsize
#         for i in tqdm(inds):
#             pred = torch.zeros_like(ys[:, 0])
#             if i > 0:
#                 pred = torch.zeros_like(ys[:, 0])
#                 for j in range(ys.shape[0]):
#                     train_xs, train_ys = xs[j, :i], ys[j, :i]

#                     clf = xgb.XGBRegressor()

#                     clf = clf.fit(train_xs, train_ys)
#                     test_x = xs[j, i : i + 1]
#                     y_pred = clf.predict(test_x)
#                     pred[j] = y_pred[0].item()

#             preds.append(pred)

#         return torch.stack(preds, dim=1)
