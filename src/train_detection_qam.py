import os
import sys
from munch import Munch
from random import randint
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import numpy as np
import torch
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler, sample_scale
from curriculum import Curriculum
from schema import schema
from models import build_model, InverseProblemTransformerModel, InverseProblemBidirectionalTransformerModel
from eval import eval_model, load_into_model_from_run
from eval_encoders import bert_eval_with_masking
import pickle
# from KS_monomial_sets import monomial_terms
# from torch.distributions.multivariate_normal import MultivariateNormal

from samplers import SignalSampler, SignalSamplerQAM

import wandb
import pdb

torch.backends.cudnn.benchmark = True


def train_step(
    model,
    xs,
    ys,
    optimizer,
    loss_func,
    batch_idx,
    max_train_steps,
    k_steps_for_loss="all",
    num_accum_steps=1,
    masking=None,
    masking_method=None,
    s_ids=None
):
    # masking: None by default, for no masking.
    # For any other value, a mask index will be prepended to x and y.
    # "x": y's will not be masked. Some x's will be masked. The number
    # of x's masked is determined by masking_method.
    # The masked samples could be anywhere in the sequence, not necessarily at the beginning
    # and not necessarily contiguous.
    # Only the masked elements will be used in loss calculation.
    # The same mask is used for all sequences in the batch.
    # For now, other options are not implemented.
    # NOTE: for inverse problems, keep in mind that the x and y used here are pre-swap.
    # masking_method: if "uniform", n_mask is selected uniformly at random, between 1 and n_points.
    # Otherwise, this is interpreted as a positive int, which will be n_mask.
    if masking is not None:
        assert masking_method is not None
        bsize, n_points, xdim = xs.shape
        assert ys.shape[0] == bsize
        assert ys.shape[1] == n_points
        ydim = ys.shape[2]
        if masking == "x":
            if masking_method == "uniform":
                n_mask = np.random.choice(n_points) + 1
            else:
                n_mask = int(masking_method)
                assert n_mask > 0
            mask_indices = np.random.choice(a=n_points, size=(n_mask,), replace=False)
            # Masked elements have a nonzero value in the zeroth dim, unmasked elements have a 0.
            y_mask = torch.zeros(bsize, n_points, 1, dtype=ys.dtype, device=ys.device)
            x_mask_single = torch.zeros(1, n_points, 1, dtype=xs.dtype, device=xs.device)
            x_mask_single[0, mask_indices, 0] = 1
            x_mask = torch.tile(x_mask_single, (bsize, 1, 1))
            assert x_mask.shape == (bsize, n_points, 1)

            xs = torch.cat([x_mask, xs], dim=2)
            assert xs.shape == (bsize, n_points, xdim + 1)

            ys = torch.cat([y_mask, ys], dim=2)
            assert ys.shape == (bsize, n_points, ydim + 1)
        else:
            raise NotImplementedError("The only masking option available is 'x'.")

    # optimizer.zero_grad()
    pred_logits = model(xs, ys)

    # print('pred_logits = ', pred_logits.shape)
    # print('s_ids = ', s_ids)

    # print('Output = ', output.shape)

    # if isinstance(model, InverseProblemTransformerModel) or isinstance(model, InverseProblemBidirectionalTransformerModel):
    #     # Swap xs and ys for the purpose of loss computation.
    #     # TODO: better long-term solution
    #     tmp = xs
    #     xs = ys
    #     ys = tmp

    bsize, n_points, n_dims = xs.shape

    if masking is not None:
        if masking == "x":
            # We are now using post-swap variables, so this corresponds to the y's.
            bsize, n_points, out_dim = pred_logits.shape
            assert ys.shape == (bsize, n_points, out_dim + 1)
            # NOTE: torch.nonzero could be used to find the indices which were masked.
            # However, it's easier to just use mask_indices, which we already have.
            pred_logits_masked_subset = pred_logits[:, mask_indices, :]
            # Do the same for ys, but also remove the mask dim.
            s_ids_masked_subset = s_ids[:, mask_indices]
            loss = loss_func(pred_logits_masked_subset, s_ids_masked_subset)
        else:
            raise NotImplementedError("The only masking option available is 'x'.")
    elif k_steps_for_loss == "all":
        loss = loss_func(pred_logits, s_ids)    
    else:
        loss = loss_func(
            pred_logits[:, -int(k_steps_for_loss), :], s_ids[:, -int(k_steps_for_loss)]
        )

    # normalize loss to account for batch accumulation
    loss = loss / num_accum_steps

    loss.backward()
    # optimizer.step()

    if ((batch_idx + 1) % num_accum_steps == 0) or (batch_idx + 1 == max_train_steps):
        optimizer.step()
        optimizer.zero_grad()

    post_probs = torch.softmax(pred_logits, dim=-1)

    return loss.detach().item(), pred_logits.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def wandb_log_task(task, metrics_task, baseline_loss, point_wise_tags, step, suffix=""):
    wandb.log(
        {
            f"{task}_eval{suffix}/overall_loss": np.mean(metrics_task["mean"]),
            f"{task}_eval{suffix}/excess_loss": np.mean(metrics_task["mean"])
            / baseline_loss,
            f"{task}_eval{suffix}/pointwise/loss": dict(
                zip(point_wise_tags, metrics_task["mean"])
            ),
        },
        step=step,
    )


def get_n_points_eval(task, n_dims, task_kwargs, curriculum):
    return curriculum.n_points_schedule.end
    # n_points_eval = 0
    # if 'polynomials_deg2_monomials_selection' in task:
    #     n_points_eval = curriculum.n_points_schedule.end
    # elif 'polynomials' == task or 'polynomials_unbiased_points' == task:
    #     # n_points_eval = 2 * task_kwargs.max_degree + 1
    #     n_points_eval = curriculum.n_points_schedule.end
    # elif "_cs" not in task:
    #     # n_points_eval = 2 * n_dims + 1
    #     # TODO: if we choose to log the inf-norm-optimization performance while training then this will break when n_points_eval > current value of input dimensions + 1
    #     # This is because inf-norm-optimization LPP is feasible only when n_points_eval <= current value of input dimension + 1
    #     # Remedy for this is to use a different n_points_eval value for each solver such that it is always feasible
    #     n_points_eval = curriculum.n_points_schedule.end
    # else:
    #     n_points_eval = curriculum.n_points_schedule.end
    # return n_points_eval


def get_training_optimizer(model, args):
    optimizer = None
    if args.model.train_only_emb:
        # set requires_grad=False for all params
        for param in model.parameters():
            param.requires_grad = False

        # set requires_grad=True for model._read_in i.e. embedding layer
        for param in model._read_in.parameters():
            param.requires_grad = True

        # pass only the params with requires_grad=True to the optimizer
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.training.learning_rate,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    return optimizer


def get_all_deg2_term_indices(n_dims):
    all_deg2_terms = []
    for i in range(n_dims):
        all_deg2_terms.append([i, i])
        for j in range(i + 1, n_dims):
            all_deg2_terms.append([i, j])
    return all_deg2_terms


def validateTaskKwargs(args):
    taskName = args.training.task
    task_kwargs = args.training.task_kwargs
    if taskName == "polynomials_deg2_monomials_selection_unbiased":
        variant = task_kwargs["variant"]
        assert variant in ["fixedS", "fixedK", "randomS"], "invalid variant provided"
        if variant == "fixedS":
            assert (
                len(task_kwargs["fixedS"]) == task_kwargs["numDeg2Select"]
            ), "Length of fixed S is different from number of monomial degree 2 terms to be selected"
            fixedS_array = np.array(task_kwargs["fixedS"])
            assert np.all(
                (0 <= fixedS_array) & (fixedS_array <= args.model.n_dims - 1)
            ), "Some index in fixedS is out of bounds [0, n_dims-1]"
        elif variant == "fixedK":
            fixedK_array = np.array(task_kwargs["fixedK"])
            assert fixedK_array.shape == (
                task_kwargs["sizeOfK"],
                task_kwargs["numDeg2Select"],
                2,
            ), "Shape of fixed K is different from (|K|, |S|, 2) as per config"
            assert np.all(
                (0 <= fixedK_array) & (fixedK_array <= args.model.n_dims - 1)
            ), "For some S in fixedK, some index is out of bounds [0, n_dims-1]"
        elif variant == "randomS":
            assert task_kwargs["numDeg2Select"] < len(
                task_kwargs["all_deg2_terms"]
            ), "|S| must be less than the number of degree 2 terms"


def train(model, args):
    optimizer = get_training_optimizer(model, args)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()
    n_dims = model.n_dims
    bsize = args.training.batch_size
    if args.training.data_transformation_args is not None:
        scale = sample_scale(
            method=args.training.data_transformation_args.get("method", None),
            n_dims=n_dims,
            normalize=args.training.data_transformation_args.get("normalize", False),
            seed=args.training.data_transformation_args.get("seed", None),
        )
    else:
        scale = None

    data_kwargs = args.training.data_kwargs
    if args.training.data == "gaussian":
        if data_kwargs is None:
            data_kwargs = {}
        data_kwargs.update({"scale": scale})

    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims, **data_kwargs)

    excess_tensors = {}
    # excess_tensors_eval = {}
    # if args.training.task == "polynomials_unbiased_points":
    #     excess_tensors["tens"] = torch.empty(0, curriculum.n_points)
    #     excess_tensors_eval["tens"] = torch.empty(
    #         0, 2 * args.training.task_kwargs.max_degree + 1
    #     )
    #     excess_tensors["coefs"] = torch.empty(
    #         0, args.training.task_kwargs.max_degree + 1
    #     )
    #     excess_tensors_eval["coefs"] = torch.empty(
    #         0, args.training.task_kwargs.max_degree + 1
    #     )
    # elif args.training.task == "polynomials_deg2_monomials_selection_unbiased":
    #     # make a list of all the deg 2 monomial term indices
    #     # for n_dims = 20, there are 20 + 190 terms: [[0, 0], [1, 1],...,[19, 19], [0, 1], [0, 2], ....[18, 19]]
    #     all_deg2_terms = get_all_deg2_term_indices(n_dims)
    #     args.training.task_kwargs["all_deg2_terms"] = all_deg2_terms
    #     variant = args.training.task_kwargs["variant"]
    #     if variant == "fixedK":
    #         numDeg2Select = args.training.task_kwargs["numDeg2Select"]
    #         sizeOfK = args.training.task_kwargs["sizeOfK"]
    #         args.training.task_kwargs["fixedK"] = torch.tensor(
    #             monomial_terms[f"{n_dims}-{sizeOfK}-{numDeg2Select}"], dtype=torch.int64
    #         )
    # elif args.training.task == "gaussian_mixture_linear_regression":
    #     mean = torch.zeros(size=(n_dims,))
    #     mean[0] = args.training.task_kwargs["gaussian_centre_abs"]
    #     cov = torch.eye(n_dims)
    #     cov[0, 0] = 1e-8
    #     distrib1 = MultivariateNormal(loc=mean, covariance_matrix=cov)
    #     distrib2 = MultivariateNormal(loc=-mean, covariance_matrix=cov)
    #     args.training.task_kwargs["distrib1"] = distrib1
    #     args.training.task_kwargs["distrib2"] = distrib2
    # elif args.training.task == "haar_wavelets":
    #     # create a dummy object to access haar methods
    #     hw = HaarWavelets(n_dims=1, batch_size=4)
    #     max_level = args.training.task_kwargs["max_level"]
    #     # create the vectorized basis based on max_level of haar wavelets
    #     vectorized_basis = [np.vectorize(f) for f in hw.haar_basis(max_level=max_level)]
    #     args.training.task_kwargs["vectorized_basis"] = vectorized_basis

    validateTaskKwargs(args)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))
    # log also when i+1 == args.training.train_steps

    num_training_examples = args.training.num_training_examples
    num_accum_steps = args.training.num_accum_steps
    optimizer.zero_grad()
    log_loss = 0.0
    log_point_wise_loss = 0.0
    # outputs_list = []
    for i in pbar:
        if i % num_accum_steps == 0:
            log_loss = 0.0
            log_point_wise_loss = torch.zeros(
                size=(curriculum.n_points,), dtype=torch.float32
            ).cuda()

        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse_linear_regression" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        if "fourier_series" in args.training.task:
            args.training.task_kwargs["max_frequency"] = curriculum.max_freq
            task_sampler = get_task_sampler(
                args.training.task,
                n_dims,
                bsize,
                num_tasks=args.training.num_tasks,
                **args.training.task_kwargs,
            )
        elif args.training.task == "random_fourier_features":
            args.training.task_kwargs["rff_dim"] = curriculum.rff_dim
            task_sampler = get_task_sampler(
                args.training.task,
                n_dims,
                bsize,
                num_tasks=args.training.num_tasks,
                **args.training.task_kwargs,
            )

        task = task_sampler(**task_sampler_args)
        # curriculum.n_points = (
        #     task.get_bound() + 1 if "_cs" in args.training.task else curriculum.n_points
        # )
        if isinstance(data_sampler, SignalSampler) or isinstance(data_sampler, SignalSamplerQAM):
            xs, s_ids = data_sampler.sample_xs(
                curriculum.n_points,
                bsize,
                curriculum.n_dims_truncated,
                **data_sampler_args,
            )

            # print('xs shape = ', xs.shape)
        else:
            xs = data_sampler.sample_xs(
                curriculum.n_points,
                bsize,
                curriculum.n_dims_truncated,
                **data_sampler_args,
            )
            s_ids = None
            print('Inside wrong conditional')

        # import time
        # start = time.time()
        # if isinstance(task, PolynomialsUnbiasedPoints):
        #     assert (
        #         n_dims == 1
        #     ), "n_dims is not 1, please change sampling logic for ys s.t. it is from same distribution as xs but is of shape [batch, n_points]"
        #     # form the batch of ys w/ or w/o rejection sampling as needed
        #     ys, _ = task.rejection_sample_to_form_batch(
        #         xs,
        #         data_sampler,
        #         data_sampler_args,
        #         bsize,
        #         curriculum.n_points,
        #         excess_tensors,
        #     )

        ys = task.evaluate(xs)

        # print('ys shape = ', ys.shape)

        # end = time.time()
        # print("time",end - start)
        # outputs_list.append(ys)
        # fname='save_op.pkl'
        # if i % 1000 == 0:
        #     with open(fname, 'wb') as handle:
        #         pickle.dump(outputs_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


        loss_func = task.get_training_metric()
        loss, pred_logits = train_step(
            model,
            xs.cuda(),
            ys.cuda(),
            optimizer,
            loss_func,
            batch_idx=i,
            max_train_steps=args.training.train_steps,
            k_steps_for_loss=args.training.k_steps_for_loss,
            num_accum_steps=num_accum_steps,
            masking=args.training.masking,
            masking_method=args.training.masking_method, 
            s_ids=s_ids.cuda(),
        )

        log_loss += loss
        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()

        # if isinstance(model, InverseProblemTransformerModel) or isinstance(model, InverseProblemBidirectionalTransformerModel):
        #     # Swap xs and ys for the purpose of loss computation.
        #     # TODO: better long-term solution
        #     point_wise_loss = point_wise_loss_func(output, xs.cuda()).mean(dim=0)
        # elif model.is_estimation:
        #     # print('In correct loss computation')
        #     point_wise_loss = point_wise_loss_func(output, ws_targets.cuda()).mean(dim=(0, 2))
        # else:

        point_wise_loss = point_wise_loss_func(pred_logits, s_ids.cuda()).mean(dim=0)
        point_wise_loss = point_wise_loss / num_accum_steps
        log_point_wise_loss += point_wise_loss
        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )
        if args.training.task in ["demodulation", "demodulation_time_varying"]:
            # Baseline loss computation assumes that the problem is linear regression, which this is not.
            # For this task, just set excess_loss = log_loss.
            baseline_loss = 1.0

        if (
            i + 1 == num_accum_steps
            or (  # first log when num_accum_steps are over -- this is equiv. to log at step=0 for non-accumulation training
                i > 0 and (i + 1) % args.wandb.log_every_steps == 0
            )
            or i + 1  # log during training whenever we pass the logging interval
            == args.training.train_steps
        ) and not args.test_run:  # log at the last train step
            wandb.log(
                {
                    "overall_loss": log_loss,
                    "excess_loss": log_loss / baseline_loss,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, log_point_wise_loss.cpu().numpy())
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                    "max_freq": curriculum.max_freq,
                    "rff_dim": curriculum.rff_dim,
                },
                step=(i + 1) // num_accum_steps,
            )

        # if (
        #     i + 1 == num_accum_steps
        #     or (  # first log when num_accum_steps are over -- this is equiv. to log at step=0 for non-accumulation training
        #         i > 0 and (i + 1) % args.training.eval_every_steps == 0
        #     )
        #     or i + 1  # log during training whenever we pass the logging interval
        #     == args.training.train_steps
        # ) and not args.test_run:  # log at the last train step
            
            
        #     n_dims = args.model.n_dims
        #     metrics = eval_model(
        #         model,
        #         task_name=args.training.task,
        #         data_name=args.training.data,
        #         n_dims=args.model.n_dims,
        #         n_points=get_n_points_eval(
        #             args.training.task,
        #             args.model.n_dims,
        #             args.training.task_kwargs,
        #             curriculum,
        #         ),
        #         prompting_strategy="standard",
        #         batch_size=64,
        #         data_sampler_kwargs=data_kwargs,
        #         task_sampler_kwargs=args.training.task_kwargs,
        #         excess_tensors_eval=excess_tensors_eval,
        #     )

        #     wandb_log_task(
        #         args.training.task,
        #         metrics,
        #         baseline_loss,
        #         point_wise_tags,
        #         step=(i + 1) // num_accum_steps,
        #     )

        #     if args.training.eval_ood:
        #         assert args.training.task in [
        #             "polynomials_deg2_monomials_selection_unbiased"
        #         ], "task is not in the list of tasks for OOD evaluation"
        #         metrics_ood = eval_model(
        #             model,
        #             task_name=args.training.task,
        #             data_name=args.training.data,
        #             n_dims=args.model.n_dims,
        #             n_points=get_n_points_eval(
        #                 args.training.task,
        #                 args.model.n_dims,
        #                 args.training.task_kwargs,
        #                 curriculum,
        #             ),
        #             prompting_strategy="standard",
        #             batch_size=64,
        #             data_sampler_kwargs=data_kwargs,
        #             task_sampler_kwargs=args.training.task_kwargs,
        #             excess_tensors_eval=excess_tensors_eval,
        #             eval_ood=True,
        #         )
        #         wandb_log_task(
        #             args.training.task,
        #             metrics_ood,
        #             baseline_loss,
        #             point_wise_tags,
        #             step=(i + 1) // num_accum_steps,
        #             suffix="_ood",
        #         )

            # wandb.log(
            #     {
            #         # f"in-context-score@{curriculum.n_points//4 + 1}": metrics["mean"][
            #         #     curriculum.n_points // 4
            #         # ],
            #         # f"in-context-score@{curriculum.n_points//2 + 1}": metrics["mean"][
            #         #     curriculum.n_points // 2
            #         # ],
            #         # f"in-context-score@{3 * curriculum.n_points//4 + 1}": metrics[
            #         #     "mean"
            #         # ][3 * curriculum.n_points // 4],
            #         f"in-context-score@{curriculum.n_points}": metrics["mean"][
            #             curriculum.n_points - 1
            #         ]
            #         # f"in-context-score@{int(1.5*curriculum.n_points)}": metrics["mean"][
            #         #     int(1.5 * curriculum.n_points)
            #         # ],
            #         # f"in-context-score@{int(2*curriculum.n_points)}": metrics["mean"][
            #         #     int(2 * curriculum.n_points)
            #         # f"in-context-score@{-1}": metrics["mean"][-1],
            #         # f"in-context-score@{n_dims//2}": metrics["mean"][n_dims // 2],
            #         # f"in-context-score@{n_dims}": metrics["mean"][n_dims],
            #         # f"in-context-score@{int(1.5*n_dims)}": metrics["mean"][
            #         #     int(1.5 * n_dims)
            #         # ],
            #         # f"in-context-score@{int(2*n_dims)}": metrics["mean"][
            #         #     int(2 * n_dims)
            #     },
            #     step=i,
            # )

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if (
            i % args.training.save_every_steps == 0
            or i + 1 == args.training.train_steps
        ) and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and (
                i % args.training.keep_every_steps == 0
                or i + 1 == args.training.train_steps
            )
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    

    model = build_model(args.model)
    if args.model.load_model_path is not None:
        run_path = os.path.join(
            "../models", args.training.task, args.model.load_model_path
        )
        load_into_model_from_run(model, run_path)
    model.cuda()
    model.train()

    train(model, args)

    if not args.test_run:
        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


def replaceTaskKwargs(args, taskKwargsToReplace):
    if args.training.task in [
        "polynomials_deg2_monomials_selection_unbiased",
        "decision_tree",
        "relu_2nn_regression",
        "relu_2nn_regression_with_bias",
    ]:
        args.training.task_kwargs = Munch.fromDict(taskKwargsToReplace)
        # n_points = (args.model.n_dims+1)*args.training.task_kwargs.hidden_layer_size+20
        # args.model.n_positions = n_points
        # args.training.curriculum.points.start = n_points
        # args.training.curriculum.points.end = n_points


def is_int_tryexcept(s):
    """Returns int(s) if s is an integer-string else returns s."""
    try:
        int(s)
        return int(s)
    except ValueError:
        return s


def convertToNumber(keyVal):
    key = is_int_tryexcept(keyVal[0])
    val = is_int_tryexcept(keyVal[1])
    return [key, val]


def extractTaskKwargsFromCliArgs():
    # if sys.argv contains task_kwargs return it as a dict else return None
    # A string 'foo=bar,number=6' is made into a dict {'foo': 'bar', 'number': 6}
    # Note that only integer-strings are converted to int; float-strings remain as-is
    # for more complex and nested dict structures, this function does not work
    cli_args = sys.argv[1:]
    task_kwargs_str = None
    task_kwargs = None
    for i in range(len(cli_args)):
        if cli_args[i] == "--training.task_kwargs":
            task_kwargs_str = cli_args[i + 1]
            del sys.argv[i + 1]
            del sys.argv[i + 1]
            break

    if task_kwargs_str is not None:
        # convert keys and values into ints where possible
        task_kwargs = dict(
            convertToNumber(pair.split("=")) for pair in task_kwargs_str.split(",")
        )

    return task_kwargs


def updateStepsByGradAccumSteps(args):
    num_accum_steps = args.training.num_accum_steps
    args.training.train_steps *= num_accum_steps
    args.training.eval_every_steps *= num_accum_steps
    args.training.save_every_steps *= num_accum_steps
    args.training.keep_every_steps *= num_accum_steps

    args.training.curriculum.dims.interval *= num_accum_steps
    args.training.curriculum.points.interval *= num_accum_steps

    args.wandb.log_every_steps *= num_accum_steps


if __name__ == "__main__":
    # taskKwargsToReplace = extractTaskKwargsFromCliArgs()
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    # replaceTaskKwargs(args, taskKwargsToReplace)
    assert args.model.family in ["gpt2", "lstm", "gpt2_inverse_problem", "bert_inverse_problem", "gpt2_estimation", "gpt2_detection"]
    print(f"Running with: {args}")
    updateStepsByGradAccumSteps(args)
    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
