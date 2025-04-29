from constants import *

from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import sample_simplex
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.acquisition.logei import qLogNoisyExpectedImprovement

import torch


def optimize_qehvi_and_get_observation(problem, model, train_x, train_obj, sampler):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles

    normalized_bounds = torch.stack([
                torch.zeros_like(problem.bounds[0]),
                torch.ones_like(problem.bounds[1]),
                ])

    with torch.no_grad():
        pred = model.posterior(normalize(train_x, problem.bounds)).mean
    partitioning = FastNondominatedPartitioning(
        ref_point=problem.ref_point,
        Y=pred,
    )
    acq_func = qLogExpectedHypervolumeImprovement(
        model=model,
        ref_point=problem.ref_point,
        partitioning=partitioning,
        sampler=sampler,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=normalized_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj_true = torch.stack([problem.evaluate(x) for x in new_x])
    #new_noise = new_x[:, -2:]

    #new_obj = new_obj_true + torch.randn_like(new_obj_true) * new_noise
    return new_x, new_obj_true




def optimize_qnehvi_and_get_observation(problem, model, train_x, train_obj, sampler):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    normalized_bounds = torch.stack([
                torch.zeros_like(problem.bounds[0]),
                torch.ones_like(problem.bounds[1]),
                ])

    acq_func = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=problem.ref_point.tolist(),  # use known reference point
        X_baseline=normalize(train_x, problem.bounds),
        prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
        sampler=sampler,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=normalized_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj_true = torch.stack([problem.evaluate(x) for x in new_x])
    #new_noise = new_x[:, -2:]

    #new_obj = new_obj_true + torch.randn_like(new_obj_true) * new_noise
    return new_x, new_obj_true



def optimize_qnparego_and_get_observation(problem, model, train_x, train_obj, sampler):
    """Samples a set of random weights for each candidate in the batch, performs sequential greedy optimization
    of the qNParEGO acquisition function, and returns a new candidate and observation."""

    normalized_bounds = torch.stack([
                torch.zeros_like(problem.bounds[0]),
                torch.ones_like(problem.bounds[1]),
                ])

    train_x = normalize(train_x, problem.bounds)

    with torch.no_grad():
        pred = model.posterior(train_x).mean
    acq_func_list = []
    for _ in range(BATCH_SIZE):
        weights = sample_simplex(problem.num_objectives).squeeze()
        objective = GenericMCObjective(
            get_chebyshev_scalarization(weights=weights, Y=pred)
        )
        acq_func = qLogNoisyExpectedImprovement(  # pyre-ignore: [28]
            model=model,
            objective=objective,
            X_baseline=train_x,
            sampler=sampler,
            prune_baseline=True,
        )
        acq_func_list.append(acq_func)
    # optimize
    candidates, _ = optimize_acqf_list(
        acq_function_list=acq_func_list,
        bounds=normalized_bounds,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
    )
    # observe new values

    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj_true = torch.stack([problem.evaluate(x) for x in new_x])
    #new_noise = new_x[:, -2:]

    #new_obj = new_obj_true + torch.randn_like(new_obj_true) * new_noise
    return new_x, new_obj_true