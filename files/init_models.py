from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import normalize
from botorch.models.multitask import MultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

import torch


def initialize_model_SGP(problem, train_x, train_obj_true):
    # define models for objective and constraint
    train_x = normalize(train_x, problem.bounds)
    models = []
    for i in range(train_obj_true.shape[-1]):
        train_y = train_obj_true[..., i : i + 1]
        #train_yvar = torch.full_like(train_y, NOISE_SE[i] ** 2)
        models.append(
            SingleTaskGP(
                train_x, train_y, outcome_transform=Standardize(m=1)
            )
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model


def initialize_model_MGP(problem, train_x, train_obj_true):
    train_x = normalize(train_x, problem.bounds)

    t = train_obj_true.shape[-1]
    n = train_x.shape[0]
    d = problem.dim

    # Create task indices: one per row per task, so shape (n * t, 1)
    task_indices = torch.arange(t).repeat(n).unsqueeze(-1)

    # Repeat train_X for each task
    train_X_tiled = train_x.repeat_interleave(t, dim=0)

    # Concatenate task index as last column (shape: (n*t, d+1))
    train_X_mt = torch.cat([train_X_tiled, task_indices.float()], dim=1)

    # Flatten Y to (n*t, 1)
    train_Y_mt = train_obj_true.T.reshape(-1, 1)

    # Set the index of task feature (it's the last column, so index d)
    task_feature_index = d

    # Initialize the model
    model = MultiTaskGP(
        train_X=train_X_mt,
        train_Y=train_Y_mt,
        task_feature=task_feature_index,
    )

    # Fit the model
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    return mll, model

def initialize_model_MGP1(problem, train_x, train_obj_true):
    train_x = normalize(train_x, problem.bounds)

    t = train_obj_true.shape[-1]
    n = train_x.shape[0]
    d = problem.dim

    # Create task indices: one per row per task, so shape (n * t, 1)
    task_indices = torch.arange(t).repeat(n).unsqueeze(-1)

    # Repeat train_X for each task
    train_X_tiled = train_x.repeat_interleave(t, dim=0)

    # Concatenate task index as last column (shape: (n*t, d+1))
    train_X_mt = torch.cat([train_X_tiled, task_indices.float()], dim=1)

    # Flatten Y to (n*t, 1)
    train_Y_mt = train_obj_true.T.reshape(-1, 1)

    # Set the index of task feature (it's the last column, so index d)
    task_feature_index = d

    # Initialize the model
    model = MultiTaskGP(
        train_X=train_X_mt,
        train_Y=train_Y_mt,
        task_feature=task_feature_index,
        rank=1,
    )

    # Fit the model
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    return mll, model
