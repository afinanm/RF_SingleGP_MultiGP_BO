import argparse
import pandas as pd
import numpy as np
import random
import time
import torch

from utils import generate_initial_data, check_hv_early_stopping, save_plot_hv_progress, save_pareto_plot_2d, save_single_scatter_plot
# from constants import *
from init_models import initialize_model_MGP, initialize_model_SGP, initialize_model_MGP1
from load_data import load_lowdim_data, load_lowdim_neuro_data,load_highdim_data
from problems import RFOptimizationProblem
from optim_functions import optimize_qehvi_and_get_observation, optimize_qnehvi_and_get_observation, optimize_qnparego_and_get_observation

from botorch import fit_gpytorch_mll
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated


#################

SMOKE_TEST = False
BATCH_SIZE = 4
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4

N_BATCH = 70 if not SMOKE_TEST else 5
MC_SAMPLES = 128 if not SMOKE_TEST else 16
verbose = True

#################



def main():
    parser = argparse.ArgumentParser(description="Run BO with different GP models and datasets.")
    parser.add_argument('--gp_model_type', type=str, required=True, choices=["SingleGP", "MultiGP", "MultiGPR1"], help='Type of GP model to use.')
    parser.add_argument('--dataset', type=str, required=True, choices=["LD", "LD_NEURO", "HD"], help='Dataset to use.')
    parser.add_argument('--seed', type=int, required=True, help='Seed for random number generation.')

    args = parser.parse_args()
    gp_model_type = args.gp_model_type
    dataset = args.dataset
    seed = args.seed

    torch.set_default_dtype(torch.double)
    torch.set_default_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


    # TODO: ALBERT ESTÁN LOS BOUNDS IGUAL QUE PUSIMOS PARA EL DATASET LD DE ANTES??
    if dataset == "LD" or dataset == "LD_NEURO":
        bounds = torch.tensor([
            [10.0, 2.0, 2.0, 1.0],
            [300.0, 200.0, 15.0, 15.0]
        ])
    elif dataset == "HD":
        bounds = torch.tensor([
            [10.0, 2.0, 2.0, 1.0],
            [300.0, 600.0, 15.0, 15.0]
        ])
    else:
        raise(ValueError, "Dataset {dataset} is not supported")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


    if dataset == 'LD':
        X_train, X_test, y_train, y_test = load_lowdim_data(seed)
    elif dataset == 'LD_NEURO':
        X_train, X_test, y_train, y_test = load_lowdim_neuro_data(seed)
    else:
        X_train, X_test, y_train, y_test = load_highdim_data(seed)


    problem = RFOptimizationProblem(seed, bounds, X_train, X_test, y_train, y_test)

    print(f"Starting optimization for {gp_model_type} and {dataset} data, with seed {seed}")

    train_x, train_obj_true = generate_initial_data(problem, n=2 * (problem.dim + 1))
    base_train_data = (train_x.clone(), train_obj_true.clone())

    for acquisition in ["qParEGO", "qQEHVI", "qQNEHVI", "Sobol"]:
        log_rows = []
        metadata = {
            "dataset": dataset,
            "gp_model": gp_model_type,
            "acquisition": acquisition,
            "seed": seed,
            "batch_size": BATCH_SIZE,
        }

        train_x, train_obj_true = base_train_data
        hvs = []

        if acquisition == "qParEGO":
            optimizer = optimize_qnparego_and_get_observation
        elif acquisition == "qQEHVI":
            optimizer = optimize_qehvi_and_get_observation
        elif acquisition == "qQNEHVI":
            optimizer = optimize_qnehvi_and_get_observation
        else:
            optimizer = None  # For Sobol

        if acquisition != "Sobol":
            if gp_model_type == "MultiGP":
                mll, model = initialize_model_MGP(problem, train_x, train_obj_true)
            elif gp_model_type == "MultiGPR1":
                mll, model = initialize_model_MGP1(problem, train_x, train_obj_true)
            else:
                mll, model = initialize_model_SGP(problem, train_x, train_obj_true)

        bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_obj_true)
        hvs.append(bd.compute_hypervolume().item())

        iteration = 0
        cumulative_time = 0.0
        while iteration < N_BATCH and check_hv_early_stopping(hvs):
            t0 = time.monotonic()

            if acquisition != "Sobol":
                fit_gpytorch_mll(mll)
                sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
                new_x, new_obj = optimizer(problem, model, train_x, train_obj_true, sampler)
            else:
                new_x, new_obj = generate_initial_data(problem, n=BATCH_SIZE)

            train_x = torch.cat([train_x, new_x])
            train_obj_true = torch.cat([train_obj_true, new_obj])

            bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_obj_true)
            volume = bd.compute_hypervolume().item()
            hvs.append(volume)

            t1 = time.monotonic()
            pareto_front = train_obj_true[is_non_dominated(train_obj_true)]

            iteration_time = round(t1 - t0, 3)
            cumulative_time += iteration_time
            log_rows.append({
                "iteration": iteration + 1,
                "n_evaluations": (iteration + 1) * BATCH_SIZE,
                "suggested_candidates": new_x.cpu().numpy().tolist(),
                "objective_values": new_obj.cpu().numpy().tolist(),
                "objective_mean": new_obj.mean().item(),
                "objective_std": new_obj.std().item(),
                "candidate_norms": torch.norm(new_x, dim=1).cpu().numpy().tolist(),
                "hypervolume": volume,
                "time_sec": iteration_time,
                "cumulative_time_sec": round(cumulative_time, 3),
                "pareto_front": pareto_front.cpu().numpy().tolist(),
                **metadata,
            })

            if acquisition != "Sobol":
                if gp_model_type == "MultiGP":
                    mll, model = initialize_model_MGP(problem, train_x, train_obj_true)
                elif gp_model_type == "MultiGPR1":
                    mll, model = initialize_model_MGP1(problem, train_x, train_obj_true)
                else:
                    mll, model = initialize_model_SGP(problem, train_x, train_obj_true)

            if verbose:
                print(f"[{acquisition}] Batch {iteration + 1}: HV = {volume:.4f}, time = {t1 - t0:.2f}s")

            iteration += 1
        
        #TODO: CAMBIAR RUTAS DE SAVE
        save_plot_hv_progress({acquisition: hvs}, filename=f"../plots/{gp_model_type}/hvs_evolution_{dataset.upper()}_{acquisition.lower()}_{seed}.png")
        save_pareto_plot_2d(train_obj_true.cpu(), filename=f"../plots/{gp_model_type}/final_pareto_front_{dataset.upper()}_{acquisition.lower()}_{seed}.png")
        save_single_scatter_plot(train_obj_true, acquisition, problem.dim, BATCH_SIZE, filename=f"../plots/{gp_model_type}/obj_evolution_{dataset.upper()}_{acquisition.lower()}_{seed}.png")

        df = pd.DataFrame(log_rows)
        df.to_parquet(f'../logs/{gp_model_type}/{dataset.upper()}_{acquisition.lower()}_run_log{seed}.parquet', index=False, engine='pyarrow')


if __name__ == "__main__":
    main()
