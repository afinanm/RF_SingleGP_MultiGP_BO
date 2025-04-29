from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.multi_objective.pareto import is_non_dominated
from IPython.display import clear_output
from matplotlib.cm import ScalarMappable

import numpy as np
import torch
import matplotlib.pyplot as plt

# tkwargs = {'dtype': torch.double, 
#            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         }

def generate_initial_data(problem, n=6):
    # Generate training data
    train_x = draw_sobol_samples(bounds=problem.bounds, n=n, q=1).squeeze(1)

    #Evaluate the objective function
    train_obj_true = torch.stack([problem.evaluate(x) for x in train_x])

    #train_obj = train_obj_true + torch.randn_like(train_obj_true)

    return train_x, train_obj_true



def check_hv_early_stopping(hvs_list, threshold=1e-2, patience=5):
    """
    Early stopping if HV has not improved more than `threshold`
    for `patience` consecutive iterations.

    Parameters:
    - hvs_list: list of HV values per iteration
    - threshold: min improvement to be considered "progress"
    - patience: number of allowed consecutive stagnant iterations
    """
    if len(hvs_list) < patience:
        return True

    # Count how many iterations since last significant improvement
    stagnant_iters = 0
    for i in range(len(hvs_list) - 1, 0, -1):
        delta = hvs_list[i] - hvs_list[i - 1]
        if delta < threshold:
            stagnant_iters += 1
        else:
            break  # reset count on any significant improvement

        if stagnant_iters >= patience:
            return False

    return True


# def should_continue(hvs_list, current_volume, tol_window=3, tolerance = 0.0001):
#     tol_window += 1
#     if len(hvs_list) < tol_window:
#         return True
#     prev_mean = sum(hvs_list[-tol_window:-1]) / tol_window
#     return abs(current_volume - prev_mean) < tolerance


def save_plot_hv_progress(hv_dict, filename):
    clear_output(wait=True)
    plt.figure(figsize=(10, 5))

    for label, values in hv_dict.items():
        plt.plot(values, label=label)

    plt.xlabel("Iteration")
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



def save_pareto_plot_2d(Y, filename):
    pareto_mask = is_non_dominated(Y)
    pareto_Y = Y[pareto_mask]

    plt.figure(figsize=(6, 5))
    plt.scatter(Y[:, 0].numpy(), Y[:, 1].numpy(), label='All Points', alpha=0.3)
    plt.scatter(pareto_Y[:, 0].numpy(), pareto_Y[:, 1].numpy(), color='red', label='Pareto Front')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title("Final Pareto Front")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()




def save_single_scatter_plot(train_obj, method_name, problem_dim, batch_size, filename):
    """
    Save a scatter plot for a single optimization method.

    Parameters:
    - train_obj (torch.Tensor): Tensor of objective values.
    - method_name (str): Name of the method (e.g., "qNEHVI").
    - problem_dim (int): Dimensionality of the problem.
    - batch_size (int): Number of points added per iteration.
    - obj_x (int): Index of the objective for the x-axis.
    - obj_y (int): Index of the objective for the y-axis.
    - filename (str): Output filename (e.g., "qnehvi_plot.png").
    """
    cm = plt.get_cmap("viridis")
    obj_np = train_obj.cpu().numpy()

    # Determine initial and added samples
    n_total = obj_np.shape[0]
    n_initial = 2 * (problem_dim + 1)
    n_added = n_total - n_initial
    n_batches = n_added // batch_size

    # Color coding by batch number
    batch_number = np.concatenate([
        np.zeros(n_initial),
        np.repeat(np.arange(1, n_batches + 1), batch_size)
    ])

    # Normalize colors
    norm = plt.Normalize(batch_number.min(), batch_number.max())
    sm = ScalarMappable(norm=norm, cmap=cm)
    sm.set_array([])

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        obj_np[:, 0],
        obj_np[:, 1],
        c=batch_number,
        cmap=cm,
        alpha=0.8
    )

    ax.set_title(method_name)
    ax.set_xlabel(f"Objective {0 + 1}")
    ax.set_ylabel(f"Objective {1 + 1}")
    ax.grid(True)

    # Add colorbar
    cbar = fig.colorbar(sm, ax=ax)
    cbar.ax.set_title("Iteration")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
