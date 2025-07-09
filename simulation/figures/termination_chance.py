import numpy as np
import matplotlib.pyplot as plt

# set matplotlib parameters for better visibility
plt.rcParams.update({'font.size': 14, 'font.family': 'arial'})

def plot_termination_bifurcation_curve(
    max_cells: int = 3_000_000,
    bifurcation_prob: float = 0.01,
    final_termination_prob: float = 0.55,
    initial_termination_prob: float = 0.25,
    n_points: int = 601
) -> None:
    """
    Plot the per-time-step probabilities of (i) termination and
    (ii) successful branching as a function of the *current* total
    cell count N.

    Termination (given a branch attempt) is defined in your code as
        pₜ(N) = p₀ + N / N_max
    (capped to 1).
    Therefore the actual event probabilities are

    •  **Termination**    P_term(N) = p_b · pₜ(N)
    •  **Branch**         P_branch(N) = p_b · (1 − pₜ(N))

    where `p_b = bifurcation_prob` and `p₀ = initial_termination_prob`.
    """
    N = np.linspace(0, max_cells*2, n_points)

    N_max = max_cells * 2 * (final_termination_prob - 0.5) / (final_termination_prob - initial_termination_prob) ** 2

    p_t_cond = np.clip(initial_termination_prob + N / N_max, 0.0, final_termination_prob)

    P_term   = bifurcation_prob * p_t_cond
    P_branch = bifurcation_prob * (1.0 - p_t_cond)

    plt.figure(figsize=(6, 4))
    plt.plot(N, P_term,   label="Termination")
    plt.plot(N, P_branch, label="Bifurcation")
    plt.xlabel("Total cells in simulated gland")
    plt.ylabel("Probability")
    # plt.title("Termination vs. Branching probabilities\nas a function of total cell count")
    plt.legend()
    plt.tight_layout()

    # save the figure
    plt.savefig("termination_bifurcation_curve.png", dpi=300)
    # also save vector
    plt.savefig("termination_bifurcation_curve.svg")
    # also save as PDF
    plt.savefig("termination_bifurcation_curve.pdf")


    plt.show()

if __name__ == "__main__":
    plot_termination_bifurcation_curve(
        max_cells=3_000_000,
        bifurcation_prob=0.01,
        initial_termination_prob=0.25,
        final_termination_prob=0.55,
        n_points=6001
    )