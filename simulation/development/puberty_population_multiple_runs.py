import matplotlib.pyplot as plt
import numpy as np
import random

from simulation.puberty_deposit_elimination import simulate_ductal_tree
from simulation.utils.plotting_simulated_ducts import plotting_ducts
from simulation.adulthood import simulate_adulthood


plt.rcParams.update({'font.size': 16,
                     'font.family': 'Arial',
                     'axes.titlesize': 16,
                     'axes.labelsize': 16,
                     'xtick.labelsize': 16,
                     'ytick.labelsize': 16})

def _pad_forward_fill(x, L):
    """Pad 1-D array/list x to length L with its last value."""
    x = np.asarray(x, dtype=float)
    if x.size < L:
        last = x[-1]
        pad = np.full(L - x.size, last, dtype=float)
        return np.concatenate([x, pad])
    return x

# ── puberty ↔ adult-0 metrics (incl. eliminated-clone size) ──────────────
import numpy as np, collections as _c

def puberty_vs_adult0(G_puberty, progress_adult):
    """Return dict with averages, totals, retention and avg_cells_lost."""
    ad_counts = progress_adult["pubertal_id_counts"][0]          # adult-0
    ad_avg    = np.mean(list(ad_counts.values())) if ad_counts else 0.
    ad_total  = len(ad_counts)

    pub_counts = _c.Counter()
    for _, _, e in G_puberty.edges(data=True):
        pub_counts.update(e.get("duct_clones", []))
    pub_avg   = np.mean(list(pub_counts.values())) if pub_counts else 0.
    pub_total = len(pub_counts)

    lost_ids  = set(pub_counts) - set(ad_counts)
    lost_avg  = np.mean([pub_counts[i] for i in lost_ids]) if lost_ids else 0.

    return dict(
        avg_puberty     = pub_avg,
        avg_adult0      = ad_avg,
        delta_avg       = ad_avg - pub_avg,
        total_puberty   = pub_total,
        total_adult0    = ad_total,
        retention       = ad_total / pub_total if pub_total else 0.,
        avg_cells_lost  = lost_avg,          # ← what you asked for
    )

# ── extra diversity metrics ────────────────────────────────────────────────
import numpy as np, collections as _c

# ── diversity metrics ──────────────────────────────────────────────
import numpy as np, collections as _c

def diversity_metrics(G_puberty, progress_adult):
    """
    adult_avg_per_duct  – mean # unique pubertal IDs per duct at adult-0
    adult_total_unique  – total unique pubertal IDs that survive into adult-0
    puberty_avg_per_duct – mean # unique pubertal IDs per duct at puberty end
    """
    # adult-0 snapshot
    ad_per_duct = progress_adult["unique_clones_per_duct"][0].values()
    adult_avg_per_duct = float(np.mean(list(ad_per_duct))) if ad_per_duct else 0.
    adult_total_unique = len(progress_adult["pubertal_id_counts"][0])

    # puberty end
    per_duct = []
    for _, _, e in G_puberty.edges(data=True):
        ids = set(e.get("duct_clones", []))
        if ids:
            per_duct.append(len(ids))
    puberty_avg_per_duct = float(np.mean(per_duct)) if per_duct else 0.

    return adult_avg_per_duct, adult_total_unique, puberty_avg_per_duct




def _pad_to_length(x, L):
    """Return a copy of 1‑D array/list x padded with NaN to length L.
    If we want to see only the remaining values
    """
    x = np.asarray(x, dtype=float)
    if x.size < L:
        pad = np.full(L - x.size, np.nan)
        return np.concatenate([x, pad])
    return x



def run_multiple_sims(
    n_reps: int = 15,
    base_seed: int = 40,
    min_final_cells: int | None = 2_000_000,
    max_attempts: int = 100,
):
    """
    Run `simulate_ductal_tree` until *n_reps* qualified replicates each reach at
    least *min_final_cells* total cells.  Aggregate the results (NaN‑robust),
    plot the ensemble statistics, and return the averaged series.

    Parameters
    ----------
    n_reps : int
        Number of successful simulations to collect.
    base_seed : int
        Baseline RNG seed.  Each *attempt* increments this by one.
    min_final_cells : int | None
        Minimum total cell count required at the final iteration.  Pass ``None``
        to accept every run regardless of size.
    max_attempts : int
        Safety cap on the number of simulation attempts to avoid infinite loops
        if the threshold is set too high.

    Returns
    -------
    avg_series : dict[str, np.ndarray]
        Averaged time‑series with keys
        ``'iterations', 'total_cells', 'num_active_tebs',
        'avg_dom_frac', 'avg_dom_stem_frac', 'unique_stem_counts'``.
    all_progress : list[dict]
        Raw ``progress_data`` dict from every **accepted** replicate.
    """
    all_progress: list[dict] = []
    graphs = []
    metrics_list = []
    lost_avgs = []
    adult_unique_avgs, puberty_unique_avgs, puberty_unique_totals = [], [], []

    max_len = 0
    attempt = 0

    while len(all_progress) < n_reps and attempt < max_attempts:
        random.seed(base_seed + attempt)
        n_clones = 170
        bifurcation_prob = 0.01
        initial_termination_prob = 0.25
        final_termination_prob = 0.55
        max_cells = 3_000_000

        G, prog = simulate_ductal_tree(
            max_cells=max_cells,
            bifurcation_prob=bifurcation_prob,
            initial_side_count=n_clones // 2,
            initial_center_count=n_clones // 2,
            initial_termination_prob=initial_termination_prob,
            final_termination_prob=final_termination_prob,
        )
        attempt += 1

        final_cells = prog["total_cells"][-1]
        if (min_final_cells is None) or (final_cells >= min_final_cells):
            all_progress.append(prog)


            G_adult0, prog_adult0 = simulate_adulthood(
                G.copy(), rounds=0, progress_data=None, output_graphs=False,
                seed=base_seed + attempt)
            adult_u, pub_u, pub_tot = diversity_metrics(G, prog_adult0)
            adult_unique_avgs.append(adult_u)
            puberty_unique_avgs.append(pub_u)
            puberty_unique_totals.append(pub_tot)

            m = puberty_vs_adult0(G, prog_adult0)
            metrics_list.append(m)
            lost_avgs.append(m["avg_cells_lost"])

            graphs.append(G)
            max_len = max(max_len, len(prog["iteration"]))
            print(
                f"Run {len(all_progress)}/{n_reps} accepted "
                f"({final_cells:,d} cells)"
            )
        else:
            print(
                f"Run rejected ({final_cells:,d} cells; needs "
                f"\u2265{min_final_cells:,d})"
            )

    if len(all_progress) < n_reps:
        raise RuntimeError(
            f"Only {len(all_progress)} runs reached the required "
            f"{min_final_cells:,d} cells after {attempt} attempts."
        )

    # ------------------------------------------------------------------
    # 2) Stack and average time‑series (NaN‑aware)
    # ------------------------------------------------------------------
    keys = [
        "iteration",
        "total_cells",
        "num_active_tebs",
        "avg_dom_fraction",
        "avg_dom_stem_fraction",
        "unique_stem_counts",
    ]
    stacked = {k: [] for k in keys}

    for prog in all_progress:
        for k in keys:
            stacked[k].append(_pad_forward_fill(prog[k], max_len))

    for k in keys:
        stacked[k] = np.vstack(stacked[k])  # shape (n_runs, max_len)

    avg_series = {k: np.nanmean(stacked[k], axis=0) for k in keys}
    avg_series["iterations"] = avg_series.pop("iteration")

    # ------------------------------------------------------------------
    # 2b) Average clone-size histogram across runs
    # ------------------------------------------------------------------
    clone_sizes_per_run = []
    for prog in all_progress:
        final_counts = prog["clone_counts_over_time"][-1]
        total_cells = sum(final_counts.values())
        # fractions (≥ 1/total_cells, ≤ 1)
        sizes = np.array([cnt / total_cells for cnt in final_counts.values() if cnt > 0])
        clone_sizes_per_run.append(sizes)

    all_sizes = np.concatenate(clone_sizes_per_run)

    nbins = 40
    lo, hi = all_sizes.min(), all_sizes.max()
    bins = np.logspace(np.log10(lo), np.log10(hi), nbins + 1)

    hist_mat = np.vstack([np.histogram(s, bins=bins)[0] for s in clone_sizes_per_run])

    mean_hist = hist_mat.mean(axis=0)
    std_hist = hist_mat.std(axis=0, ddof=1)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])

    plt.figure(figsize=(6, 4))
    plt.bar(
        bin_centers,
        mean_hist,
        width=np.diff(bins),
        align="center",
        alpha=0.8,
        edgecolor="k",
        label=f"mean of {len(all_progress)} runs",
    )

    plt.xscale("log")
    plt.xlabel("Clone fraction (of gland)")
    plt.ylabel("Mean count per size bin")
    plt.title("Average clone size distribution")
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------------
    # 3) Plot ensemble time‑series
    # ------------------------------------------------------------------

    def _plot_with_mean(ykey: str, ylabel: str, title: str, color: str):
        plt.figure(figsize=(8, 5))
        # thin lines = individual runs
        for y in stacked[ykey]:
            plt.plot(avg_series["iterations"], y, lw=0.8, alpha=0.15, color=color)
        # thick line = mean
        plt.plot(
            avg_series["iterations"],
            avg_series[ykey],
            lw=2.0,
            color=color,
            label=f"Mean of {len(all_progress)} runs",
        )
        plt.xlabel("Time (simulation iterations)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()

    _plot_with_mean("total_cells", "Total # of Cells", "Total Population (mean ± individuals)", color="blue")
    _plot_with_mean("num_active_tebs", "# Active TEBs", "Active TEBs (mean ± individuals)", color="green")
    _plot_with_mean("avg_dom_fraction", "Fraction", "Dominant Clone Fraction (stem + deposited)", color="red")
    _plot_with_mean("avg_dom_stem_fraction", "Fraction", "Dominant Clone Fraction (stem only)", color="red")
    # save this one as SVG
    plt.savefig("avg_dom_fraction.svg", format="svg", bbox_inches="tight")

    _plot_with_mean("unique_stem_counts", "# Unique stem cells", "Stem cell Diversity in active TEBs", color="purple")

    teb_hist = all_progress[0]["teb_history"]
    plt.figure(figsize=(9, 5))
    for node_id, d in teb_hist.items():
        plt.plot(d["iteration"], d["dominant_clone_fraction"], alpha=0.6)
    plt.xlabel("Time (simulation iterations)")
    plt.ylabel("Dominant Clone Fraction")
    plt.title("Per TEB Dominant Clone Fraction (run 0)")
    plt.tight_layout()

    print(
        f"\nAcross {len(all_progress)} runs:"
        f"\n  ⟨unique stem cells per duct⟩ at adult-0     = {np.mean(adult_unique_avgs):.2f}"
        f"\n  ⟨unique pubertal stem cells per gland⟩        = {np.mean(puberty_unique_avgs):.2f}"
        f"\n  ⟨pubertal cells per eliminated clone⟩        = {np.mean(lost_avgs):.2f}"
    )

    plt.show()

    return avg_series, all_progress


if __name__ == "__main__":
    run_multiple_sims(n_reps=50, min_final_cells=1_000_000)
