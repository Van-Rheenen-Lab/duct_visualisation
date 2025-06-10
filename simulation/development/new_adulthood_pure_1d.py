import numpy as np
import random


def simulate_spatial_clones(L=1000000, l=1000, q=0.1, N=50, initial_label_fraction=0.001):
    """
    Simulate clonal dynamics in a 1D tissue of length L over N oestrous cycles,
    with spatial remodeling events of size l.

    Parameters:
        L: Total number of cells in the tissue (should be a multiple of l for simplicity).
        l: Remodeling domain size (number of cells affected in one remodeling event).
        q: Probability that a given domain is activated (remodeled) in each cycle.
        N: Number of cycles (rounds of remodeling) to simulate.
        initial_label_fraction: Fraction of cells initially labeled as distinct clones (others unlabeled).

    Returns:
        A numpy array of length L with clone IDs after N cycles. Clone ID 0 represents unlabeled cells.
    """
    # Initialize tissue as 1D array of cell clone IDs.
    # 0 = unlabeled cell; 1,2,... = different labeled clones.
    tissue = np.zeros(L, dtype=int)

    # Initial labeling: assign unique clone IDs to a random subset of cells.
    # We label approximately initial_label_fraction * L cells (each labeled cell is a new clone).
    num_initial_clones = int(initial_label_fraction * L)
    labeled_positions = np.random.choice(L, size=num_initial_clones, replace=False)
    for i, pos in enumerate(labeled_positions, start=1):
        tissue[pos] = i  # assign clone IDs 1..num_initial_clones

    # Pre-calculate number of non-overlapping domains
    num_domains = L // l  # assume L is divisible by l for non-overlapping domains
    # (If L is not an exact multiple of l, one could ignore or handle the remainder separately.)

    # Loop over each oestrous cycle
    for cycle in range(1, N + 1):
        # Determine which domains are activated this cycle based on probability q
        # We will treat the tissue as divided into `num_domains` segments of length l.
        active_domains = np.random.rand(num_domains) < q  # boolean array of length num_domains

        # Process each active domain
        for domain_idx, is_active in enumerate(active_domains):
            if not is_active:
                continue  # skip domains that are not activated this round

            # Calculate start and end indices of this domain in the tissue array
            domain_start = domain_idx * l
            domain_end = domain_start + l - 1  # inclusive end index

            # Remove central l/2 cells to mimic regression/remodeling.
            # Define the central region [domain_start + l/4, domain_end - l/4] as empty.
            left_border_start = domain_start
            left_border_end = domain_start + l // 4 - 1  # inclusive
            right_border_start = domain_end - l // 4 + 1  # inclusive
            right_border_end = domain_end
            center_start = left_border_end + 1  # = domain_start + l/4
            center_end = right_border_start - 1  # = domain_end - l/4
            # Mark the central region as empty (-1 to indicate no cell present after removal)
            tissue[center_start:center_end + 1] = -1

            # Neighbor-driven replacement: expand clones from border regions into the empty center.
            # We will fill the empty region by allowing cells at the edges of this empty gap to proliferate inward.
            # Use two pointers for the fill front: one from the left edge and one from the right edge of the gap.
            left_fill_idx = center_start  # first empty cell index from the left side
            right_fill_idx = center_end  # first empty cell index from the right side
            # Continue until all empty cells are filled
            # Each step, choose a side (left or right) at random and fill one cell from that side's neighbor.
            while left_fill_idx <= right_fill_idx:
                if random.random() < 0.5:
                    # Try to fill from the left side
                    if left_fill_idx <= right_fill_idx:
                        # Copy the clone ID from the immediate left neighbor (border) into the empty cell
                        tissue[left_fill_idx] = tissue[left_fill_idx - 1]
                        left_fill_idx += 1  # move the left fill front one step to the right
                    else:
                        continue  # if no empty left (should not happen if condition in while)
                else:
                    # Try to fill from the right side
                    if right_fill_idx >= left_fill_idx:
                        # Copy the clone ID from the immediate right neighbor into the empty cell
                        tissue[right_fill_idx] = tissue[right_fill_idx + 1]
                        right_fill_idx -= 1  # move the right fill front one step to the left
                    else:
                        continue
            # At this point, the central region has been filled by clones from the left and right borders.
            # (If a clone was present in the border region, it has expanded into the gap;
            # if border cells were unlabeled (0), they fill the gap with 0, etc.)

            # Additional stochastic neighbor replacement events within this domain
            # to introduce variability (simulating l/4 small random loss-replacement events).
            num_extra_events = l // 4
            for _ in range(num_extra_events):
                # Pick a random adjacent pair within the domain
                i = random.randrange(domain_start, domain_end)  # pick a start index of a neighbor pair
                # Ensure i and i+1 are within domain
                if i >= domain_end:
                    continue  # skip if we somehow picked the last cell (no neighbor to the right)
                # Randomly decide which neighbor replaces which
                if random.random() < 0.5:
                    # right neighbor replaces left neighbor
                    tissue[i] = tissue[i + 1]
                else:
                    # left neighbor replaces right neighbor
                    tissue[i + 1] = tissue[i]
            # End of processing for this active domain
        # (Inactive domains remain unchanged this cycle)
    # End of all cycles

    return tissue


# Example usage:
result_tissue = simulate_spatial_clones(L=100000, l=1000, q=0.1, N=33, initial_label_fraction=1)
# Collect clone sizes for surviving clones (exclude unlabeled ID 0)
unique, counts = np.unique(result_tissue[result_tissue != 0], return_counts=True)
clone_sizes = dict(zip(unique, counts))
print(f"Number of surviving clones: {len(clone_sizes)}")
# print("Sample clone size distribution (clone_id: size):", {k: v for k, v in list(clone_sizes.items())})

print(f"Average clone size: {np.mean(list(clone_sizes.values()))}")
print(f"Maximum clone size: {np.max(list(clone_sizes.values()))}")