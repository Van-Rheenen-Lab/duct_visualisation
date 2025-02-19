import random
import numpy as np

def simulate_adulthood(G, rounds=33, progress_data=None):
    """
    Simulates the adulthood phase, given a ductal tree G from puberty.
    Creates adult_cells on each edge, which track (pubertal_id, adult_id) pairs.
    Finally, replicates these into separate arrays: duct_clones (the puberty IDs)
    and adult_clones (the new adult IDs).
    """
    if progress_data is None:
        progress_data = {
            "iteration": [],
            "pubertal_id_counts": [],  # list of dicts
            "adult_id_counts": [],     # list of dicts
            "unique_clones_per_duct": []  # list of dicts (one dict per iteration)
        }

    # 1) Reduce each duct to 10% of the original cells, discarding the rest
    #    Then store them as (pubertal_id, adult_id) in "adult_cells".
    global_adult_id_counter = 0
    for (u, v, edata) in G.edges(data=True):
        pubertal_list = edata.get("duct_clones", [])
        if not pubertal_list:
            edata["adult_cells"] = []
            continue

        # Pick 10% (at least 1 cell)
        num_keep = max(1, int(np.round(0.1 * len(pubertal_list))))
        chosen_indices = random.sample(range(len(pubertal_list)), k=num_keep)

        new_cells = []
        for i in chosen_indices:
            pub_id = pubertal_list[i]
            global_adult_id_counter += 1
            a_id = f"A_{global_adult_id_counter}"
            new_cells.append([pub_id, a_id])

        edata["adult_cells"] = new_cells
        # We can clear puberty clones if we don't need them anymore,
        # or keep them if you prefer. Typically we do this:
        edata["duct_clones"] = []

    print(f"Total adult stem cells: {global_adult_id_counter}")

    # -------------------------------------------------------------------------
    # Helper to record progress
    # -------------------------------------------------------------------------
    def record_progress(iteration):
        pub_id_map = {}
        adult_id_map = {}
        unique_per_duct = {}  # track unique pubertal IDs per duct

        for (u, v, edata) in G.edges(data=True):
            adult_cells = edata.get("adult_cells", [])
            for (pub_id, a_id) in adult_cells:
                # Count pubertal IDs
                pub_id_map[pub_id] = pub_id_map.get(pub_id, 0) + 1
                # Count adult IDs
                adult_id_map[a_id] = adult_id_map.get(a_id, 0) + 1

            # Count unique pubertal IDs in this duct
            unique_pub_ids = set(cell[0] for cell in adult_cells)
            unique_per_duct[(u, v)] = len(unique_pub_ids)

            if len(unique_pub_ids) == 0:
                print(f"Warning: duct {u} to {v} has no unique pubertal IDs!")

        progress_data["iteration"].append(iteration)
        progress_data["pubertal_id_counts"].append(dict(pub_id_map))
        progress_data["adult_id_counts"].append(dict(adult_id_map))
        progress_data["unique_clones_per_duct"].append(unique_per_duct)

    # Record iteration 0
    record_progress(iteration=0)

    # -------------------------------------------------------------------------
    # 2) Define a helper function to find neighboring cells that have an adult_id
    # -------------------------------------------------------------------------
    def get_neighbors(u, v, idx):
        edata = G[u][v]
        cells = edata["adult_cells"]
        max_idx = len(cells) - 1

        neighbor_positions = []
        # same-duct neighbors
        if (idx - 1) >= 0:
            neighbor_positions.append((u, v, idx - 1))
        if (idx + 1) <= max_idx:
            neighbor_positions.append((u, v, idx + 1))

        # boundary check
        if idx == 0 or idx == max_idx:
            boundary_node = u if idx == 0 else v
            # for each neighbor duct
            for nbr_node in G[boundary_node]:
                # skip if it's the same edge
                if (nbr_node == v and boundary_node == u) or \
                   (nbr_node == u and boundary_node == v):
                    continue
                nbr_edata = G[boundary_node][nbr_node]
                nbr_cells = nbr_edata["adult_cells"]
                if not nbr_cells:
                    continue
                # boundary positions in that neighbor duct
                boundary_positions = [0, len(nbr_cells) - 1]
                for b_idx in boundary_positions:
                    neighbor_positions.append((boundary_node, nbr_node, b_idx))

        # Filter out any neighbors that don't have an adult_id
        valid_neighbors = []
        for (xx_u, xx_v, xx_i) in neighbor_positions:
            c = G[xx_u][xx_v]["adult_cells"][xx_i]
            if c[1] is not None:  # must have an adult ID
                valid_neighbors.append((xx_u, xx_v, xx_i))
        return valid_neighbors

    # -------------------------------------------------------------------------
    # 3) Main adulthood simulation loop
    # -------------------------------------------------------------------------
    for iteration in range(1, rounds + 1):
        # For each duct, pick 10% of adult-labeled cells -> attempt neighbor takeover
        for (u, v, edata) in G.edges(data=True):
            cells_list = edata.get("adult_cells", [])
            n_cells = len(cells_list)
            if n_cells == 0:
                continue

            k = max(1, int(round(0.1 * n_cells)))
            chosen_indices = random.sample(range(n_cells), k=k)

            for idx in chosen_indices:
                pub_id, a_id = cells_list[idx]
                # get neighbors that have an adult_id
                nbrs = get_neighbors(u, v, idx)
                if nbrs:
                    (nx_u, nx_v, nx_i) = random.choice(nbrs)
                    nbr_pub_id, nbr_adult_id = G[nx_u][nx_v]["adult_cells"][nx_i]
                    # adopt neighbor's IDs
                    cells_list[idx][0] = nbr_pub_id
                    cells_list[idx][1] = nbr_adult_id
                else:
                    pass  # no neighbor => keep existing ID

        # record iteration data
        record_progress(iteration=iteration)

    # -------------------------------------------------------------------------
    # 4) Final step: replicate each adult cell into separate lists:
    #    - duct_clones (the pubertal IDs)
    #    - adult_clones (the newly assigned adult IDs)
    # -------------------------------------------------------------------------
    for (u, v, edata) in G.edges(data=True):
        pubertal_clones = []
        new_adult_clones = []

        for (pub_id, a_id) in edata.get("adult_cells", []):
            pubertal_clones.extend([pub_id] * 10)   # replicate each pub_id 10x
            new_adult_clones.extend([a_id] * 10)    # replicate each adult_id 10x

        edata["duct_clones"] = pubertal_clones    # or rename if you like
        edata["adult_clones"] = new_adult_clones  # keep them in a separate list

    return G, progress_data
