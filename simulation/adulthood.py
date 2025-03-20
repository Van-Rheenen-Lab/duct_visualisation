    import random
    import numpy as np
    import copy
    import matplotlib.pyplot as plt
    import networkx as nx


    def convert_cells_to_clones(G):
        """
        Update each edge in the graph G so that:
          - "duct_clones" is set to 10 copies of each pubertal ID from "adult_cells"
          - "adult_clones" is set to 10 copies of each adult ID from "adult_cells"
        """
        for (u, v, edata) in G.edges(data=True):
            pubertal_clones = []
            new_adult_clones = []
            for (pub_id, a_id) in edata.get("adult_cells", []):
                pubertal_clones.extend([pub_id] * 10)
                new_adult_clones.extend([a_id] * 10)
            edata["duct_clones"] = pubertal_clones
            edata["adult_clones"] = new_adult_clones
        return G


    def simulate_adulthood(G, rounds=33, progress_data=None, output_graphs=False, seed=42):
        """
        Simulates the adulthood phase on a pubertal ductal tree G.
        This function:
          - Reduces each edge’s cells to 10% and stores them as (pubertal_id, adult_id) pairs in "adult_cells".
          - Iterates for a given number of rounds, updating adult_cells based on neighbor adoption.
          - Optionally records a snapshot of G after every iteration if output_graphs is True.

        Returns:
          - If output_graphs is False: (G, progress_data)
          - If output_graphs is True: (G, progress_data, round_graphs)
        """
        if progress_data is None:
            progress_data = {
                "iteration": [],
                "pubertal_id_counts": [],
                "adult_id_counts": [],
                "unique_clones_per_duct": []
            }
        round_graphs = {} if output_graphs else None

        if seed:
            random.seed(seed)

        # Initial conversion: reduce each edge to 10% of its cells. They will be the adult stem cells.
        # Currently, we are selecting these at random
        global_adult_id_counter = 0
        for (u, v, edata) in G.edges(data=True):
            pubertal_list = edata.get("duct_clones", [])
            if not pubertal_list:
                edata["adult_cells"] = []
                continue
            # Pick 10% (at least 1 cell)
            num_keep = max(1, int(np.round(0.1 * len(pubertal_list))))
            chosen_indices = random.sample(range(len(pubertal_list)), k=num_keep)
            # sort so the duct cells are in order
            chosen_indices.sort()
            new_cells = []
            for i in chosen_indices:
                pub_id = pubertal_list[i]
                global_adult_id_counter += 1
                a_id = f"A_{global_adult_id_counter}"
                new_cells.append([pub_id, a_id])
            edata["adult_cells"] = new_cells

        print(f"Total adult stem cells: {global_adult_id_counter}")

        def record_progress(iteration):
            pub_id_map = {}
            adult_id_map = {}
            unique_per_duct = {}
            for (u, v, edata) in G.edges(data=True):
                for (pub_id, a_id) in edata.get("adult_cells", []):
                    pub_id_map[pub_id] = pub_id_map.get(pub_id, 0) + 1
                    adult_id_map[a_id] = adult_id_map.get(a_id, 0) + 1
                unique_pub_ids = set(cell[0] for cell in edata.get("adult_cells", []))
                unique_per_duct[(u, v)] = len(unique_pub_ids)
            progress_data["iteration"].append(iteration)
            progress_data["pubertal_id_counts"].append(dict(pub_id_map))
            progress_data["adult_id_counts"].append(dict(adult_id_map))
            progress_data["unique_clones_per_duct"].append(unique_per_duct)

        # Record initial state
        record_progress(iteration=0)
        if output_graphs:
            snapshot = copy.deepcopy(G)
            convert_cells_to_clones(snapshot)
            round_graphs[0] = snapshot

        def get_neighbors(u, v, idx):
            edata = G[u][v]
            cells = edata["adult_cells"]
            max_idx = len(cells) - 1

            neighbor_positions = []
            # Same-duct neighbors
            if (idx - 1) >= 0:
                neighbor_positions.append((u, v, idx - 1))
            if (idx + 1) <= max_idx:
                neighbor_positions.append((u, v, idx + 1))

            # If at boundary, check neighboring ducts.
            if idx == 0 or idx == max_idx:
                boundary_node = u if idx == 0 else v
                for nbr_node in G[boundary_node]:
                    if (nbr_node == v and boundary_node == u) or (nbr_node == u and boundary_node == v):
                        continue
                    nbr_edata = G[boundary_node][nbr_node]
                    nbr_cells = nbr_edata.get("adult_cells", [])
                    if not nbr_cells:
                        continue
                    boundary_positions = [0, len(nbr_cells) - 1]
                    for b_idx in boundary_positions:
                        neighbor_positions.append((boundary_node, nbr_node, b_idx))

            valid_neighbors = []
            for (xx_u, xx_v, xx_i) in neighbor_positions:
                cell = G[xx_u][xx_v]["adult_cells"][xx_i]
                if cell[1] is not None:
                    valid_neighbors.append((xx_u, xx_v, xx_i))
            return valid_neighbors

        # Main simulation loop.
        for iteration in range(1, rounds + 1):
            for (u, v, edata) in G.edges(data=True):
                cells_list = edata.get("adult_cells", [])
                n_cells = len(cells_list)
                if n_cells == 0:
                    continue
                k = max(1, int(round(0.1 * n_cells)))
                chosen_indices = random.sample(range(n_cells), k=k)
                for idx in chosen_indices:
                    pub_id, a_id = cells_list[idx]
                    nbrs = get_neighbors(u, v, idx)
                    if nbrs:
                        (nx_u, nx_v, nx_i) = random.choice(nbrs)
                        nbr_pub_id, nbr_adult_id = G[nx_u][nx_v]["adult_cells"][nx_i]
                        # Adopt neighbor's IDs.
                        cells_list[idx][0] = nbr_pub_id
                        cells_list[idx][1] = nbr_adult_id
            record_progress(iteration=iteration)
            if output_graphs:
                snapshot = copy.deepcopy(G)
                convert_cells_to_clones(snapshot)
                round_graphs[iteration] = snapshot

        convert_cells_to_clones(G)
        if output_graphs:
            return G, progress_data, round_graphs
        else:
            return G, progress_data