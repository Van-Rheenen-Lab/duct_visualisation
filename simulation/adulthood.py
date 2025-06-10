import random
import numpy as np
import copy


def convert_cells_to_clones(G):
    """
    Update each edge in the graph G so that:
      - "duct_clones" is set to 10 copies of each pubertal ID from "adult_cells"
      - "adult_clones" is set to 10 copies of each adult ID from "adult_cells"

    We do this to make the stem cell -> adult cell mapping clear.
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

pass
def duplicate_border_cells(border_cells, target_length):
    """
    Given a list of border cell identities (each cell is a [pub_id, a_id] pair),
    repeatedly choose one at random and insert a duplicate immediately after its first occurrence,
    until the list reaches target_length. This ensures that if a cell is chosen to duplicate,
    its duplicate appears adjacent to it.
    """
    new_list = list(border_cells)
    while len(new_list) < target_length:
        i = random.randrange(len(new_list))
        new_list.insert(i + 1, copy.deepcopy(new_list[i]))
    return new_list


def simulate_adulthood(G, rounds=33, progress_data=None, output_graphs=False,
                       seed=42, remodel_length=20, remodel_prop=0.10):
    """
    Simulates the adulthood phase on a pubertal ductal tree G with a spatially-aware remodeling model.
    Adapted from the supplementary data 4 of the paper "Mechanisms that clear mutations drive field cancerization in
    mammary tissue", Ciwinska, Hristova and Messal et al. (2024).

    The graph is conceptually ordered as a 1D array of adult cells.

    In each round (representing an oestrous cycle):
      - We continue selecting non-overlapping contiguous domains (each of up to remodel_length cells)
        until the total number of cells in their central (to-be-remodeled) regions reaches at least
        remodel_prop * (total_cells). (If a duct ends with no adjacent neighbor, the domain is truncated.)

      - For each selected domain, the central half is removed, leaving the border cells.
        The left border is taken as ceil(remodel_length/4) cells and the right border as floor(remodel_length/4) cells.

      - Then the missing cells are replaced by duplicating the border cells:
            The combined border list is duplicated by randomly choosing a cell from it and inserting its duplicate
            immediately after it, until the total number of cells in the domain equals the original domain length.

    Progress data (under key "iteration") is recorded each round.

    Parameters:
      G: networkx.Graph with edges having an "adult_cells" list (each cell is a [pub_id, adult_id] pair).
      rounds: Number of remodeling rounds to simulate.
      progress_data: Dictionary to record progress (if None, one is created).
      output_graphs: If True, deep-copy snapshots of G are saved after each round.
      seed: Random seed.
      remodel_length: Domain size for each remodeling event (default = 1000).
      remodel_prop: Target fraction of total cells (as central region cells) to remodel each round.

    Returns:
      If output_graphs is False: (G, progress_data)
      If output_graphs is True: (G, progress_data, round_graphs)
    """
    # Set up progress data.
    if progress_data is None:
        progress_data = {
            "iteration": [],
            "pubertal_id_counts": [],
            "adult_id_counts": [],
            "unique_clones_per_duct": []
        }
    round_graphs = {} if output_graphs else None

    # Set seed.
    if seed:
        random.seed(seed)
        np.random.seed(seed)

    # Initial conversion: reduce each edge's cells to 10% of its duct_clones. Then we're left with only adult MaSCs
    global_adult_id_counter = 0
    for (u, v, edata) in G.edges(data=True):
        pubertal_list = edata.get("duct_clones", [])
        if not pubertal_list:
            edata["adult_cells"] = []
            continue
        num_keep = max(1, int(round(0.1 * len(pubertal_list))))
        chosen_indices = random.sample(range(len(pubertal_list)), k=num_keep)
        chosen_indices.sort()
        new_cells = []
        for i in chosen_indices:
            pub_id = pubertal_list[i]
            global_adult_id_counter += 1
            a_id = global_adult_id_counter
            new_cells.append([pub_id, a_id])
        edata["adult_cells"] = new_cells

    print(f"Total adult stem cells initially: {global_adult_id_counter}")

    # Build global ordering of adult cells.
    edges_info = []
    total_cells = 0
    for u, v, data in G.edges(data=True):
        cell_count = len(data.get('adult_cells', []))
        if cell_count == 0:
            continue
        edges_info.append(((u, v), cell_count))
        total_cells += cell_count

    # Create a global list of positions: each is (edge, local_index).
    global_positions = []
    for (edge, count) in edges_info:
        for i in range(count):
            global_positions.append((edge, i))

    def pick_contiguous_domain_from_global(start_idx):
        """
        Starting from a given global index in global_positions, follow connectivity to collect
        a contiguous domain of up to remodel_length cells. If a duct boundary is reached with no
        adjacent edge available, the domain is truncated.
        Returns a list of (edge, local_index) positions.
        """
        domain = []
        try:
            current_position = global_positions[start_idx]
        except IndexError:
            return None
        domain.append(current_position)
        cells_needed = remodel_length - 1
        current_edge, current_index = current_position
        visited_edges = set()
        visited_edges.add(tuple(sorted(current_edge)))
        current_direction = 1  # moving forward along the edge.
        while cells_needed > 0:
            edge_cells = G.edges[current_edge]['adult_cells']
            next_index = current_index + current_direction
            if 0 <= next_index < len(edge_cells):
                current_index = next_index
                domain.append((current_edge, current_index))
                cells_needed -= 1
            else:
                u, v = current_edge
                current_node = v if current_direction == 1 else u
                next_edge_options = []
                for nbr in G.neighbors(current_node):
                    e = (current_node, nbr)
                    if not G.has_edge(*e):
                        e = (nbr, current_node)
                    if tuple(sorted(e)) in visited_edges:
                        continue
                    if e == current_edge or not G.has_edge(*e):
                        continue
                    next_edge_options.append(e)
                if not next_edge_options:
                    break  # domain is truncated.
                next_edge = random.choice(next_edge_options)
                visited_edges.add(tuple(sorted(next_edge)))
                u2, v2 = next_edge
                if current_node == u2:
                    current_direction = 1
                    current_index = 0
                else:
                    current_direction = -1
                    current_index = len(G.edges[next_edge]['adult_cells']) - 1
                current_edge = next_edge
                domain.append((current_edge, current_index))
                cells_needed -= 1
        return domain

    def record_progress(iteration):
        pub_id_map = {}
        adult_id_map = {}
        unique_per_duct = {}
        for (u, v, data) in G.edges(data=True):
            for (pub_id, a_id) in data.get("adult_cells", []):
                pub_id_map[pub_id] = pub_id_map.get(pub_id, 0) + 1
                adult_id_map[a_id] = adult_id_map.get(a_id, 0) + 1
            unique_pub_ids = set(cell[0] for cell in data.get("adult_cells", []))
            unique_per_duct[(u, v)] = len(unique_pub_ids)
        progress_data["iteration"].append(iteration)
        progress_data["pubertal_id_counts"].append(dict(pub_id_map))
        progress_data["adult_id_counts"].append(dict(adult_id_map))
        progress_data["unique_clones_per_duct"].append(unique_per_duct)

    record_progress(iteration=0)
    if output_graphs:
        snapshot = copy.deepcopy(G)
        convert_cells_to_clones(snapshot)
        round_graphs[0] = snapshot

    # Target: remodel at least remodel_prop * total_cells cells (in central regions) per round.
    target_remodeled = remodel_prop * total_cells

    for iteration in range(1, rounds + 1):
        used_positions = set()
        selected_domains = []
        remodeled_cells = 0
        attempts = 0
        while remodeled_cells < target_remodeled:
            start_idx = random.randrange(total_cells)
            domain = pick_contiguous_domain_from_global(start_idx)
            if domain is None or len(domain) == 0:
                attempts += 1
                continue
            canonical = {(tuple(sorted(edge)), idx) for (edge, idx) in domain}
            if canonical & used_positions:
                attempts += 1
                continue
            selected_domains.append(domain)
            remodeled_cells += len(domain)
            used_positions.update(canonical)
            attempts += 1

        for domain in selected_domains:
            l = len(domain)
            if l < 4:
                continue
            # Determine border sizes using rounding:
            left_border_count = int(np.ceil(l / 4))
            right_border_count = int(np.floor(l / 4))
            # Extract border cells from domain positions.
            border_left_cells = [G.edges[edge]['adult_cells'][idx] for (edge, idx) in domain[:left_border_count]
                                 if G.edges[edge]['adult_cells'][idx] is not None]
            border_right_cells = [G.edges[edge]['adult_cells'][idx] for (edge, idx) in domain[-right_border_count:]
                                  if G.edges[edge]['adult_cells'][idx] is not None]
            initial_border = border_left_cells + border_right_cells
            if not initial_border:
                continue
            # The final remodeled domain should have length l.
            # The gap to fill is l - len(initial_border).
            final_domain_cells = duplicate_border_cells(initial_border, l)
            # Write the final cells back to the positions in the domain.
            for i, (edge, idx) in enumerate(domain):
                G.edges[edge]['adult_cells'][idx] = copy.deepcopy(final_domain_cells[i])
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
