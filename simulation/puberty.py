import random
import networkx as nx
import numpy as np
import random

class TEB:
    def __init__(self, side_cells=None, center_cells=None):
        self.side_cells = side_cells if side_cells is not None else []
        self.center_cells = center_cells if center_cells is not None else []

    def pick_side_clone(self):
        return random.choice(self.side_cells)

    def expand_population(self):
        """
        Duplicates both side and center populations. Then flips ~70% side->center,
        and 70% center->side. Returns (new_side, new_center).
        """
        # Duplicate
        all_side = self.side_cells + self.side_cells
        all_center = self.center_cells + self.center_cells

        # Shuffle so that the flips are random
        random.shuffle(all_side)
        random.shuffle(all_center)

        new_side, new_center = [], []

        # Flip 70% of side -> center
        side_flip_count = int(0.7 * len(all_side))
        side_to_center = all_side[:side_flip_count]
        side_to_remain_side = all_side[side_flip_count:]

        # Flip 70% of center -> side
        center_flip_count = int(0.7 * len(all_center))
        center_to_side = all_center[:center_flip_count]
        center_to_remain_center = all_center[center_flip_count:]

        # Build new side/center
        new_side.extend(side_to_remain_side)   # 30% that remain side
        new_side.extend(center_to_side)        # 70% that flip from center->side
        new_center.extend(side_to_center)      # 70% that flip from side->center
        new_center.extend(center_to_remain_center)  # 30% that remain center

        return new_side, new_center

    def split_population_into_two(self, side_list, center_list):
        """
        Splits side_list and center_list independently into two equal halves.
        Returns two new TEBs with equal halves.
        """
        random.shuffle(side_list)
        random.shuffle(center_list)

        half_side = len(side_list) // 2
        half_center = len(center_list) // 2

        side1 = side_list[:half_side]
        side2 = side_list[half_side:]
        center1 = center_list[:half_center]
        center2 = center_list[half_center:]

        return TEB(side1, center1), TEB(side2, center2)

    def expand_for_n_children(self, n_children):
        """
        Create a list of TEBs for the n_children branches:
          - If n_children == 1, pass this TEB unchanged (NO expansion).
          - If n_children == 2, do one expand_population and then split in half.
          - If n_children > 2, repeat expansions until we have enough total clones
            so we can give each child TEB exactly the same total (#side + #center)
            as the *current* TEB (i.e., 'keep TEB size the same' for each child).
        """

        parent_side = list(self.side_cells)
        parent_center = list(self.center_cells)
        original_size = len(parent_side) + len(parent_center)

        # -- Case 1: Only 1 child => no expansion
        if n_children == 1:
            return [TEB(parent_side, parent_center)]

        # -- Case 2: Exactly 2 children => standard expand + split
        if n_children == 2:
            new_side, new_center = self.expand_population()  # doubles population
            teb1, teb2 = self.split_population_into_two(new_side, new_center)
            return [teb1, teb2]

        # -- Case 3: More than 2 children =>
        # We want each child TEB to have the same total size as the current TEB,
        # i.e. original_size. So we need n_children * original_size total.
        # We'll keep expanding until we reach that total or exceed it.

        target_total = n_children * original_size
        side_current = parent_side
        center_current = parent_center
        current_total = len(side_current) + len(center_current)

        # Expand repeatedly until we have enough clones
        while current_total < target_total:
            side_new, center_new = TEB(side_current, center_current).expand_population()
            side_current = side_new
            center_current = center_new
            current_total = len(side_current) + len(center_current)

        # Now we have at least target_total clones in side_current+center_current.
        # We'll partition them into n_children groups, each of exactly original_size total.
        labeled_combined = [("S", s) for s in side_current] + [("C", c) for c in center_current]
        random.shuffle(labeled_combined)

        children_tebs = []
        offset = 0
        for _ in range(n_children):
            # Slice out exactly original_size items for this child
            child_slice = labeled_combined[offset : offset + original_size]
            offset += original_size

            child_side = [cid for (lab, cid) in child_slice if lab == "S"]
            child_center = [cid for (lab, cid) in child_slice if lab == "C"]
            children_tebs.append(TEB(child_side, child_center))

        return children_tebs

    def stochastic_side_replacement(self, rep_prob: float = 0.10, rng=random):
        """
        With probability `rep_prob` pick a random side cell, delete it,
        and replace it by a clone chosen uniformly from *all* current cells
        (side + center).  Side-pool size is preserved.
        """
        if rng.random() >= rep_prob or not self.side_cells:
            return                     # no event this elongation

        # victim is an *index* in the side list
        victim_idx = rng.randrange(len(self.side_cells))

        # donor clone id can come from either compartment
        donor_id = rng.choice(self.side_cells + self.center_cells)

        # execute replacement
        self.side_cells[victim_idx] = donor_id

def simulate_ductal_tree(
    max_cells=10000,
    bifurcation_prob=0.01,
    replacement_prob=0.10,
    initial_side_count=50,
    initial_center_count=50,
    initial_termination_prob=0.05
):
    """
    Returns (G, progress_data).
    G is a DiGraph whose edges store 'duct_clones'.
    """
    G = nx.DiGraph()

    # Start node
    node_counter = 0
    root_node = node_counter

    # Initial TEB
    initial_teb = TEB(
        side_cells=list(range(int(initial_side_count))),
        center_cells=list(range(int(initial_side_count), int(initial_side_count + initial_center_count)))
    )

    # Initialize the first 2 nodes, the first duct and the first TEB
    G.add_node(root_node, description="branch")
    node_counter += 1
    G.add_node(node_counter, description="branch")
    G.add_edge(root_node, node_counter, duct_clones=[])
    active_ends = [(node_counter, initial_teb, [])]



    deposited_clones_map = {}  # If you want to store clones from terminated TEBs

    progress_data = {
        "iteration": [],
        "total_cells": [],
        "clone_counts_over_time": [],
        "num_active_tebs": [],
        "avg_dom_fraction": [],
        "avg_dom_stem_fraction": [],
        "teb_history": {},
        "duct_creation_history": []
    }

    iteration_count = 0

    # Helper: deposit the buffer clones into the duct from the "parent" to start_node.
    # If no parent edge (i.e. root?), do nothing or store them in the node's attribute if you prefer.
    def deposit_buffer_in_duct(G, start_node, buffer_clones):
        parents = list(G.predecessors(start_node))
        if parents:
            parent = parents[0]
            # Extend the duct_clones on that edge
            if "duct_clones" not in G[parent][start_node]:
                G[parent][start_node]["duct_clones"] = []
            G[parent][start_node]["duct_clones"].extend(buffer_clones)

        # Immediately add these clones to deposited_clones_map permanently
        for cid in buffer_clones:
            deposited_clones_map[cid] = deposited_clones_map.get(cid, 0) + 1

        buffer_clones.clear()

    def deposit_teb_in_duct(G, node_id, clones_list):
        """Extend the parent's duct_clones by these clones."""
        parents = list(G.predecessors(node_id))
        if not parents:
            return
        parent = parents[0]
        if "duct_clones" not in G[parent][node_id]:
            G[parent][node_id]["duct_clones"] = []
        G[parent][node_id]["duct_clones"].extend(clones_list)

        # # Also add to 'deposited_clones_map' if your counting
        for cid in clones_list:
            deposited_clones_map[cid] = deposited_clones_map.get(cid, 0) + 1

    def get_current_duct_clones(G, start_node):
        parents = list(G.predecessors(start_node))
        if parents:
            parent = parents[0]
            return G[parent][start_node].get("duct_clones", [])
        return []

    termination_side_count = []
    while active_ends:
        iteration_count += 1

        # A map of all clones in current active TEBs
        active_clone_map = {}
        dom_fraction_sum = 0.0
        dom_stem_fraction_sum = 0.0


        for (node_id, teb, buffer_clones) in active_ends:
            # Count local clones
            local_map = {}
            for c in teb.side_cells:
                local_map[c] = local_map.get(c, 0) + 1
            for c in teb.center_cells:
                local_map[c] = local_map.get(c, 0) + 1
            for c in buffer_clones:
                local_map[c] = local_map.get(c, 0) + 1

            # Track TEB history
            if node_id not in progress_data["teb_history"]:
                progress_data["teb_history"][node_id] = {
                    "iteration": [],
                    "dominant_clone_fraction": [],
                    "dominant_stem_cell_fraction": []
                }

            stem_cells = teb.center_cells + teb.side_cells
            stem_map = {}
            for c in stem_cells:
                stem_map[c] = stem_map.get(c, 0) + 1

            local_total = sum(local_map.values())
            if local_total > 0:
                local_dom_fraction = max(local_map.values()) / local_total
                if len(stem_map) > 0:
                    local_dom_stem_fraction = max(stem_map.values()) / sum(stem_map.values())
                else:
                    local_dom_stem_fraction = 0.0
            else:
                local_dom_fraction = 0.0
                local_dom_stem_fraction = 0.0

            progress_data["teb_history"][node_id]["iteration"].append(iteration_count)
            progress_data["teb_history"][node_id]["dominant_clone_fraction"].append(local_dom_fraction)
            progress_data["teb_history"][node_id]["dominant_stem_cell_fraction"].append(local_dom_stem_fraction)

            dom_fraction_sum += local_dom_fraction
            dom_stem_fraction_sum += local_dom_stem_fraction

            # Merge into active_clone_map
            for cid, count in local_map.items():
                active_clone_map[cid] = active_clone_map.get(cid, 0) + count

        num_tebs = len(active_ends)
        if num_tebs > 0:
            avg_dom_fraction = dom_fraction_sum / num_tebs
            avg_dom_stem_fraction = dom_stem_fraction_sum / num_tebs
        else:
            avg_dom_fraction = 0.0
            avg_dom_stem_fraction = 0.0



        combined_map = dict(deposited_clones_map)
        for cid, count in active_clone_map.items():
            combined_map[cid] = combined_map.get(cid, 0) + count

        total_cells = sum(combined_map.values())

        # Save iteration data
        progress_data["iteration"].append(iteration_count)
        progress_data["total_cells"].append(total_cells)
        progress_data["clone_counts_over_time"].append(dict(combined_map))
        progress_data["num_active_tebs"].append(num_tebs)
        progress_data["avg_dom_fraction"].append(avg_dom_fraction)
        progress_data["avg_dom_stem_fraction"].append(avg_dom_stem_fraction)

        next_active_ends = []

        for (start_node, teb, buffer_clones) in active_ends:
            # 1) First deposit any buffer clones into the "current duct" (parent->start_node).
            deposit_buffer_in_duct(G, start_node, buffer_clones)  # clears buffer_clones

            # 2) Decide whether to branch or elongate
            p_terminate = initial_termination_prob + (
                total_cells / max_cells
            )


            if random.random() < bifurcation_prob:

                if random.random() < p_terminate:
                    # TERMINATE
                    termination_side_count.append(len(teb.side_cells))

                    parents = list(G.predecessors(start_node))
                    if parents:
                        parent = parents[0]
                    progress_data["duct_creation_history"].append((parent, start_node, iteration_count))


                    parent = list(G.predecessors(start_node))[0]
                    duct_clones = get_current_duct_clones(G, start_node)
                    if not duct_clones:
                        # delete the current edge
                        G.remove_edge(parent, start_node)

                    else:
                        deposit_teb_in_duct(G, start_node, teb.side_cells + teb.center_cells)
                        # append begin, end and iteration number to duct_creation_history

                else:
                    # BRANCH
                    parent = list(G.predecessors(start_node))[0]
                    duct_clones = G[parent][start_node].get("duct_clones", [])
                    if not duct_clones:
                        # delete the current edge
                        G.remove_edge(parent, start_node)
                        # set start_node to parent
                        start_node = parent

                    progress_data["duct_creation_history"].append((parent, start_node, iteration_count))

                    # Expand and split TEB population
                    side_expanded, center_expanded = teb.expand_population()
                    new_teb1, new_teb2 = teb.split_population_into_two(side_expanded, center_expanded)

                    node_counter += 1
                    child_node_1 = node_counter
                    G.add_node(child_node_1, description="branch")

                    node_counter += 1
                    child_node_2 = node_counter
                    G.add_node(child_node_2, description="branch")

                    # Create the new edges (children) with empty duct_clones
                    G.add_edge(start_node, child_node_1, duct_clones=[])
                    G.add_edge(start_node, child_node_2, duct_clones=[])

                    # Now the TEBs continue on each child branch with no 'buffer_clones' yet
                    next_active_ends.append((child_node_1, new_teb1, []))
                    next_active_ends.append((child_node_2, new_teb2, []))


            else:
                # ELONGATE
                side_clone_id = teb.pick_side_clone()
                num_daughters = random.randint(4, 8)
                buffer_clones.extend([side_clone_id] * num_daughters)

                teb.stochastic_side_replacement(rep_prob=replacement_prob)

                next_active_ends.append((start_node, teb, buffer_clones))

        # Update active ends
        active_ends = next_active_ends

        if not active_ends:
            print("No more active TEBs, stopping. Total cells:", sum(deposited_clones_map.values()))
            break
    #
    assert np.unique(termination_side_count).shape[0] == 1 # all TEBs should have the same number of side cells
    print("Total existed TEBs:", len(termination_side_count))

    return G, progress_data


def simulate_ductal_tree_on_existing_graph(
    existing_graph: nx.DiGraph,
    root_node,
    bifurcation_prob=0.01,
    replacement_prob=0.10,
    initial_side_count=50,
    initial_center_count=50
):
    """
    Runs the 'puberty simulation' on a predetermined DiGraph structure.
    - If random < bifurcation_prob, we see if the current node has children:
         * If it has children, we do an expand_for_n_children and pass TEBs to each child.
         * If leaf => TEB terminates.
    - If we do NOT branch, we elongate by replicating a side clone into buffer_clones.
    - buffer_clones are deposited into the duct at the start of each iteration for that TEB.
    """

    if not isinstance(existing_graph, nx.DiGraph):
        raise TypeError("The provided 'existing_graph' must be a nx.DiGraph.")

    # Initialize the TEB on the root
    initial_teb = TEB(
        side_cells=list(range(initial_side_count)),
        center_cells=list(range(initial_side_count, initial_side_count + initial_center_count))
    )

    # Each active end is (node_id, teb, buffer_clones)
    active_ends = [(root_node, initial_teb, [])]
    deposited_clones_map = {}

    # We'll track data over iterations
    progress_data = {
        "iteration": [],
        "total_cells": [],
        "clone_counts_over_time": [],
        "num_active_tebs": [],
        "avg_dom_fraction": [],
        "avg_dom_stem_fraction": [],
        "teb_history": {},
        "duct_creation_history": []
    }

    iteration_count = 0

    # Helper functions

    def deposit_buffer_in_duct(current_node, buffer_clones):
        """Append buffer_clones into parent's edge 'duct_clones' for current_node."""
        parents = list(existing_graph.predecessors(current_node))
        if parents:
            parent = parents[0]
            if "duct_clones" not in existing_graph[parent][current_node]:
                existing_graph[parent][current_node]["duct_clones"] = []
            existing_graph[parent][current_node]["duct_clones"].extend(buffer_clones)

        # Also increment global deposit count
        for cid in buffer_clones:
            deposited_clones_map[cid] = deposited_clones_map.get(cid, 0) + 1

        buffer_clones.clear()

    def deposit_teb_in_duct(current_node, clones_list):
        """Deposit final TEB side+center clones into parent's edge (like on termination)."""
        parents = list(existing_graph.predecessors(current_node))
        if not parents:
            return
        parent = parents[0]
        if "duct_clones" not in existing_graph[parent][current_node]:
            existing_graph[parent][current_node]["duct_clones"] = []
        existing_graph[parent][current_node]["duct_clones"].extend(clones_list)

        for cid in clones_list:
            deposited_clones_map[cid] = deposited_clones_map.get(cid, 0) + 1

    # Main loop
    while active_ends:
        iteration_count += 1

        # Collect overall stats on all active TEBs
        active_clone_map = {}
        dom_fraction_sum = 0.0
        dom_stem_fraction_sum = 0.0

        # --- Gather data from each TEB ---
        for (node_id, teb, buffer_clones) in active_ends:
            local_map = {}
            for c in teb.side_cells:
                local_map[c] = local_map.get(c, 0) + 1
            for c in teb.center_cells:
                local_map[c] = local_map.get(c, 0) + 1
            for c in buffer_clones:
                local_map[c] = local_map.get(c, 0) + 1

            # Initialize TEB history if not present
            if node_id not in progress_data["teb_history"]:
                progress_data["teb_history"][node_id] = {
                    "iteration": [],
                    "dominant_clone_fraction": [],
                    "dominant_stem_cell_fraction": []
                }

            local_total = sum(local_map.values())
            local_dom_fraction = max(local_map.values()) / local_total if local_total > 0 else 0.0

            # For "stem fraction," consider all side+center
            stem_map = {}
            for c in (teb.side_cells + teb.center_cells):
                stem_map[c] = stem_map.get(c, 0) + 1
            local_dom_stem_fraction = (
                max(stem_map.values()) / sum(stem_map.values()) if stem_map else 0.0
            )

            # Record TEB-specific stats
            progress_data["teb_history"][node_id]["iteration"].append(iteration_count)
            progress_data["teb_history"][node_id]["dominant_clone_fraction"].append(local_dom_fraction)
            progress_data["teb_history"][node_id]["dominant_stem_cell_fraction"].append(local_dom_stem_fraction)

            dom_fraction_sum += local_dom_fraction
            dom_stem_fraction_sum += local_dom_stem_fraction

            # Merge local_map into active_clone_map
            for cid, count in local_map.items():
                active_clone_map[cid] = active_clone_map.get(cid, 0) + count

        num_tebs = len(active_ends)
        avg_dom_fraction = dom_fraction_sum / num_tebs if num_tebs else 0.0
        avg_dom_stem_fraction = dom_stem_fraction_sum / num_tebs if num_tebs else 0.0

        # Combine with deposited clones to get total cell count
        combined_map = dict(deposited_clones_map)
        for cid, count in active_clone_map.items():
            combined_map[cid] = combined_map.get(cid, 0) + count
        total_cells = sum(combined_map.values())

        # Save iteration data
        progress_data["iteration"].append(iteration_count)
        progress_data["total_cells"].append(total_cells)
        progress_data["clone_counts_over_time"].append(dict(combined_map))
        progress_data["num_active_tebs"].append(num_tebs)
        progress_data["avg_dom_fraction"].append(avg_dom_fraction)
        progress_data["avg_dom_stem_fraction"].append(avg_dom_stem_fraction)

        # --- Process the next step for each TEB ---
        next_active_ends = []

        for (node_id, teb, buffer_clones) in active_ends:
            # 1) Deposit any buffer clones from last iteration into the duct now
            deposit_buffer_in_duct(node_id, buffer_clones)  # This calls buffer_clones.clear()

            # 2) Check if we "branch" at this node
            if random.random() < bifurcation_prob:
                children = list(existing_graph.successors(node_id))
                if not children:
                    # Leaf node => TEB terminates
                    deposit_teb_in_duct(node_id, teb.side_cells + teb.center_cells)
                else:
                    # Expand TEB for the number of children
                    # deposit_teb_in_duct(node_id, teb.side_cells + teb.center_cells)
                    child_tebs = teb.expand_for_n_children(len(children))

                    # Each child becomes an active end
                    for child_node, child_teb in zip(children, child_tebs):
                        next_active_ends.append((child_node, child_teb, []))
            else:
                # Elongate: pick a side clone and replicate it
                side_clone_id = teb.pick_side_clone()
                num_daughters = random.randint(4, 8)
                buffer_clones.extend([side_clone_id] * num_daughters)

                teb.stochastic_side_replacement(rep_prob=replacement_prob)

                next_active_ends.append((node_id, teb, buffer_clones))

        # Update active_ends
        active_ends = next_active_ends

    print("No more active TEBs; simulation finished. Final total cells:", sum(deposited_clones_map.values()))
    return existing_graph, progress_data