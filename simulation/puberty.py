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
        Duplicates both side and center populations. Then, from the doubled side
        cells, flips exactly ~70% to center (and keeps ~30% as side). Likewise,
        from the doubled center cells, flips 70% to side (and 30% remain center).

        Returns:
            new_side, new_center (lists of clone IDs)
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

        # Extend our new lists
        new_side.extend(side_to_remain_side)   # the 30% that stay side
        new_side.extend(center_to_side)        # the 70% that flip from center -> side
        new_center.extend(side_to_center)      # the 70% that flip from side -> center
        new_center.extend(center_to_remain_center)  # the 30% that stay center

        assert len(new_side) == len(new_center)

        return new_side, new_center


    def split_population_into_two(self, side_list, center_list):
        """
        Splits side_list and center_list independently into two equal halves.
        Ensures side1 and side2 have the same length (and likewise for center).
        """

        # Shuffle side_list and split in half
        random.shuffle(side_list)
        half_side = len(side_list) // 2
        side1 = side_list[:half_side]
        side2 = side_list[half_side:]

        # Shuffle center_list and split in half
        random.shuffle(center_list)
        half_center = len(center_list) // 2
        center1 = center_list[:half_center]
        center2 = center_list[half_center:]

        # Create two new TEB objects
        return TEB(side1, center1), TEB(side2, center2)


def simulate_ductal_tree(
    max_cells=10000,
    bifurcation_prob=0.01,
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

    triple_nodes = []
    quad_nodes = []
    termination_side_count = []
    missing_stem_cells = 0
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

        # ------------------------------------------------
        # Main growth / branching logic
        # ------------------------------------------------
        next_active_ends = []




        for (start_node, teb, buffer_clones) in active_ends:
            # 1) First deposit any buffer clones into the "current duct" (parent->start_node).
            deposit_buffer_in_duct(G, start_node, buffer_clones)  # clears buffer_clones

            # 2) Decide whether to branch or elongate
            p_terminate = initial_termination_prob + (
                total_cells / max_cells
            )



            # # replace p_terminate with an initially quickly increasing probability of termination, with an asymptote at 0.6,
            # # still related to the total number of cells.
            # actually, we should either simulate on top of an existing ductal experimental tree, or model the spatial
            # behavior like in

            # if total_cells < max_cells/6:
            #
            #     p_terminate = initial_termination_prob * (total_cells / (max_cells / 6)) ** 0.5
            # else:
            #     p_terminate = 0.6

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
                        # note that doing this basically removes the teb without depositing its clones. This is a bug.
                        missing_stem_cells += len(teb.center_cells) + len(teb.side_cells)
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
                        # note that doing this basically removes the teb without depositing its clones. This is a bug.
                        missing_stem_cells += len(teb.center_cells) + len(teb.side_cells)
                        # delete the current edge
                        G.remove_edge(parent, start_node)
                        # set start_node to parent
                        start_node = parent

                        if start_node not in triple_nodes:
                            print("Triple point! at node", start_node)
                            triple_nodes.append(start_node)
                        elif start_node not in quad_nodes:
                            print("Quadruple point! at node", start_node)
                            quad_nodes.append(start_node)
                        else:
                            print("More than quadruple point! at node", start_node)

                    progress_data["duct_creation_history"].append((parent, start_node, iteration_count))

                    deposit_teb_in_duct(G, start_node, teb.side_cells + teb.center_cells)

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
                side_clone_id = teb.pick_side_clone()  # choose a side clone
                num_daughters = random.randint(4, 8)
                new_clones = [side_clone_id] * num_daughters
                # Place these new clones into buffer_clones
                buffer_clones.extend(new_clones)

                # Remain an active end
                next_active_ends.append((start_node, teb, buffer_clones))

        # Update active ends
        active_ends = next_active_ends

        if not active_ends:
            print("No more active TEBs, stopping. Total cells:", sum(deposited_clones_map.values()))
            print("Missing stem cells:", missing_stem_cells)
            break
    #
    # # plot histogram of termination side counts
    # import matplotlib.pyplot as plt
    # plt.hist(termination_side_count, bins=range(25, 75, 1))
    # plt.xlabel("Number of side cells at termination")
    # plt.ylabel("Frequency")
    # plt.title("Histogram of side cell counts at termination")
    # plt.show()

    return G, progress_data
