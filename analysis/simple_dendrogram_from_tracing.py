from analysis.utils.loading_saving import load_duct_systems, create_directed_duct_graph, select_biggest_duct_system
from analysis.utils.fixing_annotations import simplify_graph
from analysis.utils.plotting_trees import plot_hierarchical_graph
import matplotlib.pyplot as plt


# json_path = r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_carmine\06012025_carminegood images\export files\2516525-slide1_9weeks_branch.json"
json_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\hierarchy tree.json'

graphs = load_duct_systems(json_path)
g = create_directed_duct_graph(select_biggest_duct_system(graphs))
g = simplify_graph(g)
plot_hierarchical_graph(G=g,vert_gap=4, linewidth=0.4, font_size=10)
plt.savefig("2473536_Cft_24W_uncolored.png", dpi=800)
plt.show()