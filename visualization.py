import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

def visualize_graph(data, index):
    """
    Visualize a single graph using NetworkX.
    """
    G = to_networkx(data[index], to_undirected=True)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightblue")
    plt.title(f"Graph Visualization for Trial {index}")
    plt.show()
