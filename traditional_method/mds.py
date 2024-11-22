import numpy as np
import networkx as nx
import model
import matplotlib.pyplot as plt
# import scipy.linalg as la


class MDS(model.Base):
    def __init__(self, graph: nx.Graph) -> None:
        super().__init__(graph)
        self.init_pos()
        self.ideal_dist = np.zeros((self.graph.number_of_nodes(), self.graph.number_of_nodes()))
        dist_dict = dict(nx.all_pairs_dijkstra_path_length(self.graph))
        for i in range(self.graph.number_of_nodes()):
            for j in range(self.graph.number_of_nodes()):
                self.ideal_dist[i][j] = dist_dict[i][j]
        self.pos = self.calculate_production()

    
    def calculate_production(self):
        print(self.ideal_dist)
        self.B = self.ideal_dist**2
        u, Q = np.linalg.eig(self.B)
        print(u, Q)
        idx = np.argsort(u)[::-1]
        u = u[idx]
        Q = Q[:, idx]
        d = int(1/2*self.graph.number_of_nodes())
        L = np.diag(np.sqrt(u[:d]))
        V = Q[:, :d]
        return V @ L


if __name__ == "__main__":
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2, 3, 4])
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)

    # graph = nx.erdos_renyi_graph(10, 0.8)
    print(graph)
    layout = MDS(graph=graph)
    plt.figure(figsize=(8, 8))
    nx.draw(graph, pos={i: layout.pos[i] for i in range(len(graph.nodes()))}, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')
    plt.title("Graph Layout using Stress Model")
    plt.show()

