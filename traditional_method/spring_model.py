import numpy as np
import networkx as nx
import model
import matplotlib.pyplot as plt

class SpringModel(model.Base):
    def __init__(self, graph: nx.Graph, K = 0.5) -> None:
        super().__init__(graph)
        self.init_pos()
        self.K = K

    def force_direct_algo(self, lr: float = 1e-2, threshold = 5e-10, init_values=None):
        convenge = False
        step = 0.05
        while convenge == False:
            forces = np.zeros((self.graph.number_of_nodes(), 2))
            deltas = np.zeros((self.graph.number_of_nodes(),2))
            for i, node_i in enumerate(self.graph.nodes):
                for j, node_j in enumerate(self.graph.nodes):
                    if i != j:
                        forces[i] += self.fr(i, j)
                for j in self.graph.neighbors(i):
                    forces[i] += self.fa(i,j)
                deltas[i] = step*forces[i]/np.linalg.norm(forces[i]) if np.linalg.norm(forces[i]) > 0 else np.zeros(2)
                self.pos[i] += deltas[i]
            step = self.update_step_length(step, lr, deltas)
            if np.linalg.norm(deltas) < threshold:
                convenge = True

    def fa(self, i, j):
        distance = np.linalg.norm(self.pos[i]-self.pos[j])
        if distance == 0:
            return np.zeros(2)
        return -distance*(self.pos[i]-self.pos[j])/self.K

    def fr(self, i, j):
        distance = np.linalg.norm(self.pos[i]-self.pos[j])
        if distance == 0:
            return np.zeros(2)
        return (self.pos[i]-self.pos[j])*np.pow(self.K, 2)/np.pow(distance, 2)

  
    
if __name__ == "__main__":
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2, 3, 4])
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)

    graph = nx.erdos_renyi_graph(10, 0.8)
    layout = SpringModel(graph=graph)
    layout.force_direct_algo()
    plt.figure(figsize=(8, 8))
    nx.draw(graph, pos={i: layout.pos[i] for i in range(len(graph.nodes()))}, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')
    plt.title("Graph Layout using Stress Model")
    plt.show()


