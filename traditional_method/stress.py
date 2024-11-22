import numpy as np
import networkx as nx
import model
import matplotlib.pyplot as plt

class StressModel(model.Base):
    def __init__(self, graph: nx.Graph) -> None:
        super().__init__(graph)
        self.ideal_dist = np.zeros((self.graph.number_of_nodes(), self.graph.number_of_nodes()))
        dist_dict = dict(nx.all_pairs_dijkstra_path_length(self.graph))
        for i in range(self.graph.number_of_nodes()):
            for j in range(self.graph.number_of_nodes()):
                self.ideal_dist[i][j] = dist_dict[i][j]
        self.init_pos()
        self.K = 0.5

    def force_direct_algo(self, lr: float = 1e-3, threshold = 5e-7, init_values=None):
        convenge = False
        step = 1-lr
        while convenge == False:
            forces = np.zeros((self.graph.number_of_nodes(), 2))
            deltas = np.zeros((self.graph.number_of_nodes(),2))
            for i, node_i in enumerate(self.graph.nodes):
                for j, node_j in enumerate(self.graph.nodes):
                    if i != j:
                        forces[i] += self.calculate_force(i, j)
                deltas[i] = step*forces[i]/np.linalg.norm(forces[i]) if np.linalg.norm(forces[i]) > 0 else np.zeros(2)
                self.pos[i] += deltas[i]
            step = self.update_step_length(step, lr, deltas)
            if np.linalg.norm(deltas) < threshold:
                convenge = True

    def calculate_force(self, i, j):
        ideal_d_ij = self.ideal_dist[i][j]
        real_d_ij = np.linalg.norm(self.pos[i]-self.pos[j])
        return -1/(ideal_d_ij**2)*(real_d_ij-ideal_d_ij)*((self.pos[i]-self.pos[j])/real_d_ij)
    
    def stress_majority(self, lr: float = 1e-3, threshold=5e-9):
        step = 1 - lr
        while True:
            self.weights = np.zeros((self.graph.number_of_nodes(), self.graph.number_of_nodes()))
            for i in range(self.graph.number_of_nodes()):
                for j in range(self.graph.number_of_nodes()):
                    if i != j:
                        self.weights[i][j] = 1 / (self.ideal_dist[i][j] ** 2)
                    else:
                        self.weights[i][j] = 1
            L_w = np.zeros((self.graph.number_of_nodes(), self.graph.number_of_nodes()))
            for i in range(self.graph.number_of_nodes()):
                for j in range(self.graph.number_of_nodes()):
                    L_w[i][j] = -self.weights[i][j] if i != j else np.sum(self.weights[i]) - self.weights[i][j]
            L_wd = np.zeros((self.graph.number_of_nodes(), self.graph.number_of_nodes()))
            for i in range(self.graph.number_of_nodes()):
                for j in range(self.graph.number_of_nodes()):
                    norm = np.linalg.norm(self.pos[i] - self.pos[j])
                    if i != j:
                        if norm <= 0:
                            L_wd[i][j] = 0
                        else:
                            L_wd[i][j] = -self.weights[i][j] * self.ideal_dist[i][j] / norm
                    else:
                        L_wd[i][j] = np.dot(self.weights[i], self.ideal_dist[i]) / norm if norm > 0 else 1e-10
            L_wd += np.eye(L_wd.shape[0]) * 1e-9
            deltas = np.dot(np.dot(np.linalg.inv(L_wd), L_w), self.pos)
            self.pos += deltas * step
            step = self.update_step_length(step, lr, deltas)
            if np.linalg.norm(deltas) < threshold:
                break

        
if __name__ == "__main__":
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2, 3, 4])
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)

    graph = nx.erdos_renyi_graph(10, 0.5)
    print(graph)
    layout = StressModel(graph=graph)
    layout.force_direct_algo()
    plt.figure(figsize=(8, 8))
    nx.draw(graph, pos={i: layout.pos[i] for i in range(len(graph.nodes()))}, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')
    plt.title("Graph Layout using Stress Model")
    plt.show()


