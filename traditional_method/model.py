import networkx as nx
import numpy as np
class Base:
    def __init__(self, graph: nx.graph.Graph) -> None:
        self.graph = graph
        pass
    
    def force_direct_algo(self, lr: float = 5e-5, threshold = 5e-9, init_values=None):
        raise NotImplementedError
    
    def update_step_length(self, step, lr, deltas):
        return step*(1-lr)

    # def calculate_force(self):
    #     raise NotImplementedError

    def init_pos(self):
        self.pos = np.random.rand(self.graph.number_of_nodes(), 2)