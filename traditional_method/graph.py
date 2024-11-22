import numpy as np
class G:
    def __init__(self, V, E) -> None:
        self.V = V
        self.E = E
    
    def random_init(self):
        x = np.random.rand(len(self.V), 2)
        return x
    
if __name__ == "__main__":
    V = [1,2,3,4]
    E = [[2,3], [1,4], [1], [2]]
    g = G(V, E)
    print(g.V, g.E, g.random_init())