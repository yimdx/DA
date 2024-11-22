import numpy as np
import graph
import matplotlib.pyplot as plt
K = 1
def fa(x, i, j):
    distance = np.linalg.norm(x[i]-x[j])
    if distance == 0:
        return np.zeros(2)
    return -distance*(x[i]-x[j])/K

def fr(x, i, j):
    distance = np.linalg.norm(x[i]-x[j])
    if distance == 0:
        return np.zeros(2)
    return (x[i]-x[j])*np.pow(K, 2)/np.pow(distance, 2)

def update_step_length(step):
    return 0.999*step

def force_directed_algo(G, x, tol):
    converge = True
    step = 0.05
    while converge == True:
        deltas = []
        for i in G.V:
            f = 0
            for j in G.E[i]:
                f = f + fa(x, i, j)
            for j in G.V:
                if j != i:
                    f = f + fr(x, i, j)
            delta = step*f/np.linalg.norm(f) if np.linalg.norm(f) > 0 else 0
            deltas.append(delta)
            x[i] = x[i] + delta
        step = update_step_length(step)
        print(deltas)
        if np.linalg.norm(deltas) < tol * K: 
            converge = False
            break
    return x

if __name__ == "__main__":
    V = [0, 1, 2, 3, 4]
    E = [[1], [0, 2], [1, 3], [2, 4], [3]]

    g = graph.G(V, E)
    first = g.random_init()
    for i in range(len(V)):
        for j in E[i]:
            plt.plot([first[i][0], first[j][0]], [first[i][1], first[j][1]], 'bo-')
    plt.title("Initial Positions")
    plt.savefig("origin.png")
    plt.clf()

    print("Initial positions:", first)

    res = force_directed_algo(g, first, tol=5e-10)
    print("Final positions:", res)

    for i in range(len(V)):
        for j in E[i]:
            plt.plot([res[i][0], res[j][0]], [res[i][1], res[j][1]], 'bo-')
    plt.title("Final Positions after Force-Directed Algorithm")
    plt.savefig("res.png")
