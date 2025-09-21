import matplotlib.pyplot as plt
import networkx as nx
from aima3.search import Problem, breadth_first_tree_search, greedy_best_first_graph_search, astar_search

class WaterJugProblem(Problem):
    def __init__(self, initial=(8, 0, 0), goal=(4, 4, 0), capacities=(8, 5, 3)):
        super().__init__(initial, goal)
        self.capacities = capacities

    def actions(self, state):
        actions = []
        for i in range(3):
            for j in range(3):
                if i != j and state[i] > 0 and state[j] < self.capacities[j]:
                    actions.append((i, j))
        return actions

    def result(self, state, action):
        state = list(state)
        i, j = action
        amount = min(state[i], self.capacities[j] - state[j])
        state[i] -= amount
        state[j] += amount
        return tuple(state)

    def goal_test(self, state):
        return state[0] == 4 and state[1] == 4

    def h(self, node):

        return abs(node.state[0] - 4) + abs(node.state[1] - 4)


problem = WaterJugProblem()

# Resolver com BFS
solution_bfs = breadth_first_tree_search(problem)

# Resolver com Gulosa
solution_greedy = greedy_best_first_graph_search(problem, problem.h)

# Resolver com A*
solution_astar = astar_search(problem, problem.h)


def draw_solution(path, title, pos=None, ax=None):
    G = nx.DiGraph()
    edges = []
    nodes = []

    for i in range(len(path) - 1):
        edges.append((path[i].state, path[i+1].state))
        nodes.append(path[i].state)
    nodes.append(path[-1].state)

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue",
            font_size=8, font_weight="bold", arrows=True, ax=ax)
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels={(path[i].state, path[i+1].state): str(i+1) for i in range(len(path)-1)}, ax=ax
    )
    ax.set_title(title)
    return pos


fig, axes = plt.subplots(1, 3, figsize=(18, 6))

pos = draw_solution(solution_bfs.path(), "Busca em Largura (BFS)", ax=axes[0])
draw_solution(solution_greedy.path(), "Busca Gulosa", pos=pos, ax=axes[1])
draw_solution(solution_astar.path(), "Busca A*", pos=pos, ax=axes[2])

plt.show()
