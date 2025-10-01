import matplotlib.pyplot as plt
import networkx as nx
import heapq
from collections import deque

# Configuração do problema
capacities = (8, 5, 3)
initial_state = (8, 0, 0)
goal_state = (4, 4, 0)

def actions(state):
    acts = []
    for i in range(3):
        for j in range(3):
            if i != j and state[i] > 0 and state[j] < capacities[j]:
                acts.append((i, j))
    return acts

def result(state, action):
    state = list(state)
    i, j = action
    amount = min(state[i], capacities[j] - state[j])
    state[i] -= amount
    state[j] += amount
    return tuple(state)

def goal_test(state):
    return state == goal_state

def heuristic(state):
    return abs(state[0] != 4) + abs(state[1] != 4)

# BFS
def breadth_first_search(initial_state):
    frontier = deque([(initial_state, [initial_state])])
    explored = set()
    tree_edges = []
    while frontier:
        state, path = frontier.popleft()
        if state in explored:
            continue
        explored.add(state)
        if goal_test(state):
            return path, tree_edges
        for action in actions(state):
            new_state = result(state, action)
            if new_state not in explored:
                frontier.append((new_state, path + [new_state]))
                tree_edges.append((state, new_state))
    return None, tree_edges

# Gulosa
def greedy_best_first(initial_state):
    frontier = [(heuristic(initial_state), initial_state, [initial_state])]
    explored = set()
    tree_edges = []
    while frontier:
        _, state, path = heapq.heappop(frontier)
        if state in explored:
            continue
        explored.add(state)
        if goal_test(state):
            return path, tree_edges
        for action in actions(state):
            new_state = result(state, action)
            if new_state not in explored:
                heapq.heappush(frontier, (heuristic(new_state), new_state, path + [new_state]))
                tree_edges.append((state, new_state))
    return None, tree_edges

# A*
def astar_search(initial_state):
    frontier = [(heuristic(initial_state), 0, initial_state, [initial_state])]
    explored = {}
    tree_edges = []
    while frontier:
        f, g, state, path = heapq.heappop(frontier)
        if state in explored and explored[state] <= g:
            continue
        explored[state] = g
        if goal_test(state):
            return path, tree_edges
        for action in actions(state):
            new_state = result(state, action)
            new_g = g + 1
            new_f = new_g + heuristic(new_state)
            heapq.heappush(frontier, (new_f, new_g, new_state, path + [new_state]))
            tree_edges.append((state, new_state))
    return None, tree_edges


def draw_tree(ax, tree_edges, path, title):
    G = nx.DiGraph()
    for parent, child in tree_edges:
        G.add_edge(parent, child)

    pos = nx.spring_layout(G, seed=42, k=1.5, iterations=100)
    nx.draw(G, pos, with_labels=True, node_size=2500, node_color="skyblue",
            font_size=10, font_weight="bold", arrows=True, ax=ax)
    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color="orange", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=2, ax=ax)
    ax.set_title(title)
    ax.axis("off")

path_bfs, edges_bfs = breadth_first_search(initial_state)
path_greedy, edges_greedy = greedy_best_first(initial_state)
path_astar, edges_astar = astar_search(initial_state)

fig, axes = plt.subplots(1, 3, figsize=(24, 10))
draw_tree(axes[0], edges_bfs, path_bfs, "Busca em Largura (BFS)")
draw_tree(axes[1], edges_greedy, path_greedy, "Busca Gulosa")
draw_tree(axes[2], edges_astar, path_astar, "Busca A*")

plt.show()