import networkx as nx

def build_graph(digits):
    G = nx.Graph()
    positions = list(digits.keys())

    # Add nodes
    for pos, num in digits.items():
        G.add_node(pos, value=num)

    # Connect neighbors (adjacency threshold)
    for (x1, y1), num1 in digits.items():
        for (x2, y2), num2 in digits.items():
            if (x1, y1) != (x2, y2):
                if abs(x1 - x2) < 150 and abs(y1 - y2) < 150:
                    G.add_edge((x1, y1), (x2, y2))

    return G

def solve_zip(G):
    start_nodes = [n for n, d in G.nodes(data=True) if d['value'] == 1]
    if not start_nodes:
        return []

    path = []
    current = start_nodes[0]
    path.append(current)
    current_value = 1

    # Greedy consecutive search
    while True:
        neighbors = [n for n in G.neighbors(current) if G.nodes[n]['value'] == current_value + 1]
        if not neighbors:
            break
        current = neighbors[0]
        path.append(current)
        current_value += 1

    return path
