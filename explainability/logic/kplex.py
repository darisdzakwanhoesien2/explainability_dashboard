def is_k_plex(graph, nodes, k):
    subgraph = graph.subgraph(nodes)
    for node in nodes:
        if subgraph.degree(node) < len(nodes) - k:
            return False
    return True


def find_k_plexes(graph, k, max_size=None):
    k_plexes = []
    nodes = list(graph.nodes())

    def bron_kerbosch(R, P, X):
        if not P and not X:
            if is_k_plex(graph, R, k):
                k_plexes.append(set(R))
            return

        pivot = max(
            P | X,
            key=lambda u: len(set(graph.neighbors(u)) & (P | X)),
            default=None,
        )
        if pivot is None:
            return

        for v in P - set(graph.neighbors(pivot)):
            bron_kerbosch(
                R | {v},
                P & set(graph.neighbors(v)),
                X & set(graph.neighbors(v)),
            )
            P.remove(v)
            X.add(v)

    bron_kerbosch(set(), set(nodes), set())

    if max_size:
        k_plexes = [p for p in k_plexes if len(p) <= max_size]

    return k_plexes
