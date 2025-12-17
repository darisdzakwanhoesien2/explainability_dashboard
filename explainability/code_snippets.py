from pathlib import Path

BASE_DIR = Path(__file__).parent / "logic"

def load_code(filename: str) -> str:
    return (BASE_DIR / filename).read_text()

CLEAN_BAG_OF_PHRASES_CODE = load_code("text_cleaning.py")
ANALYZE_POPULARITY_CODE = load_code("popularity.py")
FIND_K_PLEXES_CODE = load_code("kplex.py")


# # explainability/code_snippets.py

# CLEAN_BAG_OF_PHRASES_CODE = """
# def clean_bag_of_phrases(text):
#     if isinstance(text, list):
#         return ' '.join(map(str, text))
#     elif isinstance(text, str):
#         text = text.replace("'", "").replace('"', "")
#         return ''.join(c for c in text if c.isalnum() or c.isspace())
#     else:
#         return str(text)
# """

# POWER_LAW_CODE = """
# def power_law(x, a, b):
#     return a * x ** b
# """

# ANALYZE_POPULARITY_CODE = """
# def analyze_popularity(df, metric):
#     all_tokens = []
#     for index, row in df.iterrows():
#         try:
#             tokens = ast.literal_eval(row['bag_of_phrases'])
#         except (SyntaxError, ValueError):
#             tokens = []

#         entities = row['entities']
#         if entities is None or not isinstance(entities, list):
#             entities = []

#         all_tokens.extend(tokens)
#         all_tokens.extend(entities)

#     top_10_tokens = Counter(all_tokens).most_common(10)

#     top_tweets = df.nlargest(5, metric)
#     bottom_tweets = df.nsmallest(5, metric)

#     return top_10_tokens, top_tweets, bottom_tweets
# """

# IS_K_PLEX_CODE = """
# def is_k_plex(graph, nodes, k):
#     subgraph = graph.subgraph(nodes)
#     for node in nodes:
#         degree = subgraph.degree(node)
#         if degree < len(nodes) - k:
#             return False
#     return True
# """

# FIND_K_PLEXES_CODE = """
# def find_k_plexes(graph, k, max_size=None):
#     k_plexes = []
#     nodes = list(graph.nodes())

#     def bron_kerbosch(R, P, X, k):
#         if not P and not X:
#             if is_k_plex(graph, R, k):
#                 k_plexes.append(set(R))
#             return
#         pivot = max(P | X, key=lambda u: len(set(graph.neighbors(u)) & (P | X)), default=None)
#         if pivot is None:
#             return
#         for v in P - set(graph.neighbors(pivot)):
#             new_R = R | {v}
#             new_P = P & set(graph.neighbors(v))
#             new_X = X & set(graph.neighbors(v))
#             bron_kerbosch(new_R, new_P, new_X, k)
#             P.remove(v)
#             X.add(v)

#     bron_kerbosch(set(), set(nodes), set(), k)

#     if max_size:
#         k_plexes = [plex for plex in k_plexes if len(plex) <= max_size]

#     return k_plexes
# """
