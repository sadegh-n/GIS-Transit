"""
Commute network analysis using LODES origin-destination data.
Builds a directed graph of commute flows and runs community detection
and centrality analysis.
"""

import pandas as pd
import networkx as nx
import community as community_louvain


def build_commute_graph(corridors_df, county_fips, min_workers=5):
    # min_workers=5: filters out noise from very small commute pairs
    # (1-4 workers could be data artifacts or very rare commutes)
    """
    Build a directed weighted graph from LODES commute flows.
    Nodes = block groups, edges = commute flows, weights = worker count.
    """
    internal = corridors_df[
        (corridors_df["home_bg"].str[:5] == county_fips) &
        (corridors_df["work_bg"].str[:5] == county_fips) &
        (corridors_df["total_workers"] >= min_workers)
    ]

    G = nx.DiGraph()
    for _, row in internal.iterrows():
        G.add_edge(
            row["home_bg"], row["work_bg"],
            weight=row["total_workers"]
        )

    return G


def analyze_network(G):
    """
    Run community detection and centrality metrics on the commute graph.
    Returns a node-level DataFrame and summary stats.
    """
    if len(G.nodes) == 0:
        return pd.DataFrame(), {}

    # Community detection on undirected version
    G_und = G.to_undirected()
    communities = community_louvain.best_partition(G_und, weight="weight", random_state=42)

    # Centrality
    pagerank = nx.pagerank(G, weight="weight")
    betweenness = nx.betweenness_centrality(G, k=min(200, len(G.nodes)), seed=42)
    in_strength = dict(G.in_degree(weight="weight"))

    nodes = list(G.nodes)
    node_df = pd.DataFrame({
        "GEOID": nodes,
        "community": [communities.get(n, -1) for n in nodes],
        "pagerank": [pagerank.get(n, 0) for n in nodes],
        "betweenness": [betweenness.get(n, 0) for n in nodes],
        "in_strength": [in_strength.get(n, 0) for n in nodes],
    })

    stats = {
        "nodes": len(G.nodes),
        "edges": len(G.edges),
        "communities": len(set(communities.values())),
        "density": round(nx.density(G), 5),
    }

    return node_df, stats
