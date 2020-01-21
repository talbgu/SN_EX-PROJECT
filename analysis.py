import networkx as nx
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

grapg_csv_src = "ENGB_edges.csv"


def load_graph(csv_name):
    """
    Loads a graph from a text file to the memory.
    :param path: path of file.
    :return:
    """
    print("start")

    graph = nx.Graph()
    graph.nodes(data=True)

    df = pd.read_csv(csv_name, header=None)
    for index, row in df.iterrows():
        u = row[0]
        v = row[1]

        if u not in graph.nodes():
            graph.add_node(u, {"degree": 0})
        if v not in graph.nodes():
            graph.add_node(v, {"degree": 0})

        graph.add_edge(u, v)
        graph.node[u]["degree"] += 1
        graph.node[v]["degree"] += 1
    return graph


def reconstract_grapg_after_filter(graph, node_list, csv_name):
    """

    :param graph: graph whs loded
    :param node_list: node after filtering by creteria
    :param csv_name: the original  data file
    :return:
    """
    graph.clear()
    for i in node_list:
        graph.add_node(i)

    df = pd.read_csv(csv_name, header=None)
    for index, row in df.iterrows():
        u = row[0]
        v = row[1]
        if u in node_list and v in node_list:
            graph.add_edge(u, v)

    return graph


def betweeness(graph):
    bc_results_dict = nx.betweenness_centrality(graph)
    values = Counter(bc_results_dict)
    top_10_betweennes_centr = values.most_common(10)
    for node_id_bc in top_10_betweennes_centr:
        node_id = node_id_bc[0]
        node_betweenness_centrality = node_id_bc[1]
        print("node_id", " :",node_id , " :", node_betweenness_centrality, " ")


def main():
    graph = load_graph(grapg_csv_src)
    print("num of node:", len(graph.nodes()))
    print("num of edges:", len(graph.edges()))

    node_filtered = [n for n in graph.node if graph.node[n]['degree'] >= 5]
    graph = reconstract_grapg_after_filter(graph, node_filtered, grapg_csv_src)

    print("num of node after filtering:", len(graph.nodes()))
    print("num of edges after filtering:", len(graph.edges()))

    print("the density of the graph:", nx.density(graph))

    print("the Betweeness Centrality:")
    betweeness(graph)


if __name__ == '__main__':
    main()
