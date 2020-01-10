import networkx as nx
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


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


def main():
    graph = load_graph("ENGB_edges.csv")
    print("num of node:", len(graph.nodes()))
    print("num of edges:", len(graph.edges()))


    nodes = [n for n in graph.node if graph.node[n]['degree'] >= 5]


    edges_tag = set()
    for n in graph.edge():
        all_node.add(n)
    graph.clear()
    for i in edges:
        u_tag = i[0]
        v_tag = i[1]
        if u_tag not in graph.nodes():
            graph.add_node(u_tag)
        if v_tag not in graph.nodes():
            graph.add_node(v_tag)
        graph.add_edge(u_tag, v_tag, id=i[2]['id'], weight=i[2]['weight'])
    for n in all_node:
        if n not in graph.nodes():
            graph.add_node(n)

    print("num of node after filtering:", len(graph.nodes()))
    print("num of edges after filtering:", len(graph.edges()))

    print("the density of the graph:", nx.density(graph))


if __name__ == '__main__':
    main()
