import networkx as nx
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

graph = nx.read_gexf('net_il2015-2018.gexf')
graph.nodes(data=True)

print ("num of node:", len(graph.nodes()))
print("num of edges:", len(graph.edges()))

edges = [edge for edge in graph.edges(data=True) if (edge[2]['weight'] > 2)]


selected_nodes = set()
for edge in edges:
    selected_nodes.add(edge[0])
    selected_nodes.add(edge[1])
# len(selected_nodes)
len(graph.nodes())




#graph = graph.subgraph(selected_nodes)
all_node = set()
for n in graph.nodes():
    all_node.add(n)

graph.clear()
for i in edges:
    u = i[0]
    v = i[1]
    if u not in graph.nodes():
        graph.add_node(u)
    if v not in graph.nodes():
        graph.add_node(v)
    graph.add_edge(u, v, id=i[2]['id'], weight=i[2]['weight'])
for n in all_node:
    if n not in graph.nodes():
        graph.add_node(n)



print ("num of node after filtering:", len(graph.nodes()))
print("num of edges after filtering:", len(graph.edges()))


print ("the density of the graph:", nx.density(graph))
