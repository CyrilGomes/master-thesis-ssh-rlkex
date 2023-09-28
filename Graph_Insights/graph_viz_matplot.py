import networkx as nx
import graphistry

# Read graph.ml file with NetworkX
G = nx.read_graphml('Generated_Graphs/64/18038-1643986141-heap.graphml')

#use BFS to get the tree from the node with id 0
G = nx.subgraph(G, nx.bfs_tree(G, '0'))

labels = {}
for node in G.nodes:
    labels[node] = node + " " + G.nodes[node]['label']  + ' ' + str(G.nodes[node]['cat'])
#use matplotlib to plot the graph
import matplotlib.pyplot as plt
#nx.draw(G, with_labels=True, labels=labels)
#spring layout
nx.draw_spring(G, with_labels=True, labels=labels)
plt.show()
