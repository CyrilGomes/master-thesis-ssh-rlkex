import networkx as nx
import graphistry

# Read graph.ml file with NetworkX
G = nx.read_graphml('Generated_Graphs/64/18038-1643986141-heap.graphml')
graphistry.register(api=3, protocol="https", server="hub.graphistry.com", username="Meneur07", password="*X35fdR-:gnzJ6!")



#remove all individual vertices with in_degree = 0 and out_degree = 0

for node in list(G.nodes):
    if G.in_degree(node) == 0 and G.out_degree(node) == 0:
        G.remove_node(node)

for node in G.nodes:
    if G.nodes[node]['label'] == 'root':
        print(node)


# Plot graph with Graphistry
g = graphistry.bind(source='src', destination='dst')
g.plot(G)