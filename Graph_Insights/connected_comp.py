import networkx as nx
import graphistry

# Read graph.ml file with NetworkX
G = nx.read_graphml('Generated_Graphs/64/18041-1643986141-heap.graphml')
graphistry.register(api=3, protocol="https", server="hub.graphistry.com", username="Meneur07", password="*X35fdR-:gnzJ6!")

G_undirected = G.to_undirected()
G_undirected = nx.bfs_tree(G, '0')

# Plot graph with Graphistry
g = graphistry.bind(source='src', destination='dst')
g.plot(G_undirected)