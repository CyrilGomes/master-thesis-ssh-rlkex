import networkx as nx


G = nx.read_graphml('Generated_Graphs/64/18041-1643986141-heap.graphml')


#print the number of nodes that has more that k out edges
def print_nodes_with_more_than_k_out_edges(G, k):
    for node in G.nodes():
        if G.out_degree(node) > k:
            print(node)

print_nodes_with_more_than_k_out_edges(G, 50)