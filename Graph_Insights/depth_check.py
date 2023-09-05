import networkx as nx
import graphistry


folder_path = 'Generated_Graphs/64/'

def delete_nodes_by_depth(graph, root_node, max_depth):
    nodes_to_delete = []
    for node in graph.nodes():
        if nx.has_path(graph, root_node, node):
            depth = nx.shortest_path_length(graph, root_node, node)
            if depth > max_depth:
                nodes_to_delete.append(node)
    graph.remove_nodes_from(nodes_to_delete)

def check_for_cat_1_nodes(graph):
    for node in graph.nodes():
        if graph.nodes[node].get("cat") == 1:
            return True
    return False

    
G = nx.read_graphml('Generated_Graphs/64/18041-1643986141-heap.graphml')

old_length = G.number_of_nodes()

delete_nodes_by_depth(G, "root", 4)

graphistry.register(api=3, protocol="https", server="hub.graphistry.com", username="Meneur07", password="*X35fdR-:gnzJ6!")
# Plot graph with Graphistry
g = graphistry.bind(source='src', destination='dst')
print(G.nodes())  # Output: [1, 2, 3]
print(check_for_cat_1_nodes(G))  # Output: True
print(f"Removed nodes : {old_length - G.number_of_nodes()}")
g.plot(G)


