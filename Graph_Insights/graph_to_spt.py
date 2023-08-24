import networkx as nx
import graphistry


def shortest_path_tree(graph, source):
    # Compute shortest paths from the source to all other nodes
    paths = nx.single_source_shortest_path(graph, source)
    print(len(paths))
    # Create a new graph for the shortest path tree
    tree = nx.DiGraph()
    
    # Populate the tree with edges from the shortest paths
    for target, path in paths.items():
        for i in range(len(path) - 1):
            tree.add_edge(path[i], path[i + 1])
            #add node attributes
            tree.nodes[path[i]].update(graph.nodes[path[i]])
            tree.nodes[path[i+1]].update(graph.nodes[path[i+1]])
    
    return tree

def main():

    # Read graph.ml file with NetworkX
    G = nx.read_graphml('Generated_Graphs/64/18041-1643986141-heap.graphml')
    graphistry.register(api=3, protocol="https", server="hub.graphistry.com", username="Meneur07", password="*X35fdR-:gnzJ6!")

    source_node = ""
    for node in G.nodes:
        if G.nodes[node]['label'] == 'root':
            source_node = node
            break
    
    # Compute shortest path tree
    G_prime = shortest_path_tree(G, source_node)

    print(G_prime)
    # Plot graph with Graphistry
    g = graphistry.bind(source='src', destination='dst')
    g.plot(G)

if __name__ == '__main__':
    main()