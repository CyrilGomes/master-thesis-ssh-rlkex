import networkx as nx
from pyvis.network import Network
import random
import os

def load_random_graph(directory):
    files = [file for file in os.listdir(directory) if file.endswith('.graphml')] # or other format
    random_file = random.choice(files)
    path = os.path.join(directory, random_file)
    return nx.read_graphml(path)  # Change this according to your graph format

def get_root_nodes(G):
    return [n for n, d in G.in_degree() if d == 0]

def plot_graph(G, root_nodes):
    net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white', notebook=True)

    # Add nodes and edges
    for node in G.nodes:
        net.add_node(node, label=str(node), title=str(node))

    for edge in G.edges:
        net.add_edge(edge[0], edge[1])

    # Set options for nodes
    for node in root_nodes:
        net.get_node(node)['color'] = 'red'

    # Generate network with specific layout
    net.from_nx(G)
    net.show('graph.html')

def get_target_nodes(root, G):
    target_nodes = [node for node, attributes in G.nodes(data=True) if attributes['cat'] == 1]
    return target_nodes
    

directory = 'Generated_Graphs/output/'
G = load_random_graph(directory)
root_nodes = get_root_nodes(G)
plot_graph(G, root_nodes)
