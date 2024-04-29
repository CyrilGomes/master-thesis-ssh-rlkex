import networkx as nx
import graphistry

def get_target_nodes( G):
    target_nodes = [node for node, attributes in G.nodes(data=True) if attributes['cat'] == '1']
    return target_nodes
    
# Read graph.ml file with NetworkX
G = nx.read_graphml('/home/cyril/ssh-rlkex/Generated_Graphs/output/basic/V_6_8_P1/24/25740-1643890740.graphml')
graphistry.register(api=3, protocol="https", server="hub.graphistry.com", username="Meneur07", password="*X35fdR-:gnzJ6!")
#print if the graph is directed
print(nx.is_directed(G))

#make sure the graph is directed
G = G.to_directed()

#remove all individual vertices with in_degree = 0 and out_degree = 0

for node in list(G.nodes):
    if G.in_degree(node) == 0 and G.out_degree(node) == 0:
        G.remove_node(node)

#get the bfs subgraph starting from the target nodes, then combine them into one graph
target_nodes = get_target_nodes(G)
subgraphs = []
G = G.reverse()
for target_node in target_nodes:
    subgraphs.append(nx.bfs_tree(G, target_node))
G = nx.compose_all(subgraphs)
G = G.reverse()


#apply colors for target nodes
for target_node in target_nodes:
    G.nodes[target_node]['color'] = 'rgba(0, 255, 0, 0.5)'

#apply colors for root nodes
root_nodes = [node for node in G.nodes() if len(list(G.predecessors(node))) == 0]
for root_node in root_nodes:
    G.nodes[root_node]['color'] = 'rgba(255, 0, 0, 0.5)'


#apply colors
#print nodes data from the graph
for node, attributes in G.nodes(data=True):
    print(node, attributes)

# Plot graph with Graphistry
g = graphistry.bind(source='src', destination='dst', node='id', point_color='color', )
g.plot(G)