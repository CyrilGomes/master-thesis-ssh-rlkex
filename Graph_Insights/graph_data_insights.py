import os
import networkx as nx
import graphistry
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
# Set path to folder containing graph.ml files

folder_path = 'Generated_Graphs/64/'


def check_degrees():
    # Initialize list to store out-degree values
    out_degrees_list = []

    # Loop over all graph.ml files in folder and show progress bar
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.graphml'):
            # Read graph.ml file with NetworkX
            G = nx.read_graphml(os.path.join(folder_path, filename))

            # Compute out-degree of each node
            out_degrees = dict(G.out_degree())

            # Append out-degree values to list
            out_degrees_list.append(list(out_degrees.values()))

    # Convert out-degree list to Pandas dataframe
    df = pd.DataFrame(out_degrees_list).transpose()
    df.columns = [f'graph_{i+1}' for i in range(len(out_degrees_list))]

    # Plot histogram of out-degree distribution using Seaborn
    sns.histplot(data=df, bins=range(int(df.max().max())+2), alpha=0.5, kde=True)
    plt.title('Out-degree distribution of all graphs')
    plt.xlabel('Out-degree')
    plt.ylabel('Frequency')
    plt.show()


def check_degree_distribution():
    graph = nx.read_graphml('Generated_Graphs/64/18041-1643986141-heap.graphml')

    # Compute out-degree of each node
    out_degrees = dict(graph.out_degree())
     
    # Append out-degree values to np array

    out_degrees_array = np.array(list(out_degrees.values()))
    #out_degrees_list = list(out_degrees.values())
     

    print(out_degrees)
    # Plot histogram of out-degree distribution using Seaborn
    sns.displot(data=out_degrees, kde = True)
    plt.title('Out-degree distribution of all graphs')
    plt.xlabel('Out-degree')
    plt.ylabel('Frequency')
    plt.show()

def check_all_degree_distribution():
    aggregated_out_degrees = []
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.graphml'):

            # Read graph.ml file with NetworkX
            G = nx.read_graphml(os.path.join(folder_path, filename))

            # Compute out-degree of each node
            out_degrees = dict(G.out_degree())
            aggregated_out_degrees += list(out_degrees.values())

    #convert to np array
    aggregated_out_degrees = np.array(aggregated_out_degrees)
    
            
    # Append out-degree values to np array

    #out_degrees_list = list(out_degrees.values())
     

    print(aggregated_out_degrees)
    # Plot histogram of out-degree distribution using Seaborn
    sns.displot(data=aggregated_out_degrees, kde = True)
    plt.title('Out-degree distribution of all graphs')
    plt.xlabel('Out-degree')
    plt.ylabel('Frequency')
    plt.show()


def check_can_reach_dest():

    can_reach_count = 0
    total_try = 0
    path_lengths = []
    # Read graph.ml file with NetworkX
    # loop over files
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.graphml'):
            src = ""
            dst = ""
            path_length = 0
            G = nx.read_graphml(os.path.join(folder_path, filename))



            #get dst
            for node in G.nodes:
                if G.nodes[node]['label'] == 'root':
                    src = node
                    
                if G.nodes[node]['cat'] == 1:
                    dst = node
                
                if(src != "" and dst != ""):
                    break
            


            total_try+=1
            #check if src can reach dst
            if nx.has_path(G, src, dst):
                path_length = nx.shortest_path_length(G, src, dst)
                path_lengths.append(path_length)
                can_reach_count+=1
    
    print(f"average path length {sum(path_lengths)/len(path_lengths)}")
    print(f"can reach destination {can_reach_count/total_try*100}% times")
    #plot histogram of path lengths
    sns.histplot(data=path_lengths, bins=range(int(max(path_lengths))+2), alpha=0.5, kde=True)
    plt.title('Path length distribution of all graphs')
    plt.xlabel('Path length')
    plt.ylabel('Frequency')
    plt.show()
    


if __name__ == '__main__':
    check_all_degree_distribution()
