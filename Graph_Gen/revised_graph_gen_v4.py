import os
import json
import re
import struct
import networkx as nx
from tqdm import tqdm

POINTER_REGEX = re.compile(r"0x[1-9a-zA-Z][0-9a-zA-Z]{11}")
POINTER_SIZE = 8
DATA_SIZE = 16
MALLOC_HEADER_SIZE = 2 * POINTER_SIZE  
VISITED = set()

def is_potential_pointer(value):
    return POINTER_REGEX.fullmatch(hex(value))

def is_valid_pointer(source, heap_start, raw_data):
    offset = source - heap_start
    if offset < 0 or offset >= len(raw_data):
        return False
    
    return True

def get_chunk_size(raw_data, source, heap_start):

    offset = source - heap_start
    if offset < 0 or offset + MALLOC_HEADER_SIZE > len(raw_data):
        return None
    size = struct.unpack("Q", raw_data[offset-POINTER_SIZE:offset])[0]
    if size > 2561:
        return None
    
    return size & ~0x07  

def recursive_edge_addition(graph, raw_data, source, json_data, heap_start, offset_to_struct):

    if source in VISITED:
        return

    chunk_size = get_chunk_size(raw_data, source, heap_start)

    offset = source - heap_start
    update_node(graph, source, chunk_size, raw_data, json_data, offset, heap_start, offset_to_struct)

def update_node(graph, source, chunk_size, raw_data, json_data, offset, heap_start, offset_to_struct):
    pointer_count = 0
    valid_pointer_count = 0
    invalid_pointer_count = 0
    first_valid_pointer_offset = -1
    first_pointer_offset = -1
    last_valid_pointer_offset = -1
    last_pointer_offset = -1

    if chunk_size is not None:
        for i in range(0, chunk_size, POINTER_SIZE):
            potential_pointer = struct.unpack("Q", raw_data[offset+i:offset+i+POINTER_SIZE])[0]
            if is_potential_pointer(potential_pointer):
                pointer_count += 1
                if is_valid_pointer(potential_pointer, heap_start, raw_data):
                    last_valid_pointer_offset = i/chunk_size
                    if first_valid_pointer_offset is None:
                        first_valid_pointer_offset = i/chunk_size



                    valid_pointer_count += 1
                else:
                    last_pointer_offset = i/chunk_size
                    if first_pointer_offset is None:
                        first_pointer_offset = i/chunk_size
                    invalid_pointer_count += 1

    struct_size_normalized = chunk_size if chunk_size is not None else 0
    #divide struct size by the heap size
    struct_size_normalized = struct_size_normalized/ len(raw_data)

    valid_pointer_count_normalized = valid_pointer_count / pointer_count if pointer_count != 0 else 0
    invalid_pointer_count_normalized = invalid_pointer_count / pointer_count if pointer_count != 0 else 0
    graph.add_node(source,
                   label=hex(source),
                   cat=0,
                   struct_size=struct_size_normalized,
                   valid_pointer_count=valid_pointer_count_normalized,
                   invalid_pointer_count=invalid_pointer_count_normalized,
                   first_pointer_offset=first_pointer_offset,
                   last_pointer_offset=last_pointer_offset,
                   first_valid_pointer_offset=first_valid_pointer_offset,
                   last_valid_pointer_offset=last_valid_pointer_offset)

    if hex(source)[2:] in str(json_data["KEY_C_ADDR"]) or hex(source)[2:] in str(json_data["KEY_D_ADDR"]):
        graph.nodes[source].update({'label': "KEY", 'cat': 1})

    if hex(source)[2:] in str(json_data["SSH_STRUCT_ADDR"]):
        graph.nodes[source].update({'label': "SSH_STRUCT_ADDR"})
    
    VISITED.add(source)
 
    if chunk_size is None:
        return
    
    for i in range(0, chunk_size, POINTER_SIZE):
        update_potential_pointer(graph, raw_data, source, offset, i, json_data, heap_start, offset_to_struct, chunk_size)

def update_potential_pointer(graph, raw_data, source, offset, i, json_data, heap_start, offset_to_struct, chunk_size):
    potential_pointer = struct.unpack("Q", raw_data[offset+i:offset+i+POINTER_SIZE])[0]
    if is_potential_pointer(potential_pointer):
        if is_valid_pointer(potential_pointer, heap_start, raw_data):
            graph.add_edge(source, potential_pointer, offset= (offset+i)/chunk_size)

            recursive_edge_addition(graph, raw_data, potential_pointer, json_data, heap_start, i)


def remove_cycles(graph):
    while True:
        cycles = list(nx.simple_cycles(graph))
        if not cycles:
            break  # Exit the loop if there are no cycles
        cycle = cycles[0]  # Only handle one cycle at a time
        for i in range(len(cycle)):
            if i == len(cycle) - 1:
                if graph.has_edge(cycle[i], cycle[0]):
                    graph.remove_edge(cycle[i], cycle[0])
            else:
                if graph.has_edge(cycle[i], cycle[i+1]):
                    graph.remove_edge(cycle[i], cycle[i+1])

def process_raw_file(raw_file, json_file):
    with open(json_file) as f, open(raw_file, "rb") as f2:
        json_data, raw_data, graph = json.load(f), f2.read(), nx.DiGraph()
    heap_start = int(json_data["HEAP_START"], 16)

    ssh_struct_addr = int(json_data["SSH_STRUCT_ADDR"], 16)
    recursive_edge_addition(graph, raw_data, ssh_struct_addr, json_data, heap_start, 0)
    for i in range(0, len(raw_data), POINTER_SIZE):
        global_i, potential_pointer = i + heap_start, struct.unpack("Q", raw_data[i:i+POINTER_SIZE])[0]
        recursive_edge_addition(graph, raw_data, potential_pointer, json_data, heap_start, i) if is_potential_pointer(potential_pointer) else None

    return graph

def add_root_node(graph,json,  is_ssh_struct = False):
    if is_ssh_struct:
        return
         
    root_node = "root"
    graph.add_node(root_node, label=root_node, cat=0, struct_size=0, pointer_count=0, valid_pointer_count=0, invalid_pointer_count=0)
    [graph.add_edge(root_node, node, offset=0) for node in graph.nodes() if len(list(graph.predecessors(node))) == 0 and node != root_node]

def main():
    base_path, output_path = "Data/", "Generated_Graphs"
    os.makedirs(output_path, exist_ok=True)
    for folder in os.listdir(base_path):
        os.makedirs(os.path.join(output_path, folder), exist_ok=True)
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            print("Processing folder:", folder_path)
            raw_files = [f for f in os.listdir(folder_path) if f.endswith(".raw")]
            [generate_graph(folder, raw_file, output_path, folder_path) for raw_file in tqdm(raw_files)]
             



def generate_graph(folder, raw_file, output_path, folder_path):
    VISITED.clear()
    json_file = raw_file.replace("-heap.raw", ".json")
    raw_file_path, json_file_path = os.path.join(folder_path, raw_file), os.path.join(folder_path, json_file)
    graph = process_raw_file(raw_file_path, json_file_path)


    #add_root_node(graph, json.load(open(json_file_path)), is_ssh_struct = False)
    #remove_cycles(graph)
    graph = nx.convert_node_labels_to_integers(graph)
    nx.write_graphml(graph, os.path.join(output_path, folder, raw_file.replace(".raw", ".graphml")))
    

if __name__ == "__main__":
    main()