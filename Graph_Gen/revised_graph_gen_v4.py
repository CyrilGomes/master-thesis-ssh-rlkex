import os
import json
import re
import struct
import networkx as nx
from tqdm import tqdm
import argparse

POINTER_REGEX = re.compile(r"0x[1-9a-zA-Z][0-9a-zA-Z]{11}")
POINTER_SIZE = 8
DATA_SIZE = 16
MALLOC_HEADER_SIZE = 2 * POINTER_SIZE  
VISITED = set()


def is_potential_pointer(value):
    """Check if a value is a potential pointer."""
    return POINTER_REGEX.fullmatch(hex(value))

def is_valid_pointer(source, heap_start, raw_data):
    """Check if a pointer is valid within the heap context."""
    offset = source - heap_start
    return 0 <= offset < len(raw_data)

def get_chunk_size(raw_data, source, heap_start):
    """Calculate the size of a memory chunk."""
    offset = source - heap_start
    if offset < 0 or offset + MALLOC_HEADER_SIZE > len(raw_data):
        return None
    size = struct.unpack("Q", raw_data[offset-POINTER_SIZE:offset])[0]
    return (size & ~0x07) if size <= len(raw_data) else None

def recursive_edge_addition(graph, raw_data, source, json_data, heap_start, training = False):

    if source in VISITED:
        return

    chunk_size = get_chunk_size(raw_data, source, heap_start)

    offset = source - heap_start
    update_node(graph, source, chunk_size, raw_data, json_data, offset, heap_start, training)


def update_node(graph, source, chunk_size, raw_data, json_data, offset, heap_start, training):
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
                    if first_valid_pointer_offset == -1:
                        first_valid_pointer_offset = i/chunk_size



                    valid_pointer_count += 1
                else:
                    last_pointer_offset = i/chunk_size
                    if first_pointer_offset == -1:
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

    if(training):
        cat_counter = 0
        for key, value in json_data.items():
            # Check if key matches regex "KEY_*_ADDR"
            match = re.match(r"KEY_([0-9a-zA-Z]*)_ADDR", key)
            if match:
                letter = match.group(1)
                key_attribute = f"KEY_{letter}"
                # Check if the corresponding KEY_* exists and is not empty
                if key_attribute in json_data and json_data[key_attribute]:
                    if hex(source)[2:] in str(value):
                        cat_counter += 1
                        graph.nodes[source].update({'label': key, 'cat': cat_counter})

        if hex(source)[2:] in str(json_data["SSH_STRUCT_ADDR"]):
            graph.nodes[source].update({'label': "SSH_STRUCT_ADDR"})
    
    VISITED.add(source)
 
    if chunk_size is None:
        return
    
    for i in range(0, chunk_size, POINTER_SIZE):
        update_potential_pointer(graph, raw_data, source, offset, i, json_data, heap_start, chunk_size)

def update_potential_pointer(graph, raw_data, source, offset, i, json_data, heap_start, chunk_size):
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



def add_root_node(graph,json,  is_ssh_struct = False):
    if is_ssh_struct:
        return
         
    root_node = "root"
    graph.add_node(root_node, label=root_node, cat=0, struct_size=0, pointer_count=0, valid_pointer_count=0, invalid_pointer_count=0)
    [graph.add_edge(root_node, node, offset=0) for node in graph.nodes() if len(list(graph.predecessors(node))) == 0 and node != root_node]


def iterative_edge_addition(graph, raw_data, start_node, json_data, heap_start, training=False):
    stack = [start_node]

    while stack:
        current_node = stack.pop()
        if current_node in VISITED:
            continue

        chunk_size = get_chunk_size(raw_data, current_node, heap_start)
        offset = current_node - heap_start
        update_node(graph, current_node, chunk_size, raw_data, json_data, offset, heap_start, training)

        VISITED.add(current_node)

        if chunk_size is not None:
            for i in range(0, chunk_size, POINTER_SIZE):
                potential_pointer = struct.unpack("Q", raw_data[offset+i:offset+i+POINTER_SIZE])[0]
                if is_potential_pointer(potential_pointer):
                    if is_valid_pointer(potential_pointer, heap_start, raw_data):
                        graph.add_edge(current_node, potential_pointer, offset=(offset+i)/chunk_size)
                        stack.append(potential_pointer)

def process_raw_file(raw_file, json_file, training):
    """Process raw and JSON files to construct the graph."""
    with open(json_file) as f, open(raw_file, "rb") as f2:
        json_data, raw_data, graph = json.load(f), f2.read(), nx.DiGraph()
    heap_start = int(json_data["HEAP_START"], 16)
    
    iterative_edge_addition(graph, raw_data, int(json_data["SSH_STRUCT_ADDR"], 16), json_data, heap_start, training)

    return graph

def generate_graph(raw_file, json_file, output_folder, training):
    """Generate graph from raw file and json file."""
    VISITED.clear()
    graph = process_raw_file(raw_file, json_file, training)
    nx.write_graphml(graph, os.path.join(output_folder, os.path.basename(raw_file).replace(".raw", ".graphml")))

def main(folder_path, output_folder, training):
    """Main function to process folder of files and generate graphs."""
    os.makedirs(output_folder, exist_ok=True)
    for raw_file in tqdm(os.listdir(folder_path)):
        if raw_file.endswith("-heap.raw"):
            json_file = raw_file.replace("-heap.raw", ".json")
            generate_graph(os.path.join(folder_path, raw_file), 
                           os.path.join(folder_path, json_file), 
                           output_folder, training)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw memory data and generate graphs.")
    parser.add_argument("folder_path", help="Path to the folder containing raw and JSON files")
    parser.add_argument("output_folder", help="Folder to save the generated graphs")
    parser.add_argument("--training", action="store_true", help="Flag to indicate training mode")
    args = parser.parse_args()

    main(args.folder_path, args.output_folder, args.training)