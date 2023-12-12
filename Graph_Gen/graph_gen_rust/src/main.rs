use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::dot::{Dot, Config};
use serde_json::Value;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};
use regex::Regex;
use indicatif::{ProgressBar, ProgressStyle};
use petgraph_graphml::GraphMl;
use std::time::Instant;
use std::borrow::Cow;
use lazy_static::lazy_static;
use rayon::prelude::*;
use walkdir::WalkDir;
use std::path::PathBuf;
use std::collections::HashMap;
use rand::{thread_rng, seq::SliceRandom};

fn main() {
    let args: Vec<String> = env::args().collect();

    let input_folder = &args[1];
    let output_folder = &args[2];
    let max_files_per_subfolder: usize = args[3].parse().expect("Invalid number for max_files_per_subfolder");
    let subfolder_cache = Arc::new(Mutex::new(HashMap::new()));
    
    if !Path::new(output_folder).exists() {
        fs::create_dir_all(output_folder).unwrap();
    }

    if !Path::new(input_folder).exists() {
        panic!("Input folder does not exist");
    }

    let mut files_by_subfolder = HashMap::new();
    for ientry in WalkDir::new(input_folder)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().map(|ext| ext == "json").unwrap_or(false))
    {
        files_by_subfolder.entry(ientry.path().parent().unwrap().to_path_buf())
            .or_insert_with(Vec::new)
            .push(ientry.path().to_path_buf());
    }
    // Randomly select files from each subfolder
    let mut rng = thread_rng();
    let entries: Vec<_> = files_by_subfolder.values_mut()
        .flat_map(|files| {
            files.shuffle(&mut rng);
            files.iter().take(max_files_per_subfolder).cloned()
        })
        .collect();
    /* 
    // Collect all relevant entries
    let entries: Vec<_> = WalkDir::new(input_folder)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|ext| ext == "json").unwrap_or(false))
        .collect();
    */
    let total_files = entries.len();
    let progress_bar = ProgressBar::new(total_files as u64);
    let progress_style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap_or_else(|e| panic!("Error creating progress bar style: {:?}", e))
        .progress_chars("#>-");

    progress_bar.set_style(progress_style);

    entries.into_par_iter().for_each(|entry| {
        let path = entry;
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let raw_file_name = format!("{}-heap.raw", stem);
        let raw_file_path = path.with_file_name(raw_file_name);

        if raw_file_path.exists() {
            let cache = subfolder_cache.clone();

            let output_subfolder = create_output_subfolder(
                input_folder,
                output_folder,
                path.parent().unwrap(),
                &cache,
            );
            process_files(&path, &raw_file_path, &output_subfolder, stem);
        }

        progress_bar.inc(1);
    });

    progress_bar.finish_with_message("Processing complete");
}

fn create_output_subfolder(
    base_input_folder: &str,
    base_output_folder: &str,
    input_subfolder: &Path,
    cache: &Arc<Mutex<HashMap<String, String>>>,
) -> String {
    let input_subfolder_str = input_subfolder.to_str().unwrap().to_owned();

    {
        let mut cache = cache.lock().unwrap();

        // Check if the path is already in the cache
        if let Some(cached_output_subfolder) = cache.get(&input_subfolder_str) {
            return cached_output_subfolder.clone();
        }
    } // Release the lock here

    let relative_path = input_subfolder.strip_prefix(Path::new(base_input_folder))
        .unwrap_or_else(|_| panic!("Subfolder is not a subdirectory of the input base directory"));

    let output_subfolder = Path::new(base_output_folder).join(relative_path);

    if !output_subfolder.exists() {
        fs::create_dir_all(&output_subfolder).unwrap();
    }

    let output_subfolder_str = output_subfolder.to_str().unwrap().to_owned();

    // Cache the computed output subfolder
    let mut cache = cache.lock().unwrap();
    cache.insert(input_subfolder_str, output_subfolder_str.clone());

    output_subfolder_str
}

lazy_static! {
    static ref POTENTIAL_POINTER_REGEX: Regex = Regex::new(r"0x[1-9a-fA-F][0-9a-fA-F]{11}").unwrap();
    static ref KEY_REGEX: Regex = Regex::new(r"KEY_([A-Z])_ADDR").unwrap();
}

fn process_files(json_path: &Path, raw_path: &Path, output_folder: &str, base_name: &str) {
    let json_data = fs::read_to_string(json_path).unwrap();
    let json: Value = serde_json::from_str(&json_data).unwrap();

    //heap start is the address of the first byte of the heap, get it as hex
    let heap_start = json["HEAP_START"].as_str().unwrap();

    let heap_start_usize = usize::from_str_radix(&heap_start, 16).unwrap();
    //print the usize in hex in format 0x...

    let heap_size = fs::metadata(raw_path).unwrap().len();


    let mut graph: petgraph::prelude::Graph<Pointer, Edge> = DiGraph::new();
    let raw_data = fs::read(raw_path).unwrap();

    let mut iterated_addresses = HashSet::new();
    let mut processed_addresses = HashSet::new();
    for offset in (0..raw_data.len()).step_by(8) {

        if let Some(pointer_value) = read_memory(&raw_data, offset) {

            //check if the pointer is 0, if it is, skip it

            let is_within_bounds: bool = pointer_value >= heap_start_usize && pointer_value < heap_start_usize + heap_size as usize;
            if !is_within_bounds {
                continue;
            }

            if processed_addresses.contains(&pointer_value) {
                continue;
            }


            if is_potential_pointer(pointer_value) && is_within_bounds{
                //compate pointer value to "0x5621094a7500"


                
                let before = Instant::now();

                iterated_addresses.insert(offset); // Only insert if it's a potential pointer
                process_struct(&raw_data, pointer_value, &mut graph, &mut processed_addresses, heap_start_usize, heap_size as usize, &json, false);

            }

        }
    }
    save_graph(&graph, output_folder, base_name);
}




fn is_key_pointer(pointer_address: usize, json: &Value) -> bool {

    // Iterate through the JSON keys
    for (key, value) in json.as_object().unwrap() {
        if let Some(caps) = KEY_REGEX.captures(key) {
            // Check if the key matches the pattern and the value matches the address
            if let Some(addr_str) = value.as_str() {
                if let Ok(addr) = usize::from_str_radix(&addr_str, 16) {
                    if addr == pointer_address {
                        // Check if "KEY_X" exists
                        let key_name = format!("KEY_{}", caps.get(1).unwrap().as_str());
                        if let Some(value) = json.get(&key_name) {
                            if let Some(value_str) = value.as_str() {
                                if !value_str.is_empty() {
                                    return true; // The pointer address matches a key and its value is not empty
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    false
}

fn process_struct(heap_data: &[u8], pointer_address: usize, graph: &mut DiGraph<Pointer, Edge>, processed_addresses: &mut HashSet<usize>, heap_start: usize, heap_size: usize, json: &Value, is_iteration : bool) {


    if processed_addresses.contains(&pointer_address) {
        

        return; // Already processed this struct
    }


    // Mark all addresses within this struct as processed
    if is_iteration {

        if  let Some(target_node) = read_memory(heap_data, pointer_address - heap_start) {

            let is_within_bounds: bool = target_node >= heap_start && target_node < heap_start + heap_size;

            if is_within_bounds && is_potential_pointer(target_node){

                process_struct(heap_data, target_node, graph, processed_addresses, heap_start, heap_size, json, false);

            }
        }
        return;
    }


    // Get the struct size
    
    let struct_size = get_struct_size(heap_data, pointer_address, heap_start, heap_size);


    // Label is the address of the struct in hex
    let mut struct_pointer = Pointer {
        label: format!("{:#x}", pointer_address),
        struct_size: struct_size, // This will be set after confirming the struct size
        cat: if is_key_pointer(pointer_address, json) { 1 } else { 0 },
        valid_pointer_count: 0,
        invalid_pointer_count: 0,
        first_pointer_offset: 0,
        last_pointer_offset: 0,
        first_valid_pointer_offset: 0,
        last_valid_pointer_offset: 0,
        address: pointer_address,
    };

    processed_addresses.insert(pointer_address);

    let current_node = create_or_get_node(graph, pointer_address, struct_size, format!("{:#x}", pointer_address, ), is_key_pointer(pointer_address, json));

    if struct_size != 0 {
        for offset in (0..struct_size).step_by(8) {
            let potential_pointer_address = pointer_address + offset;

            if let Some(potential_pointer) = read_memory(heap_data, potential_pointer_address - heap_start) {

                if !is_potential_pointer(potential_pointer){
                    continue; // Not a potential pointer
                }

                
                processed_addresses.insert(pointer_address + offset);
                struct_pointer.last_pointer_offset = offset;
                if struct_pointer.first_pointer_offset == usize::MAX {
                    struct_pointer.first_pointer_offset = offset;
                }

                let is_within_bounds: bool = potential_pointer >= heap_start && potential_pointer < heap_start + heap_size;
                if is_within_bounds {
                    struct_pointer.valid_pointer_count += 1;
                    struct_pointer.last_valid_pointer_offset = offset;
                    if struct_pointer.first_valid_pointer_offset == usize::MAX {
                        struct_pointer.first_valid_pointer_offset = offset;
                    }
                } else {
                    struct_pointer.invalid_pointer_count += 1;
                }

                // Add edge and possibly recurse
                let edge = Edge { offset }; // Create an edge with the offset
                let target_node = create_or_get_node(graph, potential_pointer, 0, format!("{:#x}", potential_pointer), is_key_pointer(potential_pointer, json));
                graph.add_edge(current_node, target_node, edge); // Add the edge with offset
                if is_within_bounds && !processed_addresses.contains(&potential_pointer) {
                    process_struct(heap_data, potential_pointer, graph, processed_addresses, heap_start, heap_size, &json, false);
                }
            }
        }
        
        modify_node(graph, current_node, struct_pointer)
    }
    

}


fn modify_node(graph: &mut DiGraph<Pointer, Edge>, node: NodeIndex, pointer : Pointer) {
    graph[node].struct_size = pointer.struct_size;
    graph[node].valid_pointer_count = pointer.valid_pointer_count;
    graph[node].invalid_pointer_count = pointer.invalid_pointer_count;
    graph[node].first_pointer_offset = pointer.first_pointer_offset;
    graph[node].last_pointer_offset = pointer.last_pointer_offset;
    graph[node].first_valid_pointer_offset = pointer.first_valid_pointer_offset;
    graph[node].last_valid_pointer_offset = pointer.last_valid_pointer_offset;

}


fn get_node_by_address(graph: &mut DiGraph<Pointer, ()>, address: usize) -> NodeIndex {
    for node in graph.node_indices() {
        if graph[node].address == address {
            return node;
        }
    }
    panic!("Node not found");
}

fn create_or_get_node(graph: &mut DiGraph<Pointer, Edge>, address: usize, struct_size: usize, label: String, is_key : bool) -> NodeIndex {


    // Check if a node for this address already exists
    // For simplicity, we're assuming here that each address only corresponds to one node.
    // In a more complex scenario, you might need a map to track address-to-node relationships.
    for node in graph.node_indices() {
        if graph[node].address == address {
            return node;
        }
    }

    // If not found, create a new node
    let pointer = Pointer {
        label: label,
        struct_size,
        cat: if is_key { 1 } else { 0 },
        valid_pointer_count: 0,
        invalid_pointer_count: 0,
        first_pointer_offset: 0,
        last_pointer_offset: 0,
        first_valid_pointer_offset: 0,
        last_valid_pointer_offset: 0,
        address: address,
    };

    graph.add_node(pointer)
}

fn save_graph(graph: &DiGraph<Pointer, Edge>, output_folder: &str, base_name: &str) {
    let output_file = format!("{}/{}.graphml", output_folder, base_name);
    
    let graphml = GraphMl::new(&graph).export_node_weights(Box::new(|node: &Pointer| {

        vec![
            ("label".into(), node.label.clone().into()),
            ("address".into(), node.address.to_string().into()),
            ("struct_size".into(), node.struct_size.to_string().into()),
            ("cat".into(), node.cat.to_string().into()),
            ("valid_pointer_count".into(), node.valid_pointer_count.to_string().into()),
            ("invalid_pointer_count".into(), node.invalid_pointer_count.to_string().into()),
            ("first_pointer_offset".into(), node.first_pointer_offset.to_string().into()),
            ("last_pointer_offset".into(), node.last_pointer_offset.to_string().into()),
            ("first_valid_pointer_offset".into(), node.first_valid_pointer_offset.to_string().into()),
            ("last_valid_pointer_offset".into(), node.last_valid_pointer_offset.to_string().into()),
            
        ]
    }))
    .export_edge_weights(Box::new(|edge: &Edge| {
        vec![("offset".into(), edge.offset.to_string().into())]
    }));


    fs::write(output_file, graphml.to_string()).unwrap();
}




fn is_potential_pointer(value: usize) -> bool {
    let value_str = format!("{:#x}", value); // Formats the value as a hex string with '0x' prefix
    POTENTIAL_POINTER_REGEX.is_match(&value_str)
}

fn read_memory(heap_data: &[u8], offset: usize) -> Option<usize> {


    if offset + 8 > heap_data.len() {
        return None; // Out of bounds
    }


    let mut bytes = [0u8; 8];


    bytes.copy_from_slice(&heap_data[offset..offset + 8]);
    
    Some(usize::from_ne_bytes(bytes))
}



fn get_struct_size(heap_data: &[u8], pointer_address: usize, heap_start: usize, heap_size: usize) -> usize {
    // Ensure the pointer address is within the bounds of the heap
    if pointer_address < heap_start || pointer_address >= heap_start + heap_size {
        return 0;
    }

    // Calculate the offset of the header relative to the heap start
    let header_offset = pointer_address - heap_start - 8;

    // Check if the header offset is valid
    if header_offset >= heap_data.len() {
        return 0;
    }


    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&heap_data[header_offset..header_offset + 8]);
    let size = usize::from_ne_bytes(bytes);

    if size <= heap_data.len() {
        return size & !0x07;
    } else {
        return 0;
    }

}


#[derive(Debug, Clone)]
struct Pointer {
    label: String,                  // A descriptive label for the struct
    struct_size: usize,             // The size of the struct in bytes
    cat: u8,                        // Category flag (1 for Key, 0 otherwise)
    valid_pointer_count: usize,     // Count of valid pointers within the struct
    invalid_pointer_count: usize,   // Count of invalid pointers within the struct
    first_pointer_offset: usize,    // Offset of the first pointer in the struct
    last_pointer_offset: usize,     // Offset of the last pointer in the struct
    first_valid_pointer_offset: usize,  // Offset of the first valid pointer
    last_valid_pointer_offset: usize,   // Offset of the last valid pointer
    address: usize,                  // The address of the struct in memory
}



#[derive(Debug, Clone)]
struct Edge {
    offset: usize, // Offset from the struct to its parent
}


