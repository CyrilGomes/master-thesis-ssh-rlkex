# Graph Generation Component

This component is responsible for converting heap dumps into graph structures for further analysis. It's implemented in Rust for efficient processing of large heap dumps.

## Overview

The graph generator converts raw heap memory dumps into a structured graph representation where:
- Nodes represent allocated memory blocks
- Edges represent pointers between blocks
- Node features include size, pointer counts, and offsets
- Edge features include pointer offsets

## Structure

```
Graph_Gen/
├── revised_graph_gen_v4.py   # Python implementation (reference)
└── graph_gen_rust/          # Rust implementation (main)
    ├── Cargo.toml
    └── src/
        └── main.rs
```

## Usage

### Building

```bash
cd graph_gen_rust
cargo build --release
```

### Converting a Heap Dump

```bash
./target/release/graph_gen_rust <input_heap_dump> <output_graph.graphml>
```

### Output Format

The generator produces a GraphML file with the following attributes:

Node attributes:
- `struct_size`: Size of the memory block
- `valid_pointer_count`: Number of valid pointers
- `invalid_pointer_count`: Number of invalid pointers
- `first_pointer_offset`: Offset of the first pointer
- `last_pointer_offset`: Offset of the last pointer
- Various other memory-related features

Edge attributes:
- `offset`: Offset of the pointer within the source block
