# Graph Generation Component

This component is responsible for converting heap dumps into graph structures for further analysis. It's implemented in Rust for efficient processing of large heap dumps.

## Structure

```
Graph_Gen/
├── revised_graph_gen_v4.py   # Python implementation
└── graph_gen_rust/          # Rust implementation
    ├── Cargo.toml
    └── src/
        └── main.rs
```

## Usage

### Rust Implementation

1. Build the project:
```bash
cd graph_gen_rust
cargo build --release
```

2. Run the converter:
```bash
./target/release/graph_gen_rust <input_heap_dump> <output_graph>
```

### Python Implementation

The Python implementation (`revised_graph_gen_v4.py`) is provided for reference and development purposes. The Rust implementation should be preferred for production use due to better performance.

```bash
python revised_graph_gen_v4.py <input_heap_dump> <output_graph>
```

## Graph Format

[Add description of your graph format here]
