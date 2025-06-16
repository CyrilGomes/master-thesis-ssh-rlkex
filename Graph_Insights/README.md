# Graph Insights Component

This component contains tools and utilities for analyzing and visualizing the graph structures generated from heap dumps.

## Structure

```
Graph_Insights/
├── complexity_insights.ipynb    # Analysis of graph complexity
├── connected_comp.py           # Connected components analysis
├── depth_check.py             # Graph depth analysis
├── graph_data_insights.py     # General graph statistics
├── graph_to_spt.py           # Shortest path tree conversion
├── graph_viz_matplot.py      # Matplotlib visualization
├── graph_viz.py             # Interactive visualization
├── neighbours_check.py      # Node neighborhood analysis
└── test_gnn.py             # GNN testing utilities
```

## Features

1. **Graph Analysis**
   - Connected components analysis
   - Depth and breadth analysis
   - Node neighborhood statistics
   - Complexity metrics

2. **Visualization**
   - Interactive graph visualization
   - Static graph plots
   - Path visualization

3. **GNN Testing**
   - Basic GNN architecture testing
   - Node feature analysis

## Usage

### Graph Analysis

```python
from graph_data_insights import analyze_graph
from connected_comp import find_components

# Analyze graph properties
stats = analyze_graph(graph_path)

# Find connected components
components = find_components(graph_path)
```

### Visualization

```python
from graph_viz import visualize_graph

# Create interactive visualization
visualize_graph(graph_path, output_path='graph.html')
```

## Jupyter Notebooks

1. `complexity_insights.ipynb`: Detailed analysis of graph complexity metrics
2. `path_viz/*.ipynb`: Various notebooks for path visualization and analysis
