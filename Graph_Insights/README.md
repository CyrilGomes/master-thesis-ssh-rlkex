# Graph Insights Component

This component contains tools and utilities for analyzing and visualizing the graph structures generated from heap dumps.

## Overview

The Graph Insights tools help understand the structure and properties of memory graphs through:
- Graph complexity analysis
- Connected component analysis
- Path visualization
- Node neighborhood analysis
- Memory structure visualization

## Structure

```
Graph_Insights/
├── complexity_insights.ipynb    # Graph complexity analysis
├── connected_comp.py           # Connected components analysis
├── depth_check.py             # Graph depth analysis
├── graph_data_insights.py     # General graph statistics
├── graph_to_spt.py           # Shortest path tree conversion
├── graph_viz_matplot.py      # Static visualization
├── graph_viz.py             # Interactive visualization
├── neighbours_check.py      # Node neighborhood analysis
└── test_gnn.py             # GNN testing utilities
```

## Features

### Analysis Tools
- Graph complexity metrics
- Connected component identification
- Path existence checking
- Node neighborhood statistics
- Strongly Connected Components (SCCs) analysis

### Visualization Tools
- Interactive graph visualization
- Path highlighting
- Component coloring
- Memory structure visualization

## Usage

### Graph Analysis

```python
from graph_data_insights import analyze_graph

# Get graph statistics
stats = analyze_graph('/path/to/graph.graphml')
```

### Visualization

```python
from graph_viz import visualize_graph

# Create interactive visualization
visualize_graph('/path/to/graph.graphml', output='graph.html')
```

### Connected Components

```python
from connected_comp import analyze_components

# Analyze graph components
components = analyze_components('/path/to/graph.graphml')
```
