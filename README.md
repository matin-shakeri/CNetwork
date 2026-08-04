
<p align="center">
  <img src="assets/logo.svg" alt="CNetwork Logo" width="450">
</p>

<h1 align="center">🚀 CNetwork</h1>

<p align="center">
  <b>A High-Performance C Library for Complex Network Analysis</b>
</p>

<p align="center">
  <i>A lightweight, ultra-fast C substitute for Python's NetworkX.</i>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg">
  </a>
  <img src="https://img.shields.io/badge/C-C99-blue">
  <img src="https://img.shields.io/badge/OpenMP-supported-success">
</p>


## 📌 Overview

**CNetwork** is a high-performance C library designed for graph theory and complex network analysis. While tools like Python's `NetworkX` offer great expressiveness, they suffer from substantial memory overhead and execution limits on large-scale systems. **CNetwork** brings raw C speed, efficient memory representation, and parallel execution options (via OpenMP) to large-scale network algorithms.

---

## 🏗 Library Architecture

The diagram below illustrates the structural design of CNetwork, moving from underlying data structures up to high-level analysis routines and visualization integration:

```text
+-------------------------------------------------------------------+
|                        Application Layer                          |
|         (User Code / Benchmarks / Data Analysis Pipelines)        |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                          Algorithms Layer                         |
|   +-------------------+  +-------------------+  +---------------+ |
|   |    Centralities   |  | Shannon Entropy   |  | Line Digraph  | |
|   +-------------------+  +-------------------+  +---------------+ |
|   |     Line Graph    |  | Random Walks      |  | Assortativity | |
|   +-------------------+  +-------------------+  +---------------+ |
|   +-------------------+  +-------------------+  +---------------+ |
|   |     Clustering    |  | spectrum          |  |  Corellation  | |
|   +-------------------+  +-------------------+  +---------------+ |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                     Core Graph Representation                     |
|   +-----------------------------------------------------------+   |
|   | Adjacency Matrix / CSR (Compressed Sparse Row) / List     |   |
|   +-----------------------------------------------------------+   |
|   | Memory Allocator, Node/Edge Management, Thread Pool       |   |
|   +-----------------------------------------------------------+   |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                  IO & Visualization Integration                   |
|      (Gnuplot Pipe Export, GraphML/EdgeList File Loaders)         |
+-------------------------------------------------------------------+
```

## 🚀 Key Features

- **Extreme Performance:** Native C performance with zero runtime wrapper overhead.
- **Low Memory Footprint:** Efficient sparse matrix representations for large graphs.
- **Parallel Computing:** Built-in parallelization capabilities for heavy algorithms like Betweenness Centrality and PageRank.
- **Seamless Plotting:** Native pipe interface to export matrix heatmaps and network data directly to Gnuplot.

---

## 🛠 Installation & Compilation

### Prerequisites
- `gcc` or `clang` (C99 standard or higher)
- `make` or `cmake`
- `Gnuplot` (optional, for visualization)

### Building from Source

```bash
# Clone the repository
git clone https://github.com/matin-shakeri/CNetwork.git
cd CNetwork

# Build the library and examples using GCC
gcc -O3 -Iinclude src/CNetwork.c examples/example.c -o example -lm

# Run the example executable
./example