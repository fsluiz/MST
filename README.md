# MST README

This repository unifies two distinct quantum walk-based approaches for solving the **Minimum Spanning Tree (MST)** problem, with a focus on **degree constraints** and **classical comparisons**. It integrates functionalities from:

* [Degree-Constrained MST with Quantum Walks](../path-to-repo1)
* [Quantum Walk-Based MST Algorithm](../path-to-repo2)

These complementary approaches combine exhaustive search, classical Kruskal's algorithm, and quantum walk simulations based on graph Laplacians to tackle MST generation under varying constraints.

---

## Overview

This unified MST implementation provides:

* MST generation with and without degree constraints.
* Classical algorithms (Exhaustive Search, Kruskal) vs. Quantum-inspired MST generation.
* Quantum walk simulation for determining edge transition probabilities.
* Visual and numerical comparison of different MST construction methods.

---

## References to Integrated Modules

### 1. **Degree-Constrained MST with Quantum Walks**

This module targets the **Degree-Constrained MST** problem — a known NP-hard challenge. It:

* Uses **exhaustive search** for exact results.
* Applies **quantum walk** simulations with Hamiltonian dynamics to construct MSTs when `max_degree` constraints are present.
* Provides visual analysis and weight comparisons of classical vs. quantum results.

See full description: [DC-MST with Quantum Walks](../path-to-repo1)

### 2. **Quantum Walk-Based MST Algorithm**

This component addresses standard MST generation and compares:

* Quantum walk-based MST derived from transition probabilities.
* Classical Kruskal MST.
* Transition probability evolution over time.

See full description: [Quantum Walk-Based MST](../path-to-repo2)

---

## How to Use

### Prerequisites

```bash
pip install numpy networkx matplotlib qutip
```

### Execution

* To run with degree constraints, set `max_degree` in the DC-MST script.
* To compare Kruskal vs. quantum MSTs, run the standard quantum MST script.

### Output

* Visualized MSTs.
* Transition probability plots.
* Weight comparisons between classical and quantum approaches.

---

## Summary

This combined MST repository showcases how quantum walk dynamics can be adapted to classical graph optimization problems, especially under structural constraints like node degrees. Each method has strengths depending on graph size and constraint strictness. This integration allows side-by-side benchmarking for research or educational use.

---

## License

MIT License — See individual module READMEs for detailed attributions.
