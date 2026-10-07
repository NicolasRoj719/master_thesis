# Sparse Jacobian Chaining

**Author:** Nicolas David Rojas Rojas

**Degree** Master of Science in Simulation Sciences

**Institution** RWTH University | STCE

**Advisors** Simon Maertens and Dr.rer.nat Uwe Naumann

**Date** August, 2026

## Overview
A modern C++ 20 header-only template library designed to solve Dense Jacobian Chain Product 
Bracketing, as well as Matrix-Free Dense and Sparse Jacobian Chain Product Bracketing (with and 
without memory bound). Additionally, Binomial Checkpointing is implemented to optimally implement
adjoint mode on subchains under memory constrained scenarios. All supported problems are solved 
using tailored dynamic programming formulations.

## Key Features
**Dynamic Programming Solvers:** Three distinct dynamic programming routines developed to compute 
optimal operation count Jacobian matrix accumulation sequences. A fourth routine is provided to make 
efficient use of checkpoints during the accumulation of subchains via adjoint mode of AD. 

**Sparse Matrix Representation:** Defines an overload of * to multiply Sparse Matrix objects. 
Employs a greedy graph-coloring heuristic on column/row intersection graphs to identify structurally 
orthogonal column and row groups. Uses CSR and CSC formats for efficient sparsity pattern storage and 
propagation.

**Cost Estimation Models:** Provides computational cost estimates for tangent and adjoint AD modes, 
as well as matrix-matrix multiplications in terms of Fused Multiply-Add (fma) operations.
Underlaying subprograms execution cost in terms of fmas are to be set by the user, who has better 
knowledge of the target program and subprograms. A simple cost model ready to use, would be to 
assume that the execution cost of a subprogram is equal to the number of edges in its computational 
graph.

**Synthetic Data Generator:** Generates benchmark datasets and matrix metadata to test and evaluate 
dynamic algorithm performance.

**Modern C++20 Header-Only Design:** Clean implementation with zero external runtime dependencies.

## Directory structure
```text
.
├── CMakeLists.txt                  # Build configuration
├── Doxyfile.in                     # Template configuration for Doxygen
├── include/                        # Core header-only library sources
│   ├── binomial_checkpointing.hpp  # Checkpointing algorithms and cell structures
│   │   
│   ├── chain.hpp                   # Jacobian chain constructors. 
│   │
│   ├── fill_table.hpp              # DP table-filling algorithm solvers
│   │
│   ├── generator.hpp               # Synthetic metadata & matrix generators
│   │  
│   ├── jacobian.hpp                # Jacobian matrix classes
│   │  
│   ├── optimal_accumulation.hpp    # Dynamic programming post-processing
│   │   
│   ├── table.hpp                   # Dynamic programming lookup tables
│   │
│   ├── table_cell.hpp              # Dynamic programming table cell templates
│   │   
│   └── util_structs.hpp            # Metadata structures and helper utilities 
│   │   
│   └── external/# Metadata structures and helper utilities 
│       │   
│       └── json.hpp                # json header file to interface C++ and python (benchmarks). 
│
└── tests/                          # Unit testing suite
│   │   
│   └── fixture/                    # Test fixtures 
│   │   
│   └── unit/                       # Unit test source files 
│
└── benchmark/                      # Source file to execute benchmarks 
│
└── uml_diagrams/                   # Classes UML diagrams classified by header file
│
└── results/                        # Benchmarks results 

```

## Core API & Class Reference
* **Dynamic Programming:**

    Jacobian_chain<JacobianType, InformationType>: Class incharge of initializing the Jacobian chain 
    instance to be optimally accumulated with the dynamic programming solvers provided.

    Fill_table<JacobianType, InformationType>: Dynamic programming solver for Dense Jacobian Chain
    Product Bracketing and Matrix-Free Dense and Sparse Jacobian Chain Product Bracketing.

    Binomial_checkpointing: Dynamic programming solver for Binomial Checkpointing.

* **Jacobians & Metadata**

    Jacobian_information: Base class containing domain and codomian space dimension information.

    Matrix_free_information: Extends Jacobian_information with computational graph edge counts.

    Matrix_free_sparse_information: Extends Matrix_free_information with total non-zero (NNZ) entry 
    counts.

    Jacobian: Wrapper around Jacobian_information. 
    
    Dense_Jacobian: Wrapper around Matrix_free_information. 
    
    Sparse_Jacobian: Contains a Matrix_free_sparse_information object, stores CSC/CSR formats, and 
    computes structurally orthogonal row/column partitioning via greedy graph coloring. Overloads *
    to multiply Sparse_Jacobian objects.

* **Schedule & Generator**

    Node_jacobian, Node_matrix_free: Nodes representing subchain operations in the optimal Jacobian 
    matrix accumulation sequence.

    Node_binomial_checkpointing: Nodes representing optimal subproblems during checkpoint placement
    and reuse in split-reversal AD.

    operation_sequence_accumulation: Reconstructs optimal accumulation operation sequence from lookup
    table following a top-down left to right logic while storing optimal operations
    (Node_jacobian or Node_matrix_free objects) in an array which is returned by the function.

    subproblem_sequence_accumulation: Reconstructs optimal Checkpointing subproblems from lookup 
    table following a top-down left to right logic while storing optimal subproblems 
    (Node_binomial_checkpointing objects) in an array which is returned by the function.

    Generator<Information_Type>: Synthetic chain data generator for Jacobian_information, 
    Matrix_free_information, Matrix_free_sparse_information and Sparsity sturcture coordinates.

    Generator_data: Struct containing chain Matrix_free_sparse_information and corresponding sparsity 
    patterns.

## Requirements:

    Compiler: C++20 compliant compiler (GCC >= 10, Clang >= 11, or MSVC 2019+).

    Build System: CMake >= 3.16.

    Optional: Doxygen (for generating documentation).

## Building and Testing:

1. Configure and Build

Because this is a header-only library, linking against the target master_thesis_lib configures 
include paths automatically.

```bash
# Clone the repository
git clone https://github.com/NicolasRoj719/master_thesis 

cd master_thesis 

# Create build directory and configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build all test executables
cmake --build build
```

2. Run Tests

Unit test sources in tests/unit/ are automatically discovered and build as standalone test 
executables via CTest.
```bash
cd build
ctest --output-on-failure
```

Generating API Documentation
If Doxygen is installed on your system, CMake will automatically configure a custom doc target.
```bash
# Build the documentation target
cmake --build build --target doc
```

The generated HTML documentation will be placed in your build directory as specified by Doxyfile.

Usage Examples
```cpp
#include <cstdint>
#include <iostream>
#include <vector>
#include "chain.hpp"
#include "fill_table.hpp"
#include "optimal_accumulation.hpp"
#include "util_structs.hpp"

int main(){

    //MFDJCPB
    //Problem data:
    std::vector<std::size_t> dimension{27, 24, 35, 15, 30, 35};
    std::vector<std::size_t> number_edges{100, 150, 175, 190,200};

    //Create Matrix-free information vector
    std::vector<Matrix_free_information> problem_data;

    for(std::size_t idx = 0; idx < number_edges.size(); idx++){

        problem_data.emplace_back(dimension[idx], dimension[idx + 1], number_edges[idx]);
    }

    //Creating Jacobian Chain for the problem data
    Jacobian_chain<Dense_Jacobian, Matrix_free_information> chain{problem_data};

    //Execute dynamic programming algoritm (Matrix Free Dense Jacobian Chain Product Bracketing)
    Fill_table<Dense_Jacobian, Matrix_free_information> solver{chain};

    std::size_t optimal_cost = solver.get_optimal_cost();

    std::cout << "Optimal accumulation cost [fma]: " << optimal_cost << '\n';

    std::cout << "Open a web browser and search for Graphviz online viewer. \n";

    std::cout << "Copy the following lines to generate the optimal accumulation sequence ";
    std::cout << "visualization:\n";

    //Print DOT to terminal
    std::vector<Node_matrix_free> optimal_accumulation_sequence =
      operation_sequence_accumulation<Node_matrix_free>(solver.get_table(), chain.size(),
                                                          &std::cout);
}
```

```cpp
#include <cstdint>
#include <iostream>
#include <vector>
#include "fill_table.hpp"
#include "optimal_accumulation.hpp"

int main(){

    //Binomial Checkpointing
    //Problem defininition
    std::vector<std::size_t> execution_costs{100,120,145, 150,122};
    std::size_t available_checkpoints = 2;

    Binomial_checkpointing binomial_solver{execution_costs, available_checkpoints};
    std::size_t additional_cost = binomial_solver.get_additional_cost();

    std::cout << "Optimal additional cost [fma]: " << additional_cost << '\n';

    std::cout << "Open a web browser and search for Graphviz online viewer. \n";

    std::cout << "Copy the following lines to generate the optimal subproblem decomposition ";
    std::cout << "tree visualization:\n";

    //Print DOT to terminal
    std::vector<Node_binomial_checkpointing> accumulation_sequence =
      subproblem_sequence_accumulation(binomial_solver.get_table(), execution_costs.size(),
                                        available_checkpoints, &std::cout);

}
```

```cpp
#include <cstdint>
#include <iostream>
#include <vector>
#include "generator.hpp"
#include "chain.hpp"
#include "fill_table.hpp"
#include "optimal_accumulation.hpp"
#include "util_structs.hpp"
int main(){

    //Synthetic Data Generation
    std::size_t chain_length = 5;
    std::size_t dimension_lb = 10, dimension_ub = 25;
    std::size_t number_edges_lb = 150, number_edges_ub = 300;
    double density_lb = 0.05, density_ub = 0.1;

    //Optional if a deterministic generator is required.
    std::size_t seed = 68;
    bool is_deterministic = true;


    //Generator instance to create synthetic data.
    Generator<Matrix_free_sparse_information> gen{chain_length, dimension_lb, dimension_ub,
                                            number_edges_lb, number_edges_ub,
                                            density_lb, density_ub, is_deterministic, seed};

    //Creating Jacobian Chain 
    Jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information> chain{gen.generate_data()};

    //Construct sparse solver and fill look up table.
    Fill_table<Sparse_Jacobian, Matrix_free_sparse_information> solver{chain};

    std::size_t optimal_cost = solver.get_optimal_cost();

    std::cout << "Optimal accumulation cost [fma]: " << optimal_cost << '\n';

    std::cout << "Open a web browser and search for Graphviz online viewer. \n";

    std::cout << "Copy the following lines to generate the optimal accumulation sequence ";
    std::cout << "visualization:\n";

    //Print DOT to terminal
    std::vector<Node_matrix_free> optimal_accumulation_sequence = 
      operation_sequence_accumulation<Node_matrix_free>(solver.get_table(), chain.size(),
                                                          &std::cout);
}
```


