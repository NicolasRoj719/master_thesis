# Sparse Jacobian Chaining

**Author:** Nicolas David Rojas Rojas
**Degree** Master of Science in Simulation Sciences
**Institution** RWTH University | STCE
**Advisors** Simon Maertens and Dr.rer.nat Uwe Naumann
**Date** August 2026

## Overview
A modern C++ 20 header-only template library designed to solve Dense Jacobian Chain Product 
Bracketing, as well as Matrix-Free Dense and Sparse Jacobian Chain Product Bracketing (with and 
without memory bound). Additionally, Binomial Checkpointing is implemented to extend the feasibility 
of adjoint mode under memory-constrained scenarios. All supported problems are solved using tailored 
dynamic programming formulations.

## Key Features
**Dynamic Programming Solvers:** Four distinct dynamic programming routines developed to compute 
optimal operation accumulation sequences, or determine optimal checkpoint placement and reuse during 
Algorithmic Differentiation (AD) split reversal process.

**Sparse Matrix Representation:** Defines an overload of * to multiply Sparse Matrix objects. 
Employs a greedy graph-coloring heuristic on column/row intersection graphs to identify structurally 
orthogonal column and row groups. Uses CSR and CSC formats for efficient sparsity pattern storage and 
propagation.

**Cost Estimation Models:** Provides computational cost estimates for tangent and adjoint AD modes, 
as well as matrix-matrix multiplications in terms of Fused Multiply-Add (FMA) operations. Since 
runtime execution cost for general subprograms is code-dependent, checkpointing costs can be 
customized by users with domain knowledge of the target code.

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
│   ├── jacobian.hpp                # Jacobian matrix types & metadata classes
│   │  
│   ├── optimal_accumulation.hpp    # Dynamic programming post-processing.
│   │   
│   ├── table.hpp                   # Dynamic programming lookup tables
│   │
│   ├── table_cell.hpp              # Dynamic programming table cell templates
│   │   
│   └── util_structs.hpp            # Metadata structures and helper utilities 
└── tests/                          # Unit testing suite
    ├── fixture/                    # Test fixtures
    └── unit/                       # Unit test source files
```

## Core API & Class Reference
* **Dynamic Programming:**

    fill_table<JacobianType, InformationType>: Dynamic programming solver for Dense Jacobian Chain
    Product Bracketing and Matrix-Free Dense and Sparse Jacobian Chain Product Bracketing.

    binomial_checkpointing<SplitType, Information_Type>: Dynamic programming solver for Binomial 
    Checkpointing.

* **Jacobians & Metadata**

    Jacobian_information: Base class containing domain and codomian space dimension information.

    Matrix_free_information: Extends Jacobian_information with computational graph edge counts for 
    matrix-free evaluations.

    Matrix_free_sparse_information: Extends Matrix_free_information with total non-zero (NNZ) entry 
    counts.

    Jacobian: Wrapper around Jacobian_information. 
    
    Dense_Jacobian: Wrapper around Matrix_free_information. 
    
    Sparse_Jacobian: Contains a Matrix_free_sparse_information object, stores CSC/CSR formats, and 
    computes structurally orthogonal row/column partitioning via greedy graph coloring.

    Split_dense_Jacobian: Extends Dense_Jacobian with subprogram execution cost estimate in fused 
    multiply-add operations.

    Split_sparse_Jacobian: Extends Sparse_Jacobian with subprogram execution cost estimate in fused 
    multiply-add operations.

* **Chain & Views**

    jacobian_chain<JacobianType, InformationType>: Main container wrapping an ordered sequence of
    JacobianType objects.

    View_chain: Lightweight, non-owning subrange view over a segment of a JacobianType chain.


* **Schedule & Generator**

    Node_jacobian, Node_matrix_free: Nodes representing subchain operations in the optimal accumulation
    sequence.

    Node_binomial_checkpointing: Node representing optimal subproblems during checkpoint placement
    and reuse in split-reversal AD.

    operation_sequence_accumulation: Reconstructs optimal accumulation operation sequence from lookup
    table by storing them in Node_jacobian or Node_matrix_free objects.

    subproblem_sequence_accumulation: Reconstructs optimal Checkpointing subproblems from lookup 
    tables storing them in Node_binomial_checkpointing objects.

    Generator<Information_Type>: Synthetic chain data generator for Jacobian_information, 
    Matrix_free_information, Matrix_free_sparse_information and Sparsity pattern coordinates.

    Generator_data: struct containing chain Matrix_free_sparse_information and corresponding sparsity 
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

Example Usages
Simple example usages are to be developed. However inside the unit test there are plenty of examples 
on how to use the different classes an methods. Feel free to navigate through the executables and 
header files in tests/unit/ .
