Description:
    This first working version of the code implements dynamic programming solvers for: 
    Dense Jacobian Chain Product Bracketing problem (DJCPB) 
    Matrix-Free Dense Jacobian Chain Product Bracketing (MFDJCPB) 
    and its memory constrained variant. 
    
    The implementation is written in C++17 and was compiled and tested on Ubuntu 24.04 using
    the GNU Compiler Collection (g++ 13.3.0).

    Given the user-specified bounds (see Usage below), the program generates a random problem instance.
    Dimensions and edge counts are sampled uniformly within the provided ranges. The generated instance
    is then solved using all three solvers.

Build instructions:
    This project requires a compiler with C++17 support. There are no external dependencies beyond
    the standard library.

    To compile the project run the following command from the root directory
    g++ -std=c++17 -Wall -Wextra tests/solver.cpp -o ./tests/solver

Usage:
    The executable requires specific positional arguments to define the problem space.
    
    ./solver [length] [min_dim] [max_dim] [min_edges] [max_edges] [memory_limit]

    1. Chain length: Number of Jacobians in the chain.
    2. Dimension Range: Lower and upper bounds for the Jacobian input/output dimensions.
    3. Edge Range: Lower and upper bounds for the number of edges in the DAG representation.
    4. Memory Limit: upper bound on the number of DAG edges that may be stored during
    computation.

    Example:
        Inside tests directory run the following command.
        ./solver 4 1 50 10 120 300
        Chain length: 4.
        Input and output dimension within the range [1 , 50].
        Number of edges within the range [10 , 120].
        Memory limit: 300.

        Similarly outside tests directory simply run.
        ./tests/solver 4 1 50 10 120 300

Author:
Nicolas David Rojas Rojas.
Mail: nicolas.rojas@rwth-aachen.de
