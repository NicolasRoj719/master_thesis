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
    
    ./solver [length] [min_dim] [max_dim] [min_edges] [max_edges] [is_determistic] [memory_limit] [seed]

    1. Chain length: Number of Jacobians in the chain.
    2. Dimension Range: Lower and upper bounds for the Jacobian input/output dimensions.
    3. Edge Range: Lower and upper bounds for the number of edges in the DAG representation.
    4. is_deterministic bool value, either 1 or 0. If is_determistic is given the value of 0
    a random seed is given to the generator. Otherwise the seed given as command line argument 
    is given to generator. This guarantees reproducibility of the results.
    4. Memory Limit: upper bound on the number of DAG edges that may be stored during
    computation.

    Example assuming solver.cpp has solver as executable inside tests directory:
        Inside tests directory run the following command.
        ./solver 4 1 50 10 120 1 300 150
        Chain length: 4.
        Input and output dimension within the range [1 , 50].
        Number of edges within the range [10 , 120].
        The generator is seeded with the user provided seed value 150.
        Memory limit: 300.


        Inside tests directory one may also call the solver with the following arguments.
        ./solver 4 1 50 10 120 0 300
        Chain length: 4.
        Input and output dimension within the range [1 , 50].
        Number of edges within the range [10 , 120].
        The generator is seeded with a random value no user provided seed is required.
        Memory limit: 300.

Author:
Nicolas David Rojas Rojas.
Mail: nicolas.rojas@rwth-aachen.de
