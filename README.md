Description:
    This first working version of the code implements dynamic programming solvers to the
    Dense Jacobian Chain Product Bracketing problem (DJCPB), Matrix-Free Dense Jacobian Chain
    Product Bracketing (MFDJCPB) and its memory constrained sibling.

Build instructions:
    This first version constains an executable inside tests directory called solver. This
    executable runs the solver for all the three problems DJCPB, MFDJCPB and memory constrained
    MFDJCPB. No building process is requiered.

Run instructions:
    To run the executable solver. Provide the following information in the given order.
    Jacobian chain length, lower and upper bound for the input ad output dimension for the
    Jacobians that will be generated, lower and upper bound for the number of edges in the 
    DAG representation of the differentiable subpogram and 
    memory limit (given in terms of number of edges in the DAG).

    Example:
        Inside tests directory run the following command.
        ./solver.cpp 4 1 50 10 120 300
        Chain length: 4.
        Input and output dimension within the range [1 , 50].
        Number of edges within the range [10 , 120].
        Memory limit: 300.

        Similarly outside tests directory simply run.
        ./tests/solver.cpp 4 1 50 10 120 300

Dependencies:
    For this first version of the code no dependencies are requiered to installed. Only
    libraries of the standard library are used. To compile the code one requieres c++17.

Author:
Nicolas David Rojas Rojas.
Mail: nicolas.rojas@rwth-aachen.de

