#include <cstdint>
#include <iostream>
#include <vector>
#include "generator.hpp"
#include "chain.hpp"
#include "fill_table.hpp"
#include "optimal_accumulation.hpp"
#include "util_structs.hpp"

int main(){
  
  /* { */
  /*   //MFDJCPB */
  /*   //Problem data: */
  /*   std::vector<std::size_t> dimension{27, 24, 35, 15, 30, 35}; */
  /*   std::vector<std::size_t> number_edges{100, 150, 175, 190,200}; */

  /*   //Create Matrix-free information vector */
  /*   std::vector<Matrix_free_information> problem_data; */

  /*   for(std::size_t idx = 0; idx < number_edges.size(); idx++){ */

  /*     problem_data.emplace_back(dimension[idx], dimension[idx + 1], number_edges[idx]); */
  /*   } */

  /*   //Creating Jacobian Chain for the problem data */
  /*   Jacobian_chain<Dense_Jacobian, Matrix_free_information> chain{problem_data}; */

  /*   //Execute dynamic programming algoritm (Matrix Free Dense Jacobian Chain Product Bracketing) */
  /*   Fill_table<Dense_Jacobian, Matrix_free_information> solver{chain}; */

  /*   std::size_t optimal_cost = solver.get_optimal_cost(); */

  /*   std::cout << "Optimal accumulation cost [fma]: " << optimal_cost << '\n'; */

  /*   std::cout << "Open a web browser and search for Graphviz online viewer. \n"; */

  /*   std::cout << "Copy the following lines to generate the optimal accumulation sequence "; */
  /*   std::cout << "visualization:\n"; */

  /*   //Print DOT to terminal */
  /*   std::vector<Node_matrix_free> optimal_accumulation_sequence = */
  /*     operation_sequence_accumulation<Node_matrix_free>(solver.get_table(), chain.size(), */
  /*                                                         &std::cout); */
  /* } */
  /* std::cout << '\n'; */
  /* { */
  /*   //Binomial Checkpointing */
  /*   //Problem defininition */
  /*   std::vector<std::size_t> execution_costs{100,120,145, 150,122}; */
  /*   std::size_t available_checkpoints = 2; */

  /*   Binomial_checkpointing binomial_solver{execution_costs, available_checkpoints}; */
  /*   std::size_t additional_cost = binomial_solver.get_additional_cost(); */

  /*   std::cout << "Optimal additional cost [fma]: " << additional_cost << '\n'; */

  /*   std::cout << "Open a web browser and search for Graphviz online viewer. \n"; */

  /*   std::cout << "Copy the following lines to generate the optimal subproblem decomposition "; */
  /*   std::cout << "tree visualization:\n"; */

  /*   //Print DOT to terminal */
  /*   std::vector<Node_binomial_checkpointing> accumulation_sequence = */
  /*     subproblem_sequence_accumulation(binomial_solver.get_table(), execution_costs.size(), */
  /*                                       available_checkpoints, &std::cout); */
  /* } */
  /* std::cout << '\n'; */
  {
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

  return 0;
}
