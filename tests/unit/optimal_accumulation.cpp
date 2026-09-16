#include <iostream>
#include <optional>
#include <vector>

#include "binomial_checkpointing.hpp"
#include "class_test_optimal_accumulation.hpp"
#include "jacobian.hpp"
#include "table.hpp"
#include "table_cell.hpp"
#include "optimal_accumulation.hpp"

int main(){

  test_optimal_accumulation test{};
  

  //Homogeneous tangent test.
  {
    const std::size_t chain_length = 2;
    Table<Dense_Jacobian> table{chain_length};
    // Constructor arguments: domain space dimension, codomain space dimension, number of edges.
    Dense_Jacobian jac_0{5, 15, 100};
    Dense_Jacobian jac_1{15, 24, 150};

    //Emplace back node with pointers.
    //Constructor arguments: jacobian_pointer, preaccumulation_cost, operation.
    table.emplace_back(&jac_0, jac_0.domain_dim() * jac_0.number_edges(), Operation::TANGENT);
    table.emplace_back(&jac_1, jac_1.domain_dim() * jac_1.number_edges(), Operation::TANGENT);
    //Emplace back node without pointer.
    //Constructor arguments: accumulation_cost, split_position, operation.
    table.emplace_back(jac_0.domain_dim() * (jac_0.number_edges() + jac_1.number_edges()),
                        0, Operation::TANGENT);

    //operation_sequence_accumulation arguments: reference to table, chain length, pointer to ostream. 
    std::vector<Node_matrix_free> accumulation_sequence = 
      operation_sequence_accumulation<Node_matrix_free>(table, chain_length, nullptr);

    //Validate Node_matrix_free information.
    if(test.check_node_information(accumulation_sequence[0], 1, 0, Operation::TANGENT, 1) &&
        test.check_node_information(accumulation_sequence[1], 0, std::nullopt, Operation::TANGENT, 2)){

      test.set_matrix_free_homogeneous_tangent_result(true);
    }
    else{
      test.set_matrix_free_homogeneous_tangent_result(false);
    }
  }
  //Homogeneous adjoint test.
  {
    const std::size_t chain_length = 2;
    Table<Dense_Jacobian> table{chain_length};
    // Constructor arguments: domain space dimension, codomain space dimension, number of edges.
    Dense_Jacobian jac_0{24, 15, 100};
    Dense_Jacobian jac_1{15, 4, 150};

    //Emplace back node with pointers.
    //Constructor arguments: jacobian_pointer, preaccumulation_cost, operation.
    table.emplace_back(&jac_0, jac_0.codomain_dim() * jac_0.number_edges(), Operation::ADJOINT);
    table.emplace_back(&jac_1, jac_1.codomain_dim() * jac_1.number_edges(), Operation::ADJOINT);

    //Emplace back node without pointer.
    //Constructor arguments: accumulation_cost, split_position, operation.
    table.emplace_back(jac_1.codomain_dim() * (jac_0.number_edges() + jac_1.number_edges()),
                        0, Operation::ADJOINT);

    //operation_sequence_accumulation arguments: reference to table, chain length, pointer to ostream. 
    std::vector<Node_matrix_free> accumulation_sequence = 
      operation_sequence_accumulation<Node_matrix_free>(table, chain_length, nullptr);

    //Validate Node_matrix_free information.
    if(test.check_node_information(accumulation_sequence[0], 1, 0, Operation::ADJOINT, 1) &&
        test.check_node_information(accumulation_sequence[1], 1, std::nullopt, Operation::ADJOINT, 2)){

      test.set_matrix_free_homogeneous_adjoint_result(true);
    }
    else{
      test.set_matrix_free_homogeneous_adjoint_result(false);
    }
  }
  
  // Multiplication test
  {
    const std::size_t chain_length = 2;
    Table<Dense_Jacobian> table {chain_length};

    // Constructor arguments: domain space dimension, codomain space dimension, number of edges.
    Dense_Jacobian jac_0{24, 15, 100};
    Dense_Jacobian jac_1{15, 24, 150};

    //Emplace back node with pointers.
    //Constructor arguments: jacobian_pointer, preaccumulation_cost, operation.
    table.emplace_back(&jac_0, jac_0.codomain_dim() * jac_0.number_edges(), Operation::ADJOINT);
    table.emplace_back(&jac_1, jac_1.domain_dim() * jac_1.number_edges(), Operation::TANGENT);

    //Emplace back node without pointer.
    //Constructor arguments: accumulation_cost, split_position, operation.
    //Warning: this operation does not correspond to the optimal accumulation method.
    table.emplace_back(jac_0.codomain_dim() * jac_0.number_edges() + 
                      jac_1.domain_dim() * jac_1.number_edges() + 
                      jac_1.codomain_dim()* jac_0.codomain_dim() * jac_0.domain_dim(),
                      0, Operation::MULTIPLICATION);

    //operation_sequence_accumulation arguments: reference to table, chain length, pointer to ostream. 
    std::vector<Node_matrix_free> accumulation_sequence = 
      operation_sequence_accumulation<Node_matrix_free>(table, chain_length, nullptr);

    //Validate Node_matrix_free information.
    if(test.check_node_information(accumulation_sequence[0], 1, 0, Operation::MULTIPLICATION, 1) &&
        test.check_node_information(accumulation_sequence[1], 1, std::nullopt, Operation::TANGENT, 2) &&
        test.check_node_information(accumulation_sequence[2], 0, std::nullopt, Operation::ADJOINT, 2)){

      test.set_matrix_free_multiplication_result(true);
    }
    else{
      test.set_matrix_free_multiplication_result(false);
    }
  }
  
  //Matrix-free test_case_0
  {
    const std::size_t chain_length = 3;
    Table<Dense_Jacobian> table {chain_length};

    // Constructor arguments: domain space dimension, codomain space dimension, number of edges.
    Dense_Jacobian jac_0{24, 15, 100};
    Dense_Jacobian jac_1{15, 24, 150};
    Dense_Jacobian jac_2{24, 20, 80};

    //Emplace back node with pointers.
    //Constructor arguments: jacobian_pointer, preaccumulation_cost, operation.
    table.emplace_back(&jac_0, jac_0.codomain_dim() * jac_0.number_edges(), Operation::ADJOINT);
    table.emplace_back(&jac_1, jac_1.domain_dim() * jac_1.number_edges(), Operation::TANGENT);
    table.emplace_back(&jac_2, jac_2.codomain_dim() * jac_2.number_edges(), Operation::ADJOINT);

    //Warning: the following data is not intended to match the actual optimal data. It is simply 
    //mock data to test operation_sequence_accumulation method.
    //Problem instance (1,0)
    table.emplace_back(150, 0, Operation::TANGENT);
    //Problem (2,1)
    table.emplace_back(300, 1, Operation::TANGENT);
    //Problem (2,0)
    table.emplace_back(315, 1, Operation::MULTIPLICATION);

    //operation_sequence_accumulation arguments: reference to table, chain length, pointer to ostream. 
    std::vector<Node_matrix_free> accumulation_sequence = 
      operation_sequence_accumulation<Node_matrix_free>(table, chain_length, nullptr);

    //Validate Node_matrix_free information.
    if(test.check_node_information(accumulation_sequence[0], 2, 0, Operation::MULTIPLICATION, 1) &&
        test.check_node_information(accumulation_sequence[1], 2, std::nullopt, Operation::ADJOINT, 2) &&
        test.check_node_information(accumulation_sequence[2], 1, 0, Operation::TANGENT, 2) &&
        test.check_node_information(accumulation_sequence[3], 0, std::nullopt, Operation::ADJOINT, 3)){

      test.set_matrix_free_test_case_0(true);
    }
    else{
      test.set_matrix_free_test_case_0(false);
    }
  }

  //Matrix-free test_case_1
  {
    const std::size_t chain_length = 4;
    Table<Dense_Jacobian> table {chain_length};

    // Constructor arguments: domain space dimension, codomain space dimension, number of edges.
    Dense_Jacobian jac_0{24, 15, 100};
    Dense_Jacobian jac_1{15, 24, 150};
    Dense_Jacobian jac_2{24, 20, 80};
    Dense_Jacobian jac_3{20, 18, 180};

    //Emplace back node with pointers.
    //Constructor arguments: jacobian_pointer, preaccumulation_cost, operation.
    table.emplace_back(&jac_0, jac_0.codomain_dim() * jac_0.number_edges(), Operation::ADJOINT);
    table.emplace_back(&jac_1, jac_1.domain_dim() * jac_1.number_edges(), Operation::TANGENT);
    table.emplace_back(&jac_2, jac_2.codomain_dim() * jac_2.number_edges(), Operation::ADJOINT);
    table.emplace_back(&jac_3, jac_2.codomain_dim() * jac_2.number_edges(), Operation::ADJOINT);

    //Warning: the following data is not intended to match the actual optimal data. It is simply 
    //mock data to test operation_sequence_accumulation method.
    //Problem instance (1,0)
    table.emplace_back(150, 0, Operation::MULTIPLICATION);
    //Problem (2,1)
    table.emplace_back(300, 1, Operation::TANGENT);
    //Problem (2,0)
    table.emplace_back(315, 1, Operation::ADJOINT);
    //Problem (3,2)
    table.emplace_back(400, 2, Operation::MULTIPLICATION);
    //Problem (3,1)
    table.emplace_back(500, 1, Operation::TANGENT);
    //Problem (3,0)
    table.emplace_back(600, 1, Operation::MULTIPLICATION);

    //operation_sequence_accumulation arguments: reference to table, chain length, pointer to ostream. 
    std::vector<Node_matrix_free> accumulation_sequence = 
      operation_sequence_accumulation<Node_matrix_free>(table, chain_length, nullptr);

    //Validate Node_matrix_free information.
    if(test.check_node_information(accumulation_sequence[0], 3, 0, Operation::MULTIPLICATION, 1) &&
        test.check_node_information(accumulation_sequence[1], 3, 2, Operation::MULTIPLICATION, 2) &&
        test.check_node_information(accumulation_sequence[2], 3, std::nullopt, Operation::ADJOINT, 3) &&
        test.check_node_information(accumulation_sequence[3], 2, std::nullopt, Operation::ADJOINT, 3) &&
        test.check_node_information(accumulation_sequence[4], 1, 0, Operation::MULTIPLICATION, 2) &&
        test.check_node_information(accumulation_sequence[5], 1, std::nullopt, Operation::TANGENT, 3) &&
        test.check_node_information(accumulation_sequence[6], 0, std::nullopt, Operation::ADJOINT, 3)){

      test.set_matrix_free_test_case_1(true);
    }
    else{
      test.set_matrix_free_test_case_1(false);
    }
  }

  //-----DENSE JACOBIAN CHAIN PRODUCT BRACKETING OPERATION ACCUMULATION TEST-----
  //Test case 0
  {
    const std::size_t chain_length = 3;
    Table<Jacobian> table{chain_length};

    //Constructor arguments: domain space dimension and codomain space dimension.
    Jacobian jac_0{5, 10};
    Jacobian jac_1{10, 8};
    Jacobian jac_2{8, 8};

    //Emplace back node with pointers.
    //Constructor arguments: reference to a cell.
    table.emplace_back(&jac_0);
    table.emplace_back(&jac_1);
    table.emplace_back(&jac_2);

    //Cell without pointers constructor arguments: accumulation cost, split position.
    //Problem instance (1,0)
    table.emplace_back(jac_1.codomain_dim() * jac_0.domain_dim() * jac_0.codomain_dim(), 0);
    //Problem instance (2,1)
    table.emplace_back(jac_2.codomain_dim() * jac_1.domain_dim() * jac_1.codomain_dim(), 1);
    //Warning to simplify the script the optimal cost value will not be calculated. A arbitrary 
    //number is used to initialize the optimal cost.
    //Problem instance (2,0)
    table.emplace_back(200, 1);

    //operation_sequence_accumulation arguments: reference to table, chain length and pointer to ostream.
    std::vector<Node_jacobian> accumulation_sequence =
      operation_sequence_accumulation<Node_jacobian>(table, chain_length, nullptr);

    //Validate Node_jacobian information.
    if(test.check_node_information(accumulation_sequence[0], 2, 0, 1) &&
        test.check_node_information(accumulation_sequence[1], 2, std::nullopt, 2) &&
        test.check_node_information(accumulation_sequence[2], 1, 0, 2)){
      
      test.set_jacobian_test_case_0(true);
    }
    else{test.set_jacobian_test_case_0(false);}
  }

  //Test case 1
  {
    const std::size_t chain_length = 3;
    Table<Jacobian> table{chain_length};

    //Constructor arguments: domain space dimension and codomain space dimension.
    Jacobian jac_0{8, 8};
    Jacobian jac_1{8, 10};
    Jacobian jac_2{10, 5};

    //Emplace back node with pointers.
    //Constructor arguments: reference to a cell.
    table.emplace_back(&jac_0);
    table.emplace_back(&jac_1);
    table.emplace_back(&jac_2);

    //Cell without pointers constructor arguments: accumulation cost, split position.
    //Problem instance (1,0)
    table.emplace_back(jac_1.codomain_dim() * jac_0.domain_dim() * jac_0.codomain_dim(), 0);
    //Problem instance (2,1)
    table.emplace_back(jac_2.codomain_dim() * jac_1.domain_dim() * jac_1.codomain_dim(), 1);
    //Warning to simplify the script the optimal cost value will not be calculated. A arbitrary 
    //number is used to initialize the optimal cost.
    //Problem instance (2,0)
    table.emplace_back(200, 0);

    //operation_sequence_accumulation arguments: reference to table, chain length and pointer to ostream.
    std::vector<Node_jacobian> accumulation_sequence =
      operation_sequence_accumulation<Node_jacobian>(table, chain_length, nullptr);

    //Validate Node_jacobian information.
    if(test.check_node_information(accumulation_sequence[0], 2, 0, 1) &&
        test.check_node_information(accumulation_sequence[1], 2, 1, 2) &&
        test.check_node_information(accumulation_sequence[2], 0, std::nullopt, 2)){

      test.set_jacobian_test_case_1(true);
    }
    else{test.set_jacobian_test_case_1(false);}
  }

  //Test case 2
  {
    const std::size_t chain_length = 4;
    Table<Jacobian> table{chain_length};

    //Constructor arguments: domain space dimension and codomain space dimension.
    Jacobian jac_0{8, 8};
    Jacobian jac_1{8, 10};
    Jacobian jac_2{10, 7};
    Jacobian jac_3{7, 9};

    //Cell without pointers constructor arguments: accumulation cost, split position.
    //Problem instance (1,0)
    table.emplace_back(jac_1.codomain_dim() * jac_0.domain_dim() * jac_0.codomain_dim(), 0);
    //Problem instance (2,1)
    table.emplace_back(jac_2.codomain_dim() * jac_1.domain_dim() * jac_1.codomain_dim(), 1);
    //Warning: from this point on, accumulated costs are set to mock values. 
    //Problem instance (2,0)
    table.emplace_back(250, 0);
    //Problem instance (3,2)
    table.emplace_back(300, 2);
    //Problem instance (3,1)
    table.emplace_back(700, 1);
    //Problem instance (3,0)
    table.emplace_back(450, 2);

    //operation_sequence_accumulation arguments: reference to table, chain length and pointer to ostream.
    std::vector<Node_jacobian> accumulation_sequence =
      operation_sequence_accumulation<Node_jacobian>(table, chain_length, nullptr);

    //Validate Node_jacobian information.
    if(test.check_node_information(accumulation_sequence[0], 3, 0, 1) &&
        test.check_node_information(accumulation_sequence[1], 3, std::nullopt, 2) &&
        test.check_node_information(accumulation_sequence[2], 2, 0, 2) &&
        test.check_node_information(accumulation_sequence[3], 2, 1, 3) &&
        test.check_node_information(accumulation_sequence[4], 0, std::nullopt, 3)){
      
      test.set_jacobian_test_case_2(true);
    }
    else{test.set_jacobian_test_case_2(false);}
  }

  //---------------BINOMIAL_CHECKPOINTING---------------
  //Test case 0
  {
    const std::size_t chain_length = 5;
    const std::size_t available_checkpoints = 2;
    binomial_table table{chain_length, available_checkpoints};

    std::vector<Matrix_free_information> problem_dense_data;
    problem_dense_data.reserve(chain_length);

    //Matrix_free_information constructor arguments: domain space dimension, codomian space dimension,
    // number of edges.
    problem_dense_data.emplace_back(10, 15, 150);
    problem_dense_data.emplace_back(15, 13, 200);
    problem_dense_data.emplace_back(13, 20, 180);
    problem_dense_data.emplace_back(20, 23, 280);
    problem_dense_data.emplace_back(23, 18, 250);

    //Subprograms execution cost estimate
    std::vector<std::size_t> execution_costs;
    execution_costs.reserve(chain_length);
    execution_costs.emplace_back(100);
    execution_costs.emplace_back(200);
    execution_costs.emplace_back(300);
    execution_costs.emplace_back(400);
    execution_costs.emplace_back(500);

    //Build Split dense.
    jacobian_chain<Dense_Jacobian, Matrix_free_information> chain{problem_dense_data};

    //Call dynamic programming algorithm to fill the look up table.
    binomial_checkpointing 
      algorithm{execution_costs, chain_length - 1, 0, available_checkpoints, 0};

    //Accumulate operation nodes.
    //operation_sequence_accumulation arguments: reference to lookup table, chain length, number of
    //available checkpoints and address of a ostream object (e.g. std::cout).
    std::vector<Node_binomial_checkpointing> accumulation_sequence =
      subproblem_sequence_accumulation(algorithm.get_table(), chain_length,
                                        available_checkpoints, nullptr);

    //Validate Node_binomial_checkpointing information (j, i, c) and level.
    if(test.check_node_information(accumulation_sequence[0], 4, 0, 2, 1) &&
        test.check_node_information(accumulation_sequence[1], 4, 2, 1, 2) &&
        test.check_node_information(accumulation_sequence[2], 4, 3, 0, 3) &&
        test.check_node_information(accumulation_sequence[3], 2, 2, 1, 3) &&
        test.check_node_information(accumulation_sequence[4], 1, 0, 2, 2)){

      test.set_binomial_case_0(true);
    }
    else{test.set_binomial_case_0(false);}
  }

  //Test case 1
  {
    const std::size_t chain_length = 5;
    const std::size_t available_checkpoints = 2;
    binomial_table table{chain_length, available_checkpoints};

    std::vector<Matrix_free_information> problem_dense_data;
    problem_dense_data.reserve(chain_length);

    //Matrix_free_information constructor arguments: domain space dimension, codomian space dimension,
    // number of edges.
    problem_dense_data.emplace_back(10, 15, 150);
    problem_dense_data.emplace_back(15, 13, 200);
    problem_dense_data.emplace_back(13, 20, 180);
    problem_dense_data.emplace_back(20, 23, 280);
    problem_dense_data.emplace_back(23, 18, 250);

    //Subprograms execution cost estimate
    std::vector<std::size_t> execution_costs;
    execution_costs.reserve(chain_length);
    execution_costs.emplace_back(500);
    execution_costs.emplace_back(400);
    execution_costs.emplace_back(300);
    execution_costs.emplace_back(200);
    execution_costs.emplace_back(100);

    //Build Split dense.
    jacobian_chain<Dense_Jacobian, Matrix_free_information> chain{problem_dense_data};

    //Call dynamic programming algorithm to fill the look up table.
    binomial_checkpointing 
      algorithm{execution_costs, chain_length - 1, 0, available_checkpoints, 0};

    //Accumulate operation nodes.
    //operation_sequence_accumulation arguments: reference to lookup table, chain length, number of
    //available checkpoints and address of a ostream object (e.g. std::cout).
    std::vector<Node_binomial_checkpointing> accumulation_sequence =
      subproblem_sequence_accumulation(algorithm.get_table(), chain_length,
                                        available_checkpoints, nullptr);

    //Validate Node_binomial_checkpointing information (j, i, c) and level.
    if(test.check_node_information(accumulation_sequence[0], 4, 0, 2, 1) &&
        test.check_node_information(accumulation_sequence[1], 4, 1, 1, 2) &&
        test.check_node_information(accumulation_sequence[2], 4, 2, 0, 3) &&
        test.check_node_information(accumulation_sequence[3], 1, 1, 1, 3) &&
        test.check_node_information(accumulation_sequence[4], 0, 0, 2, 2)){

      test.set_binomial_case_1(true);
    }
    else{test.set_binomial_case_1(false);}
  }

  //Test case 2
  {
    const std::size_t chain_length = 6;
    const std::size_t available_checkpoints = 2;
    binomial_table table{chain_length, available_checkpoints};

    std::vector<Matrix_free_information> problem_dense_data;
    problem_dense_data.reserve(chain_length);

    //Matrix_free_information constructor arguments: domain space dimension, codomian space dimension,
    // number of edges.
    problem_dense_data.emplace_back(10, 15, 150);
    problem_dense_data.emplace_back(15, 13, 200);
    problem_dense_data.emplace_back(13, 20, 180);
    problem_dense_data.emplace_back(20, 23, 280);
    problem_dense_data.emplace_back(23, 18, 250);
    problem_dense_data.emplace_back(18, 18, 350);

    //Subprograms execution cost estimate
    std::vector<std::size_t> execution_costs;
    execution_costs.reserve(chain_length);
    execution_costs.emplace_back(100);
    execution_costs.emplace_back(200);
    execution_costs.emplace_back(300);
    execution_costs.emplace_back(400);
    execution_costs.emplace_back(500);
    execution_costs.emplace_back(600);

    //Build Split dense.
    jacobian_chain<Dense_Jacobian, Matrix_free_information> chain{problem_dense_data};

    //Call dynamic programming algorithm to fill the look up table.
    binomial_checkpointing
      algorithm{execution_costs, chain_length - 1, 0, available_checkpoints, 0};

    //Accumulate operation nodes.
    //operation_sequence_accumulation arguments: reference to lookup table, chain length, number of
    //available checkpoints and address of a ostream object (e.g. std::cout).
    std::vector<Node_binomial_checkpointing> accumulation_sequence =
      subproblem_sequence_accumulation(algorithm.get_table(), chain_length,
                                        available_checkpoints, nullptr);
    

    //Validate Node_binomial_checkpointing information (j, i, c) and level.
    if(test.check_node_information(accumulation_sequence[0], 5, 0, 2, 1) &&
        test.check_node_information(accumulation_sequence[1], 5, 3, 1, 2) &&
        test.check_node_information(accumulation_sequence[2], 5, 4, 0, 3) &&
        test.check_node_information(accumulation_sequence[3], 3, 3, 1, 3) &&
        test.check_node_information(accumulation_sequence[4], 2, 0, 2, 2) &&
        test.check_node_information(accumulation_sequence[5], 2, 1, 1, 3) &&
        test.check_node_information(accumulation_sequence[6], 0, 0, 2, 3)){

      test.set_binomial_case_2(true);
    }
    else{test.set_binomial_case_2(false);}
  }
  //Checks the state of all tests carried out. If one or more fail then the test state is going 
  //to be printed to the terminal automatically.
  test.were_all_tests_successful();
  //Print test state:
  /* test.print_test_state(); */

  return 0;
}
