#include <iostream>
#include <vector>
#include <type_traits>
#include "./../chain.hpp"
#include "./class_test_table_cell.hpp"
#include "./../jacobian.hpp"
#include "./../table_cell.hpp"

int main(){
  test_table_cell test;

  //Testing Jacobian cell
  {
    
    std::size_t cost_0 = 25, cost_1 = 30, cost_2 = 45;
    std::size_t split_pos_0 = 0, split_pos_1 = 1, split_pos_2 = 2;

    //Initialize cell objects
    cell<Jacobian> cell_0{cost_0, split_pos_0};
    cell<Jacobian> cell_1{cost_1, split_pos_1};
    cell<Jacobian> cell_2{cost_2, split_pos_2};

    bool result_test_0 = test.check_cell_initialization(cell_0, cost_0, split_pos_0);
    bool result_test_1 = test.check_cell_initialization(cell_1, cost_1, split_pos_1);
    bool result_test_2 = test.check_cell_initialization(cell_2, cost_2, split_pos_2);

    if(result_test_0 && result_test_1 && result_test_2){

      test.set_test_jacobian_cell(true);
    }

    else{
      
      test.set_test_jacobian_cell(false);
    }
  }

  //Testing Dense_Jacobian cell
  {

    std::size_t cost_0 = 25, cost_1 = 30, cost_2 = 45;

    std::size_t split_pos_0 = 0, split_pos_1 = 1, split_pos_2 = 2;

    Operation operation_0 = Operation::TANGENT, operation_1 = Operation::ADJOINT;
    Operation operation_2 = Operation::MULTIPLICATION;

    cell<Dense_Jacobian> cell_0{cost_0, split_pos_0, operation_0};
    cell<Dense_Jacobian> cell_1{cost_1, split_pos_1, operation_1};
    cell<Dense_Jacobian> cell_2{cost_2, split_pos_2, operation_2};

    bool result_test_0 = test.check_cell_initialization(cell_0,
                        cost_0, split_pos_0, operation_0);

    bool result_test_1 = test.check_cell_initialization(cell_1,
                        cost_1, split_pos_1, operation_1);

    bool result_test_2 = test.check_cell_initialization(cell_2,
                        cost_2, split_pos_2, operation_2);

    if(result_test_0 && result_test_1 && result_test_2){

      test.set_test_dense_cell(true);
    }

    else{
      
      test.set_test_dense_cell(false);
    }
  }

  //Testing Sparse Jacobian cell
  {
    //Initialize a sparse jacobian chain
    const std::string path_to_file= 
      "./chain_test_cases/case_1_sparse";

    jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>
        chain{path_to_file};

    std::size_t cost_0 = 85, cost_1 = 31, cost_2 = 67;

    std::size_t split_pos_0 = 2, split_pos_1 = 1, split_pos_2 = 0;

    Operation operation_0 = Operation::ADJOINT, operation_1 = Operation::TANGENT;
    Operation operation_2 = Operation::MULTIPLICATION;

    /* Sparse_Jacobian sparse_0 = chain.copy(0); */
    /* Sparse_Jacobian sparse_1 = chain.copy(1); */
    /* Sparse_Jacobian sparse_2 = chain.copy(2); */

    cell<Sparse_Jacobian> cell_0{chain.copy(0), cost_0, split_pos_0, operation_0};

    cell<Sparse_Jacobian> cell_1{chain.copy(1), cost_1, split_pos_1, operation_1};

    cell<Sparse_Jacobian> cell_2{chain.copy(2), cost_2, split_pos_2, operation_2};

    bool result_test_0 = test.check_cell_initialization(cell_0, chain[0],
                          cost_0, split_pos_0, operation_0);

    bool result_test_1 = test.check_cell_initialization(cell_1, chain[1],
                          cost_1, split_pos_1, operation_1);

    bool result_test_2 = test.check_cell_initialization(cell_2, chain[2],
                          cost_2, split_pos_2, operation_2);

    if(result_test_0 && result_test_1 && result_test_2){

      test.set_test_sparse_cell(true);
    }

    else{

      test.set_test_sparse_cell(false);
    }
  }

  //Testing cell_with_pointer Jacobian
  {
    //Initializing a Jacobian chain
    std::string path_to_file =
      "./chain_test_cases/case_1_jacobian_information";

    jacobian_chain<Jacobian, Jacobian_information> chain{path_to_file};

    cell_with_pointer<Jacobian> cell_0{&chain[0]};

    cell_with_pointer<Jacobian> cell_1{&chain[1]};

    cell_with_pointer<Jacobian> cell_2{&chain[2]};

    bool result_test_0 = test.check_cell_initialization(cell_0, chain[0]);

    bool result_test_1 = test.check_cell_initialization(cell_1, chain[1]);

    bool result_test_2 = test.check_cell_initialization(cell_2, chain[2]);

    if(result_test_0 && result_test_1 && result_test_2){

      test.set_test_jacobian_cell_with_pointer(true);
    }

    else{

      test.set_test_jacobian_cell_with_pointer(false);
    }
  }

  //Testing cell_with_pointer Dense_Jacobian 
  {
    //Initializing a Jacobian chain
    std::string path_to_file =
      "./chain_test_cases/case_1_matrix_free";

    jacobian_chain<Dense_Jacobian, Matrix_free_information> chain{path_to_file};

    std::size_t cost_0 = 66, cost_1 = 11, cost_2 = 33;

    Operation operation_0 = Operation::MULTIPLICATION, operation_1 = Operation::TANGENT;
    Operation operation_2 = Operation::MULTIPLICATION;

    cell_with_pointer<Dense_Jacobian> cell_0{&chain[0], cost_0, operation_0};

    cell_with_pointer<Dense_Jacobian> cell_1{&chain[1], cost_1, operation_1};

    cell_with_pointer<Dense_Jacobian> cell_2{&chain[2], cost_2, operation_2};

    bool result_test_0 = test.check_cell_initialization(cell_0, chain[0], cost_0, operation_0);

    bool result_test_1 = test.check_cell_initialization(cell_1, chain[1], cost_1, operation_1);

    bool result_test_2 = test.check_cell_initialization(cell_2, chain[2], cost_2, operation_2);

    if(result_test_0 && result_test_1 && result_test_2){

      test.set_test_dense_cell_with_pointer(true);
    }

    else{

      test.set_test_dense_cell_with_pointer(false);
    }
  }

  //Testing cell with pointer Sparse Jacobian 
  {
    //Initialize a sparse jacobian chain
    const std::string path_to_file= 
      "./chain_test_cases/case_1_sparse";

    jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>
        chain{path_to_file};

    std::size_t cost_0 = 115, cost_1 = 221, cost_2 = 337;

    Operation operation_0 = Operation::ADJOINT, operation_1 = Operation::TANGENT;
    Operation operation_2 = Operation::MULTIPLICATION;

    cell_with_pointer<Sparse_Jacobian> cell_0{&chain[0], cost_0, operation_0};

    cell_with_pointer<Sparse_Jacobian> cell_1{&chain[1], cost_1, operation_1};

    cell_with_pointer<Sparse_Jacobian> cell_2{&chain[2], cost_2, operation_2};

    bool result_test_0 = test.check_cell_initialization(cell_0, chain[0],
                          cost_0, operation_0);

    bool result_test_1 = test.check_cell_initialization(cell_1, chain[1],
                          cost_1, operation_1);

    bool result_test_2 = test.check_cell_initialization(cell_2, chain[2],
                          cost_2, operation_2);

    if(result_test_0 && result_test_1 && result_test_2){

      test.set_test_sparse_cell_with_pointer(true);
    }

    else{

      test.set_test_sparse_cell_with_pointer(false);
    }
  }

  test.set_were_all_tests_successful();
  test.check_final_state_of_the_tests();
  test.print_test_state();

}
