#include <string>

#include "class_test_table.hpp"
#include "class_test_table_cell.hpp"
#include "chain.hpp"
#include "table_cell.hpp"
#include "table.hpp"
#include "jacobian.hpp"


int main(){
  //test_cell object provides all the necessary methods to check if
  //a cell is correctly initialized.
  test_table_cell test_cell; 

  test_table test;
  
  //Test Jacobian table
  {
    //Jacobian chain initialization
    const std::string path_to_file = 
      std::string(FIXTURE_DIR) + "/chain_test_cases/case_1_jacobian_information";

    jacobian_chain<Jacobian, Jacobian_information> chain{path_to_file};

    //Reserving memory for a Table with 3 pointer cells
    //and 3 cells
    Table<Jacobian> table{static_cast<std::size_t>(3)};

    //The cost data does not correspond to the real cost data.
    //This data will be only used to check that emplace_back and
    //data access interfaces work correctly.
    std::size_t cost_1_0 = 40, cost_2_1 = 60, cost_2_0 = 56;

    std::size_t split_pos_1_0 = 0, split_pos_2_1 = 1, split_pos_2_0 = 1;

    //Emplacing back cell_0, cell_1, cell_2
    table.emplace_back(&chain[0]); table.emplace_back(&chain[1]);
    table.emplace_back(&chain[2]);

    //Emplacing back cell_1_0, cell_2_1, cell_2_0
    table.emplace_back(cost_1_0, split_pos_1_0);
    table.emplace_back(cost_2_1, split_pos_2_1);
    table.emplace_back(cost_2_0, split_pos_2_0);

    //Checking cell initialization of cells with pointer.
    if(
        test_cell.check_cell_initialization(table.get_cell(0), chain[0]) &&
        test_cell.check_cell_initialization(table.get_cell(1), chain[1]) &&
        test_cell.check_cell_initialization(table.get_cell(2), chain[2])
      ){
          
        test.set_test_emplace_back_jacobian_with_pointer(true);
      }

      else{test.set_test_emplace_back_jacobian_with_pointer(false);}

    //Checking cell initialization of cells without pointers.
    if(
        test_cell.check_cell_initialization(table.get_cell(1,0), cost_1_0, split_pos_1_0) &&
        test_cell.check_cell_initialization(table.get_cell(2,1), cost_2_1, split_pos_2_1) &&
        test_cell.check_cell_initialization(table.get_cell(2,0), cost_2_0, split_pos_2_0)
        ){

      test.set_test_emplace_back_jacobian(true);
    }

    else{test.set_test_emplace_back_jacobian(false);}

    }

  //Test Dense_Jacobian Table
  {
    //Dense Jacobian chain initialization
    const std::string path_to_file = 
      std::string(FIXTURE_DIR) + "/chain_test_cases/case_1_matrix_free";

    jacobian_chain<Dense_Jacobian, Matrix_free_information> chain{path_to_file};

    //Reserving memory for a Table with 3 pointer cells and 3 cells
    Table<Dense_Jacobian> table{static_cast<std::size_t>(3)};

    std::size_t cost_0 = 35, cost_1 = 40, cost_2 = 48;
    std::size_t cost_1_0 = 70, cost_2_1 = 65, cost_2_0 = 78;

    //The operation assignment is used only for debugging purposese, these do not match
    //by any means the optimal operation value.
    Operation op_0 = Operation::TANGENT, op_1 = Operation::ADJOINT, op_2 = Operation::TANGENT;
    Operation op_1_0 = Operation::MULTIPLICATION, op_2_1 = Operation::TANGENT;
    Operation op_2_0 = Operation::ADJOINT;

    std::size_t split_pos_1_0 = 0, split_pos_2_1 = 1, split_pos_2_0 = 1;

    //Emplacing back cell_0, cell_1 and cell_2
    table.emplace_back(&chain[0], cost_0, op_0);
    table.emplace_back(&chain[1], cost_1, op_1);
    table.emplace_back(&chain[2], cost_2, op_2);

    //Emplacing back cell_1_0, cell_2_1, cell_2_0
    table.emplace_back(cost_1_0, split_pos_1_0, op_1_0);
    table.emplace_back(cost_2_1, split_pos_2_1, op_2_1);
    table.emplace_back(cost_2_0, split_pos_2_0, op_2_0);

    //Checking cell initialization of cells with pointers.
    if(
        test_cell.check_cell_initialization(table.get_cell(0), chain[0],
          cost_0, op_0) &&

        test_cell.check_cell_initialization(table.get_cell(1), chain[1],
          cost_1, op_1) &&

        test_cell.check_cell_initialization(table.get_cell(2), chain[2],
          cost_2, op_2)
        ){

      test.set_test_emplace_back_dense_with_pointer(true);
    }

    else{test.set_test_emplace_back_dense_with_pointer(false);}

    if(
        test_cell.check_cell_initialization(table.get_cell(1,0), cost_1_0,
          split_pos_1_0, op_1_0) &&

        test_cell.check_cell_initialization(table.get_cell(2,1), cost_2_1,
          split_pos_2_1, op_2_1) &&

        test_cell.check_cell_initialization(table.get_cell(2,0), cost_2_0,
          split_pos_2_0, op_2_0)
        ){

      test.set_test_emplace_back_dense(true);
    }

    else{test.set_test_emplace_back_dense(false);}
  }
  //Test Sparse_Jacobian Table
  {
    //Sparse chain initialization
    const std::string path_to_file = 
      std::string(FIXTURE_DIR) + "/chain_test_cases/case_1_sparse";

    jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information> chain{path_to_file};

    //Reserving memory for a Table with 3 pointer cells and 3 cells
    Table<Sparse_Jacobian> table{static_cast<std::size_t>(3)};

    std::size_t cost_0 = 99, cost_1 = 77, cost_2 = 66;
    std::size_t cost_1_0 = 130, cost_2_1 = 115, cost_2_0 = 180;

    //The operation assignment is used only for debugging purposese, these do not match
    //by any means the optimal operation value.
    Operation op_0 = Operation::TANGENT, op_1 = Operation::ADJOINT, op_2 = Operation::TANGENT;
    Operation op_1_0 = Operation::MULTIPLICATION, op_2_1 = Operation::TANGENT;
    Operation op_2_0 = Operation::ADJOINT;

    std::size_t split_pos_1_0 = 0, split_pos_2_1 = 1, split_pos_2_0 = 1;

    //Emplacing back cell_0, cell_1 and cell_2
    table.emplace_back(&chain[0], cost_0, op_0);
    table.emplace_back(&chain[1], cost_1, op_1);
    table.emplace_back(&chain[2], cost_2, op_2);

    //Emplacing back cell_1_0, cell_2_1, cell_2_0
    //For simplicity no jacobian multiplication will be considered.
    //Inside cell_1_0 jacobian_1 will be stored while in cell_2_1 and
    //cell_2_0 jacobian 2 will be stored.
    table.emplace_back(chain.copy(0), cost_1_0, split_pos_1_0, op_1_0);
    table.emplace_back(chain.copy(1), cost_2_1, split_pos_2_1, op_2_1);
    table.emplace_back(chain.copy(2), cost_2_0, split_pos_2_0, op_2_0);

    //Checking cell initialization of cells with pointers.
    if(
        test_cell.check_cell_initialization(table.get_cell(0), chain[0],
          cost_0, op_0) &&

        test_cell.check_cell_initialization(table.get_cell(1), chain[1],
          cost_1, op_1) &&

        test_cell.check_cell_initialization(table.get_cell(2), chain[2],
          cost_2, op_2)
        ){

      test.set_test_emplace_back_sparse_with_pointer(true);
    }

    else{test.set_test_emplace_back_dense_with_pointer(false);}

    if(
        test_cell.check_cell_initialization(table.get_cell(1,0), cost_1_0,
          split_pos_1_0, op_1_0) &&

        test_cell.check_cell_initialization(table.get_cell(2,1), cost_2_1,
          split_pos_2_1, op_2_1) &&

        test_cell.check_cell_initialization(table.get_cell(2,0), cost_2_0,
          split_pos_2_0, op_2_0)
        ){

      test.set_test_emplace_back_sparse(true);
    }

    else{test.set_test_emplace_back_sparse(false);}
  }

  test.set_were_all_tests_succcessful();
  /* test.print_test_state(); */
}
