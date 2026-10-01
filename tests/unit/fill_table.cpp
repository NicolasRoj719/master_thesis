#include <algorithm>
#include <iostream>
#include <string>

#include "class_test_fill_table.hpp"
#include "chain.hpp"
#include "fill_table.hpp"
#include "binomial_checkpointing.hpp"

int main(){

  //Initializing chains
  const std::string path_to_file_jacobian = 
    std::string(FIXTURE_DIR) + "/chain_test_cases/case_1_jacobian_information";

  const std::string path_to_file_dense =
    std::string(FIXTURE_DIR) + "/chain_test_cases/case_1_matrix_free";

  const std::string path_to_file_sparse =
    std::string(FIXTURE_DIR) + "/chain_test_cases/case_1_sparse";

  Jacobian_chain<Jacobian, Jacobian_information>
                chain_jacobian{path_to_file_jacobian};

  Jacobian_chain<Dense_Jacobian, Matrix_free_information>
                chain_dense{path_to_file_dense};

  Jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>
                chain_sparse{path_to_file_sparse};


  //Object that stores the information collected from the tests.
  test_fill_table test;
  
  //Testing tangent and adjoint cost functions acting on Jacobian_T pointer
  {
   //Tangent
   //The data that will be used to validate can be found under:
   //"/chain_test_cases/validate/matrix_free/case_1_" + 0, 1 or 2.
   
    bool test_tangent_dense;
    bool test_tangent_sparse;
    bool test_tangent;

    if(Fill_table<Dense_Jacobian, Matrix_free_information>::tangent_cost(&chain_dense[0]) !=
        100 * 4 ||
        Fill_table<Dense_Jacobian, Matrix_free_information>::tangent_cost(&chain_dense[1]) !=
        150 * 2 ||
        Fill_table<Dense_Jacobian, Matrix_free_information>::tangent_cost(&chain_dense[2]) !=
        80 * 3){

      test_tangent_dense = false;
    }

    else{test_tangent_dense = true;}

    if(Fill_table<Sparse_Jacobian, Matrix_free_sparse_information>::tangent_cost(&chain_sparse[0])!=
        100 * 2 ||
        Fill_table<Sparse_Jacobian, Matrix_free_sparse_information>::tangent_cost(&chain_sparse[1])!=
        150 * 1 ||
        Fill_table<Sparse_Jacobian, Matrix_free_sparse_information>::tangent_cost(&chain_sparse[2])!=
        80 * 1){

      test_tangent_sparse = false;
    }

    else{test_tangent_sparse = true;}

    if(test_tangent_dense && test_tangent_sparse){

      test_tangent = true;
    }

    //Adjoint
    bool test_adjoint_dense;
    bool test_adjoint_sparse;
    bool test_adjoint;

    if(Fill_table<Dense_Jacobian, Matrix_free_information>::adjoint_cost(&chain_dense[0]) !=
        100 * 2 ||
        Fill_table<Dense_Jacobian, Matrix_free_information>::adjoint_cost(&chain_dense[1]) !=
        150 * 3 ||
        Fill_table<Dense_Jacobian, Matrix_free_information>::adjoint_cost(&chain_dense[2]) !=
        80 * 4){

      test_adjoint_dense = false;
    }


    else{test_adjoint_dense = true;}

    if(Fill_table<Sparse_Jacobian, Matrix_free_sparse_information>::adjoint_cost(&chain_sparse[0])!= 
        100 * 1 ||
        Fill_table<Sparse_Jacobian, Matrix_free_sparse_information>::adjoint_cost(&chain_sparse[1])!=
        150 * 2 ||
        Fill_table<Sparse_Jacobian, Matrix_free_sparse_information>::adjoint_cost(&chain_sparse[2])!=
        80 * 2){

      test_adjoint_sparse = false;
    }

    else{test_adjoint_sparse = true;}

    if(test_adjoint_dense && test_adjoint_sparse){


      test_adjoint = true;
    }

    if(test_tangent && test_adjoint){

      test.set_test_cost_tangent_adjoint_with_pointer(true);
    }

    else{

      test.set_test_cost_tangent_adjoint_with_pointer(false);
    }
  }

  //Testing accumulate number of edges
  {

    bool test_edge_accumulation_dense = false;
    bool test_edge_accumulation_sparse = false;
    //Testing accumulate for Dense_Jacobian
    {
      //Initilizing dense jacobian table. 
      Table<Dense_Jacobian> table_dense{3};

      std::size_t cost_0 = 45, cost_1 = 47, cost_2 = 60;

      Operation op_0 = Operation::TANGENT, op_1 = Operation::ADJOINT;
      Operation op_2 = Operation::TANGENT;

      //Emplacing cells with pointer.
      table_dense.emplace_back(&chain_dense[0], cost_0, op_0);
      table_dense.emplace_back(&chain_dense[1], cost_1, op_1);
      table_dense.emplace_back(&chain_dense[2], cost_2, op_2);

      //Tranfering data from table_dense to table object contained
      //within the Fill_table class.
      Fill_table<Dense_Jacobian, Matrix_free_information>
        fill_table_methods_dense{std::move(table_dense)};

      if(
          fill_table_methods_dense.accumulate_edges(1,0) != 100 + 150 ||
          fill_table_methods_dense.accumulate_edges(2,1) != 150 + 80 ||
          fill_table_methods_dense.accumulate_edges(2,0) != 100 + 150 + 80 ||
          fill_table_methods_dense.accumulate_edges(0,0) != 100 ||
          fill_table_methods_dense.accumulate_edges(1,1) != 150 ||
          fill_table_methods_dense.accumulate_edges(2,2) != 80

          ){

        test_edge_accumulation_dense = false;
        
      }

      else{test_edge_accumulation_dense = true;}
    }

    //Testing accumulate for Sparse_Jacobian
    {
      //Initializing sparse jacobian table
      Table<Sparse_Jacobian> table_sparse{3};

      std::size_t cost_0 = 11, cost_1 = 22, cost_2 = 33;
      std::size_t cost_1_0 = 44, cost_2_1 = 55, cost_2_0 = 66;

      std::size_t split_pos_1_0 = 0, split_pos_2_1 = 1, split_pos_2_0 = 1;

      Operation op_0 = Operation::ADJOINT, op_1 = Operation::TANGENT;
      Operation op_2 = Operation::ADJOINT;
      Operation op_1_0 = Operation::TANGENT, op_2_1 = Operation::MULTIPLICATION;
      Operation op_2_0 = Operation::ADJOINT;

      //Emplacing cells with pointer
      table_sparse.emplace_back(&chain_sparse[0], cost_0, op_0);
      table_sparse.emplace_back(&chain_sparse[1], cost_1, op_1);
      table_sparse.emplace_back(&chain_sparse[2], cost_2, op_2);

      //Emplacing other cells
      table_sparse.emplace_back(chain_sparse[1] * chain_sparse[0],
          cost_1_0, split_pos_1_0, op_1_0);

      table_sparse.emplace_back(chain_sparse[2] * chain_sparse[1],
          cost_2_1, split_pos_2_1, op_2_1);

      table_sparse.emplace_back(table_sparse.get_cell(2,1).get_jacobian() *
          chain_sparse[0], cost_2_0, split_pos_2_0, op_2_0);

      //Tranfering data from table_dense to table object contained
      //within the Fill_table class.
      Fill_table<Sparse_Jacobian, Matrix_free_sparse_information>
        fill_table_methods_sparse{std::move(table_sparse)};

      if(
          fill_table_methods_sparse.accumulate_edges(1,0) != 100 + 150 ||
          fill_table_methods_sparse.accumulate_edges(2,1) != 150 + 80 ||
          fill_table_methods_sparse.accumulate_edges(2,0) != 100 + 150 + 80 ||
          fill_table_methods_sparse.accumulate_edges(0,0) != 100 ||
          fill_table_methods_sparse.accumulate_edges(1,1) != 150 ||
          fill_table_methods_sparse.accumulate_edges(2,2) != 80
          ){

        test_edge_accumulation_sparse = false;
      }

      else{test_edge_accumulation_sparse = true;}
    }

    if(test_edge_accumulation_dense && test_edge_accumulation_sparse){

      test.set_test_accumulate_number_edges(true);
    }

    else{
      test.set_test_accumulate_number_edges(false);
    }
  }

  //Test multiplication cost.
  {
    bool test_jacobian_multiplication_cost = false;
    bool test_dense_multiplication_cost = false;
    bool test_sparse_multiplication_cost = false;
    //Jacobian multiplication cost
    {
      //Fill a sample table with the data contained in 
      //"./chain_test_cases/case_1_jacobian_information"
      Table<Jacobian> table_jacobian{3};

      //Emplacing back cells with pointer
      table_jacobian.emplace_back(&chain_jacobian[0]);
      table_jacobian.emplace_back(&chain_jacobian[1]);
      table_jacobian.emplace_back(&chain_jacobian[2]);

      //Data necessary to initialize the cells.
      std::size_t cost_1_0 = 50, cost_2_1 = 60;
      std::size_t split_pos_1_0 = 0, split_pos_2_1 = 1; 

      //Emplacing back cells without pointer
      table_jacobian.emplace_back(cost_1_0, split_pos_1_0);
      table_jacobian.emplace_back(cost_2_1, split_pos_2_1);

      //Transferring dat ownership to Fill_table class.
      Fill_table<Jacobian, Jacobian_information>
          fill_table{std::move(table_jacobian)};

      //Testing multiplication_cost
      if(
          fill_table.multiplication_cost(1,0,0) != chain_jacobian[1].codomain_dim() *
          chain_jacobian[0].domain_dim() * chain_jacobian[0].codomain_dim()
          ||
          fill_table.multiplication_cost(2,1,1) != chain_jacobian[2].codomain_dim() *
          chain_jacobian[1].domain_dim() * chain_jacobian[1].codomain_dim()
          ||
          fill_table.multiplication_cost(2,0,0) != cost_2_1 + chain_jacobian[2].codomain_dim() *
          chain_jacobian[0].domain_dim() * chain_jacobian[0].codomain_dim()
          ||
          fill_table.multiplication_cost(2,1,0) != cost_1_0 + chain_jacobian[2].codomain_dim() *
          chain_jacobian[0].domain_dim() * chain_jacobian[1].codomain_dim()
          ){

        test_jacobian_multiplication_cost = false;
      }
      else{test_jacobian_multiplication_cost = true;}
    }

    //Dense multiplication cost
    {

      std::size_t cost_0 = 55, cost_1 = 62, cost_2 = 88;
      std::size_t cost_1_0 = 76, cost_2_1 = 120;

      std::size_t split_pos_1_0 = 0, split_pos_2_1 = 1;

      Operation op_0 = Operation::ADJOINT, op_1 = Operation::TANGENT;
      Operation op_2 = Operation::TANGENT;
      Operation op_1_0 = Operation::MULTIPLICATION, op_2_1 = Operation::MULTIPLICATION;

      Table<Dense_Jacobian> table_dense{3};
      
      table_dense.emplace_back(&chain_dense[0], cost_0, op_0);
      table_dense.emplace_back(&chain_dense[1], cost_1, op_1);
      table_dense.emplace_back(&chain_dense[2], cost_2, op_2);

      table_dense.emplace_back(cost_1_0, split_pos_1_0, op_1_0);
      table_dense.emplace_back(cost_2_1, split_pos_2_1, op_2_1);

      Fill_table<Dense_Jacobian, Matrix_free_information>
          fill_table{std::move(table_dense)};

      if(
          fill_table.multiplication_cost(1,0,0) != cost_0 + cost_1 +
          chain_dense[1].codomain_dim() * chain_dense[0].domain_dim() *
          chain_dense[0].codomain_dim()
          ||
          fill_table.multiplication_cost(2,1,1) != cost_1 + cost_2 +
          chain_dense[2].codomain_dim() * chain_dense[1].domain_dim() *
          chain_dense[1].codomain_dim()
          ||
          fill_table.multiplication_cost(2,0,0) != cost_0 + cost_2_1 +
          chain_dense[2].codomain_dim() * chain_dense[0].domain_dim() *
          chain_dense[0].codomain_dim()
          ||
          fill_table.multiplication_cost(2,1,0) != cost_1_0 + cost_2 +
          chain_dense[2].codomain_dim() * chain_dense[0].domain_dim() *
          chain_dense[1].codomain_dim()
          ){
        
        test_dense_multiplication_cost = false;
      }

      else{test_dense_multiplication_cost = true;}
    }

    //Sparse multiplication cost
    {
      std::size_t cost_0 = 23, cost_1 = 34, cost_2 = 45;
      std::size_t cost_1_0 = 56, cost_2_1 = 67;

      std::size_t split_pos_1_0 = 0, split_pos_2_1 = 1;

      Operation op_0 = Operation::ADJOINT, op_1 = Operation::TANGENT;
      Operation op_2 = Operation::ADJOINT;
      Operation op_1_0 = Operation::MULTIPLICATION, op_2_1 = Operation::MULTIPLICATION;

      Table<Sparse_Jacobian> table_sparse{3};

      table_sparse.emplace_back(&chain_sparse[0], cost_0, op_0);
      table_sparse.emplace_back(&chain_sparse[1], cost_1, op_1);
      table_sparse.emplace_back(&chain_sparse[2], cost_2, op_2);

      table_sparse.emplace_back(chain_sparse[1] * chain_sparse[0],
          cost_1_0, split_pos_1_0, op_1_0);

      table_sparse.emplace_back(chain_sparse[2] * chain_sparse[1],
          cost_2_1, split_pos_2_1, op_2_1);

      Fill_table<Sparse_Jacobian, Matrix_free_sparse_information>
        fill_table{std::move(table_sparse)};

      if(
          fill_table.multiplication_cost(1,0,0,6) != cost_0 + cost_1 +
          6 * std::min(chain_sparse[1].get_max_number_nnz_row(),
              chain_sparse[0].get_max_number_nnz_column())
          ||
          fill_table.multiplication_cost(2,1,1,4) != cost_1 + cost_2 +
          4 * std::min(chain_sparse[2].get_max_number_nnz_row(),
              chain_sparse[1].get_max_number_nnz_column())
          ||
          fill_table.multiplication_cost(2,0,0,8) != cost_0 + cost_2_1 +
          8 * std::min(static_cast<std::size_t>(1),
              chain_sparse[0].get_max_number_nnz_column())
          ||
          fill_table.multiplication_cost(2,1,0,8) != cost_2 + cost_1_0 +
          8 * std::min(chain_sparse[2].get_max_number_nnz_row(),
              static_cast<std::size_t>(2))
          ){
        
        test_sparse_multiplication_cost = false;
      }

      else{test_sparse_multiplication_cost = true;}
    }
    
    if(test_jacobian_multiplication_cost && test_dense_multiplication_cost &&
        test_sparse_multiplication_cost){

      test.set_test_multiplication(true);
    }

    else{
      
      test.set_test_multiplication(false);
    }
  }

  //Test tangent and adjoint cost
  {
    bool test_tangent_dense;
    bool test_adjoint_dense;
    bool test_tangent_sparse;
    bool test_adjoint_sparse;

    // Dense tangent and adjoint cost
    {

      std::size_t cost_0 = 111, cost_1 = 222, cost_2 = 333;
      std::size_t cost_1_0 = 444, cost_2_1 = 555;

      std::size_t split_pos_1_0 = 0, split_pos_2_1 = 1;

      Operation op_0 = Operation::ADJOINT, op_1 = Operation::TANGENT;
      Operation op_2 = Operation::ADJOINT;
      Operation op_1_0 = Operation::TANGENT, op_2_1 = Operation::ADJOINT;

      Table<Dense_Jacobian> table_dense{3};

      table_dense.emplace_back(&chain_dense[0], cost_0, op_0);
      table_dense.emplace_back(&chain_dense[1], cost_1, op_1);
      table_dense.emplace_back(&chain_dense[2], cost_2, op_2);

      table_dense.emplace_back(cost_1_0, split_pos_1_0, op_1_0);
      table_dense.emplace_back(cost_2_1, split_pos_2_1, op_2_1);

      Fill_table<Dense_Jacobian, Matrix_free_information>
        fill_table{std::move(table_dense)};

      //Test tangent dense.
      if(
          fill_table.tangent_cost(1,0,0) != cost_0 + chain_dense[1].number_edges() *
          chain_dense[0].domain_dim() 
          ||
          fill_table.tangent_cost(2,1,1) != cost_1 + chain_dense[2].number_edges() *
          chain_dense[1].domain_dim()
          ||
          fill_table.tangent_cost(2,0,0) != cost_0 + (chain_dense[1].number_edges() +
          chain_dense[2].number_edges()) * chain_dense[0].domain_dim()
          ||
          fill_table.tangent_cost(2,1,0) != cost_1_0 + chain_dense[2].number_edges() *
          chain_dense[0].domain_dim()
          ){

        test_tangent_dense = false;
      }

      else{test_tangent_dense = true;}

      //Test adjoint dense
      if(
          fill_table.adjoint_cost(1,0,0) != cost_1 + chain_dense[0].number_edges() *
          chain_dense[1].codomain_dim()
          ||
          fill_table.adjoint_cost(2,1,1) != cost_2 + chain_dense[1].number_edges() *
          chain_dense[2].codomain_dim()
          ||
          fill_table.adjoint_cost(2,0,0) != cost_2_1 + chain_dense[0].number_edges() *
          chain_dense[2].codomain_dim()
          ||
          fill_table.adjoint_cost(2,1,0) != cost_2 + (chain_dense[0].number_edges() +
          chain_dense[1].number_edges()) * chain_dense[2].codomain_dim()
          ){

        test_adjoint_dense = false;
      }

      else{test_adjoint_dense = true;} 
    }

    //Test sparse tangent and adjoint
    {

      std::size_t cost_0 = 11, cost_1 = 22, cost_2 = 33;
      std::size_t cost_1_0 = 44, cost_2_1 = 55;

      std::size_t split_pos_1_0 = 0, split_pos_2_1 = 1;

      Operation op_0 = Operation::ADJOINT, op_1 = Operation::TANGENT;
      Operation op_2 = Operation::ADJOINT;
      Operation op_1_0 = Operation::ADJOINT, op_2_1 = Operation::TANGENT;

      Table<Sparse_Jacobian> table_sparse{3};

      table_sparse.emplace_back(&chain_sparse[0], cost_0, op_0);
      table_sparse.emplace_back(&chain_sparse[1], cost_1, op_1);
      table_sparse.emplace_back(&chain_sparse[2], cost_2, op_2);

      table_sparse.emplace_back(chain_sparse[1] * chain_sparse[0], cost_1_0,
                                split_pos_1_0, op_1_0);
      table_sparse.emplace_back(chain_sparse[2] * chain_sparse[1], cost_2_1,
                                split_pos_2_1, op_2_1);

      Fill_table<Sparse_Jacobian, Matrix_free_sparse_information>
        fill_table{std::move(table_sparse)};

      //Test tangent sparse
      if(
          fill_table.tangent_cost(1,0,0,2) != cost_0 + chain_sparse[1].number_edges() * 2
          ||
          fill_table.tangent_cost(2,1,1,1) != cost_1 + chain_sparse[2].number_edges() * 1
          ||
          fill_table.tangent_cost(2,0,0,2) != cost_0 + (chain_sparse[1].number_edges() +
          chain_sparse[2].number_edges()) * 2
          ||
          fill_table.tangent_cost(2,1,0,2) != cost_1_0 + chain_sparse[2].number_edges() * 2

          ){

        test_tangent_sparse = false;
      }
      
      else{test_tangent_sparse = true;}

      //Test adjoint sparse
      if(
          fill_table.adjoint_cost(1,0,0,2) != cost_1 + chain_sparse[0].number_edges() * 2
          ||
          fill_table.adjoint_cost(2,1,1,3) != cost_2 + chain_sparse[1].number_edges() * 3
          ||
          fill_table.adjoint_cost(2,0,0,3) != cost_2_1 + chain_sparse[0].number_edges() * 3
          ||
          fill_table.adjoint_cost(2,1,0,3) != cost_2 + (chain_sparse[0].number_edges() +
            chain_sparse[1].number_edges()) * 3
          ){

          test_adjoint_sparse = false;
      }

      else{test_adjoint_sparse = true;}
    }

    if(test_tangent_dense && test_adjoint_dense &&
        test_tangent_sparse && test_adjoint_sparse){
      
      test.set_test_cost_tangent_adjoint_cell(true);
    }

    else{
      test.set_test_cost_tangent_adjoint_cell(false);
    }

  }


  //Fill method is tested using cases 1 and 2.
  std::size_t memory_limit = 30;
  test.run_test_cases(memory_limit);

  //Mixed formulation
  {
    Jacobian_chain<Dense_Jacobian, Matrix_free_information>
      chain{std::string(FIXTURE_DIR) + "/chain_test_cases/case_1_matrix_free"};

    std::vector<std::size_t> subprograms_cost = 
      Binomial_checkpointing::read_subprograms_cost_from_file(std::string(FIXTURE_DIR)
          + "/chain_test_cases/case_1_functions_cost", 3);

    Table<Dense_Jacobian> table{chain.size()};

    //Emplace back cells with pointers
    //table(0)
    table.emplace_back(&chain[0],
                          chain[0].codomain_dim() * chain[0].number_edges(),
                          Operation::ADJOINT);
    //table(1)
    table.emplace_back(&chain[1],
                        chain[1].domain_dim() * chain[1].number_edges(),
                        Operation::TANGENT);
    //table(2)
    table.emplace_back(&chain[2],
                        chain[2].domain_dim() * chain[2].number_edges(),
                        Operation::TANGENT);

    //test is_split_reversable
    if(tool_box_split::is_split_reversable(table, 2, 0, 200) != true ||
        tool_box_split::is_split_reversable(table, 2, 0, 150) != true ||
        tool_box_split::is_split_reversable(table, 2, 0, 150 - 1) != false ||
        tool_box_split::is_split_reversable(table, 2, 0, 30) != false ||
        tool_box_split::is_split_reversable(table, 2, 1, 160) != true ||
        tool_box_split::is_split_reversable(table, 2, 1, 150) != true ||
        tool_box_split::is_split_reversable(table, 2, 1, 150-1) != false ||
        tool_box_split::is_split_reversable(table, 1, 0, 155) != true ||
        tool_box_split::is_split_reversable(table, 1, 0, 150) != true ||
        tool_box_split::is_split_reversable(table, 1, 0, 140) != false ||
        tool_box_split::is_split_reversable(table, 2, 2, 100) != true ||
        tool_box_split::is_split_reversable(table, 2, 2, 80) != true ||
        tool_box_split::is_split_reversable(table, 2, 2, 80 - 1) != false){

        test.set_test_is_split_reversable(false);
    }
    else{test.set_test_is_split_reversable(true);}   

    //Test adjoint chain partition 
    {
      //Test array representing the number of edges of subprograms.
      std::vector<std::size_t> test_arr{70, 60, 40, 35, 30, 20, 10};

      //Test result of each one of the tests carried out to validate the correctness of 
      //the method adjoint_chain_partition.
      bool test_1, test_2, test_3;
      {

        std::size_t memory_limit = 40;
        std::size_t last_idx = test_arr.size() - 1;
        std::size_t first_idx = 2;

        //Arguments: const ref to arr, last index in range, first index in range, memory_limit
        std::vector<std::size_t> split_positions = 
          tool_box_split::adjoint_chain_partition(test_arr, last_idx, first_idx, memory_limit);

        if(split_positions.size() != 5){
          
          test_1 = false;
        }
        else{

          if(split_positions[0] != last_idx + 1 || split_positions[1] != 5 ||
              split_positions[2] != 4 || split_positions[3] != 3 ||
              split_positions[4] != first_idx){
            
            test_1 = false;
          }
          else{test_1 = true;}
        }

        /* tool_box_split::subchain_partition_parser(split_positions, &std::cout); */
      }

      {
        std::size_t memory_limit = 80;
        std::size_t last_idx = test_arr.size() - 1;
        std::size_t first_idx = 1;

        std::vector<std::size_t> split_positions = 
          tool_box_split::adjoint_chain_partition(test_arr, last_idx, first_idx, memory_limit);

        if(split_positions.size() != 4){

          test_2 = false;
        }
        else{

          if(split_positions[0] != last_idx + 1 || split_positions[1] != 4 ||
              split_positions[2] != 2 || split_positions[3] != first_idx){
            
            test_2 = false;
          }
          else{test_2 = true;}
        }

        /* tool_box_split::subchain_partition_parser(split_positions, &std::cout); */
      }

      {
        std::size_t memory_limit = 60;
        std::size_t last_idx = 4;
        std::size_t first_idx = 1;

        std::vector<std::size_t> split_positions = 
          tool_box_split::adjoint_chain_partition(test_arr, last_idx, first_idx, memory_limit);

        if(split_positions.size() != 5){

          test_3 = false;
        }
        else{

          if(split_positions[0] != last_idx + 1 || split_positions[1] != 4 ||
              split_positions[2] != 3 || split_positions[3] != 2 ||
              split_positions[4] != first_idx){

            test_3 = false;
          }
          else{test_3 = true;}
        }

        /* tool_box_split::subchain_partition_parser(split_positions, &std::cout); */
      }

      if(test_1 && test_2 && test_3){

        test.set_test_adjoint_chain_partition(true);
      }
      else{test.set_test_adjoint_chain_partition(false);}
    }
  }

  //Test build_subchain_execution_costs_array
  {
    //Test array representing the number of edges of subprograms.
    std::vector<std::size_t> subchain_number_edges{70, 60, 40, 35, 30, 20, 10};

    //Test arrar subprogram execution costs
    std::vector<std::size_t> subprograms_execution_cost{100, 150, 200, 250, 300, 350, 400};

    bool test_1, test_2, test_3;
    {
      std::size_t memory_limit = 40;
      std::size_t last_idx = subchain_number_edges.size() - 1;
      std::size_t first_idx = 2;

      std::vector<std::size_t> split_positions =
        tool_box_split::adjoint_chain_partition(subchain_number_edges, last_idx, first_idx, memory_limit);

      std::vector<std::size_t> subchain_execution_costs =
        tool_box_split::build_subchain_execution_costs_array(subprograms_execution_cost, split_positions);

      if(subchain_execution_costs.size() != 4){
        test_1 = false;  
      }
      else{

        if(subchain_execution_costs[0] != 200 || subchain_execution_costs[1] != 250 ||
            subchain_execution_costs[2] != 300 || subchain_execution_costs[3] != 350 + 400){

          test_1 = false;
        }
        else{test_1 = true;}
      }
    }

    {
      std::size_t memory_limit = 80;
      std::size_t last_idx = subchain_number_edges.size() - 1;
      std::size_t first_idx = 1;

      std::vector<std::size_t> split_positions = 
        tool_box_split::adjoint_chain_partition(subchain_number_edges, last_idx, first_idx, memory_limit);

      std::vector<std::size_t> subchain_execution_costs =
        tool_box_split::build_subchain_execution_costs_array(subprograms_execution_cost, split_positions);

      if(subchain_execution_costs.size() != 3){
        test_2 = false;  
      }
      else{

        if(subchain_execution_costs[0] != 150 || subchain_execution_costs[1] != 200 + 250||
            subchain_execution_costs[2] != 300 + 350 + 400){

          test_2 = false;
        }
        else{test_2 = true;}
      }
    }
    
    {
      std::size_t memory_limit = 60;
      std::size_t last_idx = 4;
      std::size_t first_idx = 1;

      std::vector<std::size_t> split_positions = 
        tool_box_split::adjoint_chain_partition(subchain_number_edges, last_idx, first_idx, memory_limit);

      std::vector<std::size_t> subchain_execution_costs =
        tool_box_split::build_subchain_execution_costs_array(subprograms_execution_cost, split_positions);

      if(subchain_execution_costs.size() != 4){
        test_3 = false;  
      }
      else{

        if(subchain_execution_costs[0] != 150 || subchain_execution_costs[1] != 200||
            subchain_execution_costs[2] != 250 || subchain_execution_costs[3] != 300){

          test_3 = false;
        }
        else{test_3 = true;}
      }
    }

    if(test_1 && test_2 && test_3){

      
      test.set_test_build_subchain_execution_costs_array(true);
    }
    else{test.set_test_build_subchain_execution_costs_array(false);}
  }

  //Checking if all tests were successful.
  test.were_all_test_successful();
  //Printing the final state of the test.
  /* test.print_test_state(); */

}
