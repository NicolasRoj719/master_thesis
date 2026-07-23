#include <limits>
#include <algorithm>
#include "./../jacobian.hpp"
#include "./../chain.hpp"
#include "./../table_cell.hpp"
#include "./../table.hpp"
#include "./../fill_table.hpp"

class test_fill_table{
 public:
  test_fill_table(){}

  void set_test_cost_tangent_adjoint_with_pointer(bool arg){

   test_cost_tangent_adjoint_cell_with_pointer = arg;
  }

  void set_test_accumulate_number_edges(bool arg){

    test_accumulate_number_edges = arg;
  }

  void set_test_multiplication(bool arg){

   test_multiplication = arg;
  }

  void set_test_cost_tangent_adjoint_cell(bool arg){

   test_cost_tangent_adjoint_cell = arg;
  }

  void set_test_seed_memory_load(bool arg){

   test_seed_memory_load = arg;
  }

  void run_test_case_1_tests(){
    
    bool test_jacobian = run_test_case_1_jacobian();
    bool test_dense_without_memory = false;
    bool test_dense_with_memory = false;
    bool test_sparse_without_memory = false;
    bool test_sparse_with_memory = false;


    run_test_case_1_dense(test_dense_without_memory, test_dense_with_memory);
    run_test_case_1_sparse(test_sparse_without_memory, test_sparse_with_memory);
    
    if(test_jacobian && test_dense_without_memory &&
        test_dense_with_memory && test_sparse_without_memory &&
        test_sparse_with_memory){

      tests_case_1 = true;
    }

    else{tests_case_1 = false;}
  }

  void print_test_state(){
    
    std::cout << "State of fill algorithm test:\n";

    std::cout << "test_cost_tangent_adjoint_cell_with_pointer: ";
    if(test_cost_tangent_adjoint_cell_with_pointer){
      std::cout<<"successful.\n";
    }
    else{std::cout<<"failed.\n";}

    std::cout << "test_cost_tangent_adjoint_cell: ";
    if(test_cost_tangent_adjoint_cell){
      std::cout<<"successful.\n";
    }
    else{std::cout<<"failed.\n";}

    std::cout << "test_multiplication: ";
    if(test_multiplication){
      std::cout<<"successful.\n";
    }
    else{std::cout<<"failed.\n";}

    std::cout << "test_accumulate_number_edges: ";
    if(test_accumulate_number_edges){
      std::cout<<"successful.\n";
    }
    else{std::cout<<"failed.\n";}

    std::cout << "test_seed_memory_load: ";
    if(test_seed_memory_load){
      std::cout<<"successful.\n";
    }
    else{std::cout<<"failed.\n";}

    std::cout << "tests_case_1: ";
    if(tests_case_1){
      std::cout<<"successful.\n";
    }
    else{std::cout<<"failed.\n";}

    std::cout<<'\n';

  }

  void were_all_test_successful(){

    if(test_cost_tangent_adjoint_cell_with_pointer &&
        test_cost_tangent_adjoint_cell &&
        test_multiplication &&
        test_accumulate_number_edges &&
        test_seed_memory_load &&
        tests_case_1){
      
      were_all_test_successful_ = true;
    }

    else{
      were_all_test_successful_ = false;
      print_test_state();
    }
  } 

 protected:
  bool test_cost_tangent_adjoint_cell_with_pointer;

  bool test_cost_tangent_adjoint_cell;

  bool test_multiplication;

  bool test_accumulate_number_edges;

  bool test_seed_memory_load;

  bool tests_case_1;

  bool were_all_test_successful_;

  bool run_test_case_1_jacobian();

  void run_test_case_1_dense(bool& test_case_1_dense, bool& test_case_1_dense_with_memory);

  void run_test_case_1_sparse(bool& test_case_1_sparse, bool& test_case_1_sparse_with_memory);
};

bool test_fill_table::run_test_case_1_jacobian(){

  //Create Jacobian chain from file
  jacobian_chain<Jacobian, Jacobian_information> 
    chain{"./chain_test_cases/case_1_jacobian_information"};

  std::size_t cost_1_0 = chain[1].codomain_dim() * chain[0].domain_dim() *
                        chain[0].codomain_dim();
  std::size_t cost_2_1 = chain[2].codomain_dim() * chain[1].domain_dim() *
                        chain[1].codomain_dim();

  std::size_t split_pos_1_0 = 0, split_pos_2_1 = 1;
  std::size_t split_pos_2_0;

  // Recall the formula: cost_j_k_i = cost_j_k+1 + cost_k_i + jac_j_k+1.codomain_dim() *
  // jac_k_i.domain_dim() * jac_k_i.codomain_dim()
  std::size_t cost_2_0_0 = cost_2_1 + 0 + chain[2].codomain_dim() *
                            chain[0].domain_dim() * chain[0].codomain_dim();
  
  std::size_t cost_2_1_0 = 0 + cost_1_0 + chain[2].codomain_dim() *
                            chain[0].domain_dim() * chain[1].codomain_dim();

  std::size_t cost_2_0;


  if(cost_2_0_0 < cost_2_1_0){
    
    cost_2_0 = cost_2_0_0;
    split_pos_2_0 = 0;
  }

  else{

    cost_2_0 = cost_2_1_0;
    split_pos_2_0 = 1;
  }

  //Create and fill dynamic programming table
  fill_table<Jacobian, Jacobian_information> fill_table{chain};

  //Compare costs and split position data.
  if(fill_table.get_cell(1,0).accumulated_cost() != cost_1_0 ||
      fill_table.get_cell(1,0).split_position() != split_pos_1_0 ||
      fill_table.get_cell(2,1).accumulated_cost() != cost_2_1 ||
      fill_table.get_cell(2,1).split_position() != split_pos_2_1 ||
      fill_table.get_cell(2,0).accumulated_cost() != cost_2_0 ||
      fill_table.get_cell(2,0).split_position() != split_pos_2_0
      ){
    
    return false;
  }

  return true;
}

void test_fill_table::run_test_case_1_dense(bool& test_case_1_dense,
                                            bool& test_case_1_dense_with_memory){
  
  //Create Dense chain from file
  jacobian_chain<Dense_Jacobian, Matrix_free_information>
    chain{"./chain_test_cases/case_1_matrix_free"};

  std::size_t cost_0 = chain[0].codomain_dim() * chain[0].number_edges();
  std::size_t cost_1 = chain[1].domain_dim() * chain[1].number_edges();
  std::size_t cost_2 = chain[2].domain_dim() * chain[2].number_edges();

  Operation op_0 = Operation::ADJOINT, op_1 = Operation::TANGENT;
  Operation op_2 = Operation::TANGENT;

  //Memory load
  std::size_t memory_0 = chain[0].number_edges();
  std::size_t memory_1 = 0, memory_2 = 0;

  std::size_t cost_1_0_tan = cost_0 + chain[0].domain_dim() * chain[1].number_edges();
  std::size_t cost_1_0_adj = cost_1 + chain[1].codomain_dim() * chain[0].number_edges();
  std::size_t cost_1_0_mul = cost_0 + cost_1 + chain[1].codomain_dim()*
                              chain[0].domain_dim() * chain[0].codomain_dim();

  std::size_t cost_1_0 = cost_1_0_mul;
  std::size_t split_pos_1_0 = 0;
  Operation op_1_0 = Operation::MULTIPLICATION;
  std::size_t memory_1_0 = memory_0 + memory_1;

  if(cost_1_0_tan < cost_1_0){

    cost_1_0 = cost_1_0_tan;
    op_1_0 = Operation::TANGENT;
    memory_1_0 = memory_0;
  }

  if(cost_1_0_adj < cost_1_0){

    cost_1_0 = cost_1_0_adj;
    op_1_0 = Operation::ADJOINT;
    memory_1_0 = memory_1 + chain[0].number_edges();
  }

  std::size_t cost_2_1_tan = cost_1 + chain[1].domain_dim() * chain[2].number_edges();
  std::size_t cost_2_1_adj = cost_2 + chain[2].codomain_dim() * chain[1].number_edges();
  std::size_t cost_2_1_mul = cost_2 + cost_1 + chain[2].codomain_dim() *
                              chain[1].domain_dim() * chain[1].codomain_dim();

  std::size_t cost_2_1 = cost_2_1_mul;
  std::size_t split_pos_2_1 = 1;
  Operation op_2_1 = Operation::MULTIPLICATION;
  std::size_t memory_2_1 = memory_2 + memory_1;

  if(cost_2_1_tan < cost_2_1){

    cost_2_1 = cost_2_1_tan;
    op_2_1 = Operation::TANGENT;
    memory_2_1 = memory_1;
  }

  if(cost_2_1_adj < cost_2_1){

    cost_2_1 = cost_2_1_adj;
    op_2_1 = Operation::ADJOINT;
    memory_2_1 = memory_2 + chain[1].number_edges();
  }

  std::size_t cost_2_0_0_tan = cost_0 + chain[0].domain_dim() * 
                              (chain[1].number_edges() + chain[2].number_edges());

  std::size_t cost_2_0_0_adj = cost_2_1 +chain[2].codomain_dim() * chain[0].number_edges();
  std::size_t cost_2_0_0_mul = cost_0 + cost_2_1 + chain[2].codomain_dim() *
                                chain[0].domain_dim() * chain[0].codomain_dim();

  std::size_t cost_2_0_0 = cost_2_0_0_mul;
  std::size_t split_pos_2_0_0 = 0;
  Operation op_2_0_0 = Operation::MULTIPLICATION;
  std::size_t memory_2_0_0 = memory_2_1 + memory_0;

  if(cost_2_0_0_tan < cost_2_0_0){

    cost_2_0_0 = cost_2_0_0_tan;
    op_2_0_0 = Operation::TANGENT;
    memory_2_0_0 = memory_0;
  }

  if(cost_2_0_0_adj < cost_2_0_0){

    cost_2_0_0 = cost_2_0_0_adj;
    op_2_0_0 = Operation::ADJOINT;
    memory_2_0_0 = memory_2_1 + chain[0].number_edges();
  }

  std::size_t cost_2_0 = cost_2_0_0;
  std::size_t split_pos_2_0 = split_pos_2_0_0;
  Operation op_2_0 = op_2_0_0;
  std::size_t memory_2_0 = memory_2_0_0; 

  std::size_t cost_2_1_0_tan = cost_1_0 + chain[0].domain_dim() * chain[2].number_edges();
  std::size_t cost_2_1_0_adj = cost_2 + chain[2].codomain_dim() *
                                (chain[1].number_edges() + chain[0].number_edges());

  std::size_t cost_2_1_0_mul = cost_1_0 + cost_2 + chain[2].codomain_dim() *
                                chain[0].domain_dim() * chain[1].codomain_dim();

  std::size_t cost_2_1_0 = cost_2_1_0_mul;
  Operation op_2_1_0 = Operation::MULTIPLICATION;
  std::size_t split_pos_2_1_0 = 1;
  std::size_t memory_2_1_0 = memory_2 + memory_1_0;

  if(cost_2_1_0_tan < cost_2_1_0){

    cost_2_1_0 = cost_2_1_0_tan;
    op_2_1_0 = Operation::TANGENT;
    memory_2_1_0 = memory_1_0;
  }

  if(cost_2_1_0_adj < cost_2_1_0){

    cost_2_1_0 = cost_2_1_0_adj;
    op_2_1_0 = Operation::ADJOINT;
    memory_2_1_0 = memory_2 + chain[1].number_edges() + chain[0].number_edges();
  }

  if(cost_2_1_0 < cost_2_0){

    cost_2_0 = cost_2_1_0;
    split_pos_2_0 = split_pos_2_1_0;
    op_2_0 = op_2_1_0;
    memory_2_0 = memory_2_1_0;
  }

  {
    //Create and fill table.
    fill_table<Dense_Jacobian, Matrix_free_information> method_fill{chain};

    if(
        method_fill.get_cell(0).accumulated_cost() != cost_0 ||
        method_fill.get_cell(0).operation() != op_0 ||
        method_fill.get_cell(1).accumulated_cost() != cost_1 ||
        method_fill.get_cell(1).operation() != op_1 ||
        method_fill.get_cell(2).accumulated_cost() != cost_2 ||
        method_fill.get_cell(2).operation() != op_2 ||
        method_fill.get_cell(1,0).accumulated_cost() != cost_1_0 ||
        method_fill.get_cell(1,0).split_position() != split_pos_1_0 ||
        method_fill.get_cell(1,0).operation() != op_1_0 ||
        method_fill.get_cell(2,1).accumulated_cost() != cost_2_1 ||
        method_fill.get_cell(2,1).split_position() != split_pos_2_1 ||
        method_fill.get_cell(2,1).operation() != op_2_1 ||
        method_fill.get_cell(2,0).accumulated_cost() != cost_2_0 ||
        method_fill.get_cell(2,0).split_position() != split_pos_2_0 ||
        method_fill.get_cell(2,0).operation() != op_2_0
        ){

      test_case_1_dense = false;
    }

    else{test_case_1_dense = true;}

  }
  
  {
    //Create and fill table.
    fill_table<Dense_Jacobian, Matrix_free_information>
      method_fill{chain, std::numeric_limits<std::size_t>::max()};
    
    if(
        method_fill.get_cell(0).accumulated_cost() != cost_0 ||
        method_fill.get_cell(0).operation() != op_0 ||
        method_fill.get_cell(0).accumulated_memory() != memory_0 ||
        method_fill.get_cell(1).accumulated_cost() != cost_1 ||
        method_fill.get_cell(1).operation() != op_1 ||
        method_fill.get_cell(1).accumulated_memory() != memory_1 ||
        method_fill.get_cell(2).accumulated_cost() != cost_2 ||
        method_fill.get_cell(2).operation() != op_2 ||
        method_fill.get_cell(2).accumulated_memory() != memory_2 ||
        method_fill.get_cell(1,0).accumulated_cost() != cost_1_0 ||
        method_fill.get_cell(1,0).split_position() != split_pos_1_0 ||
        method_fill.get_cell(1,0).operation() != op_1_0 ||
        method_fill.get_cell(1,0).accumulated_memory() != memory_1_0 ||
        method_fill.get_cell(2,1).accumulated_cost() != cost_2_1 ||
        method_fill.get_cell(2,1).split_position() != split_pos_2_1 ||
        method_fill.get_cell(2,1).operation() != op_2_1 ||
        method_fill.get_cell(2,1).accumulated_memory() != memory_2_1 ||
        method_fill.get_cell(2,0).accumulated_cost() != cost_2_0 ||
        method_fill.get_cell(2,0).split_position() != split_pos_2_0 ||
        method_fill.get_cell(2,0).operation() != op_2_0 ||
        method_fill.get_cell(2,0).accumulated_memory() != memory_2_0
        ){

      test_case_1_dense_with_memory = false;
    }

    else{test_case_1_dense_with_memory = true;}

  }
}

void test_fill_table::run_test_case_1_sparse(bool& test_case_1_sparse,
                                              bool& test_case_1_sparse_with_memory){

  //Create Sparse chain from file
  jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>
    chain{"./chain_test_cases/case_1_sparse"};

  std::size_t cost_0 = chain[0].get_row_number_colors() * chain[0].number_edges();
  std::size_t cost_1 = chain[1].get_column_number_colors() * chain[1].number_edges();
  std::size_t cost_2 = chain[2].get_column_number_colors() * chain[2].number_edges();

  Operation op_0 = Operation::ADJOINT, op_1 = Operation::TANGENT, op_2 = Operation::TANGENT;

  //Memory load
  std::size_t memory_0 = chain[0].number_edges();
  std::size_t memory_1 = 0, memory_2 = 0;

  std::size_t cost_1_0;
  std::size_t split_pos_1_0 = 0;
  Operation op_1_0;
  std::size_t memory_1_0;
  std::size_t max_column_number_nnz_1_0;

  {
    //Propagating sparsity pattern
    auto sparse_jacobian_1_0 = chain[1] * chain[0];
    max_column_number_nnz_1_0 = sparse_jacobian_1_0.get_max_number_nnz_column();

    //Cost calculation 
    std::size_t cost_1_0_tan = cost_0 + sparse_jacobian_1_0.get_column_number_colors() *
                              chain[1].number_edges();

    std::size_t cost_1_0_adj = cost_1 + sparse_jacobian_1_0.get_row_number_colors() *
                              chain[0].number_edges();

    std::size_t cost_1_0_mul = cost_0 + cost_1 + sparse_jacobian_1_0.number_nnz()*std::min(
        chain[1].get_max_number_nnz_row(), chain[0].get_max_number_nnz_column());

    cost_1_0 = cost_1_0_mul;
    op_1_0 = Operation::MULTIPLICATION;
    memory_1_0 = memory_0 + memory_1;

    if(cost_1_0_tan < cost_1_0){

      cost_1_0 = cost_1_0_tan;
      op_1_0 = Operation::TANGENT;
      memory_1_0 = memory_0;
    }

    if(cost_1_0_adj < cost_1_0){

      cost_1_0 = cost_1_0_adj;
      op_1_0 = Operation::ADJOINT;
      memory_1_0 = memory_1 + chain[0].number_edges();
    }
  }

  std::size_t cost_2_1;
  std::size_t split_pos_2_1 = 1;
  Operation op_2_1;
  std::size_t memory_2_1;
  std::size_t max_row_number_nnz_2_1;
  {
    //Propagating sparsity pattern
    auto sparse_jacobian_2_1 = chain[2] * chain[1];

    max_row_number_nnz_2_1 = sparse_jacobian_2_1.get_max_number_nnz_row();

    //Cost calculation
    std::size_t cost_2_1_tan = cost_1 + sparse_jacobian_2_1.get_column_number_colors() *
                                chain[2].number_edges();

    std::size_t cost_2_1_adj = cost_2 + sparse_jacobian_2_1.get_row_number_colors() *
                                chain[1].number_edges();

    std::size_t cost_2_1_mul = cost_2 + cost_1 + sparse_jacobian_2_1.number_nnz() *
      std::min(chain[2].get_max_number_nnz_row(), chain[1].get_max_number_nnz_column());

    cost_2_1 = cost_2_1_mul;
    op_2_1 = Operation::MULTIPLICATION;
    memory_2_1 = memory_2 + memory_1;

    if(cost_2_1_tan < cost_2_1){

      cost_2_1 = cost_2_1_tan;
      op_2_1 = Operation::TANGENT;
      memory_2_1 = memory_1;
    }

    if(cost_2_1_adj < cost_2_1){

      cost_2_1 = cost_2_1_adj;
      op_2_1 = Operation::ADJOINT;
      memory_2_1 = memory_2 + chain[1].number_edges();
    }
  }

  std::size_t cost_2_0;
  std::size_t split_pos_2_0;
  std::size_t memory_2_0;
  Operation op_2_0;
  {
    //Propagating sparsity pattern
    auto sparse_jacobian_2_0 = (chain[2] * chain[1]) * chain[0];

    {
      //Cost calculation
      std::size_t cost_2_0_0_tan = cost_0 + sparse_jacobian_2_0.get_column_number_colors() *
                                  (chain[1].number_edges() + chain[2].number_edges());

      std::size_t cost_2_0_0_adj = cost_2_1 + sparse_jacobian_2_0.get_row_number_colors() *
                                    chain[0].number_edges();

      std::size_t cost_2_0_0_mul = cost_2_1 + cost_0 + sparse_jacobian_2_0.number_nnz()*
      std::min(max_row_number_nnz_2_1, chain[0].get_max_number_nnz_column());

      cost_2_0 = cost_2_0_0_mul;
      split_pos_2_0 = 0;
      op_2_0 = Operation::MULTIPLICATION;
      memory_2_0 = memory_2_1 + memory_0;

      if(cost_2_0_0_tan < cost_2_0){

        cost_2_0 = cost_2_0_0_tan;
        op_2_0 = Operation::TANGENT;
        memory_2_0 = memory_0;
      }

      if(cost_2_0_0_adj < cost_2_0){

        cost_2_0 = cost_2_0_0_adj;
        op_2_0 = Operation::ADJOINT;
        memory_2_0 = memory_2_1 + chain[0].number_edges();
      }
    }

    {
     //Cost calculation
      std::size_t cost_2_1_0_tan = cost_1_0 + sparse_jacobian_2_0.get_column_number_colors() *
                                  chain[2].number_edges();

      std::size_t cost_2_1_0_adj = cost_2 + sparse_jacobian_2_0.get_row_number_colors() *
                                  (chain[1].number_edges() + chain[0].number_edges());

      std::size_t cost_2_1_0_mul = cost_2 + cost_1_0 + sparse_jacobian_2_0.number_nnz() *
      std::min(chain[2].get_max_number_nnz_row(), max_column_number_nnz_1_0);

      if(cost_2_1_0_mul < cost_2_0){

        cost_2_0 = cost_2_1_0_mul;
        split_pos_2_0 = 1;
        op_2_0 = Operation::MULTIPLICATION;
        memory_2_0 = memory_2 + memory_1_0;
      }

      if(cost_2_1_0_tan < cost_2_0){

        cost_2_0 = cost_2_1_0_tan;
        split_pos_2_0 = 1;
        op_2_0 = Operation::TANGENT;
        memory_2_0 = memory_1_0;
      }

      if(cost_2_1_0_adj < cost_2_0){

        cost_2_0 = cost_2_1_0_adj;
        split_pos_2_0 = 1;
        op_2_0 = Operation::ADJOINT;
        memory_2_0 = memory_2 + chain[1].number_edges() + chain[0].number_edges();
      }
    }
  }

  //Test without memory bound
  {

    fill_table<Sparse_Jacobian, Matrix_free_sparse_information>
      method_fill{chain};

    if(
        method_fill.get_cell(0).accumulated_cost() != cost_0 ||
        method_fill.get_cell(0).operation() != op_0 ||
        method_fill.get_cell(1).accumulated_cost() != cost_1 ||
        method_fill.get_cell(1).operation() != op_1 ||
        method_fill.get_cell(2).accumulated_cost() != cost_2 ||
        method_fill.get_cell(2).operation() != op_2 ||
        method_fill.get_cell(1,0).accumulated_cost() != cost_1_0 ||
        method_fill.get_cell(1,0).split_position() != split_pos_1_0 ||
        method_fill.get_cell(1,0).operation() != op_1_0 ||
        method_fill.get_cell(2,1).accumulated_cost() != cost_2_1 ||
        method_fill.get_cell(2,1).split_position() != split_pos_2_1 ||
        method_fill.get_cell(2,1).operation() != op_2_1 ||
        method_fill.get_cell(2,0).accumulated_cost() != cost_2_0 ||
        method_fill.get_cell(2,0).split_position() != split_pos_2_0 ||
        method_fill.get_cell(2,0).operation() != op_2_0
        ){

      test_case_1_sparse = false;
    }

    else{test_case_1_sparse= true;}

  }

  //Test with memory bound.
  {
    fill_table<Sparse_Jacobian, Matrix_free_sparse_information>
      method_fill{chain, std::numeric_limits<std::size_t>::max()};

    if(
        method_fill.get_cell(0).accumulated_cost() != cost_0 ||
        method_fill.get_cell(0).operation() != op_0 ||
        method_fill.get_cell(0).accumulated_memory() != memory_0 ||
        method_fill.get_cell(1).accumulated_cost() != cost_1 ||
        method_fill.get_cell(1).operation() != op_1 ||
        method_fill.get_cell(1).accumulated_memory() != memory_1 ||
        method_fill.get_cell(2).accumulated_cost() != cost_2 ||
        method_fill.get_cell(2).operation() != op_2 ||
        method_fill.get_cell(2).accumulated_memory() != memory_2 ||
        method_fill.get_cell(1,0).accumulated_cost() != cost_1_0 ||
        method_fill.get_cell(1,0).split_position() != split_pos_1_0 ||
        method_fill.get_cell(1,0).operation() != op_1_0 ||
        method_fill.get_cell(1,0).accumulated_memory() != memory_1_0 ||
        method_fill.get_cell(2,1).accumulated_cost() != cost_2_1 ||
        method_fill.get_cell(2,1).split_position() != split_pos_2_1 ||
        method_fill.get_cell(2,1).operation() != op_2_1 ||
        method_fill.get_cell(2,1).accumulated_memory() != memory_2_1 ||
        method_fill.get_cell(2,0).accumulated_cost() != cost_2_0 ||
        method_fill.get_cell(2,0).split_position() != split_pos_2_0 ||
        method_fill.get_cell(2,0).operation() != op_2_0 ||
        method_fill.get_cell(2,0).accumulated_memory() != memory_2_0
        ){

      test_case_1_sparse_with_memory = false;
    }

    else{test_case_1_sparse_with_memory = true;}

  }




}

