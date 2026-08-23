#include <limits>
#include <algorithm>
#include "./../jacobian.hpp"
#include "./../chain.hpp"
#include "./../table_cell.hpp"
#include "./../table.hpp"
#include "./../fill_table.hpp"
/* #include "./../playground.hpp" */

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

  void set_test_function_cost_accumulation(bool arg){
    test_function_cost_accumulation = arg;
  }

  void set_test_split_reversed_chain(bool arg){
    test_split_reversed_chain = arg;
  }

  void set_test_is_split_reversable(bool arg){
    test_is_split_reversable = arg;
  }

  /* void set_test_seed_memory_load(bool arg){ */

  /*  test_seed_memory_load = arg; */
  /* } */

  void run_test_cases(std::size_t memory_limit = std::numeric_limits<std::size_t>::max()){
    
    bool test_jacobian = run_test_case_1_jacobian();

    bool test_dense_without_memory = 
      run_test_case_dense("./chain_test_cases/case_1_matrix_free");

    bool test_dense_with_memory =
      run_test_case_dense("./chain_test_cases/case_2_matrix_free", memory_limit);

    bool test_sparse_without_memory = 
      run_test_case_sparse("./chain_test_cases/case_1_sparse");

    bool test_sparse_with_memory =
      run_test_case_sparse("./chain_test_cases/case_2_sparse", memory_limit);
    
    if(test_jacobian && test_dense_without_memory && test_dense_with_memory &&
        test_sparse_without_memory && test_sparse_with_memory){

      test_cases = true;
    }

    else{test_cases = false;}
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

    std::cout << "test_function_cost_accumulation: ";
    if(test_function_cost_accumulation){
      std::cout<<"successful.\n";
    }
    else{std::cout<<"failed.\n";}

    std::cout << "test_split_reversed_chain: ";
    if(test_split_reversed_chain){
      std::cout<<"successful.\n";
    }
    else{std::cout<<"failed.\n";}

    std::cout << "test_is_split_reversable: ";
    if(test_is_split_reversable){
      std::cout<<"successful.\n";
    }
    else{std::cout<<"failed.\n";}

    std::cout << "test_cases: ";
    if(test_cases){
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
        test_cases){
      
      were_all_test_successful_ = true;
    }

    else{
      were_all_test_successful_ = false;
      print_test_state();
    }
  } 

  template<class cell_T>
  static bool check_cell_information(const cell_T& cell_arg, std::size_t cost,
                                std::size_t split_position){

    if(cell_arg.accumulated_cost() != cost ||
        cell_arg.split_position() != split_position){
      
      return false;
    }
    return true;
  }

  template<class cell_T>
  static bool check_cell_information(const cell_T& cell_arg, std::size_t cost,
                                      std::size_t split_position, Operation operation){

    if(cell_arg.accumulated_cost() != cost ||
        cell_arg.split_position() != split_position ||
        !(cell_arg.operation() == operation)){

      return false;
    }

    return true;
  }

  template<class cell_T>
  static bool check_cell_information(const cell_T& cell_arg, std::size_t cost,
                                      Operation operation){

    if(cell_arg.accumulated_cost() != cost ||
        !(cell_arg.operation() == operation)){
      
      return false;
    }

    return true;
  }

 protected:
  bool test_cost_tangent_adjoint_cell_with_pointer;

  bool test_cost_tangent_adjoint_cell;

  bool test_multiplication;

  bool test_accumulate_number_edges;

  bool test_cases;

  //Mixed formulation.
  bool test_function_cost_accumulation;

  bool test_split_reversed_chain;

  bool test_is_split_reversable;

  bool were_all_test_successful_;

  bool run_test_case_1_jacobian();

  bool run_test_case_dense(const std::string& path_to_chain_file,
        std::size_t memory_limit = std::numeric_limits<std::size_t>::max());

  bool run_test_case_sparse(const std::string& path_to_chain_file,
        std::size_t memory_limit = std::numeric_limits<std::size_t>::max());
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
  if(check_cell_information(fill_table.get_cell(1,0), cost_1_0, split_pos_1_0) && 
      check_cell_information(fill_table.get_cell(2,1), cost_2_1, split_pos_2_1) && 
      check_cell_information(fill_table.get_cell(2,0), cost_2_0, split_pos_2_0)){
    
    return true;
  }

  return false;
}

bool test_fill_table::run_test_case_dense(const std::string& path_to_chain_file,
    std::size_t memory_limit){

  //Create Dense chain from file
  jacobian_chain<Dense_Jacobian, Matrix_free_information>
    chain{path_to_chain_file};


  std::size_t cost_0;
  Operation op_0;
  if(chain[0].codomain_dim() < chain[0].domain_dim() &&
      chain[0].number_edges() <= memory_limit){

    cost_0 = chain[0].codomain_dim() * chain[0].number_edges();
    op_0 = Operation::ADJOINT;
  }
  else{
    
    cost_0 = chain[0].domain_dim() * chain[0].number_edges();
    op_0 = Operation::TANGENT;
  }

  std::size_t cost_1;
  Operation op_1;
  if(chain[1].codomain_dim() < chain[1].domain_dim() &&
      chain[1].number_edges() <= memory_limit){
    
    cost_1 = chain[1].codomain_dim() * chain[1].number_edges();
    op_1 = Operation::ADJOINT;
  }
  else{

    cost_1 = chain[1].domain_dim() * chain[1].number_edges();
    op_1 = Operation::TANGENT;
  }

  std::size_t cost_2;
  Operation op_2;
  if(chain[2].codomain_dim() < chain[2].domain_dim() &&
      chain[2].number_edges() <= memory_limit){
    
    cost_2 = chain[2].codomain_dim() * chain[2].number_edges();
    op_2 = Operation::ADJOINT;
  }
  else{

    cost_2 = chain[2].domain_dim() * chain[2].number_edges();
    op_2 = Operation::TANGENT;
  }

  std::size_t cost_1_0; 
  std::size_t split_pos_1_0 = 0;
  Operation op_1_0;
  std::size_t cost_tmp;
  //cost_1_0 = cost_1_0_mul
  cost_1_0 = cost_0 + cost_1 +
              chain[1].codomain_dim() * chain[0].domain_dim() * chain[0].codomain_dim();

  //cost_1_0_tan
  cost_tmp = cost_0 + chain[0].domain_dim() * chain[1].number_edges();
  op_1_0 = Operation::MULTIPLICATION;

  //cost_1_0_tan < cost_1_0_mul
  if(cost_tmp < cost_1_0){
    
    cost_1_0 = cost_tmp;
    op_1_0 = Operation::TANGENT;
  }

  //cost_1_0_adj
  cost_tmp = cost_1 + chain[1].codomain_dim() * chain[0].number_edges();
  //cost_1_0_adj < cost_1_0
  if(cost_tmp < cost_1_0 && chain[0].number_edges() <= memory_limit){

    cost_1_0 = cost_tmp;
    op_1_0 = Operation::ADJOINT;
  }

  std::size_t cost_2_1;
  std::size_t split_pos_2_1 = 1;
  Operation op_2_1;

  //cost_2_1 = cost_2_1_mul
  cost_2_1 = cost_1 + cost_2 +
              chain[2].codomain_dim() * chain[1].domain_dim() * chain[1].codomain_dim();

  op_2_1 = Operation::MULTIPLICATION;

  //cost_2_1_tan
  cost_tmp = cost_1 + chain[1].domain_dim() * chain[2].number_edges();
  //cost_2_1_tan < cost_2_1
  if(cost_tmp < cost_2_1){
    
    cost_2_1 = cost_tmp;
    op_2_1 = Operation::TANGENT;
  }

  //cost_2_1_adj
  cost_tmp = cost_2 + chain[2].codomain_dim() * chain[1].number_edges();
  //cost_2_1_adj < cost_2_1
  if(cost_tmp < cost_2_1 && chain[1].number_edges() <= memory_limit){

    cost_2_1 = cost_tmp;
    op_2_1 = Operation::ADJOINT;
  }

  std::size_t cost_2_0, split_pos_2_0;
  Operation op_2_0;

  //cost_2_1_0_mul
  cost_2_0 = cost_1_0 + cost_2 +
              chain[2].codomain_dim() * chain[0].domain_dim() * chain[1].codomain_dim();

  op_2_0 = Operation::MULTIPLICATION;
  split_pos_2_0 = 1;

  //cost_2_1_0_tan
  cost_tmp = cost_1_0 + chain[0].domain_dim() * chain[2].number_edges();

  //cost_2_1_0_tan < cost_2_0
  if(cost_tmp < cost_2_0){

    cost_2_0 = cost_tmp;
    op_2_0 = Operation::TANGENT;
    split_pos_2_0 = 1;
  }

  //cost_2_1_0_adj
  cost_tmp = cost_2 +
              chain[2].codomain_dim() * (chain[0].number_edges() + chain[1].number_edges());
  //cost_2_1_0_adj
  if(cost_tmp < cost_2_0 && (chain[0].number_edges() + chain[1].number_edges()) <= memory_limit){

    cost_2_0 = cost_tmp;
    op_2_0 = Operation::ADJOINT;
    split_pos_2_0 = 1;
  }

  //cost_2_0 = cost_2_0_0_mul
  cost_tmp= cost_0 + cost_2_1 +
              chain[2].codomain_dim() * chain[0].domain_dim() * chain[0].codomain_dim();

  if(cost_tmp < cost_2_0){
    
    cost_2_0 = cost_tmp;
    op_2_0 = Operation::MULTIPLICATION;
    split_pos_2_0 = 0;
  }

  //cost_2_0_0_tan
  cost_tmp = cost_0 +
              chain[0].domain_dim() * (chain[1].number_edges() + chain[2].number_edges());

  //cost_2_0_0_tan < cost_2_0_0_mul
  if(cost_tmp < cost_2_0){

    cost_2_0 = cost_tmp;
    op_2_0 = Operation::TANGENT;
    split_pos_2_0 = 0;
  }

  //cost_2_0_0_adj
  cost_tmp = cost_2_1 + chain[2].codomain_dim() * chain[0].number_edges();
  //cost_2_0_0_adj < cost_2_0
  if(cost_tmp < cost_2_0 && chain[0].number_edges() <= memory_limit){
  
    cost_2_0 = cost_tmp;
    op_2_0 = Operation::ADJOINT;
    split_pos_2_0 = 0;
  }

  {
    //Create and fill table.
    if(memory_limit == std::numeric_limits<std::size_t>::max()){

      fill_table<Dense_Jacobian, Matrix_free_information> method_fill{chain};

      if(check_cell_information(method_fill.get_cell(0), cost_0, op_0) && 
          check_cell_information(method_fill.get_cell(1), cost_1, op_1) && 
          check_cell_information(method_fill.get_cell(2), cost_2, op_2) && 
          check_cell_information(method_fill.get_cell(1,0), cost_1_0, split_pos_1_0, op_1_0) && 
          check_cell_information(method_fill.get_cell(2,1), cost_2_1, split_pos_2_1, op_2_1) &&
          check_cell_information(method_fill.get_cell(2,0), cost_2_0, split_pos_2_0, op_2_0)){

        return true;
      }

      return false;
    }

    fill_table<Dense_Jacobian, Matrix_free_information> method_fill{chain, memory_limit};

    if(check_cell_information(method_fill.get_cell(0), cost_0, op_0) && 
        check_cell_information(method_fill.get_cell(1), cost_1, op_1) && 
        check_cell_information(method_fill.get_cell(2), cost_2, op_2) && 
        check_cell_information(method_fill.get_cell(1,0), cost_1_0, split_pos_1_0, op_1_0) && 
        check_cell_information(method_fill.get_cell(2,1), cost_2_1, split_pos_2_1, op_2_1) &&
        check_cell_information(method_fill.get_cell(2,0), cost_2_0, split_pos_2_0, op_2_0)){

      return true;
    }

    return false;
  }
}

bool test_fill_table::run_test_case_sparse(const std::string& path_to_chain_file,
    std::size_t memory_limit){

  //Create Sparse chain from file
  jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>
    chain{path_to_chain_file};

  std::size_t cost_0;
  Operation op_0;
  if(chain[0].get_row_number_colors() < chain[0].get_column_number_colors() &&
      chain[0].number_edges() <= memory_limit){
    
    cost_0 = chain[0].get_row_number_colors() * chain[0].number_edges();
    op_0 = Operation::ADJOINT;
  }
  else{

    cost_0 = chain[0].get_column_number_colors() * chain[0].number_edges();
    op_0 = Operation::TANGENT;
  }

  std::size_t cost_1;
  Operation op_1;
  if(chain[1].get_row_number_colors() < chain[1].get_column_number_colors() &&
      chain[1].number_edges() <= memory_limit ){
    
    cost_1 = chain[1].get_row_number_colors() * chain[1].number_edges();
    op_1 = Operation::ADJOINT;
  }
  else{

    cost_1 = chain[1].get_column_number_colors() * chain[1].number_edges();
    op_1 = Operation::TANGENT;
  }

  std::size_t cost_2;
  Operation op_2;
  if(chain[2].get_row_number_colors() < chain[2].get_column_number_colors() &&
      chain[2].number_edges() <= memory_limit){
    
    cost_2 = chain[2].get_row_number_colors() * chain[2].number_edges();
    op_2 = Operation::ADJOINT;
  }
  else{

    cost_2 = chain[2].get_column_number_colors() * chain[2].number_edges();
    op_2 = Operation::TANGENT;
  }

  std::size_t cost_1_0, cost_tmp;
  std::size_t split_pos_1_0 = 0;
  Operation op_1_0;
  std::size_t max_column_number_nnz_1_0;

  {
    //Propagating sparsity pattern
    auto sparse_jacobian_1_0 = chain[1] * chain[0];
    max_column_number_nnz_1_0 = sparse_jacobian_1_0.get_max_number_nnz_column();

    //Cost calculation 
    //cost_1_0_mul
    cost_1_0 = cost_0 + cost_1 + sparse_jacobian_1_0.number_nnz() * 
      std::min(chain[1].get_max_number_nnz_row(), chain[0].get_max_number_nnz_column());
    op_1_0 = Operation::MULTIPLICATION;

    //cost_1_0_tan
    cost_tmp = cost_0 + sparse_jacobian_1_0.get_column_number_colors() *
                chain[1].number_edges();

    //cost_1_0_tan < cost_1_0_mul 
    if(cost_tmp < cost_1_0){

      cost_1_0 = cost_tmp;
      op_1_0 = Operation::TANGENT;
    }

    //cost_1_0_adj
    cost_tmp = cost_1 + sparse_jacobian_1_0.get_row_number_colors() *
                chain[0].number_edges();

    //cost_1_0_adj < cost_1_0
    if(cost_tmp < cost_1_0 && chain[0].number_edges() <= memory_limit){

      cost_1_0 = cost_tmp;
      op_1_0 = Operation::ADJOINT;
    }
  }

  std::size_t cost_2_1;
  std::size_t split_pos_2_1 = 1;
  Operation op_2_1;
  std::size_t max_row_number_nnz_2_1;

  {
    //Propagating sparsity pattern
    auto sparse_jacobian_2_1 = chain[2] * chain[1];
    max_row_number_nnz_2_1 = sparse_jacobian_2_1.get_max_number_nnz_row();

    //Cost calculation 
    //cost_2_1_mul
    cost_2_1 = cost_2 + cost_1 + sparse_jacobian_2_1.number_nnz() * 
      std::min(chain[2].get_max_number_nnz_row(), chain[1].get_max_number_nnz_column());
    op_2_1 = Operation::MULTIPLICATION;

    //cost_2_1_tan
    cost_tmp = cost_1 + sparse_jacobian_2_1.get_column_number_colors() *
                chain[2].number_edges();

    //cost_2_1_tan < cost_2_1_mul 
    if(cost_tmp < cost_2_1){

      cost_2_1 = cost_tmp;
      op_2_1 = Operation::TANGENT;
    }

    //cost_2_1_adj
    cost_tmp = cost_2 + sparse_jacobian_2_1.get_row_number_colors() *
                chain[1].number_edges();

    //cost_2_1_adj < cost_2_1
    if(cost_tmp < cost_2_1 && chain[1].number_edges() <= memory_limit){

      cost_2_1 = cost_tmp;
      op_2_1 = Operation::ADJOINT;
    }
  }

  std::size_t cost_2_0;
  std::size_t split_pos_2_0;
  Operation op_2_0;
  {
    //Propagating sparsity pattern
    auto sparse_jacobian_2_0 = (chain[2] * chain[1]) * chain[0];

    {
     //Cost calculation
     //cost_2_1_0_mul
      cost_2_0 = cost_2 + cost_1_0 + sparse_jacobian_2_0.number_nnz() *
      std::min(chain[2].get_max_number_nnz_row(), max_column_number_nnz_1_0);

      op_2_0 = Operation::MULTIPLICATION;

      split_pos_2_0 = 1;

      //cost_2_1_0_tan
      cost_tmp = cost_1_0 + sparse_jacobian_2_0.get_column_number_colors() *
                chain[2].number_edges();

      if(cost_tmp < cost_2_0){
        
        cost_2_0 = cost_tmp;
        op_2_0 = Operation::TANGENT;
        split_pos_2_0 = 1;
      }

      //cost_2_1_0_adj
      cost_tmp = cost_2 + sparse_jacobian_2_0.get_row_number_colors() *
                (chain[1].number_edges() + chain[0].number_edges());

      if(cost_tmp < cost_2_0 && (chain[1].number_edges() + chain[0].number_edges()) < memory_limit){

        cost_2_0 = cost_tmp;
        op_2_0 = Operation::ADJOINT;
        split_pos_2_0 = 1;
      }
    }

    {
      //Cost calculation
      //cost_2_0_0_mul
      cost_tmp = cost_2_1 + cost_0 + sparse_jacobian_2_0.number_nnz()*
      std::min(max_row_number_nnz_2_1, chain[0].get_max_number_nnz_column());

      if(cost_tmp < cost_2_0){
        
        cost_2_0 = cost_tmp;
        op_2_0 = Operation::MULTIPLICATION;
        split_pos_2_0 = 0;
      }

      //cost_2_0_0_tan
      cost_tmp = cost_0 + sparse_jacobian_2_0.get_column_number_colors() *
                                  (chain[1].number_edges() + chain[2].number_edges());

      //cost_2_0_0_tan < cost_2_0 
      if(cost_tmp < cost_2_0){
        
        cost_2_0 = cost_tmp;
        op_2_0 = Operation::TANGENT;
        split_pos_2_0 = 0;
      }

      //cost_2_0_0_adj
      cost_tmp = cost_2_1 + sparse_jacobian_2_0.get_row_number_colors() *
                                    chain[0].number_edges();

      if(cost_tmp < cost_2_0 && chain[0].number_edges() <= memory_limit){

        cost_2_0 = cost_tmp;
        op_2_0 = Operation::ADJOINT;
        split_pos_2_0 = 0;
      }
    }
  }

  {
    if(memory_limit == std::numeric_limits<std::size_t>::max()){

      fill_table<Sparse_Jacobian, Matrix_free_sparse_information>
        method_fill{chain};

      if(check_cell_information(method_fill.get_cell(0), cost_0, op_0) &&
          check_cell_information(method_fill.get_cell(1), cost_1, op_1) &&
          check_cell_information(method_fill.get_cell(2), cost_2, op_2) &&
          check_cell_information(method_fill.get_cell(1,0), cost_1_0, split_pos_1_0, op_1_0) &&
          check_cell_information(method_fill.get_cell(2,1), cost_2_1, split_pos_2_1, op_2_1) &&
          check_cell_information(method_fill.get_cell(2,0), cost_2_0, split_pos_2_0, op_2_0)){

        return true;
      }

      return false;
    }

    else{

      fill_table<Sparse_Jacobian, Matrix_free_sparse_information>
        method_fill{chain, memory_limit};

      if(check_cell_information(method_fill.get_cell(0), cost_0, op_0) &&
          check_cell_information(method_fill.get_cell(1), cost_1, op_1) &&
          check_cell_information(method_fill.get_cell(2), cost_2, op_2) &&
          check_cell_information(method_fill.get_cell(1,0), cost_1_0, split_pos_1_0, op_1_0) &&
          check_cell_information(method_fill.get_cell(2,1), cost_2_1, split_pos_2_1, op_2_1) &&
          check_cell_information(method_fill.get_cell(2,0), cost_2_0, split_pos_2_0, op_2_0)){

        return true;
      }

      return false;
    }
  }
}

