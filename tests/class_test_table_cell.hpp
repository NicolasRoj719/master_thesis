#include <iostream>
#include <cstdint>
#include <string>
/* #include "./../test.hpp" */
#include "./../table_cell.hpp"
#include "./../jacobian.hpp"
#include "./../chain.hpp"

class test_table_cell{
 public:
  test_table_cell(){

      test_dense_cell = is_dense_correctly_initialized();

      test_sparse_cell = is_sparse_cell_correctly_initialized();

      test_jacobian_cell_with_pointer =
          is_jacobian_cell_with_pointer_correctly_initialized();

      test_dense_cell_with_pointer =
          is_dense_cell_with_pointer_correctly_initialized();

      test_sparse_cell_with_pointer =
          is_sparse_cell_with_pointer_correctly_initialized();    

      if( (test_dense_cell && test_sparse_cell &&
            test_jacobian_cell_with_pointer && test_dense_cell_with_pointer &&
            test_sparse_cell_with_pointer) == true){

          were_all_tests_successful = true;
      }

      else{
          were_all_tests_successful = false;
          print_test_state();
      }
  }

  void print_test_state(){
    std::cout << "State of test table cell: \n";
    
    std::cout << "test_dense_cell: ";
    if(test_dense_cell){
        std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout<< "test_jacobian_cell_with_pointer: ";
    if(test_jacobian_cell_with_pointer){
        std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout<< "test_dense_cell_with_pointer: ";
    if(test_dense_cell_with_pointer){
        std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout<< "test_sparse_cell_with_pointer: ";
    if(test_sparse_cell_with_pointer){
        std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}
  }

 private:
  bool test_dense_cell;
  bool test_sparse_cell;
  bool test_jacobian_cell_with_pointer;
  bool test_dense_cell_with_pointer;
  bool test_sparse_cell_with_pointer;
  bool were_all_tests_successful;
  
  bool is_dense_correctly_initialized();

  bool is_sparse_cell_correctly_initialized(
        const Sparse_Jacobian& sparse_jacobian,
        std::size_t cost, std::size_t split_position,
        Operation operation, std::size_t memory);

  bool is_sparse_cell_correctly_initialized();

  bool is_jacobian_cell_with_pointer_correctly_initialized(
        const Jacobian& jacobian,
        std::size_t cost, std::size_t split_position);

  bool is_jacobian_cell_with_pointer_correctly_initialized();

  bool is_dense_cell_with_pointer_correctly_initialized(
        const Dense_Jacobian& jacobian,
        std::size_t cost, std::size_t split_position,
        const Operation& operation, std::size_t memory);

  bool is_dense_cell_with_pointer_correctly_initialized();

  bool is_sparse_cell_with_pointer_correctly_initialized(
        const Sparse_Jacobian& jacobian,
        std::size_t cost, std::size_t split_position,
        const Operation& operation);

  bool is_sparse_cell_with_pointer_correctly_initialized();
};

bool test_table_cell::is_dense_correctly_initialized(){
    {
        const std::size_t cost = 100, split_position = 2;
        Operation operation = Operation::TANGENT;
        cell<Dense_Jacobian> cell{cost, split_position, operation};
        
        if(cell.accumulated_cost() != cost ||
            cell.split_position() != split_position ||
            cell.operation() != operation){
            return false;
        }
    }

    {
        const std::size_t cost = 150, split_position = 1, memory = 130;
        Operation operation = Operation::ADJOINT;
        cell<Dense_Jacobian> cell{cost, split_position, operation, memory};
        
        if(cell.accumulated_cost() != cost ||
            cell.split_position() != split_position ||
            cell.operation() != operation ||
            cell.accumulated_memory().value() != memory){
            return false;
        }
    }

    return true;
}

bool test_table_cell::is_sparse_cell_correctly_initialized(
        const Sparse_Jacobian& sparse_jacobian,
        std::size_t cost, std::size_t split_position,
        Operation operation, std::size_t memory){

    cell<Sparse_Jacobian> cell{sparse_jacobian, cost, split_position, operation, memory};
    
    //Checking wrappers:
    if(cell.domain_dim() != sparse_jacobian.domain_dim() ||
        cell.codomain_dim() != sparse_jacobian.codomain_dim() ||
        cell.number_edges() != sparse_jacobian.number_edges() ||
        cell.number_nnz() != sparse_jacobian.number_nnz() ||
        cell.column_number_colors() != sparse_jacobian.get_column_number_colors() ||
        cell.row_number_colors() != sparse_jacobian.get_row_number_colors() ||
        cell.max_number_nnz_row() != sparse_jacobian.get_max_number_nnz_row() ||
        cell.max_number_nnz_column() != sparse_jacobian.get_max_number_nnz_column()){
        
        return false;
    }

    //Checking cell information
    if(cell.accumulated_cost() != cost ||
        cell.split_position() != split_position ||
        cell.operation() != operation ||
        cell.accumulated_memory().value() != memory){
        
        return false;
    }

    return true;
}

bool test_table_cell::is_sparse_cell_correctly_initialized(){

    std::string path_to_file_with_data = "./chain_test_cases/case_1_sparse";

    jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>
        chain{path_to_file_with_data};
    {
        const std::size_t cost = 240, split_position = 0, memory = 15;
        const Operation operation = Operation::MULTIPLICATION;
        
        if(!is_sparse_cell_correctly_initialized(
                chain[0], cost, split_position, operation, memory)){
            
            return false;
        }
    }

    {
        const std::size_t cost = 162, split_position = 2, memory = 80;
        const Operation operation = Operation::ADJOINT;
        
        if(!is_sparse_cell_correctly_initialized(
                chain[1], cost, split_position, operation, memory)){
            
            return false;
        }
    }

    return true;
}

bool test_table_cell::is_jacobian_cell_with_pointer_correctly_initialized(
        const Jacobian& jacobian,
        std::size_t cost, std::size_t split_position){

    cell_with_pointer<Jacobian> cell{&jacobian, cost, split_position};

    //Checking Wrappers
    if(cell.domain_dim() != jacobian.domain_dim() ||
        cell.codomain_dim() != jacobian.codomain_dim()){

        return false;
    }
    
    //Checking cell information
    if(cell.accumulated_cost() != cost ||
        cell.split_position() != split_position){

        return false;
    }

    return true;
}

bool test_table_cell::is_jacobian_cell_with_pointer_correctly_initialized(){
    
    std::string path_to_file_with_data =
        "./chain_test_cases/case_1_non_sparse_jacobian_information";

    jacobian_chain<Jacobian, Jacobian_information> chain{path_to_file_with_data};

    {
        std::size_t cost = 45, split_position= 3;

        if(!is_jacobian_cell_with_pointer_correctly_initialized(
                    chain[0], cost, split_position)){
            
            return false;
        }
    }

    {
        std::size_t cost = 63, split_position= 0;

        if(!is_jacobian_cell_with_pointer_correctly_initialized(
                    chain[2], cost, split_position)){
            
            return false;
        }
    }

    return true;
}

bool test_table_cell::is_dense_cell_with_pointer_correctly_initialized(
        const Dense_Jacobian& jacobian,
        std::size_t cost, std::size_t split_position,
        const Operation& operation, std::size_t memory){
    
    cell_with_pointer<Dense_Jacobian> 
        cell{&jacobian, cost, split_position, operation, memory};

    //Checking Wrappers
    if(cell.domain_dim() != jacobian.domain_dim() ||
        cell.codomain_dim() != jacobian.codomain_dim() ||
        cell.number_edges() != jacobian.number_edges()){
    
        return false;
    }

    //Checking cell information
    if(cell.accumulated_cost() != cost ||
        cell.split_position() != split_position ||
        cell.operation() != operation ||
        cell.accumulated_memory().value() != memory){
        
        return false;
    }

    return true;
}

bool test_table_cell::is_dense_cell_with_pointer_correctly_initialized(){
    
    std::string path_to_file_with_data =
        "./chain_test_cases/case_1_non_sparse_matrix_free";
    
    jacobian_chain<Dense_Jacobian, Matrix_free_information>
        chain{path_to_file_with_data};

    {
        const std::size_t cost = 60, split_position = 1, memory = 30;
        const Operation operation = Operation::MULTIPLICATION;
        if(!is_dense_cell_with_pointer_correctly_initialized(
                chain[0], cost, split_position, operation, memory)){
            
            return false;
        }
    }

    {
        const std::size_t cost = 111, split_position = 0, memory = 21;
        const Operation operation = Operation::ADJOINT;
        if(!is_dense_cell_with_pointer_correctly_initialized(
                chain[1], cost, split_position, operation, memory)){
            
            return false;
        }
    }

    return true;
}

bool test_table_cell::is_sparse_cell_with_pointer_correctly_initialized(
        const Sparse_Jacobian& jacobian,
        std::size_t cost, std::size_t split_position, const Operation& operation){

    cell_with_pointer<Sparse_Jacobian>
        cell{&jacobian, cost, split_position, operation};

    //Testing Wrappers
    if(cell.domain_dim() != jacobian.domain_dim() ||
        cell.codomain_dim() != jacobian.codomain_dim() ||
        cell.number_edges() != jacobian.number_edges() ||
        cell.number_nnz() != jacobian.number_nnz() ||
        cell.column_number_colors() != jacobian.get_column_number_colors() ||
        cell.row_number_colors() != jacobian.get_row_number_colors() ||
        cell.max_number_nnz_row() != jacobian.get_max_number_nnz_row() ||
        cell.max_number_nnz_column() != jacobian.get_max_number_nnz_column()){

        return false;
    }

    //Testing cell information
    if(cell.accumulated_cost() != cost ||
        cell.split_position() != split_position ||
        cell.operation() != operation){
        
        return false;
    }

    return true;
}

bool test_table_cell::is_sparse_cell_with_pointer_correctly_initialized(){

    std::string path_to_file_with_data =
        "./chain_test_cases/case_1_sparse";

    jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>
        chain{path_to_file_with_data};

    {
        const std::size_t cost = 121, split_position = 0;
        const Operation operation = Operation::TANGENT;

        if(!is_sparse_cell_with_pointer_correctly_initialized(
                chain[2], cost, split_position, operation)){

            return false;
        }
    }

    {
        const std::size_t cost = 131, split_position = 1;
        const Operation operation = Operation::ADJOINT;

        if(!is_sparse_cell_with_pointer_correctly_initialized(
                chain[1], cost, split_position, operation)){

            return false;
        }
    }

    return true;
}
