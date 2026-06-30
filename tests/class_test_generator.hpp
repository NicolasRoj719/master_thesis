#include <vector>
#include <stdexcept>
#include "./../util_structs.hpp"
#include "./../generator.hpp"

#ifndef TEST_GENERATOR_HPP
#define TEST_GENERATOR_HPP

template<class Jacobian_information_type>
class test_Generator{};

template<>
class test_Generator<Jacobian_information>{
 public:
  test_Generator(std::size_t chain_length, std::size_t dimension_lb,
      std::size_t dimension_ub, bool is_deterministic_, std::size_t seed = 45):
    generator{chain_length, dimension_lb, dimension_ub, is_deterministic_, seed}{
      if(generator.jacobian_information_size() == chain_length){
        test_size = true;
      }
      else{test_size = false;}
      test_data_bounds = is_generated_data_within_bounds(dimension_lb, dimension_ub);
      test_dimensions = rhs_output_dim_equal_lhs_input_dim();
      test_deterministic = 
        test_is_deterministic_method(chain_length, dimension_lb, dimension_ub, seed);
      
      if(test_size && test_data_bounds && test_dimensions &&
          test_deterministic){
        were_all_test_successful = true;
      }

      else{
        were_all_test_successful = false;
        print_test_state();
      }
    }

  void print_test_state() const{
    std::cout << "State of test_Generator<Jacobian_information>:\n";
    std::cout << "test_size: " ;
    if(test_size){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_data_bounds: " ;
    if(test_data_bounds){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_dimensions: ";
    if(test_dimensions){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_deterministic: ";
    if(test_deterministic){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}
    std::cout << '\n';
  }


 protected:
  const Generator<Jacobian_information> generator;
  bool test_size;
  bool test_data_bounds;
  bool test_dimensions;
  bool test_deterministic;
  bool were_all_test_successful;

  bool rhs_output_dim_equal_lhs_input_dim(){
    auto jacobian_information = generator.get_jacobian_information_reference();
    for(std::size_t i = 0; i < jacobian_information.size() - 1; i++){
      if(jacobian_information[i].codomain_dimension() !=
          jacobian_information[i+1].domain_dimension()){
        return false;
      }
    }
    return true;
  }

  bool is_generated_data_within_bounds(std::size_t dimension_lb, std::size_t dimension_ub){
    auto jacobian_information = generator.get_jacobian_information_reference();
    for(std::size_t i = 0; i < jacobian_information.size(); i++){
      if((dimension_ub < jacobian_information[i].domain_dimension()) ||
          (dimension_ub < jacobian_information[i].codomain_dimension()) ||
          (jacobian_information[i].domain_dimension() < dimension_lb) ||
          (jacobian_information[i].codomain_dimension() < dimension_lb)){
        return false;
      }
    }
    return true;
  }

  bool is_information_data_equal(
      const std::vector<Jacobian_information>& jacobian_information_0,
      const std::vector<Jacobian_information>& jacobian_information_1){
    if(jacobian_information_0.size() != jacobian_information_1.size()){
      return false;
    }
    for(std::size_t i = 0; i < jacobian_information_0.size(); i++){
      if((jacobian_information_0[i].domain_dimension() != 
          jacobian_information_1[i].domain_dimension()) ||
          (jacobian_information_0[i].codomain_dimension() !=
           jacobian_information_1[i].codomain_dimension())){
        return false;
      }
    }
    return true;
  }

  bool test_is_deterministic_method(std::size_t chain_length, std::size_t dimension_lb,
      std::size_t dimension_ub, std::size_t seed){
    const Generator<Jacobian_information> generator_0{chain_length, dimension_lb,
      dimension_ub, true, seed};
    const Generator<Jacobian_information> generator_1{chain_length, dimension_lb,
      dimension_ub, true, seed};
    return is_information_data_equal(generator_0.get_jacobian_information_reference(),
        generator_1.get_jacobian_information_reference());
  }
};

template<>
class test_Generator<Matrix_free_infomation>{
 public:
  test_Generator(std::size_t chain_length, std::size_t dimension_lb,
      std::size_t dimension_ub, std::size_t n_E_lb, std::size_t n_E_ub,
      bool is_deterministic_, std::size_t seed = 45):
    generator{chain_length, dimension_lb, dimension_ub, n_E_lb, n_E_ub,
    is_deterministic_, seed}{
      if(generator.jacobian_information_size() == chain_length){
        test_size = true;
      }
      else{test_size = false;}
      test_data_bounds = is_generated_data_within_bounds(dimension_lb, dimension_ub,
            n_E_lb, n_E_ub);
      test_dimensions = rhs_output_dim_equal_lhs_input_dim();
      test_deterministic =
        test_is_deterministic_method(chain_length, dimension_lb, dimension_ub,
            n_E_lb, n_E_ub, seed);

      if(test_size && test_data_bounds && test_dimensions &&
          test_deterministic){
        were_all_test_successful = true;
      }

      else{
        were_all_test_successful = false;
        print_test_state();
      }
    }

  void print_test_state() const{
    std::cout << "State of test_Generator<Matrix_free_infomation>:\n";
    std::cout << "test_size: " ;
    if(test_size){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_data_bounds: " ;
    if(test_data_bounds){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_dimensions: ";
    if(test_dimensions){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_deterministic: ";
    if(test_deterministic){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}
    std::cout << '\n';
  }

 protected:
  const Generator<Matrix_free_infomation> generator;  
  bool test_size;
  bool test_data_bounds;
  bool test_dimensions;
  bool test_deterministic;
  bool were_all_test_successful;

  bool rhs_output_dim_equal_lhs_input_dim(){
    auto jacobian_information = generator.get_jacobian_information_reference();
    for(std::size_t i = 0; i < jacobian_information.size() - 1; i++){
      if(jacobian_information[i].codomain_dimension() !=
          jacobian_information[i+1].domain_dimension()){
        return false;
      }
    }
    return true;
  }

  bool is_generated_data_within_bounds(std::size_t dimension_lb, std::size_t dimension_ub,
      std::size_t n_E_lb, std::size_t n_E_ub){
    auto jacobian_information = generator.get_jacobian_information_reference();
    for(std::size_t i = 0; i < jacobian_information.size(); i++){
      if((dimension_ub < jacobian_information[i].domain_dimension()) ||
          (dimension_ub < jacobian_information[i].codomain_dimension()) ||
          (jacobian_information[i].domain_dimension() < dimension_lb) ||
          (jacobian_information[i].codomain_dimension() < dimension_lb) ||
          (n_E_ub < jacobian_information[i].number_of_edges()) ||
          (jacobian_information[i].number_of_edges() < n_E_lb)){
        return false;
      }
    }
    return true;
  }

  bool is_information_data_equal(
      const std::vector<Matrix_free_infomation>& jacobian_information_0,
      const std::vector<Matrix_free_infomation>& jacobian_information_1){

    if(jacobian_information_0.size() != jacobian_information_1.size()){
      return false;
    }

    for(std::size_t i = 0; i < jacobian_information_0.size(); i++){
      if((jacobian_information_0[i].domain_dimension() != 
          jacobian_information_1[i].domain_dimension()) ||
          (jacobian_information_0[i].codomain_dimension() !=
           jacobian_information_1[i].codomain_dimension()) ||
          (jacobian_information_0[i].number_of_edges() !=
           jacobian_information_1[i].number_of_edges())){
        return false;
      }
    }
    return true;
  }

  bool test_is_deterministic_method(std::size_t chain_length, std::size_t dimension_lb,
      std::size_t dimension_ub, std::size_t n_E_lb, std::size_t n_E_ub, 
      std::size_t seed){
    const Generator<Matrix_free_infomation> generator_0{chain_length, dimension_lb,
      dimension_ub, n_E_lb, n_E_ub, true, seed};
    const Generator<Matrix_free_infomation> generator_1{chain_length, dimension_lb,
      dimension_ub, n_E_lb, n_E_ub, true, seed};
    return is_information_data_equal(generator_0.get_jacobian_information_reference(),
        generator_1.get_jacobian_information_reference());
  }
};

template<>
class test_Generator<Matrix_free_sparse_information>{
 public:
  test_Generator(std::size_t chain_length, std::size_t dimension_lb,
    std::size_t dimension_ub, std::size_t n_E_lb, std::size_t n_E_ub,
    double density_lb, double density_ub, bool is_deterministic_,
    std::size_t seed):
    generator{chain_length, dimension_lb, dimension_ub,
      n_E_lb, n_E_ub, density_lb, density_ub, is_deterministic_, seed}
  {

    if(generator.jacobian_information_size() == chain_length){
      test_jacobian_information_size = true;
    }
    else{
      test_jacobian_information_size = false;
    }

    if(generator.sparse_data_size() == chain_length){
      if(is_number_of_nnz_correct()){
      test_sparse_data_size = true;
      }
    else{test_sparse_data_size = false;}
    }
    
    else{test_sparse_data_size = false;}

    test_jacobian_information_bounds =
      is_jacobian_information_within_bounds(dimension_lb, dimension_ub, n_E_lb, n_E_ub);
    test_jacobian_information_dimensions = rhs_output_dim_equal_lhs_input_dim();
    test_sparse_data_bounds = is_sparse_data_within_bounds();
    test_repeated_entries = are_there_repeated_entries();
    test_nnz_in_every_row_column = all_columns_rows_have_nnz();
    test_deterministic = 
     is_deterministic(chain_length, dimension_lb, dimension_lb, n_E_lb, n_E_ub,
         density_lb, density_ub, seed);

    if(test_jacobian_information_size && test_jacobian_information_bounds &&
        test_jacobian_information_dimensions && test_deterministic &&
        test_sparse_data_size && test_sparse_data_bounds &&
        test_repeated_entries && test_nnz_in_every_row_column){
      were_all_test_successful = true;
    }
    else{
      were_all_test_successful = false;
      print_test_state();
    }
   
  }

  void print_test_state() const{
    std::cout << "State of test_Generator<Matrix_free_sparse_information>:\n";
    std::cout << "test_jacobian_information_size: " ;
    if(test_jacobian_information_size){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_jacobian_information_bounds: " ;
    if(test_jacobian_information_bounds){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_jacobian_information_dimensions: ";
    if(test_jacobian_information_dimensions){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout<< "test_sparse_data_size: ";
    if(test_sparse_data_size){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout<< "test_sparse_data_bounds: ";
    if(test_sparse_data_bounds){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout<< "test_repeated_entries: ";
    if(test_repeated_entries){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout<< "test_nnz_in_every_row_column: ";
    if(test_nnz_in_every_row_column){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_deterministic: ";
    if(test_deterministic){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}
    std::cout << '\n';
  }
  
 protected:
  const Generator<Matrix_free_sparse_information> generator;
  bool test_jacobian_information_size;
  bool test_jacobian_information_bounds;
  bool test_jacobian_information_dimensions;
  bool test_deterministic;
  bool test_sparse_data_size;
  bool test_sparse_data_bounds;
  bool test_repeated_entries;
  bool test_nnz_in_every_row_column;
  bool were_all_test_successful;

  bool rhs_output_dim_equal_lhs_input_dim(){
    auto jacobian_information = generator.get_jacobian_information_reference();
    for(std::size_t i = 0; i < jacobian_information.size() - 1; i++){
      if(jacobian_information[i].codomain_dimension() !=
          jacobian_information[i+1].domain_dimension()){
        return false;
      }
    }
    return true;
  }

  bool is_jacobian_information_within_bounds(std::size_t dimension_lb,
      std::size_t dimension_ub, std::size_t n_E_lb, std::size_t n_E_ub){
    auto jacobian_information = generator.get_jacobian_information_reference();
    std::size_t available_entries;
    std::size_t min_number_entries;
    std::size_t domain_dimension, codomain_dimension;
    for(std::size_t i = 0; i < jacobian_information.size(); i++){
      if((dimension_ub < jacobian_information[i].domain_dimension()) ||
          (dimension_ub < jacobian_information[i].codomain_dimension()) ||
          (jacobian_information[i].domain_dimension() < dimension_lb) ||
          (jacobian_information[i].codomain_dimension() < dimension_lb) ||
          (n_E_ub < jacobian_information[i].number_of_edges()) ||
          (jacobian_information[i].number_of_edges() < n_E_lb)){
        return false;
      }

      domain_dimension = jacobian_information[i].domain_dimension();
      codomain_dimension = jacobian_information[i].codomain_dimension();

      available_entries = domain_dimension * codomain_dimension;
      min_number_entries = (domain_dimension < codomain_dimension) ?
        codomain_dimension : domain_dimension;

      //Test checks if the number of non zeros lay between
      //between the maximum number of columns and rows
      //and the number of available entries.
      if((jacobian_information[i].number_of_nonzeros() < min_number_entries) ||
          (available_entries < jacobian_information[i].number_of_nonzeros())){
        return false;
      }
    }
    return true;
  }

  bool is_sparse_data_within_bounds(){
    auto sparse_data = generator.get_sparse_data_reference();
    auto jacobian_information = generator.get_jacobian_information_reference();
  
    if(sparse_data.size() != jacobian_information.size()){
      return false;
    }
    for(std::size_t i = 0; i < sparse_data.size(); i++){
      for(std::size_t j = 0; j < sparse_data[i].size(); j++){

        if((jacobian_information[i].codomain_dimension() <= sparse_data[i][j].row()) ||
            (jacobian_information[i].domain_dimension() <= sparse_data[i][j].col())){
          return false;
        }

      }
    }
    return true;
  }

  bool are_there_repeated_entries(){
    auto sparse_data_copy = generator.get_sparse_data_copy();
    std::sort(sparse_data_copy.begin(), sparse_data_copy.end());
    auto it =
      std::unique(sparse_data_copy.begin(), sparse_data_copy.end());
    return it == sparse_data_copy.end();
  }

  bool all_columns_rows_have_nnz(){
    auto jacobian_information = generator.get_jacobian_information_reference();
    auto sparse_data = generator.get_sparse_data_reference();

    if(jacobian_information.size() != sparse_data.size()){
      return false;
    }

    for(std::size_t i = 0; i < generator.sparse_data_size(); i++){
      //Checks that every rows has a non zero entry.
      for(std::size_t row = 0; row < jacobian_information[i].codomain_dimension();
          row++){
        if(std::find_if(sparse_data[i].begin(), sparse_data[i].end(),
              [row](const NNZ& nnz){ return nnz.row() == row;}) ==
            sparse_data[i].end()){
          return false;
        }
      }

      //Cheks that every column has a non zero entry.
      for(std::size_t column = 0; column < jacobian_information[i].domain_dimension();
          column++){
        if(std::find_if(sparse_data[i].begin(), sparse_data[i].end(),
              [column](const NNZ& nnz){ return nnz.col() == column;}) ==
            sparse_data[i].end()){
          return false;
        }
      }
    } 
    return true;
  }

  bool is_number_of_nnz_correct(){
    auto sparse_data = generator.get_sparse_data_reference();
    auto jacobian_information = generator.get_jacobian_information_reference();
    if(sparse_data.size() != jacobian_information.size()){
      return false;
    }

    for(std::size_t jac_idx= 0; jac_idx< sparse_data.size(); jac_idx++){
      if(sparse_data[jac_idx].size() != jacobian_information[jac_idx].number_of_nonzeros()){
        return false;
      }
    }
    return true;
  }

  bool is_information_data_equal(
      const std::vector<Matrix_free_sparse_information>& jacobian_information_0,
      const std::vector<Matrix_free_sparse_information>& jacobian_information_1){

    if(jacobian_information_0.size() != jacobian_information_1.size()){
      return false;
    }

    for(std::size_t i = 0; i < jacobian_information_0.size(); i++){
      if((jacobian_information_0[i].domain_dimension() != 
          jacobian_information_1[i].domain_dimension()) ||
          (jacobian_information_0[i].codomain_dimension() !=
           jacobian_information_1[i].codomain_dimension()) ||
          (jacobian_information_0[i].number_of_edges() !=
           jacobian_information_1[i].number_of_edges())){
        return false;
      }
    }
    return true;
  }
  
  bool is_sparse_data_equal(
      const std::vector<std::vector<NNZ>>& sparse_information_0,
      const std::vector<std::vector<NNZ>>& sparse_information_1){

    if(sparse_information_0.size() != sparse_information_1.size()){
      return false;
    }

    for(std::size_t jac_idx = 0; jac_idx < sparse_information_0.size(); jac_idx++){
        if(sparse_information_0[jac_idx].size() != sparse_information_1[jac_idx].size()){
          return false;
        }
      for(std::size_t nnz_idx = 0; nnz_idx < sparse_information_0[jac_idx].size(); nnz_idx++){
        if(!(sparse_information_0[jac_idx][nnz_idx] == sparse_information_1[jac_idx][nnz_idx])){
          return false;
        }
      }
    }
    return true;
  }

  bool is_deterministic(std::size_t chain_length, std::size_t dimension_lb,
      std::size_t dimension_ub, std::size_t n_E_lb, std::size_t n_E_ub,
      double density_lb, double density_ub, std::size_t seed){
    bool is_deterministic_ = true;
    Generator<Matrix_free_sparse_information> gen_0{chain_length, dimension_lb, dimension_ub,
      n_E_lb, n_E_ub, density_lb, density_ub, is_deterministic_, seed};
    auto jacobian_information_0 = gen_0.get_jacobian_information_reference();
    auto sparse_information_0 = gen_0.get_sparse_data_reference();

    Generator<Matrix_free_sparse_information> gen_1{chain_length, dimension_lb, dimension_ub,
    n_E_lb, n_E_ub, density_lb, density_ub, is_deterministic_, seed};
    auto jacobian_information_1 = gen_1.get_jacobian_information_reference();
    auto sparse_information_1 = gen_1.get_sparse_data_reference();

    if(!is_information_data_equal(jacobian_information_0, jacobian_information_1)){
      return false;
    }
    if(!is_sparse_data_equal(sparse_information_0, sparse_information_1)){
      return false;
    }
    return true;
  }

};
#endif
