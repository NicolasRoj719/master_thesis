#ifndef TEST_GENERATOR_HPP
#define TEST_GENERATOR_HPP
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "util_structs.hpp"
#include "generator.hpp"


template<class Jacobian_info_T>
class test_Generator{
 public:
  test_Generator(std::size_t chain_length, std::size_t dimension_lb,
          std::size_t dimension_ub, bool is_deterministic, std::size_t seed):
      test_jacobian_dimension_consistency(false), test_jacobian_data_bounds(false),
      test_deterministic(false), test_no_repeated_entries(false),
      test_nnz_number(false), test_all_columns_and_rows_have_nnz(false),
      were_all_test_successful(false){

        Generator<Jacobian_info_T>
            generator{chain_length, dimension_lb, dimension_ub, is_deterministic, seed}; 

        auto jacobian_information = generator.generate_jacobian_information();

        test_jacobian_dimension_consistency =
           input_output_dimension_check(jacobian_information);

        test_jacobian_data_bounds =
           is_generated_data_within_bounds(jacobian_information, dimension_lb, dimension_ub);

        test_deterministic =
           test_is_deterministic(chain_length, dimension_lb, dimension_ub, seed);

        if(test_jacobian_dimension_consistency && test_jacobian_data_bounds &&
            test_deterministic){

          were_all_test_successful = true;
        }
        
        else{

          were_all_test_successful = false;
          print_test_state();
        }
    }

  test_Generator(std::size_t chain_length, std::size_t dimension_lb,
          std::size_t dimension_ub, std::size_t number_edges_lb,
          std::size_t number_edges_ub, bool is_deterministic,
          std::size_t seed):
      test_jacobian_dimension_consistency(false), test_jacobian_data_bounds(false),
      test_deterministic(false), test_no_repeated_entries(false),
      test_nnz_number(false), test_all_columns_and_rows_have_nnz(false),
      were_all_test_successful(false){

        Generator<Jacobian_info_T>
            generator{chain_length, dimension_lb, dimension_ub,
            number_edges_lb, number_edges_ub, is_deterministic, seed};

        auto jacobian_information = generator.generate_jacobian_information();

        test_jacobian_dimension_consistency =
           input_output_dimension_check(jacobian_information);

        test_jacobian_data_bounds =
           is_generated_data_within_bounds(jacobian_information, dimension_lb, dimension_ub,
                   number_edges_lb, number_edges_ub);

        test_deterministic =
           test_is_deterministic(chain_length, dimension_lb, dimension_ub,
                   number_edges_lb, number_edges_ub, seed);

        if(test_jacobian_dimension_consistency && test_jacobian_data_bounds &&
            test_deterministic){

          were_all_test_successful = true;
        }

        else{
          
          were_all_test_successful = false;
          print_test_state();
        }
    }

  test_Generator(std::size_t chain_length, std::size_t dimension_lb,
          std::size_t dimension_ub, std::size_t number_edges_lb,
          std::size_t number_edges_ub, double density_lb, double density_ub,
          bool is_deterministic_, std::size_t seed):
      test_jacobian_dimension_consistency(false), test_jacobian_data_bounds(false),
      test_deterministic(false), test_no_repeated_entries(false), 
      test_nnz_number(false), test_all_columns_and_rows_have_nnz(false),
      were_all_test_successful(false){

        Generator<Jacobian_info_T>
            generator{chain_length, dimension_lb, dimension_ub, number_edges_lb,
            number_edges_ub, density_lb, density_ub, is_deterministic_, seed};

        auto data = generator.generate_data();

        test_jacobian_dimension_consistency =
           input_output_dimension_check(data.jacobian_information);

        test_jacobian_data_bounds =
           is_generated_data_within_bounds(data.jacobian_information, dimension_lb,
                dimension_ub, number_edges_lb, number_edges_ub);

        test_no_repeated_entries = 
            !are_there_repeated_entries(data.sparse_data);

        test_nnz_number =
            is_number_of_nnz_correct(data.jacobian_information, data.sparse_data);

        test_all_columns_and_rows_have_nnz = 
            all_columns_rows_have_nnz(data.jacobian_information, data.sparse_data);
            
        test_deterministic =
            test_is_deterministic(chain_length, dimension_lb, dimension_ub,
                number_edges_lb, number_edges_ub, density_lb, density_ub, seed);

        if(test_jacobian_dimension_consistency && test_jacobian_data_bounds &&
            test_no_repeated_entries && test_nnz_number &&
            test_all_columns_and_rows_have_nnz && test_deterministic){

          were_all_test_successful = true;
        }

        else{

          were_all_test_successful = false;
          print_test_state();
        }
    }

  void print_test_state();

 protected:

  bool test_jacobian_dimension_consistency;

  bool test_jacobian_data_bounds;

  bool test_deterministic;

  bool test_no_repeated_entries;

  bool test_nnz_number;

  bool test_all_columns_and_rows_have_nnz;

  bool were_all_test_successful;

  bool input_output_dimension_check(const std::vector<Jacobian_info_T>& information);

  bool is_generated_data_within_bounds(const std::vector<Jacobian_info_T>& information,
            std::size_t dimension_lb, std::size_t dimension_ub);

  bool is_generated_data_within_bounds(const std::vector<Jacobian_info_T>& information,
            std::size_t dimension_lb, std::size_t dimension_ub,
            std::size_t number_edges_lb, std::size_t number_edges_ub);

  bool is_jacobian_information_equal(const std::vector<Jacobian_info_T>& information_0,
            const std::vector<Jacobian_info_T>& information_1);

  bool is_sparse_data_equal(const std::vector<std::vector<NNZ>>& information_0,
            const std::vector<std::vector<NNZ>>& information_1);

  bool is_number_of_nnz_correct(
        const std::vector<Jacobian_info_T>& jacobian_information,
        const std::vector<std::vector<NNZ>>& sparse_data);

  bool are_there_repeated_entries(std::vector<NNZ> sparse_data);

  bool are_there_repeated_entries(const std::vector<std::vector<NNZ>>& sparse_data);

  bool all_columns_rows_have_nnz(
      const std::vector<Jacobian_info_T>& jacobian_information,
      const std::vector<std::vector<NNZ>>& sparse_data);

  bool is_sparse_data_within_bounds(
      const std::vector<Jacobian_info_T>& jacobian_information,
      const std::vector<std::vector<NNZ>>& sparse_data);

  bool test_is_deterministic(std::size_t chain_length, std::size_t dimension_lb,
      std::size_t dimension_ub, std::size_t seed);

  bool test_is_deterministic(std::size_t chain_length, std::size_t dimension_lb,
      std::size_t dimension_ub, std::size_t number_edges_lb,
      std::size_t number_edges_ub, std::size_t seed);

  bool test_is_deterministic(std::size_t chain_length, std::size_t dimension_lb,
      std::size_t dimension_ub, std::size_t number_edges_lb,
      std::size_t number_edges_ub, double density_lb, double density_ub,
      std::size_t seed);

};

template<class Jacobian_info_T>
bool test_Generator<Jacobian_info_T>::input_output_dimension_check(
        const std::vector<Jacobian_info_T>& information){
    
    for(std::size_t index= 0; index < information.size() - 1; index++){

        if(information[index].codomain_dimension() !=
            information[index+1].domain_dimension()){

            return false;
        }
    }

    return true;
}

template<class Jacobian_info_T>
bool test_Generator<Jacobian_info_T>::is_generated_data_within_bounds(
        const std::vector<Jacobian_info_T>& information,
        std::size_t dimension_lb, std::size_t dimension_ub){

    for(std::size_t index = 0; index < information.size(); index++){

      if(information[index].domain_dimension() < dimension_lb ||
          information[index].codomain_dimension() > dimension_ub ||
          information[index].codomain_dimension() < dimension_lb ||
          information[index].domain_dimension() > dimension_ub){

        return false;
      }
    }

    return true;
}

template<class Jacobian_info_T>
bool test_Generator<Jacobian_info_T>::is_generated_data_within_bounds(
    const std::vector<Jacobian_info_T>& information, std::size_t dimension_lb,
    std::size_t dimension_ub, std::size_t number_edges_lb, std::size_t number_edges_ub){

  for(std::size_t idx = 0; idx < information.size(); idx++){

    if(information[idx].domain_dimension() < dimension_lb ||
        information[idx].domain_dimension() > dimension_ub ||
        information[idx].codomain_dimension() < dimension_lb ||
        information[idx].codomain_dimension() > dimension_ub ||
        information[idx].number_of_edges() < number_edges_lb ||
        information[idx].number_of_edges() > number_edges_ub){
      
      return false;
    }

    if constexpr(std::is_same_v<Jacobian_info_T, Matrix_free_sparse_information>){

      if((information[idx].number_of_nonzeros() < 
          std::min(information[idx].domain_dimension(),information[idx].codomain_dimension())) ||
          (information[idx].number_of_nonzeros() > 
           information[idx].domain_dimension() * information[idx].codomain_dimension())){
        
        return false;
      }
    }
  }

  return true;
}

template<class Jacobian_info_T>
bool test_Generator<Jacobian_info_T>::is_jacobian_information_equal(
    const std::vector<Jacobian_info_T>& information_0,
    const std::vector<Jacobian_info_T>& information_1){

  if(information_0.size() != information_1.size()){

    return false;
  }

  for(std::size_t index = 0; index < information_0.size(); index++){

    if constexpr(std::is_same_v<Jacobian_info_T, Jacobian_information>){

      if(information_0[index].domain_dimension() != information_1[index].domain_dimension() ||
          information_0[index].codomain_dimension() !=information_1[index].codomain_dimension()){

        return false;
      }
    }

    else if constexpr(std::is_same_v<Jacobian_info_T, Matrix_free_information>){

      if(information_0[index].domain_dimension() != information_1[index].domain_dimension() ||
          information_0[index].codomain_dimension() !=information_1[index].codomain_dimension() ||
          information_0[index].number_of_edges() != information_1[index].number_of_edges()){

        return false;
      }
    }

    else{

      if(information_0[index].domain_dimension() != information_1[index].domain_dimension() ||
          information_0[index].codomain_dimension() !=information_1[index].codomain_dimension() ||
          information_0[index].number_of_edges() != information_1[index].number_of_edges() ||
          information_0[index].number_of_nonzeros() != information_1[index].number_of_nonzeros()){

        return false;
      }
    }
  }

  return true;
}

template<class Jacobian_info_T>
bool test_Generator<Jacobian_info_T>::is_sparse_data_equal(
    const std::vector<std::vector<NNZ>>& information_0,
    const std::vector<std::vector<NNZ>>& information_1){

  if(information_0.size() != information_1.size()){

    return false;
  }

  for(std::size_t o_index = 0; o_index < information_0.size(); o_index++){

    if(information_0[o_index].size() != information_1[o_index].size()){
      
      return false;
    }

    for(std::size_t i_index = 0; i_index < information_0[o_index].size(); i_index++){

      if(!(information_0[o_index][i_index] == information_1[o_index][i_index])){

        return false;
      }
    }
  }

  return true;
}

template<class Jacobian_info_T>
bool test_Generator<Jacobian_info_T>::is_number_of_nnz_correct(
    const std::vector<Jacobian_info_T>& jacobian_information,
    const std::vector<std::vector<NNZ>>& sparse_data){

  if(sparse_data.size() != jacobian_information.size()){

    return false;
  }

  for(std::size_t index = 0; index < sparse_data.size(); index++){
    
    if(sparse_data[index].size() != jacobian_information[index].number_of_nonzeros()){

      return false;
    }
  }

  return true;
}

template<class Jacobian_info_T>
bool test_Generator<Jacobian_info_T>::are_there_repeated_entries(
    std::vector<NNZ> sparse_data){
  
  std::sort(sparse_data.begin(), sparse_data.end());

  auto it = std::unique(sparse_data.begin(), sparse_data.end());

  return !(it == sparse_data.end());
}

template<class Jacobian_info_T>
bool test_Generator<Jacobian_info_T>::are_there_repeated_entries(
    const std::vector<std::vector<NNZ>>& sparse_data){

  for(std::size_t index = 0; index < sparse_data.size(); index++){

    if(are_there_repeated_entries(sparse_data[index])){

      return true;
    }
  }

  return false;
}

template<class Jacobian_info_T>
bool test_Generator<Jacobian_info_T>::is_sparse_data_within_bounds(
    const std::vector<Jacobian_info_T>& jacobian_information,
    const std::vector<std::vector<NNZ>>& sparse_data){

  for(std::size_t o_index = 0; o_index < sparse_data.size(); o_index++){

    for(std::size_t i_index = 0; i_index < sparse_data[o_index].size(); i_index++){

      if((jacobian_information[o_index].codomain_dimension() <= 
            sparse_data[o_index][i_index].row()) ||
          (jacobian_information[o_index].domain_dimension() <=
            sparse_data[o_index][i_index].col())){

        return false;
      }
    }
  }

  return true;
}

template<class Jacobian_info_T>
bool test_Generator<Jacobian_info_T>::all_columns_rows_have_nnz(
    const std::vector<Jacobian_info_T>& jacobian_information,
    const std::vector<std::vector<NNZ>>& sparse_data){

  for(std::size_t o_index= 0; o_index < sparse_data.size(); o_index++){

    //Checks that every row has a non zero entry.
    for(std::size_t row = 0; row < jacobian_information[o_index].codomain_dimension(); row++){

      if(std::find_if(sparse_data[o_index].begin(), sparse_data[o_index].end(),
            [row](const NNZ& nnz){return nnz.row() == row;}) == sparse_data[o_index].end()){

        return false;
      }
    }

    //Checks that every column has a non zero entry 
    for(std::size_t column = 0; column < jacobian_information[o_index].domain_dimension();
        column++){

      if(std::find_if(sparse_data[o_index].begin(), sparse_data[o_index].end(),
            [column](const NNZ& nnz){return nnz.col() == column;}) == sparse_data[o_index].end()){
        
        return false;
      }
    }
  }

  return true;
}

template<class Jacobian_info_T>
bool test_Generator<Jacobian_info_T>::test_is_deterministic(std::size_t chain_length,
    std::size_t dimension_lb, std::size_t dimension_ub, std::size_t seed){

  Generator<Jacobian_info_T>
    generator_0{chain_length, dimension_lb, dimension_ub, true, seed};

  auto jacobian_information_0 = generator_0.generate_jacobian_information();

  Generator<Jacobian_info_T>
    generator_1{chain_length, dimension_lb, dimension_ub, true, seed};

  auto jacobian_information_1 = generator_1.generate_jacobian_information();

  return is_jacobian_information_equal(jacobian_information_0,
                                        jacobian_information_1);
}

template<class Jacobian_info_T>
bool test_Generator<Jacobian_info_T>::test_is_deterministic(
    std::size_t chain_length, std::size_t dimension_lb, std::size_t dimension_ub,
    std::size_t number_edges_lb, std::size_t number_edges_ub, std::size_t seed){

  Generator<Jacobian_info_T>
    generator_0{chain_length, dimension_lb, dimension_ub, number_edges_lb,
    number_edges_ub, true, seed};

  Generator<Jacobian_info_T>
    generator_1{chain_length, dimension_lb, dimension_ub, number_edges_lb,
    number_edges_ub, true, seed};

  auto jacobian_information_0 = generator_0.generate_jacobian_information();
  auto jacobian_information_1 = generator_1.generate_jacobian_information();

  return is_jacobian_information_equal(jacobian_information_0,
                                        jacobian_information_1); 
}

template<class Jacobian_info_T>
bool test_Generator<Jacobian_info_T>::test_is_deterministic(
    std::size_t chain_length, std::size_t dimension_lb, std::size_t dimension_ub,
    std::size_t number_edges_lb, std::size_t number_edges_ub, double density_lb,
    double density_ub, std::size_t seed){

  Generator<Jacobian_info_T>
    generator_0{chain_length, dimension_lb, dimension_ub, number_edges_lb,
    number_edges_ub, density_lb, density_ub, true, seed};

  Generator<Jacobian_info_T>
    generator_1{chain_length, dimension_lb, dimension_ub, number_edges_lb,
    number_edges_ub, density_lb, density_ub, true, seed};

  auto data_generator_0 = generator_0.generate_data();
  auto data_generator_1 = generator_1.generate_data();

  if(!is_jacobian_information_equal(data_generator_0.jacobian_information,
                                    data_generator_1.jacobian_information)){

    return false;
  }

  if(!is_sparse_data_equal(data_generator_0.sparse_data,
                            data_generator_1.sparse_data)){
    return false;
  }

  return true;
}

template<class Jacobian_info_T>
void test_Generator<Jacobian_info_T>::print_test_state(){

  if constexpr(std::is_same_v<Jacobian_info_T, Matrix_free_sparse_information>){

    std::cout << "State of test_Generator sparse:\n";

    std::cout << "test_jacobian_dimension_consistency: ";
      if(test_jacobian_dimension_consistency){
        std::cout << "successful.\n";
      }
      else{std::cout << "failed.\n";}

    std::cout << "test_jacobian_data_bounds: ";
      if(test_jacobian_data_bounds){
        std::cout << "successful.\n";
      }
      else{std::cout << "failed.\n";}

    std::cout << "test_no_repeated_entries: ";
      if(test_no_repeated_entries){
        std::cout << "successful.\n";
      }
      else{std::cout << "failed.\n";}

    std::cout << "test_nnz_number: ";
      if(test_nnz_number){
        std::cout << "successful.\n";
      }
      else{std::cout << "failed.\n";}

    std::cout << "test_all_columns_and_rows_have_nnz: ";
      if(test_all_columns_and_rows_have_nnz){
        std::cout << "successful.\n";
      }
      else{std::cout << "failed.\n";}

    std::cout << "test_deterministic: ";
      if(test_deterministic){
        std::cout << "successful.\n";
      }
      else{std::cout << "failed.\n";}

    std::cout << "were_all_test_successful: ";
      if(were_all_test_successful){
        std::cout << "successful.\n";
      }
      else{std::cout << "failed.\n";}
  }

  else{

    if constexpr(std::is_same_v<Jacobian_info_T, Jacobian_information>){

      std::cout << "State of test_Generator basic jacobian:\n";
    }

    else{

      std::cout << "State of test_Generator dense:\n";
    }

    std::cout << "test_jacobian_dimension_consistency: ";
      if(test_jacobian_dimension_consistency){
        std::cout << "successful.\n";
      }
      else{std::cout << "failed.\n";}

    std::cout << "test_jacobian_data_bounds: ";
      if(test_jacobian_data_bounds){
        std::cout << "successful.\n";
      }
      else{std::cout << "failed.\n";}

    std::cout << "test_deterministic: ";
      if(test_deterministic){
        std::cout << "successful.\n";
      }
      else{std::cout << "failed.\n";}

    std::cout << "were_all_test_successful: ";
      if(were_all_test_successful){
        std::cout << "successful.\n";
      }
      else{std::cout << "failed.\n";}
  }
}

#endif
