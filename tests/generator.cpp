#include <cstdint>
#include <iostream>
#include <stdexcept>
/* #include "./class_test_playground.hpp" */
#include "./class_test_generator.hpp"
#include "./../generator.hpp"
/* #include "./../playground.hpp" */

int main(){
  {
    const std::size_t chain_length = 3;
    const std::size_t dimension_lower_bound = 4, dimension_upper_bound = 10;
    const bool is_deterministic = 0;
    const std::size_t seed = 60;
    test_Generator<Jacobian_information> 
      test{chain_length, dimension_lower_bound, dimension_upper_bound, is_deterministic, seed};
    /* test.print_test_state(); */
  } 

  {
    const std::size_t chain_length = 10;
    const std::size_t dimension_lower_bound = 9, dimension_upper_bound = 20;
    const bool is_deterministic = 1;
    const std::size_t seed = 60;
    test_Generator<Jacobian_information> 
      test{chain_length, dimension_lower_bound, dimension_upper_bound, is_deterministic, seed};
    /* test.print_test_state(); */
  } 

  {
    const std::size_t chain_length = 8;
    const std::size_t dimension_lower_bound = 6, dimension_upper_bound = 20;
    const std::size_t number_edges_lower_bound = 100;
    const std::size_t number_edges_upper_bound = 200;
    const bool is_deterministic = 0;
    const std::size_t seed = 60;
    test_Generator<Matrix_free_information>
      test{chain_length, dimension_lower_bound, dimension_upper_bound,
      number_edges_lower_bound, number_edges_upper_bound,
      is_deterministic, seed};
    /* test.print_test_state(); */
  }

  {
    const std::size_t chain_length = 8;
    const std::size_t dimension_lower_bound = 6, dimension_upper_bound = 20;
    const std::size_t number_edges_lower_bound = 100;
    const std::size_t number_edges_upper_bound = 200;
    const double density_lower_bound = 0.25, density_upper_bound = 0.5;
    const bool is_deterministic = 0;
    const std::size_t seed = 60;
    test_Generator<Matrix_free_sparse_information>
      test{chain_length, dimension_lower_bound, dimension_upper_bound, number_edges_lower_bound,
          number_edges_upper_bound, density_lower_bound, density_upper_bound, is_deterministic,
          seed};
    /* test.print_test_state(); */
  }

  {
    const std::size_t chain_length = 12;
    const std::size_t dimension_lower_bound = 4, dimension_upper_bound = 15;
    const std::size_t number_edges_lower_bound = 50;
    const std::size_t number_edges_upper_bound = 150;
    const double density_lower_bound = 0.25, density_upper_bound = 0.5;
    const bool is_deterministic = 0;
    const std::size_t seed = 20;
    test_Generator<Matrix_free_sparse_information>
      test{chain_length, dimension_lower_bound, dimension_upper_bound, number_edges_lower_bound,
          number_edges_upper_bound, density_lower_bound, density_upper_bound, is_deterministic,
          seed};
    /* test.print_test_state(); */
  }

}
