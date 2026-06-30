#include <iostream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <string>
#include "./class_test_jacobian.hpp"
#include "./../jacobian.hpp"

int main(){
  {
    const std::size_t domain_dimension = 3, codomain_dimension = 7;
    test_Jacobian test{domain_dimension, codomain_dimension};
    test.print_test_state();
  }
  
  {
    const std::size_t domain_dimension = 3, codomain_dimension = 7;
    const std::size_t number_of_edges = 150;
    test_Dense_Jacobian test{domain_dimension, codomain_dimension, number_of_edges};
    test.print_test_state();
  }
  
  {
    const std::size_t domain_dimension = 3, codomain_dimension = 7;
    const std::size_t number_of_edges = 150;
    test_Dense_Jacobian test{domain_dimension, codomain_dimension, number_of_edges};
    test.print_test_state();
  }

  {
    const std::string case_name = "case_0";
    test_Sparse_Jacobian test{case_name};
    test.print_test_state();
  }

  {
    const std::string case_name = "case_1";
    test_Sparse_Jacobian test{case_name};
    test.print_test_state();
  }

  {
    const std::string case_name = "case_2";
    test_Sparse_Jacobian test{case_name};
    test.print_test_state();
  }

}
