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
    /* test.print_test_state(); */
  }
  
  {
    const std::size_t domain_dimension = 3, codomain_dimension = 7;
    const std::size_t number_of_edges = 150;
    test_Dense_Jacobian test{domain_dimension, codomain_dimension, number_of_edges};
    /* test.print_test_state(); */
  }
  
  {
    const std::size_t domain_dimension = 3, codomain_dimension = 7;
    const std::size_t number_of_edges = 150;
    test_Dense_Jacobian test{domain_dimension, codomain_dimension, number_of_edges};
    /* test.print_test_state(); */
  }

  {
    std::size_t domain_dimension = 3, codomain_dimension = 7;
    std::size_t number_of_edges = 150, function_cost = 223;
    test_split_dense_Jacobian 
      test{domain_dimension, codomain_dimension, number_of_edges, function_cost};

    test.print_test_state();
  }

  {
    const std::string case_name = "case_0";
    test_Sparse_Jacobian test{case_name};
    /* test.print_test_state(); */
  }

  {
    const std::string case_name = "case_1";
    test_Sparse_Jacobian test{case_name};
    /* test.print_test_state(); */
  }

  {
    const std::string case_name = "case_2";
    test_Sparse_Jacobian test{case_name};
    /* test.print_test_state(); */
  }

  {
    const std::string lhs_case = "case_0_MxM_lhs";
    test_Sparse_Jacobian test_lhs{lhs_case};
    /* test_lhs.print_test_state(); */

    const std::string rhs_case = "case_0_MxM_rhs";
    test_Sparse_Jacobian test_rhs{rhs_case};
    /* test_rhs.print_test_state(); */

    const std::string product_case = "case_0_MxM";
    test_Sparse_Jacobian 
      test_product{product_case,
        test_lhs.get_sparse_jacobian(), test_rhs.get_sparse_jacobian()};
    /* test_product.print_test_state(); */
  }

  {
    const std::string lhs_case = "case_1_MxM_lhs";
    test_Sparse_Jacobian test_lhs{lhs_case};
    /* test_lhs.print_test_state(); */

    const std::string rhs_case = "case_1_MxM_rhs";
    test_Sparse_Jacobian test_rhs{rhs_case};
    /* test_rhs.print_test_state(); */

    const std::string product_case = "case_1_MxM";
    test_Sparse_Jacobian 
      test_product{product_case,
        test_lhs.get_sparse_jacobian(), test_rhs.get_sparse_jacobian()};
    /* test_product.print_test_state(); */
  }

  {
    const std::string lhs_case = "case_2_MxM_lhs";
    test_Sparse_Jacobian test_lhs{lhs_case};
    /* test_lhs.print_test_state(); */

    const std::string rhs_case = "case_2_MxM_rhs";
    test_Sparse_Jacobian test_rhs{rhs_case};
    /* test_rhs.print_test_state(); */

    const std::string product_case = "case_2_MxM";
    test_Sparse_Jacobian 
      test_product{product_case,
        test_lhs.get_sparse_jacobian(), test_rhs.get_sparse_jacobian()};
    /* test_product.print_test_state(); */
  }

  {
    std::string path_to_file = "./sparse_jacobian_sample_tests/case_1";
    std::size_t function_cost = 234;
    test_split_sparse_Jacobian test{path_to_file, function_cost};

    test.print_test_state();
  }

}
