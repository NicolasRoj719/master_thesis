#include <string>
/* #include "./playground.hpp" */
#include "./class_test_chain.hpp"
#include "./../chain.hpp"
/* #include "./../test.hpp" */
#include "./../jacobian.hpp"
#include "./../util_structs.hpp"

int main(){

  {
    const std::string case_name = "case_0";
    const std::string validate_name = case_name;
    test_chain<Jacobian, Jacobian_information> test{case_name, validate_name};
    /* test.print_test_state(); */
  }

  {
    const std::string case_name = "case_1";
    const std::string validate_name = case_name;
    test_chain<Jacobian, Jacobian_information> test{case_name, validate_name};
    /* test.print_test_state(); */
  }

  {
    const std::string case_name = "case_0";
    const std::string validate_name = case_name;
    test_chain<Dense_Jacobian, Matrix_free_information> test{case_name, validate_name};
    /* test.print_test_state(); */
  }

  {
    const std::string case_name = "case_1";
    const std::string validate_name = case_name;
    test_chain<Dense_Jacobian, Matrix_free_information> test{case_name, validate_name};
    /* test.print_test_state(); */
  }

  {
    test_chain<Jacobian, Jacobian_information> test{};
    /* test.print_test_state(); */
  }

  {
    test_chain<Dense_Jacobian, Matrix_free_information> test{};
    /* test.print_test_state(); */
  }

  {
    const std::string case_name = "case_0";
    const std::string validate_name = case_name;
    test_chain<Sparse_Jacobian, Matrix_free_sparse_information>
      test{case_name, validate_name};
    /* test.print_test_state(); */
  }

  {
    const std::string case_name = "case_1";
    const std::string validate_name = case_name;
    test_chain<Sparse_Jacobian, Matrix_free_sparse_information>
      test{case_name, validate_name};
    /* test.print_test_state(); */
  }

  {
    test_chain<Sparse_Jacobian, Matrix_free_sparse_information>
      test{};
    /* test.print_test_state(); */
  }

  return 0;
}
