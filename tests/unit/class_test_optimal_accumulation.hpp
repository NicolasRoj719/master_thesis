#include <iostream>
#include <optional>

#include "optimal_accumulation.hpp"
class test_optimal_accumulation{
 public:
  test_optimal_accumulation(){}

  static bool check_node_information(const Node_matrix_free& node, std::size_t j_index,
      std::optional<std::size_t> i_index, Operation operation, std::size_t level){
    
    if(node.j_index() != j_index || node.i_index() != i_index ||
        node.operation() != operation || node.level() != level){
        
      return false;
    }

    return true;
  }

  static bool check_node_information(const Node_binomial_checkpointing& node, std::size_t j_index,
      std::size_t i_index, std::size_t number_checkpoints, std::size_t level){

    if(node.j_index() != j_index || node.i_index() != i_index ||
        node.number_checkpoints() != number_checkpoints || node.level() != level){

      return false;
    }

    return true;
  }

  static bool check_node_information(const Node_jacobian& node, std::size_t j_index,
      std::optional<std::size_t> i_index, std::size_t level){

    if(node.j_index() != j_index || node.i_index() != i_index || node.level() != level){

      return false;
    }

    return true;
  }
  
  void print_test_state(){
    std::cout << "State of test optimal accumulation:\n";

    std::cout << "test_matrix_free_homogeneous_tangent:";
    if(test_matrix_free_homogeneous_tangent){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_matrix_free_homogeneous_adjoint:";
    if(test_matrix_free_homogeneous_adjoint){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_matrix_free_multiplication:";
    if(test_matrix_free_multiplication){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_matrix_free_test_case_0:";
    if(test_matrix_free_test_case_0){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_matrix_free_test_case_1:";
    if(test_matrix_free_test_case_1){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_jacobian_case_0:";
    if(test_jacobian_case_0){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}
    
    std::cout << "test_jacobian_case_1:";
    if(test_jacobian_case_1){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_jacobian_case_2:";
    if(test_jacobian_case_2){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_binomial_case_0:";
    if(test_binomial_case_0){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_binomial_case_1:";
    if(test_binomial_case_1){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_binomial_case_2:";
    if(test_binomial_case_2){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}
  }

  //test state setters
  void set_matrix_free_homogeneous_tangent_result(bool test_result){
    
    test_matrix_free_homogeneous_tangent = test_result;
  }

  void set_matrix_free_homogeneous_adjoint_result(bool test_result){

    test_matrix_free_homogeneous_adjoint = test_result;
  }

  void set_matrix_free_multiplication_result(bool test_result){

    test_matrix_free_multiplication = test_result;
  }

  void set_matrix_free_test_case_0(bool test_result){

    test_matrix_free_test_case_0 = test_result;
  }

  void set_matrix_free_test_case_1(bool test_result){
    
    test_matrix_free_test_case_1 = test_result;
  }

  void set_jacobian_test_case_0(bool test_result){
    
    test_jacobian_case_0 = test_result;
  }

  void set_jacobian_test_case_1(bool test_result){
    
    test_jacobian_case_1 = test_result;
  }

  void set_jacobian_test_case_2(bool test_result){
    
    test_jacobian_case_2 = test_result;
  }

  void set_binomial_case_0(bool test_result){

    test_binomial_case_0 = test_result;
  }

  void set_binomial_case_1(bool test_result){

    test_binomial_case_1 = test_result;
  }

  void set_binomial_case_2(bool test_result){
    
    test_binomial_case_2 = test_result;
  }

  void were_all_tests_successful(){
    if(test_matrix_free_homogeneous_tangent && test_matrix_free_homogeneous_adjoint &&
        test_matrix_free_multiplication && test_matrix_free_test_case_0 &&
        test_matrix_free_test_case_1 && test_jacobian_case_0 && 
        test_jacobian_case_1 && test_jacobian_case_2 && test_binomial_case_0 &&
        test_binomial_case_1 && test_binomial_case_2){}

    else{print_test_state();}
  }

 
 protected:
  bool test_matrix_free_homogeneous_tangent;
  bool test_matrix_free_homogeneous_adjoint;
  bool test_matrix_free_multiplication;
  bool test_matrix_free_test_case_0;
  bool test_matrix_free_test_case_1;

  bool test_jacobian_case_0;
  bool test_jacobian_case_1;
  bool test_jacobian_case_2;

  bool test_binomial_case_0;
  bool test_binomial_case_1;
  bool test_binomial_case_2;

};
