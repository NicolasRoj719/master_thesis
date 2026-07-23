#include <iostream>
#include "./../table_cell.hpp"

class test_optimal_accumulation{
 public:
  test_optimal_accumulation():
    test_jacobian_small(false), test_jacobian_medium(false),
    test_tangent_dense_jacobian(false), test_adjoint_dense_jacobian(false),
    test_multiplication_dense_jacobian(false), test_dense_jacobian_accumulation(false){}

  void set_test_jacobian_small(bool test_state){

    test_jacobian_small = test_state;
  }

  void set_test_jacobian_medium(bool test_state){

    test_jacobian_medium = test_state;
  }

  void set_test_tangent_dense_jacobian(bool test_state){

    test_tangent_dense_jacobian = test_state;
  }

  void set_test_adjoint_dense_jacobian(bool test_state){

    test_adjoint_dense_jacobian = test_state;
  }

  void set_test_multiplication_dense_jacobian(bool test_state){

    test_multiplication_dense_jacobian = test_state;
  }

  void set_test_dense_jacobian_accumulation(bool test_state){

    test_dense_jacobian_accumulation = test_state;
  }

  void set_were_all_tests_successful(){

    if(test_jacobian_small && test_jacobian_medium &&
        test_tangent_dense_jacobian && test_adjoint_dense_jacobian &&
        test_multiplication_dense_jacobian && test_dense_jacobian_accumulation){

      were_all_tests_successful = true;
    }

    else{
      were_all_tests_successful = false;
      print_test_state();
    }
  }

  
   void print_test_state(){
    std::cout << "Testing optimal accumulation dense:\n";

    std::cout << "test_jacobian_small: ";
    if(test_jacobian_small){
    
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_jacobian_medium: ";
    if(test_jacobian_medium){
    
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_tangent_dense: ";
    if(test_tangent_dense_jacobian){
    
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_adjoint_dense: ";
    if(test_adjoint_dense_jacobian){
    
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_multiplication_dense: ";
    if(test_multiplication_dense_jacobian){
    
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_dense_jacobian: ";
    if(test_dense_jacobian_accumulation){
    
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

   }
 
 protected:
   bool test_jacobian_small;

   bool test_jacobian_medium;

   bool test_tangent_dense_jacobian;

   bool test_adjoint_dense_jacobian;

   bool test_multiplication_dense_jacobian;

   bool test_dense_jacobian_accumulation;

   bool were_all_tests_successful;
};
