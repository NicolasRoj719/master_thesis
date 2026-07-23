#include <cstdint>
#include <cassert>
#include <iostream>
#include <vector>
#include "./class_test_util_structs.hpp"
#include "./../util_structs.hpp"

int main(){
  //Test0 NNZ class
  {
    std::size_t row = 3, column = 4;
    test_NNZ test{row, column};
    if(!test.test_outcome()){
      test.print_test_state();
    }
  }

  //Test1 NNZ class
  {
    std::size_t row = 3, column = 2;
    test_NNZ test{row, column};
    if(!test.test_outcome()){
      test.print_test_state();
    }
  }

  //Test0 jacobian_infomation class and inherited classes
  {
    const std::size_t domain_dimension = 5, codomain_dimension = 20;
    const std::size_t number_of_edges = 150, number_of_nonzeros = 25;
    test_jacobian_information test{domain_dimension, codomain_dimension,
    number_of_edges, number_of_nonzeros};

    if(!test.test_outcome()){
      test.print_test_state();
    }
  }

  //Test1 jacobian_infomation class and inherited classes
  {
    const std::size_t domain_dimension = 5, codomain_dimension = 20;
    const std::size_t number_of_edges = 150, number_of_nonzeros = 15;
    try{
      test_jacobian_information test{domain_dimension, codomain_dimension,
      number_of_edges, number_of_nonzeros};
    }
    catch(const std::invalid_argument& e){
      
    }
  }

  return 0;
}

