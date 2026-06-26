#include <cstdint>
#include <iostream>
#include "./../util_structs.hpp"

#ifndef TEST_UTIL_STRUCTS_HPP
#define TEST_UTIL_STRUCTS_HPP

class test_NNZ{
 public: 
  test_NNZ(std::size_t row, std::size_t column):
    object(row, column){
      was_constructor_test_successful = constructor_test(row, column);
      was_order_test_sucessful = order_test(row, column);
  }

  bool test_outcome(){return was_constructor_test_successful && was_order_test_sucessful;}
  bool constructor_test_outcome(){return was_constructor_test_successful;}
  bool order_test_outcome(){return was_order_test_sucessful;}

  void print_test_state() const{
    std::cout << "State of test_NNZ object:\n";
    
    std::cout << "Constructor test: ";
    if(was_constructor_test_successful){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "Order test: ";
    if(was_order_test_sucessful){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}
  }
  
 protected:
  bool constructor_test(std::size_t row, std::size_t column) const{
    if(row != object.row()){
      return false;
    }
    if(column != object.col()){
      return false;
    }
    return true;
  }

  bool order_test(std::size_t row, std::size_t column) const{
    NNZ less_0{row - 1, column};
    NNZ less_1{row, column - 1};
    NNZ less_2{row - 1, column + 1};
    NNZ greater_0{row + 1, column};
    NNZ greater_1{row, column + 1};
    NNZ greater_2{row + 1, column - 1};
    NNZ equal{row, column};
    
    if(!(less_0 < object)){return false;}
    if(!(less_1 < object)){return false;}
    if(!(less_2 < object)){return false;}
    if(greater_0 < object){return false;}
    if(greater_1 < object){return false;}
    if(greater_2 < object){return false;}
    if(!(equal == object)){return false;}
    return true;
  }

  const NNZ object;
  bool was_constructor_test_successful;
  bool was_order_test_sucessful;
};

class test_jacobian_information{
 public:
  test_jacobian_information(std::size_t domain_dim, std::size_t codomain_dim,
      std::size_t num_of_edges, std::size_t num_nonzeros)
  try: object(domain_dim, codomain_dim, num_of_edges, num_nonzeros){
    was_constructor_test_successful = 
      constructor_test(domain_dim, codomain_dim, num_of_edges, num_nonzeros);
  }
  catch(const std::invalid_argument& e){throw;}

  void print_test_state() const{
    std::cout << "State of test_jacobian_information object:\n";
    
    std::cout << "Constructor test: ";
    if(was_constructor_test_successful){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}
  }

  bool test_outcome(){return was_constructor_test_successful;}
 
 protected:
  const Matrix_free_sparse_information object;
  bool constructor_test(std::size_t domain_dim, std::size_t codomain_dim,
      std::size_t num_of_edges, std::size_t num_nonzeros){

    if(object.domain_dimension() != domain_dim){return false;}
    if(object.codomain_dimension() != codomain_dim){return false;}
    if(object.number_of_edges() != num_of_edges){return false;}
    if(object.number_of_nonzeros() != num_nonzeros){return false;}
    return true;
  }
  bool was_constructor_test_successful;
};
#endif
