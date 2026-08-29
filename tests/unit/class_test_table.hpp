#include <iostream>

class test_table{
 public:
  test_table():
    test_emplace_back_jacobian(false), test_emplace_back_dense(false),
    test_emplace_back_sparse(false), were_all_tests_succcessful(false){}

  void set_test_emplace_back_jacobian(bool argument){

    test_emplace_back_jacobian = argument;
  }

  void set_test_emplace_back_jacobian_with_pointer(bool argument){

    test_emplace_back_jacobian_with_pointer = argument;
  }

  void set_test_emplace_back_dense(bool argument){

    test_emplace_back_dense = argument;
  }

  void set_test_emplace_back_dense_with_pointer(bool argument){

    test_emplace_back_dense_with_pointer = argument;
  }

  void set_test_emplace_back_sparse(bool argument){

    test_emplace_back_sparse = argument;
  }

  void set_test_emplace_back_sparse_with_pointer(bool argument){

    test_emplace_back_sparse_with_pointer = argument;
  }

  void set_were_all_tests_succcessful(){

    if(test_emplace_back_jacobian && test_emplace_back_dense &&
        test_emplace_back_sparse){

      were_all_tests_succcessful = true;
    }

    else{

      were_all_tests_succcessful = false;
      print_test_state();
    }
  }

  void print_test_state(){
    
    std::cout<< "State of table test:\n";

    std::cout << "test_emplace_back_jacobian: ";
    if(test_emplace_back_jacobian){

      std::cout << "successful.\n";
    }
    else{ std::cout << "failed.\n";}

    std::cout << "test_emplace_back_dense: ";
    if(test_emplace_back_dense){

      std::cout << "successful.\n";
    }
    else{ std::cout << "failed.\n";}

    std::cout << "test_emplace_back_sparse: ";
    if(test_emplace_back_sparse){

      std::cout << "successful.\n";
    }
    else{ std::cout << "failed.\n";}

    std::cout<< '\n';
  }


 protected:
   bool test_emplace_back_jacobian;

   bool test_emplace_back_jacobian_with_pointer;

   bool test_emplace_back_dense;

   bool test_emplace_back_dense_with_pointer;

   bool test_emplace_back_sparse;

   bool test_emplace_back_sparse_with_pointer;

   bool were_all_tests_succcessful;
};
