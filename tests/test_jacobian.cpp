//#include <iostream> included in jacobian.hpp
#include "./../jacobian.hpp"

int main(){
  Base_Jacobian jac_0{5,20};
  Dense_Jacobian jac_1{7,7,20};

  std::cout<<"print jac_0:\n";
  jac_0.print();

  std::cout<<"print jac_1:\n";
  jac_1.print();

  return 0;
}
