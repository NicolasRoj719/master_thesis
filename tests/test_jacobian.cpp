//#include <iostream> included in jacobian.hpp
#include <vector>
#include "./../jacobian.hpp"

int main(){
  //Testing constructor.
  Base_Jacobian jac_0{5,20};
  Dense_Jacobian jac_1{7,7,20};

  //Testing default constructor
  Base_Jacobian jac_2;
  Base_Jacobian jac_3;
  Dense_Jacobian jac_4;
  Dense_Jacobian jac_5;

  //Testing print function.
  std::cout<<"print jac_0:\n";
  jac_0.print();

  std::cout<<"print jac_1:\n";
  jac_1.print();

  //Testing print function on uninitiallized objects
  std::cout<<"uninitiallized jac_2:\n";
  jac_2.print();
  std::cout<<"uninitiallized jac_4:\n";
  jac_4.print();

  //Testing operator overload to initialize jac_2 and jac_3
  std::vector<std::size_t> n_m_values={10, 15, 25, 45};
  std::vector<std::size_t> n_m_n_E_values= {25, 30, 40, 60, 55, 65};

  //Printing n_m_values and n_m_n_E_values
  std::cout<<"Printing n_m_values:\n";
  for(std::size_t data: n_m_values){
    std::cout<<data<<' ';
  }
  std::cout<<'\n';

  std::cout<<"Printing n_m_n_E_values:\n";
  for(std::size_t data: n_m_n_E_values){
    std::cout<<data<<' ';
  }
  std::cout<<'\n';

  std::size_t len_loop = n_m_values.size() / 2; 
  for(std::size_t i=0; i<len_loop; i++){
    jac_2[i]= n_m_values[2*i];
    jac_3[i]= n_m_values[2*i+1];
  }

  std::size_t len_loop2 = n_m_n_E_values.size() / 2; 
  for(std::size_t i=0; i<len_loop2; i++){
    jac_4[i]= n_m_n_E_values[2*i];
    jac_5[i]= n_m_n_E_values[2*i+1];
  }

  //Printing initialized objects.
  std::cout<<"initiallized jac_2:\n";
  jac_2.print();
  std::cout<<"initiallized jac_3:\n";
  jac_3.print();
  std::cout<<"initiallized jac_4:\n";
  jac_4.print();
  std::cout<<"initiallized jac_5:\n";
  jac_5.print();

  return 0;
}
