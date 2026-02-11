//#include <iostream> included in table_cell.hpp
//#include <cstdint> included in table_cell.hpp
#include "./../table_cell.hpp"

int main(){

  Operation MUL= Operation::MULTIPLICATION;
  Operation TAN = Operation::TANGENT;
  Operation ADJ = Operation::ADJOINT;
  Operation NONE = Operation::NONE;

  //Testing different cell type constructors.
  cell_DJCPB cell_0_0{120, 2, TAN};
  cell_MFDJCPB cell_0_1{100, 3, ADJ, 10};
  cell_DJCPB cell_0_2{80, 8, MUL};
  cell_MFDJCPB cell_0_3{40, 5, NONE, 20};

  //Testing print method.
  std::cout<<"cell_0_0 {120, 2, TAN}";
  cell_0_0.print();

  std::cout<<"cell_0_1 {100, 3, ADJ, 10}";
  cell_0_1.print();

  std::cout<<"cell_0_2 {80, 8, MUL}";
  cell_0_2.print();

  std::cout<<"cell_0_3 {40, 5, NONE, 20}";
  cell_0_3.print();

  //Checking default initialized objects.
  cell_DJCPB cell_1_0{};
  std::cout<<"default initialized object cell_DJCPB cell_1_0.\n";
  cell_1_0.print();

  cell_MFDJCPB cell_1_1{};
  std::cout<<"default initialized object cell_MFDJCPB cell_1_1.\n";
  cell_1_1.print();

  cell_MFDJCPB cell_2_0{};
  

  //Testing operator overload.
  std::cout<<"Initializing object using () operator overload cell_1_0(120,1,Operation::MULTIPLICATION).\n";
  cell_1_0(120,1,MUL);
  cell_1_0.print();

  // Testing operator overload given three arguments the fourth argument is set to 0 (m=0)
  // check table_cell.hpp.
  std::cout<<"Initializing object using () operator overload cell_1_1(220,0,Operation::TANGENT).\n";
  cell_1_1(220,0,TAN);
  cell_1_1.print();

  //Testing operator overload given all four arguments.
  std::cout<<"Initializing object using () operator overload cell_2_0(300,2,Operation::ADJOINT, 120).\n";
  cell_2_0(300, 2, ADJ, 120);
  cell_2_0.print();

  return 0;
}
