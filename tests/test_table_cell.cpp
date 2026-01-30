//#include <iostream> included in table_cell.hpp
//#include <cstdint> included in table_cell.hpp
#include "./../table_cell.hpp"

int main(){
  /* auto& TAN = Operation::TANGENT; */
  /* auto& ADJ = Operation::ADJOINT; */
  /* auto& MUL = Operation::MULTIPLICATION; */
  /* auto& NONE = Operation::NONE; */

  /* cell_DJCPB cell_0_0{120, 2, Operation::TANGENT}; */
  /* cell_MFDJCPB cell_0_1{100, 3, Operation::ADJOINT, 10}; */
  /* cell_DJCPB cell_0_2{80, 8, Operation::MULTIPLICATION}; */
  /* cell_MFDJCPB cell_0_3{40, 5, Operation::NONE, 20}; */

  Operation MUL= Operation::MULTIPLICATION;
  Operation TAN = Operation::TANGENT;
  Operation ADJ = Operation::ADJOINT;
  Operation NONE = Operation::NONE;

  cell_DJCPB cell_0_0{120, 2, TAN};
  cell_MFDJCPB cell_0_1{100, 3, ADJ, 10};
  cell_DJCPB cell_0_2{80, 8, MUL};
  cell_MFDJCPB cell_0_3{40, 5, NONE, 20};

  std::cout<<"cell_0_0 ";
  cell_0_0.print();

  std::cout<<"cell_0_1 ";
  cell_0_1.print();

  std::cout<<"cell_0_2 ";
  cell_0_2.print();

  std::cout<<"cell_0_3 ";
  cell_0_3.print();

  return 0;
}
