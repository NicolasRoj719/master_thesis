#include <cstdint>
#include "./../table.hpp"
#include "./../table_cell.hpp"

int main(){
  std::size_t chain_len = 2;
  std::size_t table_len = (chain_len + 1) * chain_len / 2;

  //Testing table constructor with cell type cell_DJCPB.
  Table<cell_DJCPB> table{chain_len, table_len};
  //Testing method emplace_back
  table.emplace_back(cell_DJCPB {40,0,Operation::ADJOINT});
  table.emplace_back(cell_DJCPB {60,1,Operation::TANGENT});
  table.emplace_back(cell_DJCPB {130,0,Operation::MULTIPLICATION});

  std::cout<<"Initializing cells:\n";
  std::cout<<"c_0_0 [ 40 0 ADJ ]\n" << "c_1_1 [ 60 1 TAN] \n";
  std::cout<<"c_1_0 [ 130 0 MUL ]\n";
  //Testing print method.
  std::cout<<"Executing print fuction. table.print()\n";
  table.print();

  //Testing (j,i) operator overload that returns reference to a cell.
  std::cout<<"Using the operator overload () that returns a reference to a cell.\n";
  std::cout<<"c_0_0 ";
  table(0,0).print();
  std::cout<<"c_1_1 ";
  table(1,1).print();
  std::cout<<"c_1_0 ";
  table(1,0).print();
  std::cout<<"Printing back element of table.\n";
  table.back().print();


  //Testing table constructor with cell type cell_DJCPB.
  Table<cell_MFDJCPB> table_0{chain_len, table_len};
  //Testing method emplace_back
  table_0.emplace_back(cell_MFDJCPB {40,0,Operation::ADJOINT, 100});
  table_0.emplace_back(cell_MFDJCPB {60,1,Operation::TANGENT, 230});
  table_0.emplace_back(cell_MFDJCPB {130,0,Operation::MULTIPLICATION, 40});

  std::cout<<"Initializing cells:\n";
  std::cout<<"c_0_0 [ 40 0 ADJ 100 ]\n" << "c_1_1 [ 60 1 TAN 230 ] \n";
  std::cout<<"c_1_0 [ 130 0 MUL 40 ]\n";

  //Testing print method.
  std::cout<<"Executing print fuction. table.print()\n";
  table_0.print();
  //Testing (j,i) operator overload that returns reference to a cell.
  std::cout<<"Using the operator overload () that returns a reference to a cell.\n";
  std::cout<<"c_0_0 ";
  table_0(0,0).print();
  std::cout<<"c_1_1 ";
  table_0(1,1).print();
  std::cout<<"c_1_0 ";
  table_0(1,0).print();

  std::cout<<"Printing back element of table_0.\n";
  table_0.back().print();

  std::cout<<"Clearing table table.\n";
  table.clear();
  std::cout<<"size of table after calling clear_table(): "<<table.size()<<'\n';

  std::cout<<"Clearing table table_0.\n";
  table_0.clear();
  std::cout<<"size of table_0 after calling clear_table(): "<<table_0.size()<<'\n';

  std::cout<<"Attempting to create a table with the wrong size.\n";
  Table<cell_DJCPB> table_error{chain_len, chain_len-1};

  return 0;
}
