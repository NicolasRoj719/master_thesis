//#include <iostream> included in table_cell.hpp
//#include <cstdint> included in table_cell.hpp
#include "./../table_cell.hpp"

int main(){
  {
    std::size_t cost = 10, k = 5;
    Operation op = Operation::ADJOINT;
    Table_cell table_cell{cost, k, op};
    assert(table_cell.cost() == cost);
    assert(table_cell.k() == k);
    assert(table_cell.op_to_string() == "ADJ");
  }

  {
    Table_cell table_cell_def{};
    assert(table_cell_def.cost() == 0);
    assert(table_cell_def.k() == 0);
    assert(table_cell_def.op_to_string() == "NONE");
  }

  {
    std::size_t cost = 10, k = 5, mem = 150;
    Operation op = Operation::TANGENT;
    cell_MFDJCPB mfdjcpb{cost, k, op, mem};
    assert(mfdjcpb.op_to_string() == "TAN");
    assert(mfdjcpb.mem_lim() == mem);
  }
  {
    cell_MFDJCPB mfdjcpb_def{};
    assert(mfdjcpb_def.mem_lim() == 0);
  }

  return 0;
}
