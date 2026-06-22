#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>
#include "./jacobian.hpp"
#include "./table_cell.hpp"
#include "./table.cpp"

int main(){
  //Creating a Jacobian chain
  {
    std::string file_chain_data = "./chain_test_cases/case_0_sparse_WF3";
    jacobian_chain<Sparse_Jacobian> sparse_chain{file_chain_data};
    Table table{sparse_chain}; 

    std::size_t cost_base = 100;
    std::size_t split_position = 0;
    Operation op = Operation::TANGENT;

    //Testing emplace_back
    for(std::size_t i = 0; i<sparse_chain.size(); i++){
      table.emplace_back(i, i, cost_base + 1, split_position + 1, op);
    }


    

  }

}

