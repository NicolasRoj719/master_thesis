#include <vector>
#include "table_cell.hpp"

template<class Cell_type>
class Table{
 public:
  Table(std::size_t jac_chain_len_):
    jac_chain_len(jac_chain_len_), table(jac_chain_len_*(jac_chain_len_-1)/2){}

  static constexpr std::size_t index(std::size_t j, std::size_t i){
    assert(i<=j);
    return((j + 1) * j / 2 + (j - i +1));
  }

  Cell_type& operator()(std::size_t j, std::size_t i){
    return(table[index(j,i)]);
  }

  void print(){
    std::cout<<"Printing the dynamic programming table:\n";
    std::size_t pos=0;
    std::size_t j,s,i;
    for(j=0; j<jac_chain_len; j++){
        for(s=0; s<=j; s++){
          i = j-s;
          table[pos].print()
          pos++;
        }
    }
  }

 private:
    std::vector<Cell_type> table;
    std::size_t jac_chain_len;
}
