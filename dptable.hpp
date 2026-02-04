#include <vector>
#include <cassert>
#include "table_cell.hpp"

#ifndef DPTABLE_HPP  
#define DPTABLE_HPP

template<class Cell_type>
class Table{
 public:
  Table(){}

  Table(std::size_t chain_len, std::size_t table_len): jac_chain_len(chain_len),
    table(table_len){}

  virtual ~Table() = default;

  Cell_type& operator()(std::size_t j, std::size_t i){
    return(table[index(j,i)]);
  }
  
  std::vector<Cell_type>& get_table(){return table;}

  void print() {
    std::cout<<"Printing the dynamic programming table:\n";
    std::size_t pos=0;
    std::size_t j,s,i;
    for(j=0; j<jac_chain_len; j++){
        for(s=0; s<=j; s++){
          i = j-s;
          //Base template syntax (this->).
          std::cout<<"F'_"<<j<<'_'<<i<<"= ";
          table[pos].print();
          pos++;
        }
    }
  }

 protected:

  static constexpr std::size_t index(std::size_t j, std::size_t i){
    assert(i<=j);
    return((j + 1) * j / 2 + (j - i +1));
  }

  std::size_t jac_chain_len = 0;

  std::vector<Cell_type> table;

};
#endif
