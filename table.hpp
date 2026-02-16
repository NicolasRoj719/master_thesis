#include <vector>
#include <cassert>
#include <stdexcept>
#include "table_cell.hpp"

#ifndef DPTABLE_HPP  
#define DPTABLE_HPP

template<class Cell_type>
class Table{
 public:
  Table(std::size_t chain_len, std::size_t table_len): jac_chain_len(chain_len){
    //TODO when implementing the scheduled version of the problem do not forget to 
    //include the possible sizes given by the number of available threads.
    if(table_len != chain_len * (chain_len + 1) / 2){
      throw std::invalid_argument("Invalid table_len value.");
    }
    table.reserve(table_len);
  }

  virtual ~Table() = default;

  Cell_type& operator()(std::size_t j, std::size_t i){
    return(table[index(j,i)]);
  }
  
  std::vector<Cell_type>& get(){return table;}

  void print() {
    /* std::cout<<"Printing the dynamic programming table:\n"; */
    std::size_t pos=0;
    std::size_t j,s,i;
    for(j=0; j<jac_chain_len; j++){
        for(s=0; s<=j; s++){
          i = j-s;
          //Base template syntax (this->).
          std::cout<<"c_"<<j<<'_'<<i<<" ";
          table[pos].print();
          pos++;
        }
    }
  }
  
  void emplace_back(const Cell_type& cell){
    table.emplace_back(cell);
  }

  void clear(){
    table.clear();
  }

  void reserve(std::size_t table_len){
    table.reserve(table_len);
  }

  size_t size(){
    return table.size();
  }

  Cell_type back(){
    return table.back();
  }

 protected:

  static constexpr std::size_t index(std::size_t j, std::size_t i){
    assert(i<=j);
    return((j + 1) * j / 2 + (j - i));
  }

  std::size_t jac_chain_len = 0;

  std::vector<Cell_type> table;

};
#endif
