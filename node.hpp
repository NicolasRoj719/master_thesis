#include <memory>
#include <vector>
#include <cstdint>
#include <type_traits>
#include <string>
#include <iostream>
#include "table_cell.hpp"

#ifndef NODE_HPP
#define NODE_HPP

class Node_base{
 public:
  Node_base(Table_cell* p_ptr, Table_cell* c_ptr, Table_cell* i_ptr_,
      std::size_t j_, std::size_t i_, std::size_t cost_):
    parent_ptr(p_ptr), i_ptr(i_ptr_), j(j_), i(i_), cost(cost_){
      children_ptr.push_back(c_ptr);
    }

  Node_base(Table_cell* p_ptr, Table_cell* c_ptr_0, Table_cell* c_ptr_1, Table_cell* i_ptr_,
      std::size_t j_, std::size_t i_, std::size_t cost_):
    parent_ptr(p_ptr), i_ptr(i_ptr_), j(j_), i(i_), cost(cost_){
      children_ptr.push_back(c_ptr_0);
      children_ptr.push_back(c_ptr_1);
    }

 protected:
  Table_cell* parent_ptr;
  Table_cell* i_ptr;
  std::vector<Table_cell*> children_ptr;
  std::size_t j,i;
  std::size_t cost;
};

template<class Cell_type>
class Node: public Node_base{
 public:
  Node(Table_cell* p_ptr, Table_cell* c_ptr, Table_cell* i_ptr_,
      std::size_t j_, std::size_t i_, std::size_t cost_): 
    Node_base(p_ptr, c_ptr, i_ptr_, j_, i_, cost_){}

  Node(Table_cell* p_ptr, Table_cell* c_ptr_0, Table_cell* c_ptr_1, Table_cell* i_ptr_, 
      std::size_t j_, std::size_t i_, std::size_t cost_): 
    Node_base(p_ptr, c_ptr_0, c_ptr_1, i_ptr_, j_, i_, cost_){}
  
  friend std::ostream& operator<< (std::ostream& os, Node& node){
    os << "\""+ cost +' '+ this->i_ptr->op_to_string + '('+ std::to_string(j) + ' ' +
        this->i_ptr->k + ' ' + std::to_string(i) + " )\"";

    return os;
  }
};


template<>
class Node<cell_MFDJCPB>: public Node_base{
 public:
  Node(Table_cell* p_ptr, Table_cell* c_ptr, Table_cell* i_ptr_,
      std::size_t j_, std::size_t i_, std::size_t cost_, std::size_t memory_):
    Node_base(p_ptr, c_ptr, i_ptr_, j_, i_, cost_), memory(memory_){}

  Node(Table_cell* p_ptr, Table_cell* c_ptr_0, Table_cell* c_ptr_1, Table_cell* i_ptr_, 
      std::size_t j_, std::size_t i_, std::size_t cost_, std::size_t memory_): 
    Node_base(p_ptr, c_ptr_0, c_ptr_1, i_ptr_, j_, i_, cost_), memory(memory_){}

  friend std::ostream& operator<< (std::ostream& os, Node& node){
    os << "\""+ cost +' '+ this->i_ptr->op_to_string + '('+ std::to_string(j) + ' ' +
        this->i_ptr->k + ' ' + std::to_string(i) + " ) " + this->i_ptr->memory;

    return os;
  }
 
 protected:
  std::size_t memory;
};
#endif
