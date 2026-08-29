#include <memory>
#include <cstdint>
#include <type_traits>
#include <string>
#include <fstream>
#include "table_cell.hpp"

#ifndef NODE_HPP
#define NODE_HPP

//Node_base class is not implemented to use polymorphism
//it is used to avoid code repetition.
class Node_base{
 public:
  Node_base(Table_cell* i_ptr_, std::size_t j_,
      std::size_t i_, std::size_t cost_, int idx_):
    i_ptr(i_ptr_), j(j_), i(i_), cost(cost_), parent_idx(idx_){}

  Table_cell* get_i_ptr(){
    return i_ptr;
  }

  std::size_t get_i(){
    return i;
  }

  std::size_t get_j(){
    return j;
  }

  std::size_t get_cost(){
    return cost;
  }

  int get_parent_idx(){
    return parent_idx;
  }

 protected:
  Table_cell* i_ptr;
  std::size_t j,i;
  std::size_t cost;
  int parent_idx;
};

template<class Cell_type>
class Node: public Node_base{
 public:
  Node(Table_cell* i_ptr_, std::size_t j_, std::size_t i_,
      std::size_t cost_, int idx_):
    Node_base(i_ptr_, j_, i_, cost_, idx_){}
  /* Node(Node_base* p_ptr, Table_cell* i_ptr_, */
  /*     std::size_t j_, std::size_t i_, std::size_t cost_): */ 
  /*   Node_base(p_ptr, i_ptr_, j_, i_, cost_){} */

  friend std::ostream& operator<< (std::ostream& os, Node<Cell_type>& node){
    os << "\"" << node.get_cost() << ' ' << node.get_i_ptr()->operation << " (" << node.get_j()
      <<' ' << node.get_i_ptr()->k << ' ' << node.get_i() << ")\"";
    return os;
  }
};


template<>
class Node<cell_MFDJCPB>: public Node_base{
 public:
  Node(Table_cell* i_ptr_, std::size_t j_, std::size_t i_,
      std::size_t cost_, std::size_t memory_, int idx_):
    Node_base(i_ptr_, j_, i_, cost_, idx_), memory(memory_){}

  friend std::ostream& operator<< (std::ostream& os, Node<cell_MFDJCPB>& node){
    os << "\"" << node.get_cost() << ' ' << node.get_i_ptr()->operation << " (" << node.get_j()
      <<' ' << node.get_i_ptr()->k << ' ' << node.get_i() << ") " << node.get_memory() << '\"';
    return os;
  }

  std::size_t get_memory(){
    return memory;
  }

 private:
  std::size_t memory;
};
#endif
