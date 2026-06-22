#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>
#include "./jacobian.hpp"
#include "./table_cell.hpp"

#ifndef DPTABLE_HPP  
#define DPTABLE_HPP

template<class Jacobian_type>
class Table{
 public:
  Table(const std::vector<Jacobian_type>& jacobian_chain){
    const std::size_t chain_size = jacobian_chain.size();
    single_jacobian_cells.reserve(chain_size);
    calculated_jacobian_cells.reserve(chain_size * (chain_size - 1) / 2);

    for(std::size_t i = 0; i < chain_size; i++){
      single_jacobian_cells.emplace_back(&jacobian_chain[i]);
    }
  }

  const cell_with_pointer<Jacobian_type> operator() (std::size_t i) const{
    return single_jacobian_cells[i];
  }

  const cell<Jacobian_type> operator() (std::size_t j, std::size_t i) const{
    if(j <= i){
      throw std::invalid_argument("Invalid access to calculated_jacobian cell.\n"
          "Condition j <= i does not hold. Entered j = " + std::to_string(j) +
          " and i = " + std::to_string(i) + ".");
    }
    return calculated_jacobian_cells[ j * (j-1) / 2 + ((j-1) - i)];
  }

  //calculated_jacobian_cells is constructed following a very strict order
  //which is imposed and guaranteed by another class.
  void emplace_back(std::size_t j, std::size_t i,
      std::size_t cost, std::size_t split_position){
    if(j==i){
      single_jacobian_cells[j].cell_data_initializer(cost, split_position);
    }
    else{
      calculated_jacobian_cells.emplace_back(cost, split_position);
    }
  }

  //calculated_jacobian_cells is constructed following a very strict order
  //which is imposed and guaranteed by another class.
  void emplace_back(std::size_t j, std::size_t i,
      std::size_t cost, std::size_t split_position, Operation op){

    if(j==i){
      single_jacobian_cells[j].cell_data_initializer(cost, split_position, op);
    }

    else{
      calculated_jacobian_cells.emplace_back(cost, split_position, op);
    }
  }

  //calculated_jacobian_cells is constructed following a very strict order
  //which is imposed and guaranteed by another class.
  void emplace_back(std::size_t j, std::size_t i,
      std::size_t cost, std::size_t split_position,
      Operation op, std::size_t memory){

    if(j==i){
      single_jacobian_cells[j].cell_data_initializer(cost, split_position, op, memory);
    }

    else{
      calculated_jacobian_cells.emplace_back(cost, split_position, op, memory);
    }
  }

  //calculated_jacobian_cells is constructed following a very strict order
  //which is imposed and guaranteed by another class.
  void emplace_back(std::size_t j, std::size_t i,
      const Jacobian_type jacobian_lhs, const Jacobian_type jacobian_rhs,
      std::size_t cost, std::size_t split_position, Operation op){

    if(j==i){
      throw std::invalid_argument("emplace_back cannot be used for j=i.\n"
          "The Jacobian object was created within the constructor.");
    }

    else{
      calculated_jacobian_cells.emplace_back(jacobian_lhs * jacobian_rhs,
          cost, split_position, op);
    }
  }
  
  //calculated_jacobian_cells is constructed following a very strict order
  //which is imposed and guaranteed by another class.
  void emplace_back(std::size_t j, std::size_t i,
      const Jacobian_type jacobian_lhs, const Jacobian_type jacobian_rhs,
      std::size_t cost, std::size_t split_position,
      std::size_t memory, Operation op){

    if(j==i){
      throw std::invalid_argument("emplace_back cannot be used for j=i.\n"
          "The Jacobian object was created within the constructor.");
    }

    else{
      calculated_jacobian_cells.emplace_back(jacobian_lhs * jacobian_rhs,
          cost, split_position, op, memory);
    }
  }

  void clear(){
    single_jacobian_cells.clear();
    calculated_jacobian_cells.clear();
  }

  std::size_t size(){
    return single_jacobian_cells.size() + calculated_jacobian_cells.size();
  }

 protected:
  std::vector<cell_with_pointer<Jacobian_type>> single_jacobian_cells;
  std::vector<cell<Jacobian_type>> calculated_jacobian_cells;
};
#endif
