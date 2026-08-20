#include <algorithm>
#include <cassert>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include "./jacobian.hpp"
#include "./table_cell.hpp"

#ifndef DPTABLE_HPP  
#define DPTABLE_HPP

template<class Jacobian_T>
class Table{
 public:
  Table(std::size_t chain_size){

     cells_with_pointer.reserve(chain_size);
     cells.reserve((chain_size * (chain_size - 1))/ 2);
  }

  void emplace_back(std::size_t accumulated_cost, std::size_t split_position);

  void emplace_back(std::size_t accumulated_cost, std::size_t split_position,
        Operation operation);

  void emplace_back(std::size_t accumulated_cost, std::size_t split_position,
        Operation operation, std::size_t accumulated_memory);

  void emplace_back(Jacobian_T jacobian, std::size_t accumulated_cost,
        std::size_t split_position, Operation operation);

  void emplace_back(Jacobian_T jacobian, std::size_t accumulated_cost,
        std::size_t split_position, Operation operation, std::size_t accumulated_memory);

  void emplace_back(const Jacobian_T* jacobian_pointer);

  void emplace_back(const Jacobian_T* jacobian_pointer, std::size_t accumulated_cost,
        Operation operation);

  void emplace_back(const Jacobian_T* jacobian_pointer, std::size_t accumulated_cost,
        Operation operation, std::size_t accumulated_memory);

  const cell<Jacobian_T>& get_cell(std::size_t j_index, std::size_t i_index) const{

     return cells[((j_index - 1) * j_index) / 2 + (j_index - 1 - i_index)];
  }

  const cell_with_pointer<Jacobian_T>& get_cell(std::size_t j_index) const{

      return cells_with_pointer[j_index];
  }

  void clear(){
     cells.clear();
     cells_with_pointer.clear();
  }
 
 protected:
  std::vector<cell<Jacobian_T>> cells;
  std::vector<cell_with_pointer<Jacobian_T>> cells_with_pointer;
};

template <>
void Table<Jacobian>::emplace_back(
      std::size_t accumulated_cost, std::size_t split_position){
  
   cells.emplace_back(accumulated_cost, split_position); 
}

template <>
void Table<Jacobian>::emplace_back(
      const Jacobian* jacobian_pointer){

   cells_with_pointer.emplace_back(jacobian_pointer);
}

template <>
void Table<Dense_Jacobian>::emplace_back(
      std::size_t accumulated_cost, std::size_t split_position, Operation operation){

   cells.emplace_back(accumulated_cost, split_position, operation);
}

template<>
void Table<Dense_Jacobian>::emplace_back(
      std::size_t accumulated_cost, std::size_t split_position,
      Operation operation, std::size_t memory){

   cells.emplace_back(accumulated_cost, split_position, operation, memory);
}

template <>
void Table<Dense_Jacobian>::emplace_back(
      const Dense_Jacobian* jacobian_pointer, std::size_t accumulated_cost,
      Operation operation){

   cells_with_pointer.emplace_back(
         jacobian_pointer, accumulated_cost, operation);
}

template <>
void Table<Dense_Jacobian>::emplace_back(
      const Dense_Jacobian* jacobian_pointer, std::size_t accumulated_cost,
      Operation operation, std::size_t memory){


   cells_with_pointer.emplace_back(
         jacobian_pointer, accumulated_cost, operation, memory);
}

template<>
void Table<Sparse_Jacobian>::emplace_back(
      const Sparse_Jacobian jacobian, std::size_t accumulated_cost,
      std::size_t split_position, Operation operation){

   cells.emplace_back(jacobian, accumulated_cost, split_position, operation);
}

template<>
void Table<Sparse_Jacobian>::emplace_back(
      const Sparse_Jacobian jacobian, std::size_t accumulated_cost,
      std::size_t split_position, Operation operation, std::size_t memory){

   cells.emplace_back(jacobian, accumulated_cost, split_position, operation, memory);
}

template <>
void Table<Sparse_Jacobian>::emplace_back(
      const Sparse_Jacobian* jacobian_pointer, std::size_t accumulated_cost,
      Operation operation){

   cells_with_pointer.emplace_back(
         jacobian_pointer, accumulated_cost, operation);
}

template<>
void Table<Sparse_Jacobian>::emplace_back(
      const Sparse_Jacobian* jacobian_pointer, std::size_t accumulated_cost,
      Operation operation, std::size_t memory){

   cells_with_pointer.emplace_back(jacobian_pointer, accumulated_cost,
         operation, memory);
}

#endif
