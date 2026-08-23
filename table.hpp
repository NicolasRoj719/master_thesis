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
class Table_base{
 public:
   explicit Table_base(std::size_t chain_size){

     cells_with_pointer.reserve(chain_size);
     cells.reserve((chain_size * (chain_size - 1))/ 2);
   }

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

template<class Jacobian_T>
class Table: public Table_base<Jacobian_T>{
 public:
   //Inherit base constructor.
   using Table_base<Jacobian_T>::Table_base;

   //cell<Jacobian> emplace_back
   void emplace_back(std::size_t accumulated_cost, std::size_t split_position){
      
      this -> cells.emplace_back(accumulated_cost, split_position);
   }

   //cell<Dense_Jacobian> and cell<Split_dense_Jacobian> emplace_back
   void emplace_back(std::size_t accumulated_cost, std::size_t split_position,
                     Operation operation){
      
      this -> cells.emplace_back(accumulated_cost, split_position, operation);
   }

   //cell<Sparse_Jacobian> and cell<Split_sparse_Jacobian> emplace_back
   void emplace_back(Jacobian_T&& jacobian, std::size_t accumulated_cost,
                        std::size_t split_position, Operation operation){
      
      this -> cells.emplace_back(std::move(jacobian), accumulated_cost, split_position, operation);
   }

   //Cells with pointers
   //cells_with_pointer<Jacobian>
   void emplace_back(const Jacobian_T* jacobian_ptr){
      
      this -> cells_with_pointer.emplace_back(jacobian_ptr);
   }

   //cells_with_pointer<Dense_Jacobian>, cells_with_pointer<Split_dense_Jacobian>,
   //cells_with_pointer<Sparse_Jacobian> and cells_with_pointer<Split_sparse_Jacobian>
   void emplace_back(const Jacobian_T* jacobian_ptr, std::size_t accumulated_cost,
                        Operation operation){
      
      this -> cells_with_pointer.emplace_back(jacobian_ptr, accumulated_cost, operation);
   }

};

#endif
