/**
 * @file table.hpp
 * @brief Dynamic programming (DP) lookup table storage structures for Jacobian Chain Product Bracketing.
 */

#ifndef TABLE_HPP  
#define TABLE_HPP

#include <algorithm>
#include <cassert>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "jacobian.hpp"
#include "table_cell.hpp"


/**
 * @brief Base class for dynamic programming lookup tables.
 *
 * Handles memory allocation for cell and cell_with_pointer storage arrays and provides
 * indexing lookup mechanisms for upper-triangular subchain DP data.
 *
 * @tparam Jacobian_T Jacobian matrix specialization type.
 */
template<class Jacobian_T>
class Table_base{
 public:
  /**
   * @brief Allocates storage for cell_with_pointer and cells arrays.
   *
   * @param chain_size Total number of jacobian matrices in the target chain.
   */
  explicit Table_base(std::size_t chain_size){

     cells_with_pointer.reserve(chain_size);
     cells.reserve((chain_size * (chain_size - 1))/ 2);
  }

  /**
   * @brief Access cell in cells using two indices.
   *
   * Maps 2D subchain indices (j,i) belonging to a lower triangular array to a linear 
   * storage vector cells.
   *
   * cell at (j,i) contains the optimal solution to the subchain (j,i) including end points.
   *
   * @param j_index End index of the subchain (1-based, j > i).
   * @param i_index Start index of the subchain (0-based).
   * @return Const reference to the corresponding 'cell<Jacobian_T>' object.
   */
  const cell<Jacobian_T>& get_cell(std::size_t j_index, std::size_t i_index) const{

    return cells[((j_index - 1) * j_index) / 2 + (j_index - 1 - i_index)];
  }  

  /**
   *  @brief Access subchains of length 1. 
   *
   *  Subchain collapses to a single Jacobian matrix. Accesses of the type (j, j) are done via 
   *  a single index j.
   *
   *  @param j_index Index of Jacobian_T matrix in the chain (0-based).
   *  @return Const reference to the corresponding 'cell_with_pointer<Jacobian_T>' object.
   */
  const cell_with_pointer<Jacobian_T>& get_cell(std::size_t j_index) const{

    return cells_with_pointer[j_index];
  }

  /// Clears both internal cell vectors.
  void clear(){
    cells.clear();
    cells_with_pointer.clear();
  }

 protected:
  /**
   * @brief Vector storing optimal subchain DP computational results ('cell<Jacobian_T>'),
   * for subchains of length greater than one.
   */
  std::vector<cell<Jacobian_T>> cells;

  /**
   * @brief Vector storing optimal subchain DP computational results, with subchains of length one 
   * and non-owning pointers (cell_with_pointer<Jacobian_T>) pointing to Jacobian_T matrices in the chain.
   */
  std::vector<cell_with_pointer<Jacobian_T>> cells_with_pointer;
};

/**
 * @brief Dynamic programming table providing overload constructor to allocate storage and emplace_back
 * methods to populate the table.
 *
 * @tparam Jacobian_T Jacobian matrix specialization type.
 */
template<class Jacobian_T>
class Table: public Table_base<Jacobian_T>{
 public:
   /**
    * @brief Inherits constructors from Table_base. 
    */
   using Table_base<Jacobian_T>::Table_base;

   //cell<Jacobian> emplace_back
   /**
    * @brief Initializes a cell with optimal cost and split_position.
    *
    * @note Intended primary for standard 'Jacobian' types.
    * @note Only possible operation is MULTIPLICATION.
    * @note Split position k index creates subproblems (j, k+1) and (k, i).
    *
    * @param accumulated_cost Optimal accumulated cost.
    * @param split_position Optimal split position index.
    */
   void emplace_back(std::size_t accumulated_cost, std::size_t split_position){
      
      this -> cells.emplace_back(accumulated_cost, split_position);
   }

   /**
    * @brief Constructs cell with dynamic programming (DP) metrics in cells array.
    *
    * @note Applicable for 'Dense_Jacobian' and 'Split_dense_Jacobian' cell types.
    * @note All three accumulation methods ADJOINT, TANGENT and MULTIPLICATION are available.
    * @note ADJOINT mode splitting at index k creates the subproblem (k, i).
    * @note TANGENT mode splitting at index k creates the subproblem (j, k+1).
    * @note MULTIPLICATION mode splitting at index k creates the subproblems (j, k+1) and (k, i).
    *
    * @param accumulated_cost Optimal accumulated cost.
    * @param split_position Optimal split position index.
    * @param operation Optimal accumulation method.
    */
   void emplace_back(std::size_t accumulated_cost, std::size_t split_position,
                     Operation operation){
      
      this -> cells.emplace_back(accumulated_cost, split_position, operation);
   }

   //cell<Sparse_Jacobian> and cell<Split_sparse_Jacobian> emplace_back
   /**
    * @brief Constructs cell by moving ownership of an internal sparse matrix and initializing dynamic 
    * programming (DP) metrics in cells array.
    *
    * @note Applicable for 'Sparse_Jacobian' and 'Split_sparse_Jacobian' cell types.
    * @note Subproblem creation follows the rules indicated above.
    * @note The Sparse and Split sparse Jacobian objects are propagated through the table
    * by means of multiplication.
    *
    * @param jacobian Rvalue reference to Jacobian_t instance in the chain.
    * @param accumulated_cost Optimal accumulated execution cost.
    * @param split_position Optimal split position index (k).
    * @param operation Optimal accumulation operation mode.
    */
   void emplace_back(Jacobian_T&& jacobian, std::size_t accumulated_cost,
                        std::size_t split_position, Operation operation){
      
      this -> cells.emplace_back(std::move(jacobian), accumulated_cost, split_position, operation);
   }

   /**
    * @brief Constructs cells_with_pointer containing non owning reference to Jacobian_T type in 
    * cells_with_pointer array.
    *
    * @note Intended for 'Jacobian' cell with pointer type.
    *
    * @param jacobian_ptr non owning pointer to Jacobian_T instance in the chain.
    */
   void emplace_back(const Jacobian_T* jacobian_ptr){
      
      this -> cells_with_pointer.emplace_back(jacobian_ptr);
   }

   //cells_with_pointer<Dense_Jacobian>, cells_with_pointer<Split_dense_Jacobian>,
   //cells_with_pointer<Sparse_Jacobian> and cells_with_pointer<Split_sparse_Jacobian>
   /**
    * @brief Constructs cell_with_pointer in cells_with_pointer with information from 
    * non owning pointer to Jacobian_T and dynamic programming (DP) metrics.
    *
    * @note Applicable for 'Dense_Jacobian', 'Split_dense_Jacobian', 'Sparse_Jacobian' and
    * 'Split_sparse_Jacobian' cells_with_pointer type.
    * @note Preaccumulation operation methods are restricted to either TANGENT or ADJOINT.
    *
    * @param jacobian_ptr Non owning pointer to Jacobian_T instance in the chain.
    * @param accumulated_cost Optimal accumulated cost.
    * @param operation Optimal preaccumulation method.
    */
   void emplace_back(const Jacobian_T* jacobian_ptr, std::size_t accumulated_cost,
                        Operation operation){
      
      this -> cells_with_pointer.emplace_back(jacobian_ptr, accumulated_cost, operation);
   }
};

#endif  //TABLE_HPP
