/**
 * @file fill_table.hpp 
 * @brief Dynamic programming (DP) algorithms look tables for optimal Jacobian chain bracketing.
 */
#ifndef FILL_TABLE
#define FILL_TABLE

#include <algorithm>
#include <optional>
#include <limits>
#include <type_traits>
#include <vector>

#include "binomial_checkpointing.hpp"
#include "chain.hpp"
#include "table_cell.hpp"
#include "table.hpp"
#include "jacobian.hpp"


/**
 * @brief Primary template for dynamic programming table-filling algorithms.
 *
 * Specific combinations of Jacobian representation and metadata types require 
 * explicit specilizations to evaluate optimal accumulation costs under different 
 * operational modes.
 *
 * @tparam Jacobian_T Jacobian matrix representation type.
 * @tparam Jacobian_info_T Metadata/information type corresponding to 'Jacobian_T'.
 */
template<class Jacobian_T, class Jacobian_info_T>
class Fill_table{};

/**
 * @brief Explicit specialization of 'Fill_table' for 'Jacobian' and 'Jacobian_information' types.
 *
 * Computes optimal bracketing sequence to minimize total computation cost in terms of Fused 
 * Multiply-Add (fma) operations, restricted to dense matrix-matrix multiplications.
 */
template<>
class Fill_table<Jacobian, Jacobian_information>{
 public:
   /**
    * @brief Constructs table-filler engine and executes the dynamic programming algorithm.
    *
    * @param chain Target Jacobian chain containing sequence metadata.
    */
   Fill_table(const Jacobian_chain<Jacobian, Jacobian_information>& chain):
      table(chain.size()){

      fill(chain);
   }

   /**
    * @brief Constructs a table-filler with pre-populated or mock Table instance (e.g., for testing). 
    *
    * @param table_ External Table instance moved into internal storage.
    */
   Fill_table(Table<Jacobian> table_): table(std::move(table_)){}

   /**
    * @brief Calculates dense matrix multiplication cost for subchain \f$(j, k+1)$ and \f$(k, i)$.
    *
    * @param j End index of the subchain (1-based, \f$j > i\f$).
    * @param k Split position index (\f$i \le k < j\f$).
    * @param i Start index of the subchain (0-based).
    * @return Total computational cost in fma operations. 
    */
   std::size_t multiplication_cost(std::size_t j, std::size_t k, std::size_t i){

      std::size_t accumulation_cost_lhs, accumulation_cost_rhs;
      // Access cell_with_pointer
      if(k == i){
         //Elemental Jacobians have zero prior accumulation cost. 
         accumulation_cost_rhs = 0;
      }
      else{
         
         accumulation_cost_rhs = table.get_cell(k, i).accumulated_cost();
      }

      //Access cell_with_pointer
      if(j == k+1){
         //Elemental Jacobians have zero prior accumulation cost. 
         accumulation_cost_lhs = 0;
      }
      else{

         accumulation_cost_lhs = table.get_cell(j, k+1).accumulated_cost();
      }

      return accumulation_cost_lhs + accumulation_cost_rhs +
               table.get_cell(j).codomain_dim() * 
               table.get_cell(i).domain_dim() *
               table.get_cell(k).codomain_dim();
   }

   /**
    * @brief Accesses non-owning Cell reference to an elemental Jacobian in the chain.
    *
    * @param index 0-based position in the chain.
    * @return Const reference to 'Cell_with_pointer<Jacobian>' object.
    */
   const Cell_with_pointer<Jacobian>& get_cell(std::size_t index){

      return table.get_cell(index);
   }

   /**
    * @brief Accesses matrix Cell at \f$(j,i)\f$ containing subproblem optimal data.
    *
    * @param j End index of subchain.
    * @param i Start index of subchain.
    * @return Const reference to 'Cell<Jacobian>' object.
    */
   const Cell<Jacobian>& get_cell(std::size_t j, std::size_t i){

      return table.get_cell(j,i);
   }

   /**
    * @brief Access the last Cell in the DP table.
    *
    * @pre fill method must be executed first before calling this method.
    * @return Const reference to 'Cell<Jacobian>' object.
    */
   const Cell<Jacobian>& back(){
      return table.back();
   }

   /**
    * @brief Get reference to table.
    *
    * @return Const reference to 'Table<Jacobian>'
    */
   const Table<Jacobian>& get_table(){
      return table;
   }
 
 private:
   /// Internal DP lookup table.
   Table<Jacobian> table;

   /**
    * @brief Fills DP lookup table by computing optimal computational costs over increasing
    * subproblems lengths. 
    *
    * Steps:
    * 1. Initializes Cell_with_pointer objects with non-owning pointers to chain elements.
    * 2. Initializes Cell objects with subchain optimal costs and split positions.
    * 
    * Subproblems are evaluated in the following order:
    * (1,0)
    * (2,1) (2,0)
    * @param chain Target Jacobian chain instance.
    */
   void fill(const Jacobian_chain<Jacobian, Jacobian_information>& chain);
};


void Fill_table<Jacobian, Jacobian_information>::fill(
      const Jacobian_chain<Jacobian, Jacobian_information>& chain){

   //Filling cells with pointers
   for(std::size_t elemental_idx = 0; elemental_idx < chain.size(); elemental_idx++){

      table.emplace_back(&chain[elemental_idx]);
   }

   std::size_t minimum_cost, i_index, split_position;
   std::size_t multiplication_cost_j_k_i;

   for(std::size_t j_index = 1; j_index < chain.size(); j_index++){

      for(std::size_t aux_var = 1; aux_var < j_index + 1; aux_var++){

         i_index = j_index - aux_var;
         minimum_cost = std::numeric_limits<std::size_t>::max();

         //Problem instance (j,i)
         for(std::size_t k_index = i_index; k_index < j_index; k_index++){
            

            multiplication_cost_j_k_i = multiplication_cost(j_index, k_index, i_index);
            if(multiplication_cost_j_k_i < minimum_cost){

               minimum_cost = multiplication_cost_j_k_i;
               split_position = k_index;
            }
         }

         table.emplace_back(minimum_cost, split_position);
      }
   }
}

/**
 * @namespace tool_box_dense
 * @brief Helper routines providing cost evaluation models for dense Jacobian chain subproblems. 
 *
 * Contains functions to compute computational cost estimates in fma for forward (tangent) AD, 
 * reverse (adjoint) AD, and standard dense matrix-matrix multiplication.
 */
namespace tool_box_dense{

   /**
    * @brief Computes forward-mode (tangent) AD preaccumulation cost for a single elemental Jacobian.
    * 
    * @tparam Jacobian_T Elemental Jacobian type.
    * @param elemental_jacobian_ptr Non-owning pointer to elemental Jacobian instance.
    * @pre elemental_jacobian_ptr must not be nullptr.
    * @return Operation count for tangent-mode preaccumulation. 
    */
   template<class Jacobian_T>
   std::size_t tangent_cost(const Jacobian_T* elemental_jacobian_ptr){

      
      return elemental_jacobian_ptr -> domain_dim() *
               elemental_jacobian_ptr -> number_edges();
   }
   /**
    * @brief Computes reverse-mode (adjoint) AD preaccumulation cost for a single elemental Jacobian.
    *
    * @tparam Jacobian_T Elemental Jacobian type.
    * @param elemental_jacobian_ptr Non owning pointer to the elemental Jacobian instance.
    * @pre elemental_jacobian_ptr must not be nullptr.
    * @return Operation count for adjoint-mode preaccumulation.
    */
   template<class Jacobian_T>
   std::size_t adjoint_cost(const Jacobian_T* elemental_jacobian_ptr){

      return elemental_jacobian_ptr -> codomain_dim() *
               elemental_jacobian_ptr -> number_edges();
   }

   /**
    * @brief Sums total computational graph edges across a subchain range [k_plus_1_or_i, j_or_k].
    *
    * @tparam Jacobian_T Elemental Jacobian type.
    * @param table Reference to the dynamic programming (DP) lookup table.
    * @pre Cells with non-owning pointers are already initialized.
    * @param j_or_k Inclusive upper index bound of the subchain.
    * @param k_plus_1_or_i Inclusive lower index bound of the subchain.
    * @return Accumulated sum of edges across subchain range [k_plus_1_or_i, j_or_k].
    */
   template<class Jacobian_T>
   std::size_t accumulate_edges(const Table<Jacobian_T>& table, std::size_t j_or_k,
                                 std::size_t k_plus_1_or_i){

      std::size_t sum_of_edges = 0;
      for(std::size_t idx = k_plus_1_or_i; idx < j_or_k + 1; idx++){
         
         sum_of_edges += table.get_cell(idx).number_edges();
      }

      return sum_of_edges;
   }

   /**
    * @brief Calculates tangent-mode accumulation cost for subproblem instance (j,i) split at k.
    *
    * Combines prior subchain accumulation cost (k, i) with derivative propagation costs through 
    * subchain (j, k+1).
    *
    * @tparam Jacobian_T Elemental Jacobian type.
    * @param table Reference to the dynamic programming lookup table.
    * @param j Upper index bound of the subchain (inclusive).
    * @param k Split index within range [i, j-1]
    * @param i Lower index bound of the subchain (inclusive).
    * @return Accumulated forward-mode cost for subproblem (j, i). 
    */
   template<class Jacobian_T>
   std::size_t tangent_cost(const Table<Jacobian_T>& table, std::size_t j, std::size_t k,
                              std::size_t i){
      //Access a cell_with_pointer
      if(k == i){
         
         return table.get_cell(i).accumulated_cost() +
            table.get_cell(i).domain_dim() * accumulate_edges(table, j, k+1);
      }

      else{
         
         return table.get_cell(k,i).accumulated_cost() +
            table.get_cell(i).domain_dim() * accumulate_edges(table, j, k+1);
      }
   }

   /**
    * @brief Calculates standard dense matrix-matrix multiplication cost of two preaccumulated 
    * Jacobians (j, k+1) and (k, i).
    * 
    * @note The explicit preaccumulated Jacobians are not needed to calculate the computational cost.
    *
    * @tparam Jacobian_T Elemental Jacobian type.
    * @param table Reference to the dynamic programming lookup table.
    * @param j Upper index bound of the subchain (inclusive).
    * @param k Split index within range [i, j-1]
    * @param i Lower index bound of the subchain (inclusive).
    * @return Total combined cost of preaccumulation of subchains (j, k+1) and (k, i) plus dense 
    * matrix multiplication.
    */
   template<class Jacobian_T>
   std::size_t multiplication_cost(const Table<Jacobian_T>& table, std::size_t j, std::size_t k,
                                    std::size_t i){

      std::size_t accumulated_cost_lhs, accumulated_cost_rhs;
      //Access Cell_with_pointer
      if(k == i){
         
         accumulated_cost_rhs = table.get_cell(i).accumulated_cost();
      }
      else{

         accumulated_cost_rhs = table.get_cell(k,i).accumulated_cost();
      }

      //Access Cell_with_pointer
      if(j == k+1){
         
         accumulated_cost_lhs = table.get_cell(j).accumulated_cost();
      }
      else{
         
         accumulated_cost_lhs = table.get_cell(j, k+1).accumulated_cost();
      }

      return accumulated_cost_lhs + accumulated_cost_rhs +
               table.get_cell(j).codomain_dim() * table.get_cell(i).domain_dim() *
               table.get_cell(k).codomain_dim();
   }
}

/**
 * @brief Dynamic programing table-filler for standard, unconstrained or simple memory-bounded
 * dense Jacobian chain.
 *
 * Computes optimal accumulation decisions (MULTIPLICATION, TANGENT, or ADJOINT) for all subchains 
 * (j, i) to minimize the total computational cost measured in scalar fused multiply-add operations.
 *
 * @note Only the metadata from the Jacobian matrix domain/codomain dimensions and the graph edge 
 * count of the subprograms is required to run the optimizer.
 */
template<>
class Fill_table<Dense_Jacobian, Matrix_free_information>{
 public:
   /**
    * @brief Constructs and fills dynamic programming (DP) table without memory bound.
    *
    * @param chain Sequence of elemental Jacobians to optimize.
    */
   Fill_table(const Jacobian_chain<Dense_Jacobian, Matrix_free_information>& chain):
      table(chain.size()){

      fill(chain);
   }

   /**
    * @brief Constructs and fills dynamic programming (DP) table under a memory constraint.
    *
    * @param chain Sequence of elemental Jacobians to optimize.
    * @memory_bound_ Maximum allowed edge storage limit for executing Adjoint mode accumulation.
    */
   Fill_table(const Jacobian_chain<Dense_Jacobian, Matrix_free_information>& chain,
               std::size_t memory_bound_):
      table(chain.size()), memory_bound(memory_bound_){

      fill(chain);
   }

   /**
    * @brief Constructs table filler initiallized with a pre-populated or mock dynamic programming
    * table.
    * 
    * Used primarily to test methods depending on existing data within the dynamic programming table.
    *
    * @param table_ Instance of Table<Dense_Jacobian> to be moved into internal storage.
    */
   Fill_table(Table<Dense_Jacobian> table_):
      table(std::move(table_)){}
   
   /**
    * @brief Sums total computational graph edges across a subchain range [k_plus_1_or_i, j_or_k]
    *
    * @note Forwards the execution to tool_box_dense::accumulate_edges using the internal 'table' 
    * and 'memory_bound'.
    *
    * @param j_or_k Upper index bound.
    * @param k_plus_1_or_i Lower index bound.
    * @return Accumulated sum of edges across subchain range.
    */
   std::size_t accumulate_edges(std::size_t j_or_k, std::size_t k_plus_1_or_i) const{
      
      return tool_box_dense::accumulate_edges(table, j_or_k, k_plus_1_or_i);
   }

   /**
    * @brief Computes forward-mode (tangent) AD preaccumulation cost for a single elemental Jacobian.
    * @see tool_box_dense::tangent_cost.
    */
   static std::size_t tangent_cost(const Dense_Jacobian* elemental_jacobian_ptr) {
     
      return tool_box_dense::tangent_cost(elemental_jacobian_ptr);
   }

   /**
    * @brief Computes reverse-mode (adjoint) AD preaccumulation cost for a single elemental Jacobian.
    * @see tool_box_dense::adjoint_cost.
    */
   static std::size_t adjoint_cost(const Dense_Jacobian* elemental_jacobian_ptr) {

      return tool_box_dense::adjoint_cost(elemental_jacobian_ptr);
   }

   /**
    * @brief Calculates tangent-mode accumulation cost for subproblem instance (j,i) split at k.
    *
    * @note Forwards execution to tool_box_dense::tangent_cost using internal 'table'.
    * @param j Upper index bound (inclusive).
    * @param k Split index.
    * @param i Lower index bound (inclusive).
    * @return Accumulated forward-mode cost for subproblem (j, i).
    */
   std::size_t tangent_cost(std::size_t j, std::size_t k, std::size_t i) const{

      return tool_box_dense::tangent_cost(table, j, k, i);
   }

   /**
    * @brief Calculates reverse-mode (adjoint) cost using optimal preaccumulation cost of subproblem 
    * (j, k+1).
    *
    * @param j Upper index bound of the subchain (inclusive).
    * @param k Split index within range [i, j-1]
    * @param i Lower index bound of the subchain (inclusive).
    * @return std::optional containing calculated accumulation cost of subproblem (j, i) if the
    * memory threshold is met; std::nullopt if exceeded.
    */
   std::optional<std::size_t> adjoint_cost(std::size_t j, std::size_t k, std::size_t i) const{

      std::size_t total_edges = accumulate_edges(k,i);
      //Access a Cell_with_pointer
      if(j == k+1){
         
         //If memory bound is set
         if(memory_bound){
            
            //If memory bound is not surpassed 
            if(total_edges <= *memory_bound){

               return table.get_cell(j).accumulated_cost() +
                  table.get_cell(j).codomain_dim() * total_edges;
            }

            //If memory bound is exceeded.
            return {};
         }
         //Access a normal Cell
         else{
            
            return table.get_cell(j).accumulated_cost() +
               table.get_cell(j).codomain_dim() * total_edges; 
         }
      }
      //Access normal Cell
      else{
         
         if(memory_bound){

            if(total_edges <= *memory_bound){

               return table.get_cell(j, k+1).accumulated_cost() +
                  table.get_cell(j).codomain_dim() * total_edges;
            }

            //Memory bound not fulfilled
            return {};
         }

         else{
            
            return table.get_cell(j, k+1).accumulated_cost() +
               table.get_cell(j).codomain_dim() * total_edges;
         }
      }
   }

   /**
    * @brief Calculates standard dense matrix multiplication cost of two preaccumulates Jacobians
    * (j, k+1) and (k, i).
    *
    * @note Forwards execution to tool_box_dense::multiplication_cost using internal 'table'.
    *
    * @param j Upper index bound (inclusive).
    * @param k Split index.
    * @param i Lower index bound (inclusive).
    */
   std::size_t multiplication_cost(std::size_t j, std::size_t k, std::size_t i) const{

      return tool_box_dense::multiplication_cost(table, j, k, i);
   }

   /**
    * @brief Returns the configured memory bound threshold, if set.
    * @return std::optional containing edge bound limit or std::nullopt.
    */
   std::optional<std::size_t> get_memory_bound() const{

      return memory_bound;
   }

   /**
    * @brief Accesses dynamic programming Cell with non-owning pointer to elemental Jacobian at a given index.
    *
    * @param index Jacobian matrix index in the chain.
    * @return Const reference to 'Cell_with_pointer<Dense_Jacobian>' object.
    */
   const Cell_with_pointer<Dense_Jacobian>& get_cell(std::size_t index){
      
      return table.get_cell(index);
   }

   /**
    * @brief .Accesses dynamic programming Cell at \f$(j, i)\f$ storing optimal subproblem result.
    *
    * @param j End index of the subchain (1-based, $j > i$).
    * @param i Start index of the subchain (0-based).
    * @return Const reference to 'Cell<Dense_Jacobian>' object.
    */
   const Cell<Dense_Jacobian>& get_cell(std::size_t j, std::size_t i){

      return table.get_cell(j,i);
   }

   /**
    * @brief Access the last Cell in the DP table.
    *
    * @pre fill method must be executed first before calling this method.
    * @return Const reference to 'Cell<Dense_Jacobian>' object.
    */
   const Cell<Dense_Jacobian>& back(){
      return table.back();
   }

   /**
    * @brief Get reference to table.
    *
    * @return Const reference to 'Table<Dense_Jacobian>'
    */
   const Table<Dense_Jacobian>& get_table(){
      return table;
   }

 private:
   /// Dynamic programming table storing optimal subproblem solutions.
   Table<Dense_Jacobian> table;
   /// Optional edge limit constraint for adjoint evaluations.
   std::optional<std::size_t> memory_bound;

   /**
    * @brief Populates DP table by computing optimal computational costs over increasing 
    * subproblems lengths.
    *
    * @details Steps:
    * 1. Initializes Cell_with_pointer objects with base optimal costs for preaccumulation of 
    * elemental Jacobians and corresponding non-owning pointers to chain elements.
    * 2. Initializes Cell objects with subchain optimal costs and split positions. 
    *
    * @details Subproblems are evaluated in the following order:
    * (1, 0)
    * (2, 1) (2, 0)
    * ...
    *
    * @param chain Target Jacobian chain instance.
    */
   void fill(const Jacobian_chain<Dense_Jacobian, Matrix_free_information>& chain);
};

void Fill_table<Dense_Jacobian, Matrix_free_information>::fill(
      const Jacobian_chain<Dense_Jacobian, Matrix_free_information>& chain){

   table.clear();

   //Initializing cells with pointer.
   for(std::size_t elemental_idx = 0; elemental_idx < chain.size(); elemental_idx++){

      if(chain[elemental_idx].domain_dim() <= chain[elemental_idx].codomain_dim()){

         table.emplace_back(&chain[elemental_idx], tangent_cost(&chain[elemental_idx]),
               Operation::TANGENT);
      }

      else{

         if(memory_bound){

            if(chain[elemental_idx].number_edges() <= *memory_bound){
               
               table.emplace_back(&chain[elemental_idx], adjoint_cost(&chain[elemental_idx]),
                     Operation::ADJOINT);
            }
            //Memory bound is exceeded then the elemental jacobian is accumulated using tangent
            //mode.
            else{

               table.emplace_back(&chain[elemental_idx], tangent_cost(&chain[elemental_idx]),
                     Operation::TANGENT);
            }
         }

         else{

            table.emplace_back(&chain[elemental_idx], adjoint_cost(&chain[elemental_idx]),
                  Operation::ADJOINT);
         }

      }
   }

   std::size_t minimum_cost, minimum_cost_j_k_i, split_position;
   std::size_t split_position_j_k_i;
   std::size_t i_index, k_index, tangent_cost_j_k_i;
   std::optional<std::size_t> adjoint_cost_j_k_i;
   Operation operation, operation_j_k_i;

   for(std::size_t j_index = 1; j_index < chain.size(); j_index++){
   
      for(std::size_t aux_var_i = 1; aux_var_i < j_index + 1; aux_var_i++){

         i_index = j_index - aux_var_i;
         minimum_cost = std::numeric_limits<std::size_t>::max();

         //k_index split position.
         //Problem instance (j_index, i_index)
         for(std::size_t aux_var_k = 1; aux_var_k <= j_index - i_index; aux_var_k++){ 
            //Decreasing order. From j_index -1 to i_index
            k_index = j_index - aux_var_k;
            
            minimum_cost_j_k_i = multiplication_cost(j_index, k_index, i_index);
            split_position_j_k_i = k_index;
            operation_j_k_i = Operation::MULTIPLICATION;

            //Minimum over the different accumulation methods.
            tangent_cost_j_k_i = tangent_cost(j_index, k_index, i_index);
            if(tangent_cost_j_k_i < minimum_cost_j_k_i){

               minimum_cost_j_k_i = tangent_cost_j_k_i;
               operation_j_k_i = Operation::TANGENT;
            }

            adjoint_cost_j_k_i = adjoint_cost(j_index, k_index, i_index);
            if(adjoint_cost_j_k_i){

               if(*adjoint_cost_j_k_i < minimum_cost_j_k_i){

                  minimum_cost_j_k_i = *adjoint_cost_j_k_i;
                  operation_j_k_i = Operation::ADJOINT;
               }
            }

            //Compare the minimum cost over all accumulation methods
            //for split position k_index with the current minimum for
            // the problem instance (j_index, i_index)
            if(minimum_cost_j_k_i < minimum_cost){

               minimum_cost = minimum_cost_j_k_i;
               split_position = split_position_j_k_i;
               operation = operation_j_k_i;
            }
         }
         table.emplace_back(minimum_cost, split_position, operation);
      }
   }
}

/**
 * @namespace tool_box_sparse
 * @brief Helper algorithms for estimating operational costs taylored for sparse Jacobian matrices and
 * sparse Jacobian subchains.
 *
 * @details Contains core cost models for forward (tangent) mode, reverse (adjoint) mode and sparse 
 * matrix-matrix multiplication. From compressed matrix formats (CSR/CSC) Column and Row Intersection 
 * graphs are build, which are later used run greedy coloring algorithms to obtain row and column
 * chromatic number estimates.
 *
 * @note Shared across 'Fill_table<Sparse_Jacobian, Matrix_free_sparse_information>' and 
 * 'Fill_table<Split_sparse_Jacobian, Split_reversal_sparse_information>'.
 */
namespace tool_box_sparse{

   /**
    * @brief Calculates forward-mode (tangent) preaccumulation cost for an elemental sparse Jacobian. 
    * 
    * @note Instead of scaling the number of edges in the computational graph with the domain space 
    * dimension, the column number of colors estimate is used.
    *
    * @tparam Jacobian_T Elemental sparse Jacobian type.
    * @param elemental_jacobian_ptr Pointer to the target sparse elemental Jacobian. 
    * @return std::size_t Total operations required for tangent preaccumulation. 
    */
   template<class Jacobian_T>
   std::size_t tangent_cost(const Jacobian_T* elemental_jacobian_ptr){
      
      return elemental_jacobian_ptr -> get_column_number_colors() *
               elemental_jacobian_ptr -> number_edges();
   }

   /**
    * @brief Calculates reverse-mode (adjoint) preaccumulation cost for an elemental sparse Jacobian.
    *
    * @note Instead of scaling the number of edges in the computational graph with the codomain space 
    * dimension, the row number of colors estimate is used.
    * 
    * @tparam Jacobian_T Elemental sparse Jacobian type.
    * @param elemental_jacobian_ptr Pointer to the target sparse elemental Jacobian. 
    * @return std::size_t Total operations required for adjoint preaccumulation. 
    */
   template<class Jacobian_T>
   std::size_t adjoint_cost(const Jacobian_T* elemental_jacobian_ptr){

      return elemental_jacobian_ptr -> get_row_number_colors() *
               elemental_jacobian_ptr -> number_edges();
   }

   /**
    * @brief Sums computational graph edges across a subchain range [j_or_k, k_plus_1_or_i]. 
    *
    * @tparam Jacobian_T Elemental sparse Jacobian type.
    * @param table Reference to the DP lookup table containing subproblem evaluations.
    * @param j_or_k Upper index bound (inclusive).
    * @param k_plus_1_or_i Lower index bound (inclusive).
    * @return std::size_t Total edge count across the specified subchain. 
    */
   template<class Jacobian_T>
   std::size_t accumulate_edges(const Table<Jacobian_T>& table, std::size_t j_or_k,
                                 std::size_t k_plus_1_or_i){
      //Access Cell with pointer
      if(j_or_k == k_plus_1_or_i){

         return table.get_cell(j_or_k).number_edges();
      }
      else{
         
         return table.get_cell(j_or_k, k_plus_1_or_i).number_edges();
      }
   }

   /**
    * @brief Tangent accumulation cost for subchains of size greater than one.
    *
    * @tparam Jacobian_T Elemental sparse Jacobian type.
    * @param table Reference to the DP lookup table.
    * @param j Upper index bound of the subchain (inclusive).
    * @param k Split index within range [i, j-1]
    * @param i Lower index bound of the subchain (inclusive).
    * @param j_i_column_number_colors Pre-calculated column colors for subchain range (j,i).
    * @return std::size_t Combined cost of preaccumulation (k, i) plus tangent evaluation of (j, k+1).
    */
   template<class Jacobian_T>
   std::size_t tangent_cost(const Table<Jacobian_T>& table, std::size_t j, std::size_t k,
                              std::size_t i, std::size_t j_i_column_number_colors){
      //Access cell with pointer
     if(k == i){

        return table.get_cell(i).accumulated_cost() +
               j_i_column_number_colors * accumulate_edges(table, j, k+1);
     }
     //Access normal Cell
     else{

         return table.get_cell(k, i).accumulated_cost() +
               j_i_column_number_colors * accumulate_edges(table, j, k+1);
     }

   }

   /**
    * @brief Calculates upper-bound cost CSR/CSC sparse matrix-matrix multiplication. 
    *
    * @details Models the cost of sparse matrix multiplication of sparse Jacobian objects representing
    * sparse data from (j, k+1) and (k, i). An upper bound cost estimate is developed using number 
    * of non-zero entries ('nnz') and maximum non-zero row/column bound of operands (j, k+1) and (k, i).
    *
    * @tparam Jacobian_T Elemental sparse Jacobian type.
    * @param table Reference to the DP lookup table.
    * @param j Upper index bound of the subchain (inclusive).
    * @param k Split index within range [i, j-1]
    * @param i Lower index bound of the subchain (inclusive).
    * @param number_nnz Number of non-zero elements in resulting product sparse Jacobian matrix.
    * @return std::size_t Total cost: preaccumulation plus sparse matrix multiplication cost. 
    */
   template<class Jacobian_T>
   std::size_t multiplication_cost(const Table<Jacobian_T>& table, std::size_t j, std::size_t k,
                                    std::size_t i, std::size_t number_nnz){

      std::size_t accumulated_cost_lhs, accumulated_cost_rhs;
      std::size_t max_number_nnz_row, max_number_nnz_column;
      //Access Cell_with_pointer
      if(k == i){

         accumulated_cost_rhs = table.get_cell(i).accumulated_cost();
         max_number_nnz_column = table.get_cell(i).max_number_nnz_column();
      }
      else{
         
         accumulated_cost_rhs = table.get_cell(k, i).accumulated_cost();
         max_number_nnz_column = table.get_cell(k, i).max_number_nnz_column();
      }

      //Access Cell_with_pointer
      if(j == k+1){

         accumulated_cost_lhs = table.get_cell(j).accumulated_cost();
         max_number_nnz_row = table.get_cell(j).max_number_nnz_row();
      }
      else{
         
         accumulated_cost_lhs = table.get_cell(j, k+1).accumulated_cost();
         max_number_nnz_row = table.get_cell(j, k+1).max_number_nnz_row();
      }

      return accumulated_cost_rhs + accumulated_cost_lhs +
               number_nnz * 
               std::min(max_number_nnz_column, max_number_nnz_row);
   }

   /**
    * @brief Multiplies subchain sparse Jacobians to obtain a resultant sparse Jacobian object. 
    *
    * @tparam Jacobian_T Elemental sparse Jacobian type.
    * @param table Reference to the DP lookup table. 
    * @param j Upper index bound of the subchain (inclusive).
    * @param i Lower index bound of the subchain (inclusive).
    * @return const Jacobian_T Resulting sparse Jacobian object containing relevant sparse data of
    * subchain (j,i).
    */
   template<class Jacobian_T>
   const Jacobian_T propagate_sparsity_pattern(const Table<Jacobian_T>& table, std::size_t j,
                                                std::size_t i){
      //Accessing cells with pointer
      if(j == i + 1){

         return table.get_cell(j).get_jacobian() *
                  table.get_cell(i).get_jacobian();
      }
      
      else{
         
         return table.get_cell(j, i+1).get_jacobian() *
                  table.get_cell(i).get_jacobian();
      }
   }

}

/**
 * @brief Dynamic programing table-filler for unconstrained or memory-bounded sparse Jacobian chains. 
 *
 * @details Evaluates subproblems (j, i) agains MULTIPLICATION, TANGENT, and ADJOINT strategies 
 * using matrix sparsity to minimize cumulative accumulation operations.
 */
template<>
class Fill_table<Sparse_Jacobian, Matrix_free_sparse_information>{
 public:
   /**
    * @brief Constructs and solves DP table without memory limits.
    *
    * @param chain Sequence of sparse Jacobians to optimize.
    */
   Fill_table(const Jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>& chain):
      table(chain.size()){
      
      fill(chain);
   }

   /**
    * @brief Constructs and solves DP table subject to an edge-capacity memory limit. 
    * 
    * @param chain Sequence of sparse Jacobians to optimize.
    * @param memory_bound_ Maximum allowed graph edges active during reversal-mode (adjoint). 
    */
   Fill_table(const Jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>& chain,
                  std::size_t memory_bound_):
      table(chain.size()), memory_bound(memory_bound_){

      fill(chain);
   }

   /**
    * @brief Constructs engine with a pre-populated DP table (primarily for unit testing). 
    *
    * @param table_ Storage table instance moved into class ownership. 
    */
   Fill_table(Table<Sparse_Jacobian> table_):
      table(std::move(table_)){}
   
   /**
    * @brief Sums total computational graph edges across a subchain [j_or_k, k_plus_1_or_i].
    *
    * @note Forwards the execution to tool_box_sparse::accumulate_edges using the internal 'table'.
    *
    * @param j_or_k Upper index bound.
    * @param k_plus_1_or_i Lower index bound.
    * @return Accumulated sum of edges across subchain range.
    */
   std::size_t accumulate_edges(std::size_t j_or_k, std::size_t k_plus_1_or_i) const{

      return tool_box_sparse::accumulate_edges(table, j_or_k, k_plus_1_or_i);
   }

   /**
    * @brief Computes forward-mode (tangent) AD preaccumulation cost for a single sparse elemental Jacobian.
    * @see tool_box_sparse::tangent_cost.
    */
   static std::size_t tangent_cost(const Sparse_Jacobian* elemental_jacobian_ptr) {

      return tool_box_sparse::tangent_cost(elemental_jacobian_ptr);
   }

   /**
    * @brief Computes reverse-mode (adjoint) AD preaccumulation cost for a single sparse elemental Jacobian.
    * @see tool_box_sparse::adjoint_cost.
    */
   static std::size_t adjoint_cost(const Sparse_Jacobian* elemental_jacobian_ptr) {

      return tool_box_sparse::adjoint_cost(elemental_jacobian_ptr);
   }

   /**
    * @brief Calculates tangent-mode accumulation cost for subproblem instance (j,i) split at k.
    *
    * @note Forwards execution to tool_box_sparse::tangent_cost using internal 'table'.
    *
    * @param j Upper index bound (inclusive).
    * @param k Split index.
    * @param i Lower index bound (inclusive).
    * @return Accumulated forward-mode cost for subproblem (j, i).
    */
   std::size_t tangent_cost(std::size_t j, std::size_t k, std::size_t i,
                              std::size_t j_i_column_number_colors) const{

      return tool_box_sparse::tangent_cost(table, j, k, i, j_i_column_number_colors);
   }

   /**
    * @brief Computes reverse-mode (adjoint) cost for subchain (j, i) split at k with and without 
    * memory constraint.
    *
    * @param j Upper index bound of the subchain (inclusive).
    * @param k Split index within range [i, j-1]
    * @param i Lower index bound of the subchain (inclusive).
    * @param j_i_row_number_colors Pre-calculated row colors for subchain (j, i).
    * @return std::optional Total adjoint accumulation cost if memory_bound is not exceeded; 
    * 'std::nullopt' if exceeded.
    */
   std::optional<std::size_t> adjoint_cost(std::size_t j, std::size_t k, std::size_t i,
                                             std::size_t j_i_row_number_colors) const{
      
      std::size_t total_edges = accumulate_edges(k, i);
      //Access a Cell_with_pointer
      if(j == k+1){

         //If memory bound is set
         if(memory_bound){

            //If memory bound is not surpassed. 
            if(total_edges <= *memory_bound){

               return table.get_cell(j).accumulated_cost() +
                        j_i_row_number_colors * total_edges;
            }

            //If memory bound is surpassed
            return {};
         }

         return table.get_cell(j).accumulated_cost() +
                  j_i_row_number_colors * total_edges;
      }
      //Access a normal cell
      else{

         //If memory bound is set.
         if(memory_bound){
            
            if(total_edges <= *memory_bound){

               return table.get_cell(j, k+1).accumulated_cost() +
                        j_i_row_number_colors * total_edges;
            }

            return {};
         }

         return table.get_cell(j, k+1).accumulated_cost() +
                  j_i_row_number_colors * total_edges;
      }
   }

   /**
    * @brief Calculates sparse Jacobian multiplication between sparse Jacobian objects representing 
    * the subchains (j, k+1) and (k, i).
    *
    * @note Forwards execution to tool_box_sparse::multiplication_cost using internal 'table'.
    * @param j Upper index bound (inclusive).
    * @param k Split index.
    * @param i Lower index bound (inclusive).
    * @return Total dense matrix-matrix multiplication cost.
    */
   std::size_t multiplication_cost(std::size_t j, std::size_t k, std::size_t i,
                                    std::size_t number_nnz) const{

      return tool_box_sparse::multiplication_cost(table, j, k, i, number_nnz);
   }

   /**
    * @brief Multiplies subchain sparse Jacobians to obtain a resultant sparse Jacobian object. 
    *
    * @note Forwards execution to tool_box_sparse::propagate_sparsity_pattern using internal 'table'.
    * @param j Upper index bound.
    * @param i Lower index bound.
    * @return consst Sparse_Jacobian sparse Jacobian object representation of subchain (j,i) Jacobian
    * matrix.
    */
   const Sparse_Jacobian propagate_sparsity_pattern(std::size_t j, std::size_t i) const{

      return tool_box_sparse::propagate_sparsity_pattern(table, j, i);
   }

   /// Returns configured memory bound threshold, if set.
   std::optional<std::size_t> get_memory_bound() const{
      
      return memory_bound;
   }

   /**
    * @brief Access Cell with non owning pointer to elemental sparse Jacobian at index 'index' in 
    * the chain.
    *
    * @param index sparse Jacobian matrix index in the chain.
    * @return Const reference to 'Cell_with_pointer<Sparse_Jacobian>' object.
    */
   const Cell_with_pointer<Sparse_Jacobian>& get_cell(std::size_t index){
      
      return table.get_cell(index);
   }

   /**
    * @brief Access cell at (j, i) containing optimal accumulation and sparsity data. 
    *
    * @param j End index of the subchain (1-based, $j > i$).
    * @param i Start index of the subchain (0-based).
    * @return Const reference to 'cell<Sparse_Jacobian>' object.
    */
   const Cell<Sparse_Jacobian>& get_cell(std::size_t j, std::size_t i){

      return table.get_cell(j,i);
   }

   /**
    * @brief Access the last Cell in the DP table.
    *
    * @pre fill method must be executed first before calling this method.
    * @return Const reference to 'Cell<Sparse_Jacobian>' object.
    */
   const Cell<Sparse_Jacobian>& back(){
      return table.back();
   }

   /**
    * @brief Get reference to table.
    *
    * @return Const reference to 'Table<Sparse_Jacobian>'
    */
   const Table<Sparse_Jacobian>& get_table(){
      return table;
   }


 private:
   /// Dynamic programming table structure storing subproblems optimal results.
   Table<Sparse_Jacobian> table;

   /// Optional edge limit constraint for adjoint evaluations.
   std::optional<std::size_t> memory_bound;

   /**
    * @brief Populates DP table by computing optimal computational costs over increasing 
    * subproblems lengths.
    *
    * @details Steps:
    * 1. Initializes Cell_with_pointer objects with base optimal costs for preaccumulation of 
    * elemental Jacobians and corresponding non-owning pointers to chain elements.
    * 2. Initializes Cell objects with subchain optimal costs and split positions. 
    *
    * @details Subproblems are evaluated in the following order:
    * (1, 0)
    * (2, 1) (2, 0)
    * ...
    *
    * @param chain Target Jacobian chain instance.
    */
   void fill(const Jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>& chain);
};

void Fill_table<Sparse_Jacobian, Matrix_free_sparse_information>::fill(
      const Jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>& chain){

   table.clear();

   //Initializing Cells with pointer.
   for(std::size_t elemental_idx = 0; elemental_idx < chain.size(); elemental_idx++){

      if(chain[elemental_idx].get_column_number_colors() <=
            chain[elemental_idx].get_row_number_colors()){

         table.emplace_back(&chain[elemental_idx], tangent_cost(&chain[elemental_idx]),
                              Operation::TANGENT);
      }

      else{

         if(memory_bound){


            if(chain[elemental_idx].number_edges() <= *memory_bound){
               
               table.emplace_back(&chain[elemental_idx], adjoint_cost(&chain[elemental_idx]),
                        Operation::ADJOINT);
            }
            //Memory bound is exceeded. The elemental jacobian is accumulated using tangent mode.
            else{

               table.emplace_back(&chain[elemental_idx], tangent_cost(&chain[elemental_idx]),
                     Operation::TANGENT);
            }
         }

         else{

            table.emplace_back(&chain[elemental_idx], adjoint_cost(&chain[elemental_idx]),
                  Operation::ADJOINT);
         }
      }
   }

   std::size_t minimum_cost, minimum_cost_j_k_i, split_position;
   std::size_t split_position_j_k_i;
   std::size_t i_index, k_index, tangent_cost_j_k_i;
   std::optional<std::size_t> adjoint_cost_j_k_i;
   Operation operation, operation_j_k_i;

   for(std::size_t j_index = 1; j_index < chain.size(); j_index++){

      for(std::size_t aux_var_i = 1; aux_var_i < j_index + 1; aux_var_i++){

         i_index = j_index - aux_var_i;
         minimum_cost = std::numeric_limits<std::size_t>::max();
         
         //Propagating sparsity information.
         Sparse_Jacobian sparse_jacobian_j_i{propagate_sparsity_pattern(j_index, i_index)};
         //k_index split_position.
         //Problem instance (j_index, i_index)
         for(std::size_t aux_var_k= 1; aux_var_k <= j_index - i_index; aux_var_k++){
            //Decreasing order. From j_index - 1 to i_index
            k_index = j_index - aux_var_k;

            minimum_cost_j_k_i = multiplication_cost(j_index, k_index,
                                 i_index, sparse_jacobian_j_i.number_nnz());

            split_position_j_k_i = k_index;
            operation_j_k_i = Operation::MULTIPLICATION;

            //Minimum over the different accumulation methods.
            tangent_cost_j_k_i = tangent_cost(j_index, k_index,
                                 i_index, sparse_jacobian_j_i.get_column_number_colors());

            if(tangent_cost_j_k_i < minimum_cost_j_k_i){

               minimum_cost_j_k_i = tangent_cost_j_k_i;
               operation_j_k_i = Operation::TANGENT;
            }

            adjoint_cost_j_k_i = adjoint_cost(j_index, k_index,
                                 i_index, sparse_jacobian_j_i.get_row_number_colors());

            if(adjoint_cost_j_k_i){

               if(*adjoint_cost_j_k_i < minimum_cost_j_k_i){

                  minimum_cost_j_k_i = *adjoint_cost_j_k_i;
                  operation_j_k_i = Operation::ADJOINT;
               }
            }

            //Compare the minimum cost over all accumulation methods
            //for split position k_index with the current minimum for
            // the problem instance (j_index, i_index)
            if(minimum_cost_j_k_i < minimum_cost){

               minimum_cost = minimum_cost_j_k_i;
               split_position = split_position_j_k_i;
               operation = operation_j_k_i;
            }
         }
         table.emplace_back(std::move(sparse_jacobian_j_i), minimum_cost,
                              split_position, operation);

      }
   }
}

/**
 * @namespace tool_box_split
 * @brief Helper functions providing tools to facilitate the post processing of a optimally 
 * accumulated adjoint subchain using Break points.
 *
 */
namespace tool_box_split{

   /**
    * @brief Sums the total execution cost across a subchain (j,i) (inclusive). 
    *
    * @param subprograms_execution_cost subprograms cost array in fmas.
    * @param j Upper index bound of the subchain (inclusive).
    * @param i Lower index bound of the subchain (inclusive).
    * @return std::size_t Accumulated function execution cost.
    */
   std::size_t accumulate_subprograms_cost(const std::vector<size_t>& subprograms_execution_cost,
                                          std::size_t j, std::size_t i){

      std::size_t sum_function_cost = 0;
      for(std::size_t idx = i; idx < j + 1; idx++){

         sum_function_cost += subprograms_execution_cost[idx];
      }

      return sum_function_cost;
   }

   /**
    * @brief Determines wether a subchain (j,i) can be accumulated in reverse mode via split reversal.
    *
    * @details Checks if every single subprogram in the range [i, j] satisfies the memory constraint
    * individually.
    * If any subprogram's computational graph edge count exceeds memory_bound, reverse-mode accumulation 
    * is impossible.
    *
    * @tparam Table_T Table type, either Table<Dense_Jacobian> or Table<Sparse_Jacobian>.
    * @param table Reference to the dynammic programming (DP) lookup table.
    * @pre Cells with non-owning pointers are already initialized.
    * @param j Upper index bound of the subchain (inclusive). 
    * @param i Lower index bound of the subchain (inclusive).
    * @param memory_bound Maximum allowable number of edges in the computational graph stored simultaneously. 
    * @return true If the subchain is split-reversible.
    * @return false Otherwise.
    */
   template<class Table_T>
   bool is_split_reversable(const Table_T& table, std::size_t j, std::size_t i,
                              std::size_t memory_bound){

      for(std::size_t idx = i; idx < j + 1; idx++){

         if(memory_bound < table.get_cell(idx).number_edges()){

            return false;
         }
      }

      return true;
   }

   /**
    * @brief Finds the minimal split index k_tilde such that the subchain (k, k_tilde + 1) fits 
    * in memory.
    *
    * @pre is_split_reversable must evaluate to true for the target subchain before calling this function.
    * 
    * @param table Reference to Dense_Jacobian dynamic programming (DP) lookup table.
    * @param i Lower index bound of the subchain (inclusive).
    * @param accumulated_edges_j_i Total accumulated edges across the subchain.
    * @param memory_bound Maximum allowable edge capacity.
    * @return std::size_t The resulting split index k_tilde.
    */
   std::size_t split_reversed_chain(const Table<Dense_Jacobian>& table, std::size_t j,
                                       std::size_t i, std::size_t memory_bound){

      std::size_t accumulated_edges_j_i = tool_box_dense::accumulate_edges(table, j, i);  

      std::size_t counter = 0;

      while(memory_bound < accumulated_edges_j_i){

         accumulated_edges_j_i -= table.get_cell(i + counter).number_edges();
         counter ++;
      }

      counter--;

      return i + counter;
   }

   /**
    * @brief Finds the minimal split index k_tilde such that the subchain (k, k_tilde + 1) fits 
    * in memory.
    *
    * @pre is_split_reversable must evaluate to true for the target subchain before calling this function.
    * 
    * @param table Reference to Sparse_Jacobian dynamic programming (DP) lookup table.
    * @param i Lower index bound of the subchain (inclusive).
    * @param accumulated_edges_j_i Total accumulated edges across the subchain.
    * @param memory_bound Maximum allowable edge capacity.
    * @return std::size_t The resulting split index k_tilde.
    */
   std::size_t split_reversed_chain(const Table<Sparse_Jacobian>& table, std::size_t j,
                                       std::size_t i, std::size_t memory_bound){

      std::size_t accumulated_edges_j_i = tool_box_sparse::accumulate_edges(table, j, i);  

      std::size_t counter = 0;

      while(memory_bound < accumulated_edges_j_i){

         accumulated_edges_j_i -= table.get_cell(i + counter).number_edges();
         counter ++;
      }

      counter--;

      return i + counter;
   }
}

#endif ///FILL_TABLE
