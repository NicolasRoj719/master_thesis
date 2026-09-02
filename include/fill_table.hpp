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
class fill_table{};

/**
 * @brief Explicit specialization of 'fill_table' for 'Jacobian' and 'Jacobian_information' types.
 *
 * Computes optimal bracketing sequence to minimize total computation cost in terms of Fused 
 * Multiply-Add (fma) operations, restricted to dense matrix-matrix multiplications.
 */
template<>
class fill_table<Jacobian, Jacobian_information>{
 public:
   /**
    * @brief Constructs table-filler engine and executes the dynamic programming algorithm.
    *
    * @param chain Target Jacobian chain containing sequence metadata.
    */
   fill_table(const jacobian_chain<Jacobian, Jacobian_information>& chain):
      table(chain.size()){

      fill(chain);
   }

   /**
    * @brief Constructs a table-filler with pre-populated or mock Table instance (e.g., for testing). 
    *
    * @param table_ External Table instance moved into internal storage.
    */
   fill_table(Table<Jacobian> table_): table(std::move(table_)){}

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
    * @brief Accesses non-owning cell reference to an elemental Jacobian in the chain.
    *
    * @param index 0-based position in the chain.
    * @return Const reference to 'cell_with_pointer<Jacobian>' object.
    */
   const cell_with_pointer<Jacobian>& get_cell(std::size_t index){

      return table.get_cell(index);
   }

   /**
    * @brief Accesses matrix cell at \f$(j,i)\f$ containing subproblem optimal data.
    *
    * @param j End index of subchain.
    * @param i Start index of subchain.
    * @return Const reference to 'cell<Jacobian>' object.
    */
   const cell<Jacobian>& get_cell(std::size_t j, std::size_t i){

      return table.get_cell(j,i);
   }
 
 private:
   /// Internal DP lookup table.
   Table<Jacobian> table;

   /**
    * @brief Fills DP lookup table by computing optimal computational costs over increasing
    * subproblems lengths. 
    *
    * Steps:
    * 1. Initializes cell_with_pointer objects with non-owning pointers to chain elements.
    * 2. Initializes cell objects with subchain optimal costs and split positions.
    * 
    * Subproblems are evaluated in the following order:
    * (1,0)
    * (2,1) (2,0)
    * @param chain Target Jacobian chain instance.
    */
   void fill(const jacobian_chain<Jacobian, Jacobian_information>& chain);
};


void fill_table<Jacobian, Jacobian_information>::fill(
      const jacobian_chain<Jacobian, Jacobian_information>& chain){

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
      //Access cell_with_pointer
      if(k == i){
         
         accumulated_cost_rhs = table.get_cell(i).accumulated_cost();
      }
      else{

         accumulated_cost_rhs = table.get_cell(k,i).accumulated_cost();
      }

      //Access cell_with_pointer
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
class fill_table<Dense_Jacobian, Matrix_free_information>{
 public:
   /**
    * @brief Constructs and fills dynamic programming (DP) table without memory bound.
    *
    * @param chain Sequence of elemental Jacobians to optimize.
    */
   fill_table(const jacobian_chain<Dense_Jacobian, Matrix_free_information>& chain):
      table(chain.size()){

      fill(chain);
   }

   /**
    * @brief Constructs and fills dynamic programming (DP) table under a memory constraint.
    *
    * @param chain Sequence of elemental Jacobians to optimize.
    * @memory_bound_ Maximum allowed edge storage limit for executing Adjoint mode accumulation.
    */
   fill_table(const jacobian_chain<Dense_Jacobian, Matrix_free_information>& chain,
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
   fill_table(Table<Dense_Jacobian> table_):
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
      //Access a cell_with_pointer
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
         //Access a normal cell
         else{
            
            return table.get_cell(j).accumulated_cost() +
               table.get_cell(j).codomain_dim() * total_edges; 
         }
      }
      //Access normal cell
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
    * @brief Claculates standard dense matrix multiplication cost of two preaccumulates Jacobians
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
    * @brief Accesses dynamic programming cell with non-owning pointer to elemental Jacobian at a given index.
    *
    * @param index Jacobian matrix index in the chain.
    * @return Const reference to 'cell_with_pointer<Dense_Jacobian>' object.
    */
   const cell_with_pointer<Dense_Jacobian>& get_cell(std::size_t index){
      
      return table.get_cell(index);
   }

   /**
    * @brief .Accesses dynamic programming cell at \f$(j, i)\f$ storing optimal subproblem result.
    *
    * @param j End index of the subchain (1-based, $j > i$).
    * @param i Start index of the subchain (0-based).
    * @return Const reference to 'cell<Dense_Jacobian>' object.
    */
   const cell<Dense_Jacobian>& get_cell(std::size_t j, std::size_t i){

      return table.get_cell(j,i);
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
    * 1. Initializes cell_with_pointer objects with base optimal costs for preaccumulation of 
    * elemental Jacobians and corresponding non-owning pointers to chain elements.
    * 2. Initializes cell objects with subchain optimal costs and split positions. 
    *
    * @details Subproblems are evaluated in the following order:
    * (1, 0)
    * (2, 1) (2, 0)
    * ...
    *
    * @param chain Target Jacobian chain instance.
    */
   void fill(const jacobian_chain<Dense_Jacobian, Matrix_free_information>& chain);
};

void fill_table<Dense_Jacobian, Matrix_free_information>::fill(
      const jacobian_chain<Dense_Jacobian, Matrix_free_information>& chain){

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
 * @namespace tool_box_split
 * @brief Helper functions providing tools to facilitate the embeeding of binomial checkpointig into
 * the Jacobian chain product bracketing dynamic programming formulation.
 *
 * @node Shared across 'fill_table<Split_dense_Jacobian, Split_reversal_dense_information>' and
 *  'fill_table<Split_sparse_Jacobian, Split_reversal_sparse_information>'.
 *
 */
namespace tool_box_split{

   /**
    * @brief Sums the total execution cost across a subchain (j,i) (inclusive). 
    *
    * @tparam Split_T Split elemental Jacobian type.
    * @param table Reference to the dynamic programming (DP) lookup table.
    * @pre Cells with non-owning pointers are already initialized.
    * @param j Upper index bound of the subchain (inclusive).
    * @param i Lower index bound of the subchain (inclusive).
    * @return std::size_t Accumulated function execution cost.
    */
   template<class Split_T>
   std::size_t accumulate_function_cost(const Table<Split_T>& table, std::size_t j, std::size_t i){

      std::size_t sum_function_cost = 0;
      for(std::size_t idx = i; idx < j + 1; idx++){

         sum_function_cost += table.get_cell(idx).function_cost();
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
    * @tparam Split_T Split elemental Jacobian type.
    * @param table Reference to the dynammic programming (DP) lookup table.
    * @pre Cells with non-owning pointers are already initialized.
    * @param j Upper index bound of the subchain (inclusive). 
    * @param i Lower index bound of the subchain (inclusive).
    * @param memory_bound Maximum allowable number of edges in the computational graph stored simultaneously. 
    * @return true If the subchain is split-reversible.
    * @return false Otherwise.
    */
   template<class Split_T>
   bool is_split_reversable(const Table<Split_T>& table, std::size_t j, std::size_t i,
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
    * within memory.
    *
    * @pre is_split_reversable must evaluate to true for the target subchain before calling this function.
    * 
    * @tparam Split_T Split elemental Jacobian type.
    * @param table Reference to the dynamic programming (DP) lookup table.
    * @param i Lower index bound of the subchain (inclusive).
    * @param accumulated_edges_j_i Total accumulated edges across the subchain.
    * @param memory_bound Maximum allowable edge capacity.
    * @return std::size_t Te resulting split index k_tilde.
    */
   template<class Split_T>
   std::size_t split_reversed_chain(const Table<Split_T>& table, std::size_t i,
                                    std::size_t accumulated_edges_j_i, std::size_t memory_bound){

      std::size_t counter = 0;
      while(memory_bound < accumulated_edges_j_i){

         accumulated_edges_j_i -= table.get_cell(i + counter).number_edges();
         counter ++;
      }

      counter--;

      return i + counter;
   }
}


/**
 * @brief Explicit specialization of 'fill_table' for 'Split_dense_Jacobian' and 
 * 'Split_reversal_dense_information'.
 *
 * @details Embeds Binomial Checkpointing into the Dense Jacobian Chain Product Bracketing DP
 * formulation to handle memory-constrained reverse-mode automatic differentiation.
 *
 * @details THIS CLASS IS EXPERIMENTAL. EMBEEDING BINOMIAL CHECKPOINTING INTO THE OPTIMAL 
 * BRACKETING FORMULATION HAD NO BENEFITIAL EFFECT ON THE OUTCOME OF THE ALGORITHM. THEREFORE 
 * ITS USED IS DISCOURGE. THE METHODS CONTAINED INSIDE ARE NEVER THE LESS USEFUL, WHICH JUSTIFIES 
 * THE EXISTANCE OF THIS TEMPLATE SPECIALIZATION.
 */
template<>
class fill_table<Split_dense_Jacobian, Split_reversal_dense_information>{
 public:
   /**
    * @brief Constructs the dynamic programming table-filler engine 
    *
    * @note The dynamic programming table population call 'fill()' is currently commented out in this 
    * constructor, as the embedded binomial checkpointing model does not fit correctly into the initial 
    * formulation.
    *
    * @param chain Target Split Jacobian chain containing split reversal metadata (dense reversal 
    * metadata plus subprograms execution costs). 
    * @param memory_bound Maximum allowable number of edges in the computational graph stored simultaneously. 
    * @param number_checkpoints_ Total number of available checkpoints for the optimization algorithms. 
    */
   fill_table(const jacobian_chain<Split_dense_Jacobian, Split_reversal_dense_information>& chain_,
                  std::size_t memory_bound_, std::size_t number_checkpoints_):
      table(chain_.size()), chain(chain_), memory_bound(memory_bound_),
      number_checkpoints(number_checkpoints_){

      /* fill(); */
   }

   /**
    * @brief Sums total computational graph edges across a subchain [j_or_k, k_plus_1_or_i]
    *
    * @note Forwards the execution to tool_box_dense::accumulate_edges using the internal 'table'.
    *
    * @param j_or_k Upper index bound.
    * @param k_plus_1_or_i Lower index bound.
    * @return Accumulated sum of edges across subchain range.
    *
    */
   std::size_t accumulate_edges(std::size_t j_or_k, std::size_t k_plus_1_or_i) const{

      return tool_box_dense::accumulate_edges(table, j_or_k, k_plus_1_or_i);
   }

   /**
    * @brief Computes forward-mode (tangent) AD preaccumulation cost for a single elemental Jacobian.
    * @see tool_box_dense::tangent_cost.
    */
   static std::size_t tangent_cost(const Split_dense_Jacobian* elemental_jacobian_ptr){

      return tool_box_dense::tangent_cost(elemental_jacobian_ptr);
   }

   /**
    * @brief Computes reverse-mode (adjoint) AD preaccumulation cost for a single elemental Jacobian.
    * @see tool_box_dense::adjoint_cost.
    */
   static std::size_t adjoint_cost(const Split_dense_Jacobian* elemental_jacobian_ptr){

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
   std::size_t tangent_cost(std::size_t j, std::size_t k, std::size_t i){

      return tool_box_dense::tangent_cost(table, j, k, i);
   }

   /**
    * @brief Sums the total execution cost across a subchain (j, i) (inclusive).
    *
    * @note Forwards execution to tool_box_split::accumulate_function_cost using internal 'table'.
    *
    * @param j Upper index bound (inclusive).
    * @param i Lower index bound (inclusive).
    * @return std::size_t Accumulated function execution cost.
    */
   std::size_t accumulate_function_cost(std::size_t j, std::size_t i){

      return tool_box_split::accumulate_function_cost(table, j, i);
   }

   /**
    * @brief Checks if the subchain (j, i) is split reversable. 
    *
    * @note Forwards execution to tool_box_split::is_split_reversable  using internal 'table' and 
    * 'memory_bound'.
    *
    * @param j Upper index bound (inclusive).
    * @param i Lower index bound (inclusive).
    */
   bool is_split_reversable(std::size_t j, std::size_t i){
      
      return tool_box_split::is_split_reversable(table, j, i, memory_bound);
   }

   /**
    * @brief Finds the minimal split index k_tilde such that the subchain (k, k_tilde + 1) fits 
    * within memory.
    * 
    * @note Forwards execution to tool_box_split::split_reversed_chain using the internal 'table' and
    * 'memory_bound'.
    *
    * @param i Lower index bound.
    * @param accumulated_edges_j_i Overall subchain (j, i) total number of edges.
    * @return std::size_t Split index.
    */
   std::size_t split_reversed_chain(std::size_t i, std::size_t accumulated_edges_j_i){
      
      return tool_box_split::split_reversed_chain(table, i, accumulated_edges_j_i, memory_bound);
   }


   /**
    * @brief Constructs a temporary 'Split_dense_Jacobian' metadata object for the subchain (j, i). 
    *
    * @param j End index of new split Jacobian object.
    * @param i Start index of new split Jacobian object.
    * @return Split_dense_Jacobian Container for structural metadata and total subchain execution cost. 
    */
   Split_dense_Jacobian joint_jacobian(std::size_t j, std::size_t i){

      return Split_dense_Jacobian{table.get_cell(i).domain_dim(),
                                    table.get_cell(j).codomain_dim(),
                                    accumulate_edges(j,i),
                                    accumulate_function_cost(j,i)};
   }

   /**
    * @brief Computes an unconstrained lower-bound estimate for the adjoint accumulation cost.
    *
    * @param j Upper index bound of the subchain (inclusive).
    * @param k Split index within range [i, j-1]
    * @param i Lower index bound of the subchain (inclusive).
    * @return std::size_t Total adjoint accumulation cost without memory constraint.
    */
   std::size_t pre_adjoint_cost(std::size_t j, std::size_t k, std::size_t i){
      //Access a cell with pointer
      if(j == k+1){
         
         return table.get_cell(j).accumulated_cost() +
                  table.get_cell(j).codomain_dim() * accumulate_edges(k,i); 
      }
      //Access a normal cell
      else{

         return table.get_cell(j, k+1).accumulated_cost() +
                  table.get_cell(j).codomain_dim() * accumulate_edges(k,i);
      }
   }

   /**
    * @brief Improved adjoint accumulation cost model, using binomial checkpointg if necessary.
    *
    * @details First determines if the subchain (k,i) can be accumulated standardly without exceeding 
    * memory_bound (joint reversal). If memory is exceeded, it checks wether the subchain is split-
    * reversible. If split reversal is feasible, it calculates the minimal split index k_tilde, 
    * constructs the intermediate split Jacobian metadata for (k, k_tilde + 1) and executes the 
    * binomial checkpoiting algorithm on the chain made of the intermediate split Jacobian (k, k_tilde + 1)
    * and subchain (k, i) to determine additional re-execution costs.
    *
    * @param j Upper index bound of the overall subchain (inclusive).
    * @param j Split index in the range [i, j-1]
    * @param i Lower index bound of the overall subchain (inclusive).
    * @param[out] is_adjoint_split_optimal Set to true if split reversal mode is feasible and selected 
    * for this subproblem.
    * @return std::optional<std::size_t> Total reverse-mode accumulation cost if feasible, 
    * std::nullopt if memory bound renders split reversal impossible.
    */
   std::optional<std::size_t> adjoint_cost(std::size_t j, std::size_t k, std::size_t i,
                                             bool& is_adjoint_split_optimal){

      std::size_t number_edges_k_i = accumulate_edges(k,i);

      if(number_edges_k_i <= memory_bound){

         //Access a cell_with_pointer
         if(j == k+1){

            return table.get_cell(j).accumulated_cost() +
                     table.get_cell(j).codomain_dim() * number_edges_k_i;
         }
         //Access a normal cell
         else{
            return table.get_cell(j, k+1).accumulated_cost() +
                     table.get_cell(j).codomain_dim() * number_edges_k_i;
         }
      }
      //subchain k,i is not joint reversable
      {
         if(is_split_reversable(k,i)){

            is_adjoint_split_optimal = true;
            
            std::size_t k_tilde = split_reversed_chain(i, number_edges_k_i);

            //Create split_dense_jacobian (k,k_tilde + 1)
            Split_dense_Jacobian split_dense_k_k_tilde_plus_one = joint_jacobian(k, k_tilde + 1);

            binomial_checkpointing 
               binomial_algortihm{chain, k_tilde, i, number_checkpoints - 1,
                                    &split_dense_k_k_tilde_plus_one};

            //Adjoint cost calculation
            //Access a cell with pointer
            if(j == k+1){

               return table.get_cell(j).accumulated_cost() +
                        table.get_cell(j).codomain_dim() * number_edges_k_i +
                        binomial_algortihm.get_additional_cost();
            }
            //Access a normal cell
            else{

               return table.get_cell(j, k+1).accumulated_cost() +
                        table.get_cell(j).codomain_dim() * number_edges_k_i +
                        binomial_algortihm.get_additional_cost();
            }
         }
         //Not split reversable
         else{return {};}
      }
   }

   /**
    * @brief Calculates standard dense matrix-matrix multiplication cost of two preaccumulated 
    * Jacobians (j, k+1) and (k, i).
    *
    * @note Forwards execution to tool_box_dense::multiplication_cost using internal 'table'.
    *
    * @param j Upper index bound (inclusive).
    * @param k Split index.
    * @param i Lower index bound (inclusive).
    * @return Total dense matrix-matrix multiplication cost.
    */
   std::size_t multiplication_cost(std::size_t j, std::size_t k, std::size_t i){

      return tool_box_dense::multiplication_cost(table, j, k, i);
   }

 private:
   /// Dynamic programming table structure storing subproblems optimal results.
   Table<Split_dense_Jacobian> table;
   /// Reference to target chain on which the dynamic algorithm will run. 
   const jacobian_chain<Split_dense_Jacobian, Split_reversal_dense_information>& chain;
   /// Edge limit constraint for adjoint evaluations.
   std::size_t memory_bound;
   /// Number of available checkpoints during the whole optimization process.
   std::size_t number_checkpoints;

   /**
    * @brief Populates the dynamic programming table with optimal subchain accumulation strategies.
    *
    * @details This method is a reference implementation of the ideas illustrated along the class 
    * specialization; however on embedding binomial checkpoiting in the Dense Jacobian Chain Product 
    * Bracketing formulation the "new" formulation favored split reversal cost estimation without 
    * calling the binomial checkpointig algorithm. This means that the obtained cost minimum was 
    * lacking execution additional costs incurred during split reversal.
    */
   void fill();
};

void fill_table<Split_dense_Jacobian, Split_reversal_dense_information>::fill(){

   table.clear();

   //Initializing cells with pointer.
   for(std::size_t elemental_idx = 0; elemental_idx < chain.size(); elemental_idx++){

      if(chain[elemental_idx].domain_dim() <= chain[elemental_idx].codomain_dim()){

         table.emplace_back(&chain[elemental_idx], tangent_cost(&chain[elemental_idx]),
               Operation::TANGENT);
      }

      else{
         if(chain[elemental_idx].number_edges() <= memory_bound){
            
            table.emplace_back(&chain[elemental_idx], adjoint_cost(&chain[elemental_idx]),
                  Operation::ADJOINT);
         }
         //Memory bound is exceeded then the elemental jacobian is accumulated using
         // tangent mode.
         else{

            table.emplace_back(&chain[elemental_idx], tangent_cost(&chain[elemental_idx]),
                  Operation::TANGENT);
         }
      }
   }

   std::size_t minimum_cost, minimum_cost_j_k_i, split_position;
   std::size_t split_position_j_k_i;
   std::size_t i_index, k_index, tangent_cost_j_k_i;
   std::optional<std::size_t> adjoint_cost_j_k_i;
   bool is_adjoint_split_optimal = false;
   Operation operation, operation_j_k_i;

   for(std::size_t j_index = 1; j_index < chain.size(); j_index++){

      for(std::size_t aux_var_i = 1; aux_var_i < j_index + 1; aux_var_i++){

         i_index = j_index - aux_var_i;
         minimum_cost = std::numeric_limits<std::size_t>::max();

         //k_index split position
         //Problem instance (j_index, i_index)
         for(std::size_t aux_var_k = 1; aux_var_k <= j_index - i_index; aux_var_k++){
            //Decreasing order. From j_index - 1 to i_index
            k_index = j_index - aux_var_k;

            minimum_cost_j_k_i = multiplication_cost(j_index, k_index, i_index);
            split_position_j_k_i = k_index;
            operation_j_k_i = Operation::MULTIPLICATION;

            //Minimum over the different accumulation methods
            tangent_cost_j_k_i = tangent_cost(j_index, k_index, i_index);
            if(tangent_cost_j_k_i < minimum_cost_j_k_i){

               minimum_cost_j_k_i = tangent_cost_j_k_i;
               operation_j_k_i = Operation::TANGENT;
            }

            //Assuming no memory bound: is adjoint mode potentially the optimal
            //accumulation method for the subproblem j,i?
            if(pre_adjoint_cost(j_index, k_index, i_index) < minimum_cost_j_k_i){

               adjoint_cost_j_k_i = adjoint_cost(j_index, k_index, i_index,
                                                   is_adjoint_split_optimal);

               if(adjoint_cost_j_k_i){
                  
                  if(*adjoint_cost_j_k_i < minimum_cost_j_k_i){

                     minimum_cost_j_k_i = *adjoint_cost_j_k_i;
                     if(is_adjoint_split_optimal){
                        
                        operation_j_k_i = Operation::ADJOINT_SPLIT;
                     }
                     else{
                        
                        operation_j_k_i = Operation::ADJOINT;
                     }
                  }
               }

               is_adjoint_split_optimal = false;
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
 * @note Shared across 'fill_table<Sparse_Jacobian, Matrix_free_sparse_information>' and 
 * 'fill_table<Split_sparse_Jacobian, Split_reversal_sparse_information>'.
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
      //Access cell with pointer
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
     //Access normal cell
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
      //Access cell_with_pointer
      if(k == i){

         accumulated_cost_rhs = table.get_cell(i).accumulated_cost();
         max_number_nnz_column = table.get_cell(i).max_number_nnz_column();
      }
      else{
         
         accumulated_cost_rhs = table.get_cell(k, i).accumulated_cost();
         max_number_nnz_column = table.get_cell(k, i).max_number_nnz_column();
      }

      //Access cell_with_pointer
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
class fill_table<Sparse_Jacobian, Matrix_free_sparse_information>{
 public:
   /**
    * @brief Constructs and solves DP table without memory limits.
    *
    * @param chain Sequence of sparse Jacobians to optimize.
    */
   fill_table(const jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>& chain):
      table(chain.size()){
      
      fill(chain);
   }

   /**
    * @brief Constructs and solves DP table subject to an edge-capacity memory limit. 
    * 
    * @param chain Sequence of sparse Jacobians to optimize.
    * @param memory_bound_ Maximum allowed graph edges active during reversal-mode (adjoint). 
    */
   fill_table(const jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>& chain,
                  std::size_t memory_bound_):
      table(chain.size()), memory_bound(memory_bound_){

      fill(chain);
   }

   /**
    * @brief Constructs engine with a pre-populated DP table (primarily for unit testing). 
    *
    * @param table_ Storage table instance moved into class ownership. 
    */
   fill_table(Table<Sparse_Jacobian> table_):
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
      //Access a cell_with_pointer
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
    * @brief Access cell with non owning pointer to elemental sparse Jacobian at index 'index' in 
    * the chain.
    *
    * @param index sparse Jacobian matrix index in the chain.
    * @return Const reference to 'cell_with_pointer<Sparse_Jacobian>' object.
    */
   const cell_with_pointer<Sparse_Jacobian>& get_cell(std::size_t index){
      
      return table.get_cell(index);
   }

   /**
    * @brief Access cell at (j, i) containing optimal accumulation and sparsity data. 
    *
    * @param j End index of the subchain (1-based, $j > i$).
    * @param i Start index of the subchain (0-based).
    * @return Const reference to 'cell<Sparse_Jacobian>' object.
    */
   const cell<Sparse_Jacobian>& get_cell(std::size_t j, std::size_t i){

      return table.get_cell(j,i);
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
    * 1. Initializes cell_with_pointer objects with base optimal costs for preaccumulation of 
    * elemental Jacobians and corresponding non-owning pointers to chain elements.
    * 2. Initializes cell objects with subchain optimal costs and split positions. 
    *
    * @details Subproblems are evaluated in the following order:
    * (1, 0)
    * (2, 1) (2, 0)
    * ...
    *
    * @param chain Target Jacobian chain instance.
    */
   void fill(const jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>& chain);
};

void fill_table<Sparse_Jacobian, Matrix_free_sparse_information>::fill(
      const jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>& chain){

   table.clear();

   //Initializing cells with pointer.
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

#endif ///FILL_TABLE
