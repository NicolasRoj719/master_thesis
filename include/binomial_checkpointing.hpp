#ifndef BINOMIAL_CHECKPOINTING
#define BINOMIAL_CHECKPOINTING

#include <cassert>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include "./chain.hpp"

/**
 * @brief Represents a single dynamic programming table entry for binomial checkpointing.
 *
 * Stores the computed minimum re-evaluation cost, checkpoint allowance, and split position.
 *
 * @note An empty split position indicates a base-case subproblem. 
 */
class binomial_cell{
 public:
  /**
   * @brief Constructs a cell with cost, checkpoints constraints, and an optimal split position index. 
   *
   * @param additional_cost Total re-execution cost incurred without tape recording for this subproblem. 
   * @param number_available_checkpoints Maximum checkpoints allowed for this subproblem context. 
   * @param split_position Index splitting subproblem (j, i, c) into (j, k+1, c-1) and (k, i, c).
   * Satisfies $j > k \ge i$. Defaults to 'std::nullopt'.
   */
  binomial_cell(std::size_t additional_cost, std::size_t number_available_checkpoints,
                  std::optional<std::size_t> split_position = std::nullopt):
    additional_cost_(additional_cost), available_checkpoints_(number_available_checkpoints),
    split_position_(split_position){}

  /// @brief Gets the optimal additional computational cost.
  std::size_t additional_cost() const{return additional_cost_;}
  /// @brief Gets the number of checkpoints allocated for this subproblem.
  std::size_t number_checkpoints() const{return available_checkpoints_;}
  /// @brief Gets the optimal split index, if available.
  std::optional<std::size_t> split_position() const{return split_position_;}

 private:
  /// Optimal additional computational cost due to execution without recording data.
  std::size_t additional_cost_;
  /// Available number of checkpoints to solve this problem instance.
  std::size_t available_checkpoints_;
  /// Returns the optimal split position $k$; 'std::nullopt' if no split exists.
  std::optional<std::size_t> split_position_;
};

/**
 * @brief Manages linear storage for dynamic programming lookup tables with a stacked triangular layout.
 *
 * @details Subproblems are organized by available checkpoints ($c$). For a fixed checkpoint allocation 
 * $c$, problem instances (j, i, c) are traversed in the following lower-triangular index order:
 *
 * (0, 0, c)
 * (1, 1, c) (1, 0, c)
 * (2, 2, c) (2, 1, c) (2, 0, c)
 * ...
 */
class binomial_table{
 public:
  /**
   * @brief Pre-allocates storage for dynamic programming cells based on chain length and checkpoint 
   * counts.
   *
   * @param chain_length Total length of the Jacobian length. 
   * @param number_checkpoints Total available checkpoints. 
   */
  binomial_table(std::size_t chain_length, std::size_t number_checkpoints){

    cells_per_floor = (chain_length * (chain_length + 1)) / 2;

    table.reserve((number_checkpoints + 1) * cells_per_floor);
  }

  /**
   * @brief Default constructor for unallocated tables (primarily used for unit testing).
   */
  binomial_table(){}

  /**
   * brief Maps subproblem indices (j, i, c) to the linear storage array. 
   *
   * @param j Upper bound index of the subproblem (inclusive).
   * @param i Lower bound index of the subproblem (inclusive).
   * @param c Available checkpoint count for the subproblem.
   * @return Const reference to the target 'binomial_cell'.
   */
  const binomial_cell& get_cell(std::size_t j, std::size_t i, std::size_t c) const noexcept{

    std::size_t accumulation = ((j + 1) * j) / 2;
    std::size_t position = j - i;

    return table[c * cells_per_floor + accumulation + position];
  } 

  /**
   * @brief Appends subproblem data including an optimal split position. 
   *
   * @param additional_cost Re-execution cost without tape recording. 
   * @param available_checkpoints Available checkpoint capacity. 
   * @param split_position Index of the optimal subproblem split boundary. 
   */
  void emplace_back(std::size_t additional_cost, std::size_t available_checkpoints,
                      std::size_t split_position){

    table.emplace_back(additional_cost, available_checkpoints, split_position);
  }

  /**
   * @brief Appends subproblem data without split position (base cases). 
   *
   * @note Minimum length of the subproblems. Split reversal of a single subprogram. 
   *
   * @param additional_cost Re-execution cost without tape recording. 
   * @param available_checkpoints Available checkpoint capacity. 
   */
  void emplace_back(std::size_t additional_cost, std::size_t available_checkpoints){

    table.emplace_back(additional_cost, available_checkpoints);
  }

  /**
   * @brief Returns the total count of stored solution cells. 
   *
   * @return std::size_t Total number of cell instances. 
   */
  std::size_t size() const{
    
    return table.size();
  }

  /**
   * @brief Retrieves the final cell containing problem optimal solution. 
   *
   * @return Const reference to last stored 'binomial_cell'.
   */
  const binomial_cell& back() const{
    return table.back();
  }

 private:
  /// Number of subproblem cells stored per checkpoint level. 
  std::size_t cells_per_floor;
  /// Flattened sequence of dynamic programming cells. 
  std::vector<binomial_cell> table;
};

/**
 * @brief Lightweight, non-owning subrange view of a Jacobian chain.
 *
 * @tparam Split_type Class defining individial operations, implementing 'function_cost()'.
 * @tparam Split_information_type Metadata describing structure/dimensions of the Jacobians.
 */
template<class Split_type, class Split_information_type>
class View_chain{
 public:
   /**
    * @brief Constructs a subrange view over a specified section of a Jacobian chain.
    *
    * @param chain_ Base Jacobian chain to reference. 
    * @param first_index Starting index of the view.
    * @param number_elements Number of elements included from the original chain.
    * @param ptr Optional pointer to an extra 'Split_type' element to append logically to the end of 
    * this view.
    * @throws std::invalid_argument If the requested view of range falls outside the bounds of 'chain_'. 
    */
   View_chain(const jacobian_chain<Split_type, Split_information_type>& chain_,
                std::size_t first_index, std::size_t number_elements,
                const Split_type* ptr = nullptr){
    
     if(chain_.size() + 1 < first_index + number_elements){

       throw std::invalid_argument("The subrange specified is not contained within "
                                      "the chain given as an argument");
     }

     subrange = std::span<const Split_type>(chain_).subspan(first_index, number_elements);
     split_ptr = ptr;
   }

   /**
    * @brief Accesses elementsin the view by index. 
    *
    * @param index Zero-based element index within the view range. 
    * @return Const reference to the target 'Split_type'.
    * @throws std::out_of_range If 'index' is out of bounds.
    */
   const Split_type& operator[](std::size_t index) const{
      if(index < subrange.size()){
        return subrange[index];
      }

      if(index == subrange.size() && split_ptr != nullptr){
        return *split_ptr;
      }

      throw std::out_of_range("Index out of bounds in View_chain");
   }

   /**
    * @brief Gets the total length of the active view range (including virtual appends). 
    * @return Element count accessible through this view. 
    */
   std::size_t size() const{
    
     if(split_ptr == nullptr){
      
       return subrange.size();
     }

     else{
       return subrange.size() + 1;
     }
   }

 private:
  /// Non-owning view of the subrange elements. 
  std::span<const Split_type> subrange;
  /// Pointer to an externally appended element (or 'nullptr'). 
  const Split_type* split_ptr;
};

/**
 * @brief Dynamic programming solver for optimal binomial checkpointing in Algorithmic Differentiation 
 * (AD).
 *
 * Evaluates minimum re-execution overheads for split-reversal accumulation across Jacobian chain 
 * sequences.
 *
 * @tparam Split_type Element type implementing 'std::size_t function_cost() const'. 
 * @tparam Split_information_type Jacobian structural metadata.  
 */
template<class Split_type, class Split_information_type>
class binomial_checkpointing{
 public:
  /**
   * @brief Solves the optimal checkpointing placement sequence and obtains minimum re-execution cost 
   * over a subchain (j, i) with $c$ checkpoints.
   *
   * Evaluates problem instances defined as $t(j - i, 0, c)$ when 'split_pointer == nullptr', or 
   * $t(j - i + 1, 0, c)$ when an additional element pointer is supplied.
   *
   * @param chain_ Base Jacobian chain. 
   * @param j Upper bound index of the subchain (inclusive).
   * @param i Lower bound index of the subchain (inclusive).
   * @param checkpoints Available checkpoint capacity allocated to solve the problem instance.
   * @param split_pointer Optional pointer to an extra 'Split_type' element appended to element $j$. 
   */
  binomial_checkpointing(const jacobian_chain<Split_type, Split_information_type> chain_,
                          std::size_t j, std::size_t i, std::size_t checkpoints,
                          const Split_type* split_pointer = nullptr):
    chain(chain_, i, j - i + 1, split_pointer), table(chain.size(), checkpoints){

      fill_table(chain.size(), checkpoints);
  }

  /**
   * @brief Constructs a view-only instance without solving the dynamic table (primarily for testing
   * costs).
   *
   * @param chain_ Base Jacobian chain.
   * @param j Upper bound index of the subchain (inclusive).
   * @param i Lower bound index of the subchain (inclusive).
   */
  binomial_checkpointing(const jacobian_chain<Split_type, Split_information_type> chain_,
                          std::size_t j, std::size_t i):
    chain(chain_, i, j - i + 1), table(){}

  //Constructor designed to test additional_cost
  /**
   * @brief Injects a custom or pre-populated table for testing cost functions.
   *
   * @param chain_ Base Jacobian chain.
   * @param table_ Pre-constructed lookup table moved into internal storage.
   * @param j Upper bound index of the subchain (inclusive).
   * @param i Lower bound index of the subchain (inclusive).
   */
  binomial_checkpointing(const jacobian_chain<Split_type, Split_information_type>& chain_,
                          binomial_table table_, std::size_t j, std::size_t i):
    chain(chain_, i, j - i + 1), table(std::move(table_)){}

  /**
   * @brief Computes additional re-execution cost for subchain (j, i) assuming zero available checkpoints
   * $t(j - i, 0, 0)$.
   *
   * @param j Upper bound index of the subchain (inclusive).
   * @param i Lower bound index of the subchain (inclusive).
   * @return std::size_t Total additional computational cost incurred under zero-checkpoint constraints. 
   */
  std::size_t additional_cost(std::size_t j, std::size_t i){

    std::size_t cost = 0;

    for(std::size_t idx = 1; idx < (j-i+1); idx++){

      cost += idx * chain[j - idx].function_cost();  
    }

    return cost;
  }

  /**
   * @brief Computes execution cost for subprogram segment [i, split_position] without tape recording. 
   *
   * @param split_position Target split boundary index. 
   * @param i Lower bound index of subchain (inclusive).
   * @return Summed execution costs of subprograms that make up the subprogram segment. 
   */
  std::size_t advancing_cost(std::size_t split_position, std::size_t i){

    std::size_t cost = 0;
    
    for(std::size_t idx = i; idx < split_position + 1; idx++){
      
      cost += chain[idx].function_cost();
    }

    return cost;
  }

  /**
   * @brief Retrieves the solution cell corresponding to subproblem (j, i, c). 
   *
   * @param j Upper bound index of the subchain (inclusive).
   * @param i Lower bound index of the subchain (inclusive).
   * @param c Available checkpoint count.
   * @return Const reference to target 'binomial_cell'.
   */
  const binomial_cell& get_cell(std::size_t j, std::size_t i, std::size_t c) const{

    return table.get_cell(j, i, c);
  }
  

  /**
   * @brief Calculates additional re-execution cost for subchain (j, i) with candidate split position 
   * $$k.
   *
   * @warning Public strictly to support unit testing.
   *
   * Uses optimal additional cost data from cells (k, i, c) and (j, k+1, c-1) alongside forward execution
   * cost.
   *
   * @param j Upper bound index of the subchain (inclusive).
   * @param split_position Candidate split index $k$.
   * @param i Lower bound index of the subchain (inclusive).
   * @param c Available checkpoint count.
   * @return std::size_t Total additional cost for split decision $k$. 
   */
  std::size_t additional_cost(std::size_t j, std::size_t split_position, std::size_t i,
                              std::size_t available_checkpoints){
    return table.get_cell(split_position, i, available_checkpoints).additional_cost() +
           table.get_cell(j, split_position + 1, available_checkpoints - 1).additional_cost() +
           advancing_cost(split_position, i);
  }

  /**
   * @brief Gets the total optimal additional cost evaluated for the target chain.
   * @return Optimal additional re-execution cost value stored in the final dynamic cell.
   */
  std::size_t get_additional_cost(){
    
    return table.back().additional_cost();
  }

 private:
  /// Lightweight view over target subchain range. 
  View_chain<Split_type, Split_information_type> chain;

  /// Dynamic programming solution storage table. 
  binomial_table table;

  /**
   * @brief Runs the dynamic program to fill the subproblem solution table. 
   *
   *
   * @param chain_length Effective operational chain length.
   * @param number_checkpoints Maximum total checkpoints allocated. 
   */
  void fill_table(std::size_t chain_length, std::size_t number_checkpoints);
};

template<class Split_type, class Split_information_type>
void binomial_checkpointing<Split_type, Split_information_type>::fill_table(
    std::size_t chain_length, std::size_t number_checkpoints){

  std::size_t k_min_j_k_i;
  std::size_t cost_min_j_k_i;
  std::size_t cost_j_k_i;
  std::size_t i;

  // Base Case: 0 Checkpoints available.
  for(std::size_t j = 0; j < chain_length; j++){
    
    for(std::size_t aux_var = 0; aux_var < j + 1; aux_var++){
      
      i = j - aux_var;
      table.emplace_back(additional_cost(j, i), 0);
    }
  }

  //General Case: 1 to number_checkpoints available.
  for(std::size_t checkpoints = 1; checkpoints < number_checkpoints + 1; checkpoints++){

    //Emplacing back optimal solution to (i-i, 0, c)
    table.emplace_back(additional_cost(0, 0), checkpoints);
    for(std::size_t j= 1; j < chain_length; j++){

      table.emplace_back(additional_cost(j,j), checkpoints);
      for(std::size_t aux_var = 1; aux_var < j + 1; aux_var++){

        i = j - aux_var;
        cost_min_j_k_i = std::numeric_limits<std::size_t>::max();
        for(std::size_t split_position = i; split_position < j; split_position++){

          cost_j_k_i = additional_cost(j, split_position, i, checkpoints);

          if(cost_j_k_i < cost_min_j_k_i){
            cost_min_j_k_i = cost_j_k_i;
            k_min_j_k_i = split_position;
          }
        }
        //Stores the minimum as an entry in the table.
        table.emplace_back(cost_min_j_k_i, checkpoints, k_min_j_k_i);
      }
    }
  }
}
#endif ///BINOMIAL_CHECKPOINTING
