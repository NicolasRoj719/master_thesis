/**
 * @file binomial_checkpointing.hpp
 * @brief Dynamic programming algorithm and lookup structures for optimal binomial checkpointing in
 * Algorithmic Differentiation.
 *
 * Provides classes and data structures (`Binomial_cell`, `Binomial_table`, and 
 * `binomial_checkpointing`)
 * to evaluate optimal re-execution cost and optimal checkpoint placement sequences.
 */
#ifndef BINOMIAL_CHECKPOINTING
#define BINOMIAL_CHECKPOINTING

#include <cassert>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "chain.hpp"

/**
 * @brief Represents a single dynamic programming table entry for binomial checkpointing.
 *
 * Stores the computed minimum re-evaluation cost, checkpoint allowance, and split position.
 *
 * @note An empty split position indicates a base-case subproblem. 
 */
class Binomial_cell{
 public:
  /**
   * @brief Constructs a cell with cost, checkpoints constraints, and an optimal split position index. 
   *
   * @param additional_cost Total re-execution cost incurred without tape recording for this subproblem. 
   * @param number_available_checkpoints Maximum checkpoints allowed for this subproblem context. 
   * @param split_position Index splitting subproblem (j, i, c) into (j, k+1, c-1) and (k, i, c).
   * Satisfies $j > k \ge i$. Defaults to 'std::nullopt'.
   */
  Binomial_cell(std::size_t additional_cost, std::size_t number_available_checkpoints,
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
class Binomial_table{
 public:
  /**
   * @brief Pre-allocates storage for dynamic programming cells based on chain length and checkpoint 
   * counts.
   *
   * @param chain_length Total length of the Jacobian length. 
   * @param number_checkpoints Total available checkpoints. 
   */
  Binomial_table(std::size_t chain_length, std::size_t number_checkpoints){

    cells_per_floor = (chain_length * (chain_length + 1)) / 2;

    table.reserve((number_checkpoints + 1) * cells_per_floor);
  }

  /**
   * @brief Default constructor for unallocated tables (primarily used for unit testing).
   */
  Binomial_table(){}

  /**
   * brief Maps subproblem indices (j, i, c) to the linear storage array. 
   *
   * @param j Upper bound index of the subproblem (inclusive).
   * @param i Lower bound index of the subproblem (inclusive).
   * @param c Available checkpoint count for the subproblem.
   * @return Const reference to the target 'Binomial_cell'.
   */
  const Binomial_cell& get_cell(std::size_t j, std::size_t i, std::size_t c) const noexcept{

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
   * @return Const reference to last stored 'Binomial_cell'.
   */
  const Binomial_cell& back() const{
    return table.back();
  }

 private:
  /// Number of subproblem cells stored per checkpoint level. 
  std::size_t cells_per_floor;
  /// Flattened sequence of dynamic programming cells. 
  std::vector<Binomial_cell> table;
};

/**
 * @brief Dynamic programming solver for optimal binomial checkpointing in Algorithmic Differentiation 
 * (AD).
 *
 * Evaluates minimum re-execution overheads for split-reversal accumulation across a Jacobian chain
 * via the use of the execution cost of the subprograms given as a vector.
 *
 */
class Binomial_checkpointing{
 public:
  /**
   * @brief Executes the dynamic programming algorithm given the execution cost of the 
   * subprograms.
   *
   * @param subprograms_execution_cost Array with execution costs. 
   * @param checkpoints Available checkpoint capacity allocated to solve the problem instance.
   */
  Binomial_checkpointing(const std::vector<std::size_t>& subprograms_execution_cost,
                          std::size_t checkpoints):
    chain{subprograms_execution_cost}, table{chain.size(), checkpoints}{

      fill_table(chain.size(), checkpoints);
  }

  /**
   * @brief Testing purposes. 
   *
   * @param subprograms_execution_cost Array with execution costs. 
   */
  Binomial_checkpointing(const std::vector<std::size_t>& subprograms_execution_cost):
    chain{subprograms_execution_cost}, table{}{}

  //Constructor designed to test additional_cost
  /**
   * @brief Injects a custom or pre-populated table for testing cost functions.
   *
   * @param subprograms_execution_cost Array with execution costs. 
   * @param table_ Pre-constructed lookup table moved into internal storage.
   */
  Binomial_checkpointing(const std::vector<std::size_t>& subprograms_execution_cost,
                          Binomial_table table_):
    chain{subprograms_execution_cost}, table{std::move(table_)}{}

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

      cost += idx * chain[j - idx];  
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
      
      cost += chain[idx];
    }

    return cost;
  }

  /**
   * @brief Retrieves the solution cell corresponding to subproblem (j, i, c). 
   *
   * @param j Upper bound index of the subchain (inclusive).
   * @param i Lower bound index of the subchain (inclusive).
   * @param c Available checkpoint count.
   * @return Const reference to target 'Binomial_cell'.
   */
  const Binomial_cell& get_cell(std::size_t j, std::size_t i, std::size_t c) const{

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

  /**
   * @brief gets constant reference to dynamic programming lookup table.
   * @return constant reference to table.
   */
  const Binomial_table& get_table(){
    return table;
  }

  static std::vector<std::size_t> read_subprograms_cost_from_file(
          const std::string& path_to_file, std::size_t list_length = 1){
    
    std::ifstream file(path_to_file);

    if(!file.is_open()){
      throw std::runtime_error("Could not open file located at: " + path_to_file);
    }

    std::vector<std::size_t> subprograms_execution_cost;
    subprograms_execution_cost.reserve(list_length);
    std::size_t execution_cost;

    while(file >> execution_cost){

      subprograms_execution_cost.push_back(execution_cost);
    }

    return subprograms_execution_cost;
  }

 private:
  /// Subprograms evaluation costs. 
  const std::vector<std::size_t>& chain;

  /// Dynamic programming solution storage table. 
  Binomial_table table;

  /**
   * @brief Runs the dynamic program to fill the subproblem solution table. 
   *
   *
   * @param chain_length Effective operational chain length.
   * @param number_checkpoints Maximum total checkpoints allocated. 
   */
  void fill_table(std::size_t chain_length, std::size_t number_checkpoints);
};

void Binomial_checkpointing::fill_table(std::size_t chain_length, std::size_t number_checkpoints){

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
