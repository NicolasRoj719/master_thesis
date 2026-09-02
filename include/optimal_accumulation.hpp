/**
 * @file optimal_accumulation.hpp
 * @brief Reconstructs optimal evaluation sequences and binomial checkpointing schedules.
 * @details Provides node representations and recursive traversal functions to extract optimal
 * accumulation sequences and checkpoint placements from dynamic programming lookup tables.
 * Supports Graphviz syntax generation for visual dependency tree rendering.
 */
#ifndef OPTIMAL_ACCUMULATION_HPP
#define OPTIMAL_ACCUMULATION_HPP

#include <algorithm>
#include <iostream>
#include <cstddef>
#include <optional>
#include <vector>

#include "table.hpp"
#include "table_cell.hpp"
#include "binomial_checkpointing.hpp"

/**
 * @brief Abstract base class representing a generic subchain operation node in an optimal schedule 
 * tree.
 */
class Node_base_chain{
 public:
  Node_base_chain(std::size_t j_index, std::optional<std::size_t> i_index, std::size_t level):
    j(j_index), level_number(level), i(i_index){}

  /// @brief Returns the upper bound index of the subchain (inclusive).
  std::size_t j_index() const {return j;}
  /// @brief Returns the lower bound index of the subchain (inclusive), if defined.
  std::optional<std::size_t> i_index() const {return i;}
  /// @brief Returns the depth level of this node in the scheduling tree.
  std::size_t level() const {return level_number;}

  /**
   * @brief Less-than operator to order nodes by their depth level.
   *
   * Used when sorting node vectors to place the root node at the front and leaf nodes at the back.
   *
   * @param rhs The right-hand side node to compare against.
   * @return `true` if `this->level_number` is strictly less than `rhs.level_number`, `false` otherwise.
   */
  bool operator<(const Node_base_chain& rhs) const{

    return this->level_number < rhs.level();
  }

  /**
   * @brief Equality operator comparing node depth levels.
   *
   * @param rhs The right-hand side node to compare against.
   * @return `true` if both nodes share the same depth level, `false` otherwise.
   */
  bool operator==(const Node_base_chain& rhs) const{
    
    return this->level_number == rhs.level();
  }
 
 protected:
  /// Upper index of the subchain (inclusive).
  std::size_t j;
  /// Depth level of the node in the execution schedule tree.
  std::size_t level_number;
  /// Lower bound index of the subchain (inclusive), or `std::nullopt` if unassigned.
  std::optional<std::size_t> i;
};

/**
 * @brief Represents an operation node in an optimal accumulation tree.
 *
 * `Node_matrix_free` serves two primary roles:
 * 1. Facilitates visual tracking/rendering of the optimal accumulation chain (e.g., Graphviz 
 * syntax generation).
 * 2. Provides metadata to build a serial execution schedule.
 *
 * @note Used to visualize and schedule optimal solutions for the Matrix-Free
 *       Sparse/Dense Jacobian Chain Product Bracketing problem.
 */
class Node_matrix_free: public Node_base_chain{
 public:
  /**
   * @brief Constructs a `Node_matrix_free` instance.
   *
   * Option to output Graphviz syntax immediately upon node creation if `out_stream` is provided.
   *
   * @param j_index Upper bound index of the subchain (inclusive).
   * @param i_index Lower bound index of the subchain (inclusive), or `std::nullopt` for single-leaf nodes.
   * @param operation Accumulation or preaccumulation operation type.
   * @param level Depth level of the node in the scheduling tree (1-based, root = 1).
   * @param out_stream Optional pointer to an output stream for Graphviz rendering. 
   * Pass `nullptr` to disable output.
   */
  Node_matrix_free(std::size_t j_index, std::optional<std::size_t> i_index, 
      Operation operation, std::size_t level, std::ostream* out_stream = nullptr):
      Node_base_chain(j_index, i_index, level), accumulation_operation(operation){

    if(out_stream){

      if(i){
        
        // The root node (level 1) has no parent.
        if(level == 1){

          parent_print(*out_stream);
        }
        else{

          child_print(*out_stream);
          parent_print(*out_stream);
        }
      }

      else{child_print(*out_stream);}
    }
  }

  /**
   * @brief Outputs the parent portion of a Graphviz dependency edge.
   * @param[out] out_stream Target output stream.
   */
  void parent_print(std::ostream& out_stream) const{

    out_stream << "\"( " << j << " , " << *i << " ) " << accumulation_operation << "\" -> ";
  }

  /**
   * @brief Outputs the child portion of a Graphviz dependency edge.
   * @param[out] out_stream Target output stream.
   */
  void child_print(std::ostream& out_stream) const{

    if(i){

      out_stream << "\"( " << j << " , " << *i << " ) " << accumulation_operation << "\"\n";
    }
    else{

      out_stream << "\"( " << j << " ) " << accumulation_operation << "\"\n";
    }
  }

  /// @brief Returns the accumulation operation for this node.
  Operation operation() const {return accumulation_operation;}

 private:
  /// Operation associated with this node.
  Operation accumulation_operation;
};

/**
 * @brief Represents an operation node in an optimal accumulation tree for the Dense Jacobian 
 * Chain Product Bracketing problem.
 *
 * 'Node_jacobian' serves two primary roles:
 * 1. Facilitates visual tracking/rendering of the optimal accumulation chain (e.g., Graphviz 
 * syntax generation).
 * 2. Provides metadata to build a serial execution schedule.
 */
class Node_jacobian: public Node_base_chain{
 public:
  /**
   * @brief Constructs a `Node_jacobian` instance.
   *
   * Option to output Graphviz syntax immediately upon node creation if `out_stream` is provided.
   * @param j_index Upper bound index of the subchain (inclusive).
   * @param i_index Lower bound index of the subchain (inclusive), or `std::nullopt` for single-leaf nodes.
   * @param level Depth level of the node in the scheduling tree (1-based, root = 1).
   * @param out_stream Optional pointer to an output stream for Graphviz rendering. 
   * Pass `nullptr` to disable output.
   */
  Node_jacobian(std::size_t j_index, std::optional<std::size_t> i_index,
      std::size_t level, std::ostream* out_stream = nullptr):
    Node_base_chain(j_index, i_index, level){

    if(out_stream){

      if(i){
        //Leaf
        if(j - *i == 1){
          
          child_print(*out_stream);
        }
        //The root node (level 1) has no parent.
        else if(level == 1){

          parent_print(*out_stream);
        }
        else{
          child_print(*out_stream);
          parent_print(*out_stream);
        }
      }
      //Leaf
      else{child_print(*out_stream);}
    }
  }

  /**
   * @brief Outputs the parent portion of a Graphviz dependency edge.
   * @param[out] out_stream Target output stream.
   */
  void parent_print(std::ostream& out_stream) const{

    out_stream << "\"( " << j << " , " << *i << " )\" ->";
  }

  /**
   * @brief Outputs the child portion of a Graphviz dependency edge.
   * @param[out] out_stream Target output stream.
   */
  void child_print(std::ostream& out_stream) const{

    if(i){

      out_stream << "\"( " << j << " , " << *i << " )\"\n";
    }
    else{

      out_stream << "\"( " << j << " )\"\n";
    }
  }

};

/**
 * @brief Represents an subproblem node in an optimal checkpoint placing strategy that minimizes 
 * the total re-evaluation cost during split reversal AD.
 *
 * `Node_binomial_checkpointing` facilitate visual tracking of the checkpoints placement procedure 
 * via Graphviz syntax generatrion.
 */
class Node_binomial_checkpointing{
 public:
   /**
    * @brief Constructs a 'Node_binomial_checkpointing' instance.
    *
    * Option to output Graphviz syntax immediately upon node creation if 'out_stream' is provided.
    *
   * @param j_index Upper bound index of the subchain (inclusive).
   * @param i_index Lower bound index of the subchain (inclusive), or `std::nullopt` for single-leaf nodes.
   * @param number_checkpoints Number of checkpoints available.
   * @param level Depth level of the node in the scheduling tree (1-based, root = 1).
   * @param out_stream Optional pointer to an output stream for Graphviz rendering. 
   * Pass `nullptr` to disable output.
    *
    */
  Node_binomial_checkpointing(std::size_t j_index, std::size_t i_index,
      std::size_t number_checkpoints, std::size_t level, std::ostream* out_stream = nullptr):
    j(j_index), i(i_index), c(number_checkpoints), level_number(level){

    if(out_stream){
      //The root node (level 1) has no parent.
      if(level_number == 1){
        
        parent_print(*out_stream);
      }
      //Leaf
      else if(j - i == 1 || j == i || c == 0){

        child_print(*out_stream);
      }
      else{
        
        child_print(*out_stream);
        parent_print(*out_stream);
      }
    }
  }

  /**
   * @brief Outputs the parent portion of a Graphviz dependency edge.
   * @param[out] out_stream Target output stream.
   */
  void parent_print(std::ostream& out_stream) const{

    out_stream << "\"( " << j << " , " << i << " , " << c << " )\" ->";
  }

  /**
   * @brief Outputs the child portion of a Graphviz dependency edge.
   * @param[out] out_stream Target output stream.
   */
  void child_print(std::ostream& out_stream) const{
    
    out_stream << "\"( " << j << " , " << i << " , " << c << " )\"\n";
  }

  /// @brief Returns the upper bound index of the subchain (inclusive).
  std::size_t j_index() const {return j;}
  /// @brief Returns the lower bound index of the subchain (inclusive).
  std::size_t i_index() const {return i;}
  /// @brief Returns the number of availablecheckpoints.
  std::size_t number_checkpoints() const {return c;}
  /// @brief Returns depth in the three (1-based valued).
  std::size_t level() const {return level_number;}

 private:
  /// Upper index of the subchain (inclusive).
  std::size_t j;
  /// Lower index of the subchain (inclusive).
  std::size_t i;
  /// Number of available checkpoints.
  std::size_t c;
  /// Dependency tree depth level.
  std::size_t level_number;
};

/**
 * @brief Recursively traverses optimal subproblems and populates sequence nodes.
 *
 * @tparam Table_T Lookup table type (typically `Table<Dense_Jacobian>` or `Table<Sparse_Jacobian>`).
 * @param j_index Upper bound index of the subchain (inclusive).
 * @param i_index Lower bound index of the subchain (inclusive).
 * @param[in,out] level Current depth tracking level in the recursion tree.
 * @param table Populated dynamic programming lookup table.
 * @param[out] accumulation_sequence Target container for constructed `Node_matrix_free` objects.
 * @param[out] out_stream Optional pointer to an output stream for Graphviz format generation.
 */
template<class Table_T>
void recursive_operation_accumulation(std::size_t j_index, std::size_t i_index, std::size_t& level,
      const Table_T& table, std::vector<Node_matrix_free>& accumulation_sequence,
      std::ostream* out_stream = nullptr){
  
  level++;

  const auto cell_j_i = table.get_cell(j_index, i_index);

  accumulation_sequence.emplace_back(
      j_index, i_index, cell_j_i.operation(), level, out_stream);

  auto k_split_pos = cell_j_i.split_position();

  std::size_t level_copy;

  switch(cell_j_i.operation()){
    case Operation::TANGENT:
      if(k_split_pos == i_index){
        
        level++;
        accumulation_sequence.emplace_back(
            i_index, std::nullopt, table.get_cell(i_index).operation(), level, out_stream);
        break;
      }

      else{

        recursive_operation_accumulation(
            k_split_pos, i_index, level, table, accumulation_sequence, out_stream);
        break;
      }

    case Operation::ADJOINT:
      if(k_split_pos + 1 == j_index){

        level++;
        accumulation_sequence.emplace_back(
            j_index, std::nullopt, table.get_cell(j_index).operation(), level, out_stream);
        break;
      }
      else{

        recursive_operation_accumulation(
            j_index, k_split_pos + 1, level, table, accumulation_sequence, out_stream);
        break;
      }

    case Operation::MULTIPLICATION:

      level_copy = level;

      // Copy node locally as reallocation during emplace_back invalidates references
      const auto parent_node = accumulation_sequence.back();

      if(k_split_pos + 1 == j_index){

        level_copy++;
        accumulation_sequence.emplace_back(
            j_index, std::nullopt, table.get_cell(j_index).operation(), level_copy, out_stream);

      }
      else{
        
        recursive_operation_accumulation(
            j_index, k_split_pos + 1, level_copy, table, accumulation_sequence, out_stream);
      }

      if(out_stream){

        parent_node.parent_print(*out_stream);
      }

      if(k_split_pos == i_index){
        
        level++;
        accumulation_sequence.emplace_back(
            i_index, std::nullopt, table.get_cell(i_index).operation(), level, out_stream);

      }
      else{

        recursive_operation_accumulation(
            k_split_pos, i_index, level, table, accumulation_sequence, out_stream);
      }
    }

}

/**
 * @brief Recursively traverses optimal subproblems and populates operation node array.
 *
 * @param j_index Upper bound index of the subchain (inclusive).
 * @param i_index Lower bound index of the subchain (inclusive).
 * @param[in,out] level Current depth tracking level in the recursion tree.
 * @param table Populated dynamic programming lookup table.
 * @param[out] accumulation_sequence Target container for constructed `Node_jacobian` objects.
 * @param[out] out_stream Optional pointer to an output stream for Graphviz format generation.
 */
void recursive_operation_accumulation(std::size_t j_index, std::size_t i_index, std::size_t& level,
      const Table<Jacobian>& table, std::vector<Node_jacobian>& accumulation_sequence,
      std::ostream* out_stream = nullptr){

  level++;

  accumulation_sequence.emplace_back(j_index, i_index, level, out_stream);

  auto k_split_pos = table.get_cell(j_index, i_index).split_position();

  std::size_t level_copy = level;

  const auto parent_node = accumulation_sequence.back();

  //Left branch.
  //Leaf with chain length equal 2.
  if(k_split_pos + 2 == j_index){
    
    level_copy++;
    accumulation_sequence.emplace_back(j_index, k_split_pos + 1, level_copy, out_stream);
  }
  //Leaf with chain length equal 1.
  else if(k_split_pos + 1 == j_index){

    level_copy++;
    accumulation_sequence.emplace_back(j_index, std::nullopt, level_copy, out_stream); 
  }
  else{

    recursive_operation_accumulation(
        j_index, k_split_pos + 1, level_copy, table, accumulation_sequence, out_stream);
  }

  if(out_stream){

    parent_node.parent_print(*out_stream);
  }

  // Right branch.
  //Leaf with chain length equal to 2.
  if(k_split_pos - 1 == i_index){
    
    level++;
    accumulation_sequence.emplace_back(k_split_pos, i_index, level, out_stream);
  }
  //Leaf with chain length equal to 1.
  else if(k_split_pos == i_index){

    level++;
    accumulation_sequence.emplace_back(i_index, std::nullopt, level, out_stream);
  }
  else{

    recursive_operation_accumulation(
        k_split_pos, i_index, level, table, accumulation_sequence, out_stream); 
  }
}

/**
 * @brief Traverses a filled dynamic programming lookup table to reconstruct the optimal sequence of
 * operations.
 *
 * Recursively visits optimal subproblems to populate an unordered sequence of node operations.
 * Call @ref obtain_serial_execution_sequence on the returned sequence to order operations for execution.
 *
 * @note For `Table<Jacobian>` and `Node_jacobian` chain_length must be greater than 2. A subchain 
 * of length two has zero degrees of freedom; its trivieal solution is inherently optimal.
 *
 * @tparam Node_T Operation node type, either Node_matrix_free or Node_jacobian are allowed.
 * @tparam Table_T Lookup table type (typically `Table<Dense_Jacobian>` or `Table<Sparse_Jacobian>`).
 * @param table Populated dynamic programming lookup table.
 * @param chain_length Number of elements in the Jacobian matrix chain.
 * @param[out] out_stream Optional pointer to an output stream for Graphviz file generation.
 * Pass `nullptr` to disable output.
 * @return `std::vector<Node_T>` Unordered set of operations defining the optimal sequence.
 */
template<class Node_T, class Table_T>
std::vector<Node_T> operation_sequence_accumulation(
    const Table_T& table, std::size_t chain_length, std::ostream* out_stream = nullptr){

  std::size_t level = 0;
  std::vector<Node_T> accumulation_sequence;

  accumulation_sequence.reserve(chain_length-1);

  if(out_stream){
    //Graphviz format
    *out_stream << "digraph G {\n";
  }
  //Fill up accumulation_sequence recursively.
  recursive_operation_accumulation(chain_length-1, 0, level, table, accumulation_sequence, out_stream);

  if(out_stream){
    //Graphviz format
    *out_stream << "}\n";
  }

  return accumulation_sequence;
}

/**
 * @brief Recursively traverses optimal subproblems of the binomial checkpointing algorithm and 
 * stores them in an array.
 *
 * @param j_index Upper bound index of the subchain (inclusive).
 * @param i_index Lower bound index of the subchain (inclusive).
 * @param c_index Number of available checkpoints.
 * @param level Depth of the subproblem in the subproblem dependency tree (1-based).
 * @param table Populated dynamic programming lookup table.
 * @param[out] subproblem_sequence Target container for constructed `Node_binomial_checkpointing` objects.
 * @param[out] out_stream Optional pointer to an output stream for Graphviz format generation.
 */
void recursive_subproblem_accumulation(std::size_t j_index, std::size_t i_index, std::size_t c_index,
                                std::size_t& level, const binomial_table& table, 
                                std::vector<Node_binomial_checkpointing>& subproblem_sequence,
                                std::ostream* out_stream = nullptr){
  
  level++;

  subproblem_sequence.emplace_back(j_index, i_index, c_index, level, out_stream);

  auto k_split_pos = *table.get_cell(j_index, i_index, c_index).split_position();

  std::size_t level_copy = level;

  const auto parent_node = subproblem_sequence.back();

  //Left branch
  //Leaf no more checkpoints available.
  if(c_index - 1 == 0){

    level_copy++;
    subproblem_sequence.emplace_back(j_index, k_split_pos + 1, c_index - 1, level_copy, out_stream);
  }
  //Leaf subchain of length equal to two or one.
  else if(j_index == k_split_pos + 2 || j_index == k_split_pos + 1){

    level_copy++;
    subproblem_sequence.emplace_back(j_index, k_split_pos + 1, c_index - 1, level_copy, out_stream);
  }
  else{
    
    recursive_subproblem_accumulation(
        j_index, k_split_pos + 1, c_index - 1, level_copy, table, subproblem_sequence, out_stream);
  }

  if(out_stream){

    parent_node.parent_print(*out_stream);
  }

  //Right branch
  //Leaf subchain of length equal to two or one.
  if(k_split_pos - 1 == i_index || k_split_pos == i_index){
    
    level++;
    subproblem_sequence.emplace_back(k_split_pos, i_index, c_index, level, out_stream);
  }
  else{

    recursive_subproblem_accumulation(
        k_split_pos, i_index, c_index, level, table, subproblem_sequence, out_stream);
  }
}

/**
 * @brief Traverses a filled dynamic programming lookup table to reconstruct the optimal sequence of
 * subproblem instances.
 *
 * The placement of a checkpoint generates two subproblem instances.
 *
 * @note `chain_length` must be greater than two and the number of checkpoints available must be at 
 * least one.
 *
 * @param table Populated dynamic programming lookup table.
 * @param chain_length Number of elements in the Jacobian matrix chain.
 * @param available_checkpoints Number of available checkpoints.
 * @param[out] out_stream Optional pointer to an output stream for Graphviz file generation.
 * Pass `nullptr` to disable output.
 * @return `std::vector<Node_binomial_checkpointing>` Sequence of optimal subproblems in decreasing 
 * order of subchain size.
 */
std::vector<Node_binomial_checkpointing> subproblem_sequence_accumulation(
    const binomial_table& table, std::size_t chain_length, std::size_t available_checkpoints,
    std::ostream* out_stream = nullptr){

  std::size_t level = 0;
  std::vector<Node_binomial_checkpointing> subproblem_sequence;

  subproblem_sequence.reserve(chain_length - 1);

  if(out_stream){
    //Graphviz format
    *out_stream << "digraph G {\n";
  }

  recursive_subproblem_accumulation(chain_length - 1, 0, available_checkpoints, level,
                                    table, subproblem_sequence, out_stream);

  if(out_stream){
    //Graphviz format
    *out_stream << "}\n";
  }

  return subproblem_sequence;
}

/**
 * @brief Sorts a sequence of node objects in ascending order by depth level.
 *
 * Orders operations from root (level 1) downward so that dependent operations follow 
 * their prerequisites for valid execution.
 *
 * @tparam Node_T Type of node element (must fulfill comparison operations).
 * @param[in,out] accumulation_sequence Vector of node operations to be sorted in place.
 */
template<class Node_T>
void obtain_serial_execution_sequence(std::vector<Node_T>& accumulation_sequence){

  std::sort(accumulation_sequence.begin(), accumulation_sequence.end());
}

#endif //OPTIMAL_ACCUMULATION_HPP
