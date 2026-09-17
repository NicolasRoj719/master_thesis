/**
 * @file table_cell.hpp
 * @brief Implementation of table cell base classes and specialization for dynamic programming.
 */

#ifndef TABLE_CELL_HPP
#define TABLE_CELL_HPP
#include <iostream>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>

#include "jacobian.hpp"


/**
 * @brief Operation modes for Jacobian preaccumulation or accumulation.
 */
enum class Operation: std::uint8_t{
  MULTIPLICATION = 0,
  TANGENT,
  ADJOINT,
  ADJOINT_SPLIT
};

/**
 * @brief Parses an Operation enum value into its corresponding string representation.
 *
 * @param op Accumulation or preaccumulation operation mode.
 * @return String representation of the specidied Operation.
 */
inline std::string to_string(Operation op){
  switch(op){
    case Operation::MULTIPLICATION:
      return "MUL";
    case Operation::TANGENT:
      return "TAN";
    case Operation::ADJOINT:
      return "ADJ";
    case Operation::ADJOINT_SPLIT:
      return "ADJ_SPLIT";
    default:
      return "NOT DEFINED";
  }
}

/**
 * @brief Overloads output stream insertion operator for Operation enum.
 *
 * @param os Target output stream instance.
 * @param op Accumulation or preaccumulation method.
 * @returns Reference to the updated output stream.
 */
inline std::ostream& operator<<(std::ostream& os, Operation op){
  return os << to_string(op);
}

/**
 * @brief Primary template for table cell structure.
 * @tparam Jacobian_type Jacobian matrix type.
 */
template <class Jacobian_type>
class Cell{};

/**
 * @brief Specialization of Cell for base Jacobian object storing optimal data.
 */
template <>
class Cell<Jacobian>{
 public:
  /**
   * @brief Constructs Cell from explicit dynamic progamming (DP) table results.
   * @param cost_ Optimal accumulated cost.
   * @param split_pos Optimal split position index.
   */
  Cell(std::size_t cost_, std::size_t split_pos):
    cost(cost_), k(split_pos){}

  /// Gets the optimal chain split position index.
  std::size_t split_position() const {return k;}
  /// Gets the optimal accumulated computational cost.
  std::size_t accumulated_cost() const {return cost;}
 
 protected:
  /// Optimal accumulated cost in fused multiplied-add operations.
  std::size_t cost;
  /// Optimal split position index.
  std::size_t k;
};

/**
 * @brief Specialization of Cell for Dense_Jacobian objects.
 */
template <>
class Cell<Dense_Jacobian>: public Cell<Jacobian>{
 public:
  /**
   * @brief Constructs Cell from explicit dynamic programming (DP) table with accumulation mode.
   * @param cost_ Optimal accumulated execution cost.
   * @param split_pos Optimal split position index.
   * @param op Optimal accumulation method.
   */
  Cell(std::size_t cost_, std::size_t split_pos, Operation op):
    Cell<Jacobian>(cost_, split_pos), operation_(op){}

  /// Gets the optimal accumulation operation mode.
  Operation operation() const {return operation_;}

 protected:
  /// Optimal accumulation method.
  Operation operation_;
};


/**
 * @brief Specialization of Cell for Sparse_Jacobian objects owning an internal sparse Jacobian
 * matrix instance.
 */
template<>
class Cell<Sparse_Jacobian>: public Cell<Dense_Jacobian>{
 public:
  /**
   * @brief Constructs Cell by copying an existing Sparse_Jacobian instance. 
   * @Even though std::move is called on 'sparse_jacobian_', its const qualification causes 
   * overload resolution to invoke the copy constructor.
   * @param sparse_jacobian_ Reference to Sparse_Jacobian object to copy. 
   * @param cost_ Optimal accumulated execution cost.
   * @param split_pos Optimal split position index.
   * @param op Optimal accumulation method.
   */
  Cell(const Sparse_Jacobian& sparse_jacobian_, std::size_t cost_, std::size_t split_pos,
     Operation op):
    Cell<Dense_Jacobian>(cost_, split_pos, op), sparse_jacobian(std::move(sparse_jacobian_)){}

  /**
   * @brief Constructs Cell by moving resource ownership from an rvalue Sparse_Jacobian. 
   *
   *  @param sparse_jacobian_ Rvalue reference to Sparse_Jacobian instance.
   *  @param cost_ Optimal accumulated execution cost.
   *  @param split_pos Optimal split position index.
   *  @param op Optimal accumulation operation mode.
   */
  Cell(Sparse_Jacobian&& sparse_jacobian_, std::size_t cost_, std::size_t split_pos,
      Operation op):
    Cell<Dense_Jacobian>(cost_, split_pos, op), sparse_jacobian(std::move(sparse_jacobian_)){}

  //Wrappers
  /// Gets domain dimension of the underlying sparse Jacobian matrix.
  std::size_t domain_dim() const {return sparse_jacobian.domain_dim();}
  /// Gets codomain dimension of the underlying sparse Jacobian matrix.
  std::size_t codomain_dim() const {return sparse_jacobian.codomain_dim();}
  /// Gets total number of edges in underlying computational graph.
  std::size_t number_edges() const {return sparse_jacobian.number_edges();}
  /// Gets total non-zero elements count of underlying sparse matrix.
  std::size_t number_nnz() const {return sparse_jacobian.number_nnz();}
  /// Gets total column coloring count.
  std::size_t column_number_colors() const {
    return sparse_jacobian.get_column_number_colors();
  }
  /// Gets total row coloring count.
  std::size_t row_number_colors() const {
    return sparse_jacobian.get_row_number_colors();
  }
  /// Gets maximum number of non-zero entries present in any row.
  std::size_t max_number_nnz_row() const {
    return sparse_jacobian.get_max_number_nnz_row();
  }
  /// Gets maximum number of non-zero entries present in any column.
  std::size_t max_number_nnz_column() const {
    return sparse_jacobian.get_max_number_nnz_column();
  }

  //Const references to compress data arrays.
  /// Returns reference to column indices array.
  const std::vector<std::size_t>& get_column_idx() const{
    return sparse_jacobian.get_column_idx_reference();
  }
  /// Returns reference to row pointer to indices array.
  const std::vector<std::size_t>& get_row_pointer() const{
    return sparse_jacobian.get_row_pointer_reference();
  }

  /// Returns reference to row indices array. 
  const std::vector<std::size_t>& get_row_idx() const{

    return sparse_jacobian.get_row_idx_reference();
  }
  /// Returns reference to column pointer indices array.
  const std::vector<std::size_t>& get_column_pointer() const{

    return sparse_jacobian.get_column_pointer_reference();
  }
  /// Gets const reference to underlying Sparse_Jacobian instance.
  const Sparse_Jacobian& get_jacobian() const{

    return sparse_jacobian;
  }

 private:
  /// Owned Sparse_Jacobian object.
  Sparse_Jacobian sparse_jacobian;
};


/**
 * @brief Base template for table Cell wrappers maintaining pointer references to Jacobian matrices
 * in the chain.
 * 
 * @tparam Jacobian_type Jacobian matrix type referenced by non-owning pointer.
 */
template <class Jacobian_type>
class Cell_with_pointer{};

/**
 * @brief Specialization of Cell_with_pointer referencing Jacobian object. 
 */
template <>
class Cell_with_pointer<Jacobian>{
 public:
  /**
   * @brief Constructs a Cell wrapper holding a const pointer to a chain Jacobian.
   * @param jac_ptr Pointer referencing target Jacobian in chain.
   */
  Cell_with_pointer(const Jacobian* jac_ptr):
    jacobian_ptr(jac_ptr){}

  //Wrappers
  /// Gets domain dimension of referenced Jacobian matrix.
  std::size_t domain_dim() const {return jacobian_ptr -> domain_dim();}

  /// Gets codomain dimension of referenced Jacobian matrix.
  std::size_t codomain_dim() const {return jacobian_ptr -> codomain_dim();}

  /// Returns pointer to underlying Jacobian instance.
  const Jacobian* get_pointer() const {return jacobian_ptr;}

 private:
  /// Non-owning pointer to Jacobian matrix in chain.
  const Jacobian* jacobian_ptr;

};

/**
 * @brief Specialization of Cell_with_pointer referencing Dense_Jacobian objects. 
 */
template <>
class Cell_with_pointer<Dense_Jacobian>{
 public:
  /**
   * @brief Constructs Cell wrapper referencing Dense_Jacobian with dynamic programming (DP) metrics.
   *
   * @param jac_ptr Pointer referencing target Dense_Jacobian in chain.
   * @param cost_ Optimal preaccumulation cost.
   * @param Optimal preaccumulation operation mode.
   */
  Cell_with_pointer(const Dense_Jacobian* jac_ptr, std::size_t cost_, Operation op):
    jacobian_ptr(jac_ptr), cost(cost_), operation_(op){}
  
  //Wrappers
  /// Gets domain space dimension of referenced dense Jacobian matrix.
  std::size_t domain_dim() const {return jacobian_ptr -> domain_dim();}

  /// Gets codomain space dimension of referenced dense Jacobian matrix.
  std::size_t codomain_dim() const {return jacobian_ptr -> codomain_dim();}

  /// Gets edge count of underlying computational graph.
  std::size_t number_edges() const {return jacobian_ptr -> number_edges();}

  //Cell information
  /// Gets optimal preaccumulation computational cost.
  std::size_t accumulated_cost() const {return cost;}

  /// Gets optimal accumulation operation mode. 
  Operation operation() const {return operation_;}

  /// Returns pointer to underlying Dense_Jacobian instance.
  const Dense_Jacobian* get_pointer() const {return jacobian_ptr;}

 private:
  /// Non-owning pointer to Dense_Jacobian matrix in chain.
  const Dense_Jacobian* jacobian_ptr;

  /// Optimal preaccumulated cost in fused multiply-add operations.
  std::size_t cost;

  // Optimal preaccumulation method.
  Operation operation_;
};


/**
 * @brief Specialization of Cell_with_pointer referencing Sparse_Jacobian objects.
 */
template <>
class Cell_with_pointer<Sparse_Jacobian>{
 public:
  /**
   * @brief Constructs a Cell wrapper referencing Sparse_Jacobian with dynamic programming (DP) metrics.
   *
   * @param jac_ptr Pointer referencing target Sparse_Jacobian in chain.
   * @param cost_ Optimal preaccumulation cost.
   * @param op Optimal preaccumulation method.
   */
  Cell_with_pointer(const Sparse_Jacobian* jac_ptr, std::size_t cost_, Operation op):
    jacobian_ptr(jac_ptr), cost(cost_), operation_(op){}

  //Wrappers
  /// Gets domain space dimension of referenced sparse Jacobian matrix.
  std::size_t domain_dim() const {return jacobian_ptr -> domain_dim();}
  /// Gets codomain space dimension of referenced sparse Jacobian matrix.
  std::size_t codomain_dim() const {return jacobian_ptr -> codomain_dim();}
  /// Gets edge count of referenced computational graph.
  std::size_t number_edges() const {return jacobian_ptr -> number_edges();}
  /// Gets total non-zero elements count of referenced sparse Jacobian matrix.
  std::size_t number_nnz() const {return jacobian_ptr -> number_nnz();}

  /// Gets column coloring count.
  std::size_t column_number_colors() const {
    return jacobian_ptr -> get_column_number_colors();
  }

  /// Gets row coloring count.
  std::size_t row_number_colors() const {
    return jacobian_ptr -> get_row_number_colors();
  }

  /// Gets maximum number of non-zero entries in any single row.
  std::size_t max_number_nnz_row() const {
    return jacobian_ptr -> get_max_number_nnz_row();
  }
  /// Gets maximum number of non-zero entries in any single column.
  std::size_t max_number_nnz_column() const {
    return jacobian_ptr -> get_max_number_nnz_column();
  }

  //Const references to compress data arrays
  /// Returns reference to column indices array.
  const std::vector<std::size_t>& get_column_idx() const{
    
    return jacobian_ptr -> get_column_idx_reference();
  }

  /// Returns reference to row pointer indices array.
  const std::vector<std::size_t>& get_row_pointer() const{

    return jacobian_ptr -> get_row_pointer_reference(); 
  }

  /// Returns reference to row indices array.
  const std::vector<std::size_t>& get_row_idx() const {

    return jacobian_ptr -> get_row_idx_reference();
  }

  /// Returns reference to column pointer indices array.
  const std::vector<std::size_t>& get_column_pointer() const{

    return jacobian_ptr -> get_column_pointer_reference();
  }

  //Cell information
  /// Gets optimal preaccumulation computational cost.
  std::size_t accumulated_cost() const {return cost;}

  /// Gets optimal preaccumulation method.
  Operation operation() const {return operation_;}

  /// Returns const reference to pointed Sparse_Jacobian object.
  const Sparse_Jacobian& get_jacobian() const{
    return *jacobian_ptr;
  } 

  /// Returns raw pointer to referenced Sparse_Jacobian instance.
  const Sparse_Jacobian* get_pointer() const {return jacobian_ptr;}

 
 private:
  /// Non-owning pointer to Sparse_Jacobian matrix in chain.
  const Sparse_Jacobian* jacobian_ptr;
  /// Optimal preacumulation cost in fused multiply-add operations.
  std::size_t cost;
  /// Optimal preaccumulation method.
  Operation operation_;
};

#endif //TABLE_CELL_HPP
