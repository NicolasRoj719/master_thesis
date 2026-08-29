/**
 * @file util_structs.hpp
 * @brief Base types and metadata structures used across the AD optimization
 * framework.
 */
#ifndef UTIL_STRUCTS_HPP
#define UTIL_STRUCTS_HPP

#include <cstdint>
#include <stdexcept>

/**
 * @brief Coordinates (row, column) for a single non-zero entry in a sparse matrix.
 *
 * The collection of all NNZ entries for a given matrix defines its sparsity pattern.
 */
class NNZ{
 public:
  /**
   * @brief Constructs an NNZ coordinat pair.
   * @param row Zero-based row index.
   * @param col Zero-based column index.
   */
  NNZ(std::size_t row, std::size_t col): row_(row), col_(col){}
  
  std::size_t row() const {return row_;}
  
  std::size_t col() const {return col_;}
  
  /**
   * @bried comparison operator to establish a strict ordering.
   *
   * Used to verify that sparsity pattern generators produce unique, non-repeated entries.
   * @param rhs Right-hand side NNZ instance to compare against.
   * @return True if this instance preces 'rhs' in row-major order.
   */
  bool operator<(NNZ const& rhs) const{
    if(row_ < rhs.row()){
      return true;
    }
    if(row_ == rhs.row()){
      if(col_ < rhs.col()){
        return true;
      }
    }
    return false;
  }

  bool operator==(NNZ const& rhs) const{
    if(row_ == rhs.row() && col_ == rhs.col()){
      return true;
    }
    return false;
  }

 protected:
  std::size_t row_;
  std::size_t col_;
};

/**
 * @brief Dimensions metadata required for Jacobian dense chain bracketing.
 *
 * Stores basic dimensions used by dynamic programming algorithms to optimize
 * matrix multiplication bracketing, assuming dense matrices.
 *
 */
class Jacobian_information{
 public:
   /**
    * @brief Constructs Jacobian dimension metadata.
    * @param domain_dim Dimension of the domain.
    * @param codomain_dim Dimension of the codomain.
    */
  Jacobian_information(std::size_t domain_dim, std::size_t codomain_dim):
    domain_dimension_(domain_dim), codomain_dimension_(codomain_dim){}

  std::size_t domain_dimension() const {return domain_dimension_;}
  std::size_t codomain_dimension() const {return codomain_dimension_;}
 protected:
  std::size_t domain_dimension_;
  std::size_t codomain_dimension_;
};

/**
 *
 *  @brief Extended metadata including edge counts for matrix-free evaluation.
 *
 *  Extends Jacobian_information with edge count data ('number_of_edges_') to
 *  estimate the preaccumulation cost of elemental Jacobians when solving the
 *  Matrix-Free Dense Bracketing Problem.
 */
class Matrix_free_information: public Jacobian_information{
 public:
  /**
   * @brief Constructs matrix-free metadata iwht edge counts.
   * @param domain_dim Domain dimension.
   * @param codomain_dim Codomain dimension.
   * @param number_of_edges Number of edges present in the computational 
   * graph representation of the subprogram.
   */
  Matrix_free_information(std::size_t domain_dim, std::size_t codomain_dim,
      std::size_t num_of_edges):
    Jacobian_information(domain_dim, codomain_dim), number_of_edges_(num_of_edges){}

  std::size_t number_of_edges() const {return number_of_edges_;}

 protected:
  std::size_t number_of_edges_;
};

/**
 * @brief Metadata for dense Jacobians including estimate for the function
 * evalutaion cost in terms of fused multiply add (fma) operations.
 *
 * With the function evaluation cost estimate Binomial Checkpoiting algorithm
 * may be used to obtain the additional cost incurred in accumulating the Jacobian
 * via split reverse mode of Algorithmic Differentiation (AD).
 */
class Split_reversal_dense_information: public Matrix_free_information{
 public:
   /**
    * @brief Constructs split reversal metadata from base matrix-free info.
    * @param jacobian_information Base matrix-free properties.
    * @param function_cost Evaluation cost estimateof the primal subprogram.
    */
   Split_reversal_dense_information(Matrix_free_information jacobian_information,
       std::size_t function_cost):
     Matrix_free_information(std::move(jacobian_information)), function_cost_(function_cost){}

   std::size_t function_cost() const {return function_cost_;}

 protected:
   std::size_t function_cost_;
};

/**
 * @brief Metadata for sparse matrix-free Jacobians.
 *
 * Contains non-zero counts required to pre-allocate memory for Compressed Sparse Column
 * (CSC) and Compressed Sparse Row (CSR) index and pointer arrays.
 */
class Matrix_free_sparse_information: public Matrix_free_information{
 public:
  /**
   * @brief Constructs and validates sparse Jacobian metadata.
   * @param domain_dim Domain dimension.
   * @param codomain_dim Codomain dimension.
   * @param num_of_edges Graph edge count.
   * @param num_nonzeros Count of non-zero entries.
   *
   * @throws std::invalid_argument If 'num_nonzeros' exceeds total matrix capacity
   * ('domain_dim * codomain_dim')
   * @throws std::invalid_argument If 'num_nonzeros' is less than 'std::max(domain, codomain_dim)'
   */
  Matrix_free_sparse_information(std::size_t domain_dim, std::size_t codomain_dim,
      std::size_t num_of_edges, std::size_t num_nonzeros):
    Matrix_free_information(domain_dim, codomain_dim, num_of_edges),
    number_of_nonzeros_(num_nonzeros){

      if(domain_dim * codomain_dim < num_nonzeros){
        throw std::invalid_argument(
            "The number of non zero entries must be less than "
            "the total entries of the Jacobian.\n" 
            "Number of non zeros entered: " + std::to_string(num_nonzeros) + '\n' +
            "Maximum number of non zeros entries: " +std::to_string(domain_dim * codomain_dim)+
            '\n');
      }

      std::size_t max_domain_codomain = (domain_dim >= codomain_dim)? domain_dim : codomain_dim;
      
      if(num_nonzeros < max_domain_codomain){
        throw std::invalid_argument(
            "The number of non zero entries must be the maximum between "
            "the codomain and domain dimension.\n"
            "Number of non zeros entered: " + std::to_string(num_nonzeros) + '\n' +
            "Maximum between domain and codomain dimension: "
            + std::to_string(max_domain_codomain) + '\n');
      }
    }

  std::size_t number_of_nonzeros() const {return number_of_nonzeros_;}

 protected:
  std::size_t number_of_nonzeros_;
};


/**
 * @brief Metadata for Matrix_free_sparse_information with cost estimate of
 * subprogram evaluation in terms of fused multiply add (fma).
 */
class Split_reversal_sparse_information: public Matrix_free_sparse_information{
 public:
  /**
   * @brief Construcs sparse split reversal metadata.
   * @param Matrix_free_sparse_information Base sparse matrix-free properties.
   * @param function_cost Evaluation cost of the primal subprogram.
   */
  Split_reversal_sparse_information(Matrix_free_sparse_information jacobian_information,
      std::size_t function_cost):
    Matrix_free_sparse_information(std::move(jacobian_information)),
    function_cost_(function_cost){}

  std::size_t function_cost() const {return function_cost_;}

 protected:
  std::size_t function_cost_;
};

/**
 * @brief Container structure holding synthetic generator outputs. 
 */
struct Generator_data{
  /// Metadata for each elemental Jacobian in the chain.
 std::vector<Matrix_free_sparse_information> jacobian_information;

 /// Sparsity patterns for each elemental Jacobian.
 std::vector<std::vector<NNZ>> sparse_data;
};

#endif
