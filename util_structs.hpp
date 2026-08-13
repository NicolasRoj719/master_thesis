#include <cstdint>
#include <stdexcept>

#ifndef UTIL_STRUCTS_HPP
#define UTIL_STRUCTS_HPP

class NNZ{
 public: 
  NNZ(std::size_t row, std::size_t col): row_(row), col_(col){}
  std::size_t row() const {return row_;}
  std::size_t col() const {return col_;}
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

class Jacobian_information{
 public:
  Jacobian_information(std::size_t domain_dim, std::size_t codomain_dim):
    domain_dimension_(domain_dim), codomain_dimension_(codomain_dim){}

  std::size_t domain_dimension() const {return domain_dimension_;}
  std::size_t codomain_dimension() const {return codomain_dimension_;}
 protected:
  std::size_t domain_dimension_;
  std::size_t codomain_dimension_;
};

class Matrix_free_information: public Jacobian_information{
 public:
  Matrix_free_information(std::size_t domain_dim, std::size_t codomain_dim,
      std::size_t num_of_edges):
    Jacobian_information(domain_dim, codomain_dim), number_of_edges_(num_of_edges){}

  std::size_t number_of_edges() const {return number_of_edges_;}

 protected:
  std::size_t number_of_edges_;
};

class Split_reversal_dense_information: public Matrix_free_information{
 public:
   Split_reversal_dense_information(Matrix_free_information jacobian_information,
       std::size_t function_cost):
     Matrix_free_information(std::move(jacobian_information)), function_cost_(function_cost){}

   std::size_t function_cost() const {return function_cost_;}

 protected:
   std::size_t function_cost_;
};

class Matrix_free_sparse_information: public Matrix_free_information{
 public:
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

class Split_reversal_sparse_information: public Matrix_free_sparse_information{
 public:
  Split_reversal_sparse_information(Matrix_free_sparse_information jacobian_information,
      std::size_t function_cost):
    Matrix_free_sparse_information(std::move(jacobian_information)),
    function_cost_(function_cost){}

  std::size_t function_cost() const {return function_cost_;}

 protected:
  std::size_t function_cost_;
};

struct Generator_data{
 std::vector<Matrix_free_sparse_information> jacobian_information;
 std::vector<std::vector<NNZ>> sparse_data;
};

#endif
