#include <iostream>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include "./jacobian.hpp"

#ifndef TABLE_CELL_HPP
#define TABLE_CELL_HPP

enum class Operation: std::uint8_t{
  MULTIPLICATION = 0,
  TANGENT,
  ADJOINT,
  NONE,
};

std::string to_string(Operation op){
  switch(op){
    case Operation::MULTIPLICATION:
      return "MUL";
    case Operation::TANGENT:
      return "TAN";
    case Operation::ADJOINT:
      return "ADJ";
    case Operation::NONE:
      return "NONE";
    default:
      return "NOT DEFINED";
  }
}

std::ostream& operator<<(std::ostream& os, Operation op){
  return os << to_string(op);
}

template <class Jacobian_type>
class cell{};

template <>
class cell<Jacobian>{
 public:
  cell(std::size_t cost_, std::size_t k_):
    cost(cost_), k(k_){}

  std::size_t split_position() const {return k;}
  std::size_t accumulated_cost() const {return cost;}
 
 protected:
  std::size_t cost;
  std::size_t k;
};

template <>
class cell<Dense_Jacobian>: public cell<Jacobian>{
 public:
  cell(std::size_t cost_, std::size_t k_, Operation op):
    cell<Jacobian>(cost_, k_), operation_(op){}

  cell(std::size_t cost_, std::size_t k_, Operation op,
      std::size_t memory_):
    cell<Jacobian>(cost_, k_), operation_(op), memory(memory_){}

  const Operation& operation(){return operation_;}

  const std::optional<std::size_t>& accumulated_memory() const{
    
    return memory;
  }  

 protected:
  Operation operation_;
  std::optional<std::size_t> memory;
};

template<>
class cell<Sparse_Jacobian>: public cell<Dense_Jacobian>{
 public:
  cell(Sparse_Jacobian sparse_jacobian_, std::size_t cost_, std::size_t k_,
     Operation op):
    cell<Dense_Jacobian>(cost_, k_, op), sparse_jacobian(sparse_jacobian_){}

  cell(Sparse_Jacobian sparse_jacobian_, std::size_t cost_, std::size_t k_,
     Operation op, std::size_t memory_):
    cell<Dense_Jacobian>(cost_, k_, op, memory_), sparse_jacobian(sparse_jacobian_){}

  //Wrappers

  std::size_t domain_dim() const {return sparse_jacobian.domain_dim();}

  std::size_t codomain_dim() const {return sparse_jacobian.codomain_dim();}

  std::size_t number_edges() const {return sparse_jacobian.number_edges();}

  std::size_t number_nnz() const {return sparse_jacobian.number_nnz();}

  std::size_t column_number_colors() const {
    return sparse_jacobian.get_column_number_colors();
  }

  std::size_t row_number_colors() const {
    return sparse_jacobian.get_row_number_colors();
  }

  std::size_t max_number_nnz_row() const {
    return sparse_jacobian.get_max_number_nnz_row();
  }

  std::size_t max_number_nnz_column() const {
    return sparse_jacobian.get_max_number_nnz_column();
  }

 protected:
  const Sparse_Jacobian sparse_jacobian;
};

template <class Jacobian_type>
class cell_with_pointer{};

template <>
class cell_with_pointer<Jacobian>: public cell<Jacobian>{
 public:
  cell_with_pointer(const Jacobian* jac_ptr, std::size_t cost_, std::size_t k_):
    cell<Jacobian>(cost_, k_), jacobian_ptr(jac_ptr){}

  //Wrappers
  std::size_t domain_dim() const {return jacobian_ptr -> domain_dim();}

  std::size_t codomain_dim() const {return jacobian_ptr -> codomain_dim();}

 private:
  const Jacobian* jacobian_ptr;
};

template <>
class cell_with_pointer<Dense_Jacobian>: public cell<Dense_Jacobian>{
 public:
  cell_with_pointer(const Dense_Jacobian* jac_ptr, std::size_t cost_, std::size_t k_,
      Operation op):
    cell<Dense_Jacobian>(cost_, k_, op), jacobian_ptr(jac_ptr){}

  cell_with_pointer(const Dense_Jacobian* jac_ptr, std::size_t cost_, std::size_t k_,
      Operation op, std::size_t memory_):
    cell<Dense_Jacobian>(cost_, k_, op, memory_), jacobian_ptr(jac_ptr){}

  
  //Wrappers
  std::size_t domain_dim() const {return jacobian_ptr -> domain_dim();}

  std::size_t codomain_dim() const {return jacobian_ptr -> codomain_dim();}

  std::size_t number_edges() const {return jacobian_ptr -> number_edges();}

 private:
  const Dense_Jacobian* jacobian_ptr;
};

template <>
class cell_with_pointer<Sparse_Jacobian>: public cell<Dense_Jacobian>{
 public:
  cell_with_pointer(const Sparse_Jacobian* jac_ptr, std::size_t cost_, std::size_t k_,
      Operation op):
    cell<Dense_Jacobian>(cost_, k_, op), jacobian_ptr(jac_ptr){}

  cell_with_pointer(const Sparse_Jacobian* jac_ptr, std::size_t cost_, std::size_t k_,
      Operation op, std::size_t memory_):
    cell<Dense_Jacobian>(cost_, k_, op, memory_), jacobian_ptr(jac_ptr){}

  //Wrappers

  std::size_t domain_dim() const {return jacobian_ptr -> domain_dim();}

  std::size_t codomain_dim() const {return jacobian_ptr -> codomain_dim();}

  std::size_t number_edges() const {return jacobian_ptr -> number_edges();}

  std::size_t number_nnz() const {return jacobian_ptr -> number_nnz();}

  std::size_t column_number_colors() const {
    return jacobian_ptr -> get_column_number_colors();
  }

  std::size_t row_number_colors() const {
    return jacobian_ptr -> get_row_number_colors();
  }

  std::size_t max_number_nnz_row() const {
    return jacobian_ptr -> get_max_number_nnz_row();
  }

  std::size_t max_number_nnz_column() const {
    return jacobian_ptr -> get_max_number_nnz_column();
  }
 
 private:
  const Sparse_Jacobian* jacobian_ptr;
};

#endif
