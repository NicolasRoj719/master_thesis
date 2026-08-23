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
  ADJOINT_SPLIT
};

std::string to_string(Operation op){
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

std::ostream& operator<<(std::ostream& os, Operation op){
  return os << to_string(op);
}

template <class Jacobian_type>
class cell{};

template <>
class cell<Jacobian>{
 public:
  cell(std::size_t cost_, std::size_t split_pos):
    cost(cost_), k(split_pos){}

  std::size_t split_position() const {return k;}
  std::size_t accumulated_cost() const {return cost;}
 
 protected:
  std::size_t cost;
  std::size_t k;
};

template <>
class cell<Dense_Jacobian>: public cell<Jacobian>{
 public:
  cell(std::size_t cost_, std::size_t split_pos, Operation op):
    cell<Jacobian>(cost_, split_pos), operation_(op){}

  Operation operation() const {return operation_;}

 protected:
  Operation operation_;
};

template<>
class cell<Split_dense_Jacobian>: public cell<Dense_Jacobian>{
  public:
    cell(std::size_t cost_, std::size_t split_pos, Operation op):
      cell<Dense_Jacobian>(cost_, split_pos, op){}
};

template<>
class cell<Sparse_Jacobian>: public cell<Dense_Jacobian>{
 public:
  cell(const Sparse_Jacobian& sparse_jacobian_, std::size_t cost_, std::size_t split_pos,
     Operation op):
    cell<Dense_Jacobian>(cost_, split_pos, op), sparse_jacobian(std::move(sparse_jacobian_)){}

  cell(Sparse_Jacobian&& sparse_jacobian_, std::size_t cost_, std::size_t split_pos,
      Operation op):
    cell<Dense_Jacobian>(cost_, split_pos, op), sparse_jacobian(std::move(sparse_jacobian_)){}

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

  //Const references to compress data arrays
  const std::vector<std::size_t>& get_column_idx() const{

    return sparse_jacobian.get_column_idx_reference();
  }

  const std::vector<std::size_t>& get_row_pointer() const{

    return sparse_jacobian.get_row_pointer_reference();
  }

  const std::vector<std::size_t>& get_row_idx() const{

    return sparse_jacobian.get_row_idx_reference();
  }

  const std::vector<std::size_t>& get_column_pointer() const{

    return sparse_jacobian.get_column_pointer_reference();
  }

  const Sparse_Jacobian& get_jacobian() const{

    return sparse_jacobian;
  }

 private:
  Sparse_Jacobian sparse_jacobian;
};

template<>
class cell<Split_sparse_Jacobian>: public cell<Dense_Jacobian>{
 public:
   cell(Split_sparse_Jacobian split_sparse_, std::size_t cost_,
       std::size_t split_pos, Operation op):
  cell<Dense_Jacobian>(cost_, split_pos, op), split_sparse(std::move(split_sparse_)){}

   //Wrappers
   std::size_t number_edges() const{return split_sparse.number_edges();}

   std::size_t number_nnz() const{return split_sparse.number_nnz();}

   std::size_t column_number_colors() const {
    
     return split_sparse.get_column_number_colors();
   }

   std::size_t row_number_colors() const{
    
     return split_sparse.get_row_number_colors();
   }

   std::size_t max_number_nnz_row() const{
      return split_sparse.get_max_number_nnz_row();
   }

   std::size_t max_number_nnz_column() const{
      return split_sparse.get_max_number_nnz_column();
   }

   const Split_sparse_Jacobian& get_jacobian() const{

      return split_sparse;
   }

 private:
  Split_sparse_Jacobian split_sparse;
};

template <class Jacobian_type>
class cell_with_pointer{};

template <>
class cell_with_pointer<Jacobian>{
 public:
  cell_with_pointer(const Jacobian* jac_ptr):
    jacobian_ptr(jac_ptr){}

  //Wrappers
  std::size_t domain_dim() const {return jacobian_ptr -> domain_dim();}

  std::size_t codomain_dim() const {return jacobian_ptr -> codomain_dim();}

  const Jacobian* get_pointer() const {return jacobian_ptr;}

 private:
  const Jacobian* jacobian_ptr;

};

template <>
class cell_with_pointer<Dense_Jacobian>{
 public:
  cell_with_pointer(const Dense_Jacobian* jac_ptr, std::size_t cost_, Operation op):
    jacobian_ptr(jac_ptr), cost(cost_), operation_(op){}
  
  //Wrappers
  std::size_t domain_dim() const {return jacobian_ptr -> domain_dim();}

  std::size_t codomain_dim() const {return jacobian_ptr -> codomain_dim();}

  std::size_t number_edges() const {return jacobian_ptr -> number_edges();}

  //Cell information
  std::size_t accumulated_cost() const {return cost;}

  Operation operation() const {return operation_;}

  const Dense_Jacobian* get_pointer() const {return jacobian_ptr;}

 private:
  const Dense_Jacobian* jacobian_ptr;

  std::size_t cost;

  Operation operation_;
};

template<>
class cell_with_pointer<Split_dense_Jacobian>{
 public:
   cell_with_pointer(const Split_dense_Jacobian* jac_ptr, std::size_t cost_, Operation op):
     jacobian_ptr(jac_ptr), cost(cost_), operation_(op){}

   //Wrappers
   std::size_t domain_dim() const {return jacobian_ptr -> domain_dim();}

   std::size_t codomain_dim() const {return jacobian_ptr -> codomain_dim();}

   std::size_t number_edges() const {return jacobian_ptr -> number_edges();}

   std::size_t function_cost() const {return jacobian_ptr -> function_cost();}

   //Cell information
   std::size_t accumulated_cost() const {return cost;}

   Operation operation() const {return operation_;}

   const Split_dense_Jacobian* get_pointer() const {return jacobian_ptr;}

 private:
   const Split_dense_Jacobian* jacobian_ptr;

   std::size_t cost;

   Operation operation_;
};

template <>
class cell_with_pointer<Sparse_Jacobian>{
 public:
  cell_with_pointer(const Sparse_Jacobian* jac_ptr, std::size_t cost_, Operation op):
    jacobian_ptr(jac_ptr), cost(cost_), operation_(op){}

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

  //Const references to compress data arrays
  const std::vector<std::size_t>& get_column_idx() const{
    
    return jacobian_ptr -> get_column_idx_reference();
  }

  const std::vector<std::size_t>& get_row_pointer() const{

    return jacobian_ptr -> get_row_pointer_reference(); 
  }

  const std::vector<std::size_t>& get_row_idx() const {

    return jacobian_ptr -> get_row_idx_reference();
  }

  const std::vector<std::size_t>& get_column_pointer() const{

    return jacobian_ptr -> get_column_pointer_reference();
  }

  //Cell information
  std::size_t accumulated_cost() const {return cost;}

  Operation operation() const {return operation_;}


  const Sparse_Jacobian& get_jacobian() const{
    return *jacobian_ptr;
  } 

  const Sparse_Jacobian* get_pointer() const {return jacobian_ptr;}

 
 private:
  const Sparse_Jacobian* jacobian_ptr;

  std::size_t cost;

  Operation operation_;
};

template<>
class cell_with_pointer<Split_sparse_Jacobian>{
 public:
   cell_with_pointer(const Split_sparse_Jacobian* jac_ptr, std::size_t cost_, Operation op):
     jacobian_ptr(jac_ptr), cost(cost_), operation_(op){}

   //Wrappers
   std::size_t number_edges() const {return jacobian_ptr -> number_edges();}

   std::size_t number_nnz() const {return jacobian_ptr -> number_nnz();}

   std::size_t column_number_colors() const{
      return jacobian_ptr -> get_column_number_colors();
   }

   std::size_t row_number_colors() const{
    return jacobian_ptr -> get_row_number_colors();
   }

   std::size_t max_number_nnz_row() const {
    return jacobian_ptr -> get_max_number_nnz_row(); 
   }

   std::size_t max_number_nnz_column() const{
    return jacobian_ptr -> get_max_number_nnz_column();
   }

   //Cell information
   std::size_t accumulated_cost() const {return cost;}

   Operation operation() const {return operation_;}

   const Split_sparse_Jacobian& get_jacobian() const{
    return *jacobian_ptr;
   }

   const Split_sparse_Jacobian* get_pointer() const {return jacobian_ptr;}

 private:
   const Split_sparse_Jacobian* jacobian_ptr;

   std::size_t cost;

   Operation operation_;
};

#endif
