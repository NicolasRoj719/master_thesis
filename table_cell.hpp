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
  //cell_with_pointer initializes first the pointer
  //without initializing the cell data.
  cell(){}

  std::string split_position_to_string(){
    return std::to_string(k);
  }

  std::size_t split_position(){return k;}
  std::size_t accumulated_cost(){return cost;}

  void print(){
    std::cout<< "[ " << cost << ' ' << k << " ]\n";
  }
 
 protected:
  std::size_t cost;
  std::size_t k;
};

template <>
class cell<Dense_Jacobian>: public cell<Jacobian>{
 public:
  cell(std::size_t cost_, std::size_t k_, Operation op):
    cell<Jacobian>(cost_, k_), operation(op){}

  cell(std::size_t cost_, std::size_t k_, Operation op,
      std::optional<std::size_t> memory_):
    cell<Jacobian>(cost_, k_), operation(op), memory(memory_){}

  std::string op_to_string(){
    return to_string(operation);
  }
  //cell_with_pointer initializes first the pointer
  //without initializing the cell data.
  cell(){}

  void print(){
    std::cout<< "[ " << cost << ' ' << k << ' ' << operation << " ]\n";
  }

  std::optional<std::size_t> accumulated_memory_use(){return memory;}

 protected:
  Operation operation;
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
  std::size_t num_nnz() const{return sparse_jacobian.num_nnz();}
  std::size_t col_number_colors() const {return sparse_jacobian.col_num_colors();}
  std::size_t row_number_colors() const {return sparse_jacobian.row_num_colors();}
  std::size_t rhs_inner_dimension() const {return sparse_jacobian.rhs_inner_dim();}
  std::size_t lhs_inner_dimension() const {return sparse_jacobian.lhs_inner_dim();}

  void print_basic_information() const{
    std::cout <<"Sparse jacobian information: " 
      "[n col_number_colors m row_number_colors n_E nnz]: \n";
    sparse_jacobian.print_cell_relevant_information();
  }

  void print_basic_info_plus_CSR_CSC() const{
    print_basic_information();
    sparse_jacobian.print_CSR_CSC();
  }

 protected:
  const Sparse_Jacobian sparse_jacobian;
};

template <class Jacobian_type>
class cell_with_pointer{};

template <>
class cell_with_pointer<Jacobian>: cell<Jacobian>{
 public:
  cell_with_pointer(const Jacobian* jac_ptr):
    cell<Jacobian>(), jacobian_ptr(jac_ptr){}

  cell_with_pointer(const Jacobian* jac_ptr, std::size_t cost_, std::size_t k_):
    cell<Jacobian>(cost_, k_), jacobian_ptr(jac_ptr){}

  void cell_data_initializer(std::size_t cost_, std::size_t k_){
    cell<Jacobian>(cost_, k_);
  }

  //Wrappers
  std::size_t n() const {return jacobian_ptr -> n();}
  std::size_t m() const {return jacobian_ptr -> m();}

 private:
  const Jacobian* jacobian_ptr;
};

template <>
class cell_with_pointer<Dense_Jacobian>: public cell<Dense_Jacobian>{
 public:
  cell_with_pointer(const Dense_Jacobian* jac_ptr):
    cell<Dense_Jacobian>(), jacobian_ptr(jac_ptr){}

  cell_with_pointer(const Dense_Jacobian* jac_ptr, std::size_t cost_, std::size_t k_,
      Operation op):
    cell<Dense_Jacobian>(cost_, k_, op), jacobian_ptr(jac_ptr){}

  cell_with_pointer(const Dense_Jacobian* jac_ptr, std::size_t cost_, std::size_t k_,
      Operation op, std::size_t memory_):
    cell<Dense_Jacobian>(cost_, k_, op, memory_), jacobian_ptr(jac_ptr){}

  void cell_data_initializer(std::size_t cost_, std::size_t k_, Operation op){
    cell<Dense_Jacobian>(cost_, k_, op);
  }

  void cell_data_initializer(std::size_t cost_, std::size_t k_,
      Operation op, std::size_t memory_){
    cell<Dense_Jacobian>(cost_, k_, op, memory_);
  }
  
  //Wrappers
  std::size_t n() const {return jacobian_ptr -> n();}
  std::size_t m() const {return jacobian_ptr -> m();}
  std::size_t n_E() const {return jacobian_ptr -> n_E();}

 private:
  const Dense_Jacobian* jacobian_ptr;
};

template <>
class cell_with_pointer<Sparse_Jacobian>: public cell<Dense_Jacobian>{
 public:
  cell_with_pointer(const Sparse_Jacobian* jac_ptr):
    cell<Dense_Jacobian>(), jacobian_ptr(jac_ptr){}

  cell_with_pointer(const Sparse_Jacobian* jac_ptr, std::size_t cost_, std::size_t k_,
      Operation op):
    cell<Dense_Jacobian>(cost_, k_, op), jacobian_ptr(jac_ptr){}

  cell_with_pointer(const Sparse_Jacobian* jac_ptr, std::size_t cost_, std::size_t k_,
      Operation op, std::size_t memory_):
    cell<Dense_Jacobian>(cost_, k_, op, memory_), jacobian_ptr(jac_ptr){}

  void cell_data_initializer(std::size_t cost_, std::size_t k_, Operation op){
    cell<Dense_Jacobian>(cost_, k_, op);
  }

  void cell_data_initializer(std::size_t cost_, std::size_t k_,
      Operation op, std::size_t memory_){
    cell<Dense_Jacobian>(cost_, k_, op, memory_);
  }

  //Wrappers
  std::size_t num_nnz() const{return jacobian_ptr->num_nnz();}
  std::size_t col_number_colors() const {return jacobian_ptr->col_num_colors();}
  std::size_t row_number_colors() const {return jacobian_ptr->row_num_colors();}
  std::size_t rhs_inner_dimension() const {return jacobian_ptr->rhs_inner_dim();}
  std::size_t lhs_inner_dimension() const {return jacobian_ptr->lhs_inner_dim();}

  void print_basic_information() const{
    std::cout <<"Sparse jacobian information: " 
      "[n col_number_colors m row_number_colors n_E nnz]: \n";
    jacobian_ptr->print_cell_relevant_information();
  }

  void print_basic_info_plus_CSR_CSC() const{
    print_basic_information();
    jacobian_ptr->print_CSR_CSC();
  }
 
 private:
  const Sparse_Jacobian* jacobian_ptr;
};

#endif
