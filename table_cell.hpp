#include <iostream>
#include <cstdint>
#include <string>
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


class Table_cell{
 public:
  Table_cell(std::size_t cost, std::size_t k, Operation op):
    cost_(cost), k_(k), operation(op){}
  
  Table_cell():cost_(0), k_(0),operation(Operation::NONE){}

  std::string op_to_string(){
    return to_string(operation);
  }

  std::string k_to_string(){
    return std::to_string(k_);
  }

  std::size_t k(){return k_;}
  std::size_t cost(){return cost_;}

 protected:
  std::size_t cost_;
  //split position
  std::size_t k_;
  //Operation type
  Operation operation;
};

class cell_DJCPB: public Table_cell{
 public:
  cell_DJCPB(std::size_t cost, std::size_t k, Operation op):
    Table_cell(cost, k, op){};
  
  void print() const{
    std::cout<<"[ "<<cost_<<' '<<k_<<' '<<operation<<" ]\n";
  }
};

class cell_DJCPB_p: public cell_DJCPB{
 public:
  cell_DJCPB_p(Jacobian* jac_p, std::size_t cost, std::size_t k, Operation op):
    cell_DJCPB(cost, k, op), jac_pointer(jac_p){};
  
  //Wrappers
  std::size_t n(){return jac_pointer -> n();}
  std::size_t m(){return jac_pointer -> m();}
 protected:
  Jacobian* jac_pointer;
};

class cell_MFDJCPB: public Table_cell{
 public:
  cell_MFDJCPB(std::size_t cost, std::size_t k, Operation op,
      std::size_t memory): Table_cell(cost, k, op), memory_(memory){}

  cell_MFDJCPB(): memory_(0){}

  std::string mem_to_string(){
    return std::to_string(memory_);
  }

  void print() const{
    /* Table_cell::print(); */
    /* std::cout<<memory<<" ]\n"; */
    std::cout<<"[ "<<cost_<<' '<<k_<<' '<<operation<< ' '<<memory_<<" ]\n";
  }

  std::size_t mem_lim(){return memory_;}
 protected:
  std::size_t memory_;
};

class cell_MFDJCPB_p: public cell_MFDJCPB{
 public:
  cell_MFDJCPB_p(Dense_Jacobian* jac_p, std::size_t cost, std::size_t k,
      Operation op, std::size_t memory):
    cell_MFDJCPB(cost, k, op, memory), dense_pointer(jac_p){}

  //Wrappers
  std::size_t n(){return dense_pointer -> n();}
  std::size_t m(){return dense_pointer -> m();}
  std::size_t n_E(){return dense_pointer -> n_E();}
 protected:
  Dense_Jacobian* dense_pointer;
};

class cell_MFSJCPB: public cell_MFDJCPB{
 public: 
  cell_MFSJCPB(Sparse_Jacobian jacobian, std::size_t cost, std::size_t k,
      Operation op, std::size_t memory):
    cell_MFDJCPB(cost, k, op, memory), sparse_jac(jacobian){}

  cell_MFSJCPB(Sparse_Jacobian jacobian): sparse_jac(jacobian){}

  void initiliaze_cell(std::size_t cost, std::size_t k, Operation op,
      std::size_t memory){
    cost_ = cost;
    k_ = k;
    operation = op;
    memory_ = memory;
  }
  
  //Wrappers
  std::size_t num_nnz(){return sparse_jac.num_nnz();}
  std::size_t col_num_colors(){return sparse_jac.col_num_colors();}
  std::size_t row_num_colors(){return sparse_jac.row_num_colors();}
  std::size_t rhs_inner_dim(){return sparse_jac.rhs_inner_dim();}
  std::size_t lhs_inner_dim(){return sparse_jac.lhs_inner_dim();}
  
 private:
  Sparse_Jacobian sparse_jac;
};

class cell_MFSJCPB_p: public cell_MFDJCPB{
 public:
  cell_MFSJCPB_p(Sparse_Jacobian* jac_p, std::size_t cost, std::size_t k,
      Operation op, std::size_t memory):
    cell_MFDJCPB(cost, k, op, memory), sparse_pointer(jac_p){}
  //Wrappers
  std::size_t num_nnz(){return sparse_pointer -> num_nnz();}
  std::size_t col_num_colors(){return sparse_pointer -> col_num_colors();}
  std::size_t row_num_colors(){return sparse_pointer -> row_num_colors();}
  std::size_t rhs_inner_dim(){return sparse_pointer -> rhs_inner_dim();}
  std::size_t lhs_inner_dim(){return sparse_pointer -> lhs_inner_dim();}
 private:
  Sparse_Jacobian* sparse_pointer;
};
#endif
