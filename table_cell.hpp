#include <iostream>
#include <cstdint>
#include <string>

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
  Table_cell(std::size_t op_cost_=0, std::size_t k_=0, Operation op_= Operation::NONE):
    op_cost(op_cost_), k(k_), operation(op_){}

  void operator()(std::size_t op_cost_, std::size_t k_, Operation op_){
    op_cost = op_cost_;
    k = k_;
    operation = op_;
  }

  std::string op_to_string(){
    return to_string(operation);
  }

  std::string k_to_string(){
    return std::to_string(k);
  }

  std::size_t op_cost;
  //split position
  std::size_t k;
  //Operation type
  Operation operation= Operation::NONE;

  /* void print(){ */
  /*   std::cout<<"[ "<<op_cost<<' '<<k<<' '<<operation<<' '; */
  /* }; */
};

class cell_DJCPB: public Table_cell{
 public:
  cell_DJCPB(std::size_t op_cost=0, std::size_t k_=0, Operation op_= Operation::NONE):
    Table_cell(op_cost, k_, op_){};

  void operator()(std::size_t op_cost_, std::size_t k_, Operation op_){
    Table_cell::operator()(op_cost_, k_, op_);
  }
  
  void print(){
    /* Table_cell::print(); */
    /* std::cout<<"]\n"; */
    std::cout<<"[ "<<op_cost<<' '<<k<<' '<<operation<<" ]\n";
  }
};

class cell_MFDJCPB: public Table_cell{
 public:
  cell_MFDJCPB(std::size_t op_cost_=0, std::size_t k_=0, Operation op_= Operation::NONE,
      std::size_t memory_=0): Table_cell(op_cost_, k_, op_), memory(memory_){}

  void operator()(std::size_t op_cost_, std::size_t k_, Operation op_, std::size_t memory_=0){
    Table_cell::operator()(op_cost_,k_,op_);
    memory = memory_;
  }

  std::string mem_to_string(){
    return std::to_string(memory);
  }

  std::size_t memory;

  void print(){
    /* Table_cell::print(); */
    /* std::cout<<memory<<" ]\n"; */
    std::cout<<"[ "<<op_cost<<' '<<k<<' '<<operation<< ' '<<memory<<" ]\n";
  }
};

#endif
