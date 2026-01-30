#include <iostream>
#include <cstdint>

enum class Operation: std::uint8_t{
  MULTIPLICATION = 0,
  TANGENT,
  ADJOINT,
  NONE,
};

std::ostream& operator<<(std::ostream& os, Operation op){
  switch(op){
    case Operation::MULTIPLICATION:
      return(os<<"MUL");
    case Operation::TANGENT:
      return(os<<"TAN");
    case Operation::ADJOINT:
      return(os<<"ADJ");
    case Operation::NONE:
      return(os<<"NONE");
    default:
      return(os<<"NOT DEFINED");
  }
}

class table_cell{
 public:
  table_cell(std::size_t op_cost, std::size_t k_, Operation op_):
    optimal_cost(op_cost), k(k_), operation(op_){}

  std::size_t optimal_cost;
  //split position
  std::size_t k;
  //Operation type
  Operation operation= Operation::NONE;

  virtual void print()=0;
};

class cell_DJCPB: public table_cell{
 public:
  cell_DJCPB(std::size_t op_cost, std::size_t k_, Operation op_):
    table_cell(op_cost, k_, op_){};
  
  void print() override{
    std::cout<<'['<<optimal_cost<<' '<<k<<' '<<operation<<"]\n";
  }
};

class cell_MFDJCPB: public table_cell{
 public:
  cell_MFDJCPB(std::size_t op_cost, std::size_t k_, Operation op_,
      std::size_t memory_): table_cell(op_cost, k_, op_), memory(memory_){}

  std::size_t memory;

  void print() override{
    std::cout<<'['<<optimal_cost<<' '<<k<<' '<<operation<< ' '<<memory<<"]\n";
  }
};
