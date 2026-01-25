#include <vector>
#include <fstream>

#include "jacobian.hpp"

enum class Operation: std::uint8_t{
  MULTIPLICATION = 0,
  TANGENT,
  ADJOINT,
};

class Solver{
 public:
  virtual void print_elemental_jacobians() = 0;
  virtual void build_elemental_jacobians(char* input_file) = 0;
  virtual void print_DP_table() = 0;
  virtual void build_DP_table() = 0;
  virtual void print_sequence() = 0;
 protected:
  std::vector<Jacobian> elemental_jacobians;
  std::size_t chain_length;
};

class DJCPB: public Solver{
 public:
  DJCPB(char* input_file, Jacobian* Jac_type){
    
  }
  void build_elemental_jacobians(char* input_file) override{
    std::ifstream in(input_file);
    in >> chain_length;
    elemental_jacobians.resize(chain_length);

    std::size_t i = 0;
    while(i<chain_length){
      in>> elemental_jacobians[i].m >> elemental_jacobians[i].n;
      i++;
    }
  }
 private:
  struct DP_Node{
    std::size_t optimal_cost;
    //Split position
    std::size_t k;
    //Type of Operation
    Operation op{};

    std::size_t memory;
  };
};
