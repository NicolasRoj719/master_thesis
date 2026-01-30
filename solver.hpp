#include <vector>
#include <fstream>
#include <cassert>
#include <memory>
#include <iostream>

#include "jacobian.hpp"

enum class Operation: std::uint8_t{
  MULTIPLICATION = 0,
  TANGENT,
  ADJOINT,
};

// Simple formating of Operation enum class
std::ostream& operator<<(std::ostream& os, Operation op){
  switch(op){
    case Operation::MULTIPLICATION:
      return(os<<"MUL");
    case Operation::TANGENT:
      return(os<<"TAN");
    case Operation::ADJOINT:
      return(os<<"ADJ");
  }
}

//Each entry of the dynamic programming table is a cell.
//Below are different types of cells implemented.
struct DP_cell{
  std::size_t optimal_cost;
  //Split position
  std::size_t k;
  //Type of Operation
  Operation operation{};
  //Print cell info
  void print(){
    std::cout<<optimal_cost<<' '<<k<<' '<< operation;
  }
};

struct DP_cell_MF: public DP_cell{
  std::size_t memory;
};

  struct DP_cell_Sparse: public DP_cell_MF{

};

// Constant expression that maps from a triangular array to a
// linear array.
static constexpr std::size_t index(std::size_t j, std::size_t i){
  assert(i<=j);
  return((j + 1) * j / 2 + (j - i +1));
}


template <typename Cell_type>
class table{
  std::vector<Cell_type> table;
  table(std::size_t data_size){table.resize(data_size);}

  Cell_type& operator()(std::size_t j, std::size_t i){
    return(table[index(j,i)]);
  }

  void print(std::size_t len_chain){
    std::cout<<"Printing the dynamic programming table:\n";
    std::size_t index=0;
    std::size_t j, s, i;
    for(j=0; j<len_chain;j++){
      for(s=0; s<=j; s++){
        i = j-s;
        switch(Cell_type){
          case DP_cell:
            std::cout<<"F_(j,i) [optimal_cost split_position operation]";
            std::cout"F'_("<<j<<','<<i<<") ["<< table[index].optimal_cost<< ' ' 
              << table[index].k <<' '<< table[index].operation<<"]\n";
          case DP_cell_MF:
            std::cout<<"F_(j,i) [optimal_cost split_position operation memory]";
              std::cout"F'_("<<j<<','<<i<<") ["<< table[index].optimal_cost<< ' ' 
              << table[index].k <<' '<< table[index].operation<<' '<<table[index].memory<<"]\n";
          case DP_cell_Sparse:
            //TODO
        }
        index++;
      }
    }
  }
};

template <typename Jacobian_type>
class Solver{
 public:
  virtual void print_elemental_jacs() = 0;
  virtual void build_elemental_jacs(char* input_file) = 0;
  virtual void print_DP_table() = 0;
  virtual void build_DP_table() = 0;
  virtual void print_sequence() = 0;
 protected:
  std::vector<Jacobian_type> elemental_jacs;
  std::size_t chain_length;
};

class DJCPB: public Solver<Jacobian_type>{
 public:
  DJCPB(char* input_file): {
    build_elemental_jacs(input_file);
    print_elemental_jacs();
    build_DP_table();
    print_DP_table();
  }
  
  void build_elemental_jacs(char* input_file) override{
    std::ifstream in(input_file);
    in >> chain_length;
    elemental_jacs.resize(chain_length);

    std::size_t i = 0;
    while(i<chain_length){
      auto& F_i = elemental_jacs[i]
      in>> F_i.m >> F_i.n >> F_i.n_E;
      i++;
    }
  }

  void print_elemental_jacs(){
    std::size_t i=0;
    std::cout<<"F_i: [n m n_E]\n";
    for(Jacobian_type jac: elemental_jacs){
      std::cout<<"F'_"<<i<<": ["<< jac.n <<' '
        << jac.m << jac.n_E <<"]\n";
      i++;
    }
  }

  void build_DP_table(){
    std::size_t j,i,s,k;
    std::size_t cost;
    for(j=0; j<chain_length; j++){
      for(s=0; s<=j;s++){
        // substitution
        i = j-s;
        auto& F_i = elemental_jacs[i];
        auto& F_j = elemental_jacs[j];

        if(i==j){
          if(F_i.n > F_i.m){
            dptable(i,j).optimal_cost = (F_i.m) * (F_i.n_E);
            dptable(i,j).operation = Operation::ADJOINT;
          }

          else(){
            dptable(i,j).optimal_cost = (F_i.n) * (F_i.n_E);
            dptable(i,j).operation = Operation::TANGENT;
          }

          dptable(i,j).split_position = i;
        }

        else(){
          for(k=i; k<j; k++){
            cost = dptable(j,k+1).optimal_cost + dptable(k,i).optimal_cost + (F_j.m)*(F_i.m)*(F_i.n);

            if(k=i || cost < dptable(j,i).optimal_cost){
              dptable(j,i).optimal_cost = cost;
              dptable(j,i).split_position = k;
              dptable(j,i).operation = Operation::MULTIPLICATION;
            }    
          }
        }
      }
    }
  }

  void print_DP_table() override {dptable.print();}

 private:
  table<DP_cell> dptable((chain_length+1)*chain_length/2);
};

class MFDJCPB: public Solver<Jacobian_type>{
 public:
  MFDJCPB(char* input_file, std::size_t memory_limit): mem_limit(memory_limit){
    build_elemental_jacs(input_file);
    print_elemental_jacs();
    build_DP_table();
  };

  void build_elemental_jacs(char* input_file) override{
    std::ifstream in(input_file);
    in >> chain_length;
    elemental_jacs.resize(chain_length);

    std::size_t i = 0;
    while(i<chain_length){
      auto& F_i = elemental_jacs[i];
      in>> F_i.m >> F_i.n >> F_i.n_E;
      i++;
    }
  }

  void print_elemental_jacs() override{
    std::size_t i;
    std::cout<<"F_i: [n m n_E]\n";
    for(Jacobian_type jac: elemental_jacs){
      std::cout<<"F'_"<<i<<": ["<< jac.n <<' '
        << jac.m <<' '<< jac_ptr.n_E <<"]\n";
    }
  }

  void build_DP_table() override{
    // indices
    std::size_t j,i,s,k,kk;
    std::size_t cost;
    // Variable to store the namber of edges of the subchains
    std::size_t acc_n_E = 0;
    for(j=0; j<chain_length; j++){
      for(s=0; s<=j;s++){
        // substitution
        i = j-s;

        auto& F_i = elemental_jacs[i];
        auto& F_j = elemental_jacs[j];

        if(i==j){

          if(F_i.n > F_i.m && memory_limit > F_i.n_E){
            dptable(i,j).optimal_cost = (F_i.m) * (F_i.n_E);
            dptable(i,j).memory = F_i.n_E;
            dptable(i,j).operation = Operation::ADJOINT;
          }

          else(){
            dptable(i,j).optimal_cost = (F_i.n) * (F_i.n_E);
            dptable(i,j).memory = 0;
            dptable(i,j).operation = Operation::TANGENT;
          }
          dptable(i,j).split_position = i;
        }

        else(){
          for(k=i; k<j; k++){

            // Matrix matrix MULTIPLICATION
            cost = dptable(j,k+1).optimal_cost + dptable(k,i).optimal_cost + (F_j.m)*(F_i.m)*(F_i.n);

            if(k=i || dptable(j,i).optimal_cost > cost){
              dptable(j,i).optimal_cost = cost;
              dptable(j,i).split_position = k;
              dptable(j,i).operation = Operation::MULTIPLICATION;
            }    

            // Elimination TANGENT
            for(kk=k+1; kk<=j; kk++){acc_n_E += elemental_jacs[kk].n_E;}

            cost = dptable(k,i).optimal_cost + F_i.n * acc_n_E;

            if(dptable(j,i).optimal_cost > cost){
              dptable(j,i).optimal_cost = cost;
              dptable(j,i).split_position = k;
              dptable(j,i).operation = Operation::TANGENT;
              dptable(j,i).memory = dptable(k,i).memory_limit;
            }
            
            // Elimination ADJOINT
            for(kk=i; kk<=k; kk++){acc_n_E += elemental_jacs[kk]->n_E;}

            cost = dptable(j,k+1).optimal_cost + F_j.m * acc_n_E;

            if(cost < dptable(j,i).optimal_cost && memory_limit < acc_n_E + dptable(j,k+1).memory){
              dptable(j,i).optimal_cost = cost;
              dptable(j,i).split_position = k;
              dptable(j,i).operation = Operation::TANGENT;
              dptable(j,i).memory = acc_n_E + dptable(j,k+1).memory;
            }

          }
        }
      }
    }
  }

  void print_DP_table() override{dptable.print()};

  void change_mem_limit(std::size_t mem_limit_){
    mem_limit = mem_limit_;
  }
 private:
  std::size_t mem_limit;
  table<DP_cell_MF> dptable((chain_length+1)*chain_length/2);
};
