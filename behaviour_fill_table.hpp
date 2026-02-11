#include <vector>
#include "table_cell.hpp"
#include "table.hpp"

#ifndef FILL_DPTABLE
#define FILL_DPTABLE

template<class Cell_type, class Jacobian_type>
class behaviour_fill_dptable{
 public:
  virtual ~behaviour_fill_dptable() = default;

  virtual void fill(Table<Cell_type>& table, std::vector<Jacobian_type>& elemental_jacs) = 0;
};

template<class Cell_type, class Jacobian_type>
class fill_DJCPB: public behaviour_fill_dptable<Cell_type, Jacobian_type>{
 public:
  void fill(Table<Cell_type>& table, std::vector<Jacobian_type>& elemental_jacs) override{
    
    std::size_t j,i,s,k;
    std::size_t cost;
    std::size_t chain_length = elemental_jacs.size();
    
    for(j=0; j<chain_length; j++){
      for(s=0; s<=j;s++){
        // substitution
        i = j-s;
        auto& F_i = elemental_jacs[i];
        auto& F_j = elemental_jacs[j];

        if(i==j){
          if(F_i.n() > F_i.m()){
            //Recall Cell_type{optimal_cost, split_position, Operation}
            table.emplace_back(Cell_type{F_i.m() * F_i.n_E(), i, Operation::ADJOINT});
            /* table(i,j).optimal_cost = (F_i.m) * (F_i.n_E); */
            /* table(i,j).operation = Operation::ADJOINT; */
          }

          else{
            table.emplace_back(Cell_type{F_i.n() * F_i.n_E(), i, Operation::TANGENT});
            /* table(i,j).optimal_cost = (F_i.n) * (F_i.n_E); */
            /* table(i,j).operation = Operation::TANGENT; */
          }

              /* table(i,j).split_position = i; */
        }

        else{
          Cell_type F_j_i{};
          for(k=i; k<j; k++){
            cost = table(j,k+1).op_cost + table(k,i).op_cost + F_j.m() * F_i.m() * F_i.n();
            
            if(k==i || F_j_i.op_cost > cost){
              //F_j_i(optimal_cost, split_position, operation)
              F_j_i(cost, k, Operation::MULTIPLICATION);
            }    
          }
          table.emplace_back(F_j_i);
        }
      }
    }
  }
};

template<class Cell_type, class Jacobian_type>
class fill_MFDJCPB: public behaviour_fill_dptable<Cell_type, Jacobian_type>{
 public:
  fill_MFDJCPB(std::size_t mem_limit): memory_limit(mem_limit){};

  void fill(Table<Cell_type>& table, std::vector<Jacobian_type>& elemental_jacs) override{
    // indices
    std::size_t j,i,s,k,kk;
    std::size_t cost;
    std::size_t chain_length = elemental_jacs.size();

    // Variable to store the namber of edges of the subchains
    std::size_t acc_n_E = 0;

    for(j=0; j<chain_length; j++){
      for(s=0; s<=j;s++){
        // substitution
        i = j-s;

        auto& F_i = elemental_jacs[i];
        auto& F_j = elemental_jacs[j];

        if(i==j){

          if(F_i.n() > F_i.m() && memory_limit > F_i.n_E()){
            //Cell_type{optimal_cost, split_position, operation, memory_requiered}
            table.emplace_back(Cell_type{F_i.m() * F_i.n_E(), i, Operation::ADJOINT, F_i.n_E()});
            /* table(i,j).optimal_cost = (F_i.m) * (F_i.n_E); */
            /* table(i,j).memory = F_i.n_E; */
            /* table(i,j).operation = Operation::ADJOINT; */
          }

          else{
            table.emplace_back(Cell_type{F_i.n() * F_i.n_E(), i, Operation::TANGENT, 0});
            /* table(i,j).optimal_cost = (F_i.n) * (F_i.n_E); */
            /* table(i,j).memory = 0; */
            /* table(i,j).operation = Operation::TANGENT; */
          }
          /* table(i,j).split_position = i; */
        }

        else{
          Cell_type F_j_i{};
          for(k=i; k<j; k++){

            // Matrix matrix MULTIPLICATION
            cost = table(j,k+1).op_cost + table(k,i).op_cost + F_j.m() * F_i.m() * F_i.n();

            if(k==i || F_j_i.op_cost > cost){
              //F_j_i(optimal_cost, split_position, operation, memory)
              F_j_i(cost, k, Operation::MULTIPLICATION, table(j,k+1).memory + table(k,i).memory);
              /* table(j,i).op_cost = cost; */
              /* table(j,i).split_position = k; */
              /* table(j,i).operation = Operation::MULTIPLICATION; */
            }    

            // Elimination TANGENT
            acc_n_E = 0;
            for(kk=k+1; kk<=j; kk++){acc_n_E += elemental_jacs[kk].n_E();}

            cost = table(k,i).op_cost + F_i.n() * acc_n_E;

            if(F_j_i.op_cost > cost){
              //F_j_i(optimal_cost, split_position, operation, memory)
              F_j_i(cost, k, Operation::TANGENT, table(k,i).memory);
              /* table(j,i).optimal_cost = cost; */
              /* table(j,i).split_position = k; */
              /* table(j,i).operation = Operation::TANGENT; */
              /* table(j,i).memory = table(k,i).memory_limit; */
            }
            
            // Elimination ADJOINT
            acc_n_E = 0;
            for(kk=i; kk<=k; kk++){acc_n_E += elemental_jacs[kk].n_E();}

            cost = table(j,k+1).op_cost + F_j.m() * acc_n_E;

            if(F_j_i.op_cost > cost && memory_limit > acc_n_E + table(j,k+1).memory){
              //F_j_i(optimal_cost, k, operation, memory)
              F_j_i(cost, k, Operation::ADJOINT, acc_n_E + table(j,k+1).memory);
              /* table(j,i).optimal_cost = cost; */
              /* table(j,i).split_position = k; */
              /* table(j,i).operation = Operation::TANGENT; */
              /* table(j,i).memory = acc_n_E + table(j,k+1).memory; */
            }
          }
          table.emplace_back(F_j_i);
        }
      }
    }
  }

  void change_memory_limit(std::size_t mem_limit){memory_limit = mem_limit;}

 private:
  size_t memory_limit;

};
#endif
