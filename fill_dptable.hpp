#include <vector>
#include "jacobian.hpp"
#include "table_cell.hpp"


template<class Cell_type, class Jacobian_type>
class behaviour_fill_dptable{
 public:
  virtual ~behaviour_fill_dptable() = default;

  virtual void fill(std::vector<Cell_type>& table, std::vector<Jacobian_type>& elemental_jacs) = 0;
  virtual void fill(std::vector<Cell_type>& table, std::vector<Jacobian_type>& elemental_jacs, std::size_t memory_limit) = 0;
};

template<class Cell_type, class Jacobian_type>
class fill_DJCPB: public behaviour_fill_dptable<Cell_type, Jacobian_type>{
 public:
  void fill(std::vector<Cell_type>& table, std::vector<Jacobian_type>& elemental_jacs) override{
    
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
          if(F_i.n > F_i.m){
            table(i,j).optimal_cost = (F_i.m) * (F_i.n_E);
            table(i,j).operation = Operation::ADJOINT;
          }

          else{
            table(i,j).optimal_cost = (F_i.n) * (F_i.n_E);
            table(i,j).operation = Operation::TANGENT;
          }

              table(i,j).split_position = i;
        }

        else{
          for(k=i; k<j; k++){
            cost = table(j,k+1).optimal_cost + table(k,i).optimal_cost + (F_j.m)*(F_i.m)*(F_i.n);

            if(k=i || cost < table(j,i).optimal_cost){
              table(j,i).optimal_cost = cost;
              table(j,i).split_position = k;
              table(j,i).operation = Operation::MULTIPLICATION;
            }    
          }
        }
      }
    }
  }
};

template<class Cell_type, class Jacobian_type>
class fill_MFDJCP: public behaviour_fill_dptable<Cell_type, Jacobian_type>{
 public:
  void fill(std::vector<Cell_type>& table, std::vector<Jacobian_type>& elemental_jacs, std::size_t memory_limit) override{
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

          if(F_i.n > F_i.m && memory_limit > F_i.n_E){
            table(i,j).optimal_cost = (F_i.m) * (F_i.n_E);
            table(i,j).memory = F_i.n_E;
            table(i,j).operation = Operation::ADJOINT;
          }

          else{
            table(i,j).optimal_cost = (F_i.n) * (F_i.n_E);
            table(i,j).memory = 0;
            table(i,j).operation = Operation::TANGENT;
          }
          table(i,j).split_position = i;
        }

        else{
          for(k=i; k<j; k++){

            // Matrix matrix MULTIPLICATION
            cost = table(j,k+1).optimal_cost + table(k,i).optimal_cost + (F_j.m)*(F_i.m)*(F_i.n);

            if(k=i || table(j,i).optimal_cost > cost){
              table(j,i).optimal_cost = cost;
              table(j,i).split_position = k;
              table(j,i).operation = Operation::MULTIPLICATION;
            }    

            // Elimination TANGENT
            for(kk=k+1; kk<=j; kk++){acc_n_E += elemental_jacs[kk].n_E;}

            cost = table(k,i).optimal_cost + F_i.n * acc_n_E;

            if(table(j,i).optimal_cost > cost){
              table(j,i).optimal_cost = cost;
              table(j,i).split_position = k;
              table(j,i).operation = Operation::TANGENT;
              table(j,i).memory = table(k,i).memory_limit;
            }
            
            // Elimination ADJOINT
            for(kk=i; kk<=k; kk++){acc_n_E += elemental_jacs[kk]->n_E;}

            cost = table(j,k+1).optimal_cost + F_j.m * acc_n_E;

            if(cost < table(j,i).optimal_cost && memory_limit < acc_n_E + table(j,k+1).memory){
              table(j,i).optimal_cost = cost;
              table(j,i).split_position = k;
              table(j,i).operation = Operation::TANGENT;
              table(j,i).memory = acc_n_E + table(j,k+1).memory;
            }

          }
        }
      }
    }
  } 
};
