#include <cassert>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>
#include "./chain.hpp"

#ifndef BINOMIAL_CHECKPOINTING
#define BINOMIAL_CHECKPOINTING
class binomial_cell{
 public:
  binomial_cell(std::size_t additional_cost, std::size_t number_available_checkpoints,
                  std::optional<std::size_t> split_position = std::nullopt):
    additional_cost_(additional_cost), available_checkpoints_(number_available_checkpoints),
    split_position_(split_position){}

  std::size_t additional_cost() const{return additional_cost_;}
  std::size_t number_checkpoints() const{return available_checkpoints_;}
  std::optional<std::size_t> split_position() const{return split_position_;}

 private:
  std::size_t additional_cost_, available_checkpoints_;
  std::optional<std::size_t> split_position_;
};

class binomial_table{
 public:
  //Problem instance t(j,i,c)
  binomial_table(std::size_t j, std::size_t i, std::size_t number_checkpoints):
    i_index(i), available_checkpoints(number_checkpoints)
  {
    cells_per_floor = ((j - i + 2) * (j - i + 1)) / 2;

    table.reserve((number_checkpoints + 1) * cells_per_floor);
  }

  //Testing purposes
  binomial_table(){}

  //c stands for number of available_checkpoints.
  const binomial_cell& get_cell(std::size_t j, std::size_t i, std::size_t c) const noexcept{

    std::size_t accumulation = ((j - i_index + 1) * (j - i_index)) / 2;
    std::size_t position = j - i;

    return table[c * cells_per_floor + accumulation + position];
  } 

  void emplace_back(std::size_t additional_cost, std::size_t available_checkpoints,
                      std::size_t split_position){

    table.emplace_back(additional_cost, available_checkpoints, split_position);
  }

  void emplace_back(std::size_t additional_cost, std::size_t available_checkpoints){

    table.emplace_back(additional_cost, available_checkpoints);
  }

  std::size_t size() const{
    
    return table.size();
  }

 private:
  std::size_t i_index, available_checkpoints;
  std::size_t cells_per_floor;
  std::vector<binomial_cell> table;
};

template<class Split_type, class Split_information_type>
class binomial_checkpointing{
 public:
  //Problem instance t(j,i,c)
  binomial_checkpointing(const jacobian_chain<Split_type, Split_information_type>& chain_,
                          std::size_t j, std::size_t i, std::size_t checkpoints):
    chain(chain_), table(j, i, checkpoints), j_index(j), i_index(i),
    number_checkpoints(checkpoints){

      fill_table(j, i, number_checkpoints);
  }

  //Constructor designed to test the methods additional_cost and advancing_cost. 
  binomial_checkpointing(const jacobian_chain<Split_type, Split_information_type>& chain_,
                          std::size_t j, std::size_t i):
    chain(chain_), table(), j_index(j), i_index(i), number_checkpoints(0){}

  //Constructor designed to test additional_cost
  binomial_checkpointing(const jacobian_chain<Split_type, Split_information_type>& chain_,
                          binomial_table table_, std::size_t j, std::size_t i,
                          std::size_t checkpoints):
    chain(chain_), table(std::move(table_)), j_index(j), i_index(i),
    number_checkpoints(checkpoints){}

  //Implementations:
  std::size_t additional_cost(std::size_t j, std::size_t i){

    std::size_t cost = 0;

    for(std::size_t idx = 1; idx < (j-i+1); idx++){

      cost += idx * chain[j - idx].function_cost();  
    }

    return cost;
  }

  std::size_t advancing_cost(std::size_t split_position, std::size_t i){

    std::size_t cost = 0;
    
    for(std::size_t idx = i; idx < split_position + 1; idx++){
      
      cost += chain[idx].function_cost();
    }

    return cost;
  }

  const binomial_cell& get_cell(std::size_t j, std::size_t i, std::size_t c) const{

    return table.get_cell(j, i, c);
  }
  

  //Access rights public just for debugging purposes. Refrain from calling this
  // function.
  std::size_t additional_cost(std::size_t j, std::size_t split_position, std::size_t i,
                              std::size_t available_checkpoints){
    return table.get_cell(split_position, i, available_checkpoints).additional_cost() +
           table.get_cell(j, split_position + 1, available_checkpoints - 1).additional_cost() +
           advancing_cost(split_position, i);
  }

 private:
  const jacobian_chain<Split_type, Split_information_type>& chain;
  binomial_table table;
  //Problem_instance
  std::size_t j_index, i_index, number_checkpoints;

  void fill_table(std::size_t j_index, std::size_t i_index, std::size_t number_checkpoints);

};

template<class Split_type, class Split_information_type>
void binomial_checkpointing<Split_type, Split_information_type>::fill_table(
    std::size_t j_index, std::size_t i_index, std::size_t number_checkpoints){

  std::size_t k_min_j_k_i;
  std::size_t cost_min_j_k_i;
  std::size_t cost_j_k_i;
  std::size_t i;

  //Case available_checkpoints are equal to zero.
  for(std::size_t j = i_index; j < j_index + 1; j++){
    
    for(std::size_t aux_var = 0; aux_var < j - i_index + 1; aux_var++){
      
      i = j - aux_var;
      //No checkpoints available to split the problem.
      table.emplace_back(additional_cost(j, i), 0);
    }
  }

  for(std::size_t checkpoints = 1; checkpoints < number_checkpoints + 1; checkpoints++){

    //Emplacing back problem instance (i,i,c)
    table.emplace_back(additional_cost(i_index, i_index), checkpoints);
    for(std::size_t j= i_index + 1; j < j_index + 1; j++){

      table.emplace_back(additional_cost(j,j), checkpoints);
      for(std::size_t aux_var = 1; aux_var < j - i_index + 1; aux_var++){

        i = j - aux_var;
        cost_min_j_k_i = std::numeric_limits<std::size_t>::max();
        for(std::size_t split_position = i; split_position < j; split_position++){

          cost_j_k_i = additional_cost(j, split_position, i, checkpoints);

          if(cost_j_k_i < cost_min_j_k_i){
            cost_min_j_k_i = cost_j_k_i;
            k_min_j_k_i = split_position;
          }
        }
        //Store the minimum as an entry in the table.
        table.emplace_back(cost_min_j_k_i, checkpoints, k_min_j_k_i);
      }
    }
  }
}
#endif
