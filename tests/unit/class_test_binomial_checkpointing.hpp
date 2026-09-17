#include <iostream>
#include <optional>
#include <string>

#include "binomial_checkpointing.hpp"
#include "chain.hpp"

class test_binomial{
 public:
  test_binomial(){}

  void print_test_state() const{
    
    std::cout << "State of test binomial test: \n";

    std::cout << "test_binomial_cell: ";
    if(test_binomial_cell){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_view_chain: ";
    if(test_view_chain){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_table_emplace_back: ";
    if(test_table_emplace_back){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_get_cell: ";
    if(test_get_cell){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_additional_cost_without_table: ";
    if(test_additional_cost_without_table){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_advancing_cost: ";
    if(test_advancing_cost){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_additional_cost_with_table: ";
    if(test_additional_cost_with_table){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_algorithm: ";
    if(test_algorithm){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}
  }

  bool binomial_cell_test(std::size_t additional_cost, std::size_t number_checkpoints)const{

    Binomial_cell cell{additional_cost, number_checkpoints};

    if(cell.additional_cost() != additional_cost ||
        cell.number_checkpoints() != number_checkpoints){
      
      return false;
    }

    else{return true;}
  }

  bool binomial_cell_test(std::size_t additional_cost, std::size_t number_checkpoints,
                            std::size_t split_position){

    Binomial_cell cell{additional_cost, number_checkpoints, split_position};

    if(cell.additional_cost() != additional_cost ||
        cell.number_checkpoints() != number_checkpoints ||
        *cell.split_position() != split_position){

      return false;
    }

    else{return true;}
  }

  bool check_cell(const Binomial_cell& cell, std::size_t additional_cost,
                    std::size_t number_checkpoints) const{

    if(cell.additional_cost() != additional_cost ||
        cell.number_checkpoints() != number_checkpoints){

      return false;
    }
    
    return true;
  }

  bool check_cell(const Binomial_cell& cell, std::size_t additional_cost,
                  std::size_t number_checkpoints, std::size_t split_position) const{

    if(cell.additional_cost() != additional_cost ||
        cell.number_checkpoints() != number_checkpoints ||
        cell.split_position() != split_position){

      return false;
    }
    
    return true;
  }

  void set_test_binomial_cell(bool argument){
    test_binomial_cell = argument;
  }

  void set_test_view_chain(bool argument){
    test_view_chain = argument;
  }

  void set_test_additional_cost_without_table(bool argument){
    test_additional_cost_without_table = argument;
  }

  void set_test_advancing_cost(bool argument){
    test_advancing_cost = argument;
  }

  void set_test_additional_cost_with_table(bool argument){
    test_additional_cost_with_table = argument;
  }

  void set_test_table_emplace_back(bool argument){
    test_table_emplace_back = argument;
  }

  void set_test_get_cell(bool argument){
    test_get_cell = argument;
  }

  void set_test_algorithm(bool argument){
    test_algorithm = argument;
  }

 private:
  bool test_binomial_cell;
  bool test_view_chain;
  bool test_table_emplace_back;
  bool test_get_cell;
  bool test_additional_cost_without_table;
  bool test_advancing_cost;
  bool test_additional_cost_with_table;
  bool test_algorithm;
};
