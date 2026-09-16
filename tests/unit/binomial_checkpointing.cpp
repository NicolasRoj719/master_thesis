#include <string>

#include "class_test_binomial_checkpointing.hpp"
#include "binomial_checkpointing.hpp"

int main(){

  test_binomial test{};
                      
  {
    std::size_t additional_cost = 15, number_checkpoints = 3;
    std::size_t split_position = 2;

    if(test.binomial_cell_test(additional_cost, number_checkpoints) &&
        test.binomial_cell_test(additional_cost, number_checkpoints, split_position)){

      test.set_test_binomial_cell(true);
    }

    else{test.set_test_binomial_cell(false);}
  }

  //reversal_chain
  {
    //Creating a split_dense chain from data.
    std::vector<std::size_t> subprograms_cost=
      binomial_checkpointing::read_subprograms_cost_from_file(std::string(FIXTURE_DIR) + 
          "/binomial_test_case/functions_cost", 5);


    bool test_view_3_1;
    {
      std::size_t j = 3, i = 1;
      reversal_chain view_subchain_j_i{subprograms_cost, j, i};

      if(view_subchain_j_i.size() != j - i + 1){
        test_view_3_1 = false;
      }
      else{

        if(view_subchain_j_i[0] != 222 ||
            view_subchain_j_i[1] != 333 ||
            view_subchain_j_i[2] != 444){
          
          test_view_3_1 = false;
        }
        else{test_view_3_1 = true;}
      }
    }

    bool test_view_4_2;
    {
      std::size_t j = 4, i = 2;
      reversal_chain view_subchain_j_i{subprograms_cost, j, i};

      if(view_subchain_j_i.size() != j - i + 1){
        test_view_4_2 = false;
      }
      else{

        if(view_subchain_j_i[0] != 333 ||
            view_subchain_j_i[1] != 444 ||
            view_subchain_j_i[2] !=  555){
          
          test_view_4_2 = false;
        }
        else{test_view_4_2 = true;}
      }

    }

    bool test_view_2_0;
    {
      std::size_t j = 2, i = 0;
      reversal_chain view_subchain_j_i{subprograms_cost, j, i};

      if(view_subchain_j_i.size() != j - i + 1){
        test_view_2_0 = false;
      }
      else{

        if(view_subchain_j_i[0] != 111 ||
            view_subchain_j_i[1] != 222 ||
            view_subchain_j_i[2] !=  333){
          
          test_view_2_0 = false;
        }
        else{test_view_2_0 = true;}
      }
    }

    bool test_view_with_ptr;
    {
      std::size_t j = 2, i = 0;
      reversal_chain view_subchain_j_i{subprograms_cost, j, i, subprograms_cost[j+1]};

      if(view_subchain_j_i.size() != j - i + 2){
        test_view_with_ptr = false;
      }
      else{

        if(view_subchain_j_i[0] != 111 ||
            view_subchain_j_i[1] != 222 ||
            view_subchain_j_i[2] !=  333 ||
            view_subchain_j_i[3] != 444){
          
          test_view_with_ptr = false;
        }
        else{test_view_with_ptr = true;}
      }
    }

    if(test_view_3_1 && test_view_4_2 && test_view_2_0 && test_view_with_ptr){
      test.set_test_view_chain(true);
    }

    else{test.set_test_view_chain(false);}
  }

  //Testing emplace_back and get_cell
  {
    std::size_t chain_length = 2, checkpoints = 1;

    binomial_table table{chain_length, checkpoints};

    std::size_t additional_cost = 10;

    //Ground floor.
    std::size_t c = 0;
    //(0,0,0)
    table.emplace_back(additional_cost, c);
    //(1,1,0)
    table.emplace_back(additional_cost + 1, c);
    //(1,0,0)
    table.emplace_back(additional_cost + 2, c);

    //First floor
    c++;
    //(0,0,1)
    table.emplace_back(additional_cost + 3, c);
    //(1,1,1)
    table.emplace_back(additional_cost + 4, c);
    //(1,0,1)
    table.emplace_back(additional_cost + 5, c, 0);

    if(table.size() != 6){
      test.set_test_table_emplace_back(false);
      //If emplace_back method is not working correcting then testing get_cell 
      // method may result in Segmentation fault error. Therefore set_test_get_cell
      // is set to false.
      test.set_test_get_cell(false);
    }

    else{

      if(table.get_cell(0, 0, 0).additional_cost() != additional_cost ||
          table.get_cell(1, 1, 0).additional_cost() != additional_cost + 1 ||
          table.get_cell(1, 0, 0).additional_cost() != additional_cost + 2 ||
          table.get_cell(0, 0, 1).additional_cost() != additional_cost + 3 ||
          table.get_cell(1, 1, 1).additional_cost() != additional_cost + 4 ||
          table.get_cell(1, 0, 1).additional_cost() != additional_cost + 5){

        test.set_test_get_cell(false);
      }

      else{

        test.set_test_get_cell(true);

        if(*table.get_cell(1, 0, 1).split_position() != 0){

          test.set_test_table_emplace_back(false);
        }
        else{test.set_test_table_emplace_back(true);}
      }
    }
  }

  //Testing additional_cost and advancing_cost
  {
    //Initializing the jacobian chain
    std::vector<std::size_t> subprograms_cost = 
      binomial_checkpointing::read_subprograms_cost_from_file(std::string(FIXTURE_DIR) +
          "/binomial_test_case/functions_cost", 5);

    //Problem instance (4, 0, 0)
    binomial_checkpointing binomial_algorithm{subprograms_cost, 4, 0};

    if(binomial_algorithm.advancing_cost(1, 0) != 111 + 222||
        binomial_algorithm.advancing_cost(3, 2) != 333 + 444||
        binomial_algorithm.advancing_cost(2, 2) != 333 ||
        binomial_algorithm.advancing_cost(3, 1) != 222 + 333 + 444){

      test.set_test_advancing_cost(false);
    }
    else{test.set_test_advancing_cost(true);}

    //For checkpoints = 0 or while filling up the ground floor.
    if(binomial_algorithm.additional_cost(0,0) != 0 ||
        binomial_algorithm.additional_cost(3,3) != 0 ||
        binomial_algorithm.additional_cost(1,0) != 111 ||
        binomial_algorithm.additional_cost(4,3) != 444 ||
        binomial_algorithm.additional_cost(2,0) != 111 + 222 + 111 ||
        binomial_algorithm.additional_cost(4,2) != 333 + 444 + 333 ||
        binomial_algorithm.additional_cost(3,0) != 111 + 222 + 333 + 111 + 222 + 111 ||
        binomial_algorithm.additional_cost(4,1) != 222 + 333 + 444 + 222 + 333 + 222){

      test.set_test_additional_cost_without_table(false);
    }
    else{test.set_test_additional_cost_without_table(true);}
  }

  //Testing additional cost
  {
    //Using the data from "./binomial_test_case/matrix_free_data" and
    //"./binomial_test_case/functions_cost" the subchain(3,0) will be used
    //to create the table manually and check compare the additional cost with 
    // the value returned by the functios additional_cost
    binomial_table table{3, 1};

    //Ground floor available_checkpoints = 0
    //(0,0,0)
    table.emplace_back(0, 0);
    //(1,1,0)
    table.emplace_back(0, 0);
    //(1,0,0)
    table.emplace_back(111, 0);
    //(2,2,0)
    table.emplace_back(0, 0);
    //(2,1,0)
    table.emplace_back(222, 0);
    //(2,0,0)
    table.emplace_back(111 + 222 + 111, 0);

    //First floor available_checkpoints = 1
    //(0,0,1)
    table.emplace_back(0, 1);
    //(1,1,1)
    table.emplace_back(0, 1);
    //(1,0,1)
    table.emplace_back(111, 1, 0);
    //(2,2,1)
    table.emplace_back(0, 1);
    //(2,1,1)
    table.emplace_back(222, 1, 1);
    //(2,0,1)
    //Two possible split positions at 0 or at 1.
    //Splitting at 0
    //additional_cost = advancing_cost(0,0) + table.get_cell(0,0,1).additional_cost() +
    //                  + table.get_cell(2,1,0)
    //                = 111 + 0 + 222 = 333
    //Splitting at 1
    //additional_cost = advancing_cost(1,0) + table.get_cell(1,0,1).additional_cost() +
    //                  + table.get_cell(2,2,0)
    //                = (111 + 222) + 111 + 0 = 444

    //Initializing the jacobian chain
    std::vector<std::size_t> subprograms_cost = 
      binomial_checkpointing::read_subprograms_cost_from_file(std::string(FIXTURE_DIR) + 
          "/binomial_test_case/functions_cost");
    
    //Transfering data ownership to table inside algorithm
    //Probelm instance (2,0,1)
    binomial_checkpointing algorithm{subprograms_cost, std::move(table), 2, 0};

    if(algorithm.additional_cost(1, 0, 0, 1) != 111 ||
        algorithm.additional_cost(2, 1, 1, 1) != 222 ||
        algorithm.additional_cost(2, 0, 0, 1) != 333 ||
        algorithm.additional_cost(2, 1, 0, 1) != 444){
    
      test.set_test_additional_cost_with_table(false);
    }
    else{test.set_test_additional_cost_with_table(true);}
  }

  //Test full binomial checkpoiting algorithm
  {
    //Initializing the jacobian chain
    std::vector<std::size_t> subprograms_cost = 
      binomial_checkpointing::read_subprograms_cost_from_file(std::string(FIXTURE_DIR) 
          + "/binomial_test_case/functions_cost");

    bool test_algorithm_2_0_1 = false;
    bool test_algorithm_4_2_1 = false;
    bool test_algorithm_3_0_1 = false;

    //Problem instance (2,0,1)
    {
      //Step by step calculation:
      //Ground floor c = 0. With c available_checkpoints
      //emplace_back(additional_cost, available_checkpoints)
      //Table entry: (0, 0, 0)
      //table.emplace_back(0, 0);
      //Table entry: (1, 1, 0)
      //table.emplace_back(0, 0)
      //Table entry: (1, 0, 0)
      //table.emplace_back(111, 0)
      //Table entry: (2, 2, 0)
      //table.emplace_back(0, 0)
      //Table entry: (2, 1, 0)
      //table.emplace_back(222, 0)
      //Table entry: (2, 0, 0)
      //table.emplace_back(111 + 222 + 111, 0)
      //
      //Floow c = 1
      //Table entry: (0, 0, 1)
      //table.emplace_back(0, 1)
      //Table entry: (1, 1, 1)
      //table.emplace_back(0, 1)
      //Table entry: (1, 0, 1)
      //emplace_back(additional_cost, available_checkpoints, split_position)
      //table.emplace_back(111, 1, 0)
      //Table entry: (2, 2, 1)
      //table.emplace_back(0, 1)
      //Table entry: (2, 1, 1)
      //table.emplace_back(222, 1, 1)
      //Table entry: (2, 0, 1)
      //Split at 0:
      //additional_cost = 111 + (2, 1, 0).additional_cost() + (0, 0, 1).additional_cost()
      //                = 111 + 222 + 0 
      //                = 333
      //Split at 1:
      //additional_cost = 111 + 222 + (2, 2, 0).additional_cost() + (1, 0, 1).additional_cost()
      //                = 111 + 222 + 0 + 111
      //                = 444
      //table.emplace_back(333, 1, 0)
      
      //Run the optimization algorithm with the constructor.
      binomial_checkpointing binomial_algorithm{subprograms_cost, 2, 0, 1};

      if(test.check_cell(binomial_algorithm.get_cell(0, 0, 0), 0, 0) &&
          test.check_cell(binomial_algorithm.get_cell(1, 1, 0), 0, 0) &&
          test.check_cell(binomial_algorithm.get_cell(1, 0, 0), 111, 0) &&
          test.check_cell(binomial_algorithm.get_cell(2, 2, 0), 0, 0) &&
          test.check_cell(binomial_algorithm.get_cell(2, 1, 0), 222, 0) &&
          test.check_cell(binomial_algorithm.get_cell(2, 0, 0), 111 + 222 + 111, 0) &&
          test.check_cell(binomial_algorithm.get_cell(0, 0, 1), 0, 1) &&
          test.check_cell(binomial_algorithm.get_cell(1, 1, 1), 0, 1) &&
          test.check_cell(binomial_algorithm.get_cell(1, 0, 1), 111, 1, 0) &&
          test.check_cell(binomial_algorithm.get_cell(2, 2, 1), 0, 1) &&
          test.check_cell(binomial_algorithm.get_cell(2, 1, 1), 222, 1, 1) &&
          test.check_cell(binomial_algorithm.get_cell(2, 0, 1), 333, 1, 0)){
        
        test_algorithm_2_0_1 = true;
      }
      else{test_algorithm_2_0_1 = false;}
    }

    //Problem instance (4, 2, 1) chain point of view.
    //Problem instance (2, 0, 1)
    {
      //Ground floor c = 0
      //Table entry: (0, 0, 0)
      //emplace_back(0, 0)
      //Table entry: (1, 1, 0)
      //emplace_back(0, 0)
      //Table entry: (1, 0, 0)
      //emplace_back(333, 0)
      //Table entry: (2, 2, 0)
      //emplace_back(0, 0)
      //Table entry: (2, 1, 0)
      //emplace_back(444, 0)
      //Table entry: (2, 0, 0)
      //emplace_back(333 + 444 + 333, 0)
      //
      //First floor c = 1
      //Table entry: (0, 0, 1)
      //emplace_back(0, 1)
      //Table entry: (1, 1, 1)
      //emplace_back(0, 1)
      //Table entry: (1, 0, 1)
      //emplace_back(333, 1, 0)
      //Table entry: (2, 2, 1)
      //emplace_back(0, 1)
      //Table entry: (2, 1, 1)
      //emplace_back(444, 1, 1)
      //Table entry: (2, 0, 1)
      //Split at 2: 
      //additional_cost = 333 + (2, 1, 0).additional_cost() + (0, 0, 1).additional_cost()
      //                = 333 + 444 + 0
      //                = 777
      //Split at 3:
      //additional_cost = 333 + 444 + (2, 2, 0).additional_cost() + (1, 0, 1).additional_cost()
      //                = 777 + 0 + 333 = 1110
      //emplace_back(777, 1, 0)
      //Run the optimization algorithm with the constructor.

      binomial_checkpointing binomial_algorithm{subprograms_cost, 4, 2, 1};

      if(test.check_cell(binomial_algorithm.get_cell(0, 0, 0), 0, 0) &&
          test.check_cell(binomial_algorithm.get_cell(1, 1, 0), 0, 0) &&
          test.check_cell(binomial_algorithm.get_cell(1, 0, 0), 333, 0) &&
          test.check_cell(binomial_algorithm.get_cell(2, 2, 0), 0, 0) &&
          test.check_cell(binomial_algorithm.get_cell(2, 1, 0), 444, 0) &&
          test.check_cell(binomial_algorithm.get_cell(2, 0, 0), 1110, 0) &&
          test.check_cell(binomial_algorithm.get_cell(0, 0, 1), 0, 1) &&
          test.check_cell(binomial_algorithm.get_cell(1, 1, 1), 0, 1) &&
          test.check_cell(binomial_algorithm.get_cell(1, 0, 1), 333, 1, 0) &&
          test.check_cell(binomial_algorithm.get_cell(2, 2, 1), 0, 1) &&
          test.check_cell(binomial_algorithm.get_cell(2, 1, 1), 444, 1, 1) &&
          test.check_cell(binomial_algorithm.get_cell(2, 0, 1), 777, 1, 0)){

        test_algorithm_4_2_1 = true;
      }
      else{test_algorithm_4_2_1 = false;}
    }

    //Problem instance (3,0,1)
    {
      //Ground floor c = 0.
      //Table entry: (0, 0, 0)
      //emplace_back(0, 0)
      //Table entry: (1, 1, 0)
      //emplace_back(0, 0)
      //Table entry: (1, 0, 0)
      //emplace_back(111, 0)
      //Table_entry: (2, 2, 0)
      //emplace_back(0, 0)
      //Table entry: (2, 1, 0)
      //emplace_back(222, 0)
      //Table entry (2, 0, 0)
      //emplace_back(111 + 222 + 111, 0)
      //Table entry (3, 3, 0)
      //emplace_back(0, 0)
      //Table entry (3, 2, 0)
      //emplace_back(333, 0)
      //Table entry (3, 1, 0)
      //emplace_back(222 + 333 + 222, 0)
      //Table entry (3, 0, 0)
      //emplace_back(111 + 222 + 333 + 111 + 222 + 111, 0)

      //First floor c = 1
      //Table entry (0, 0, 1)
      //emplace_back(0, 1)
      //Table entry (1, 1, 1)
      //emplace_back(0, 1)
      //Table entry (1, 0, 1)
      //emplace_back(111, 1, 0)
      //Table entry (2, 2, 1)
      //emplace_back(0, 1)
      //Table entry (2, 1, 1)
      //emplace_back(222, 1, 1)
      //Table entry (2, 0, 1)
      //emplace_back(333, 1, 0)
      //Table entry (3, 3, 1)
      //emplace_back(0, 1)
      //Table entry (3, 2, 1)
      //emplace_back(333, 1, 2)
      //Table entry (3, 1, 1)
      //Split at 1:
      //additional_cost = 222 + (3, 2, 0).additional_cost() + (1, 1, 1).additional_cost()
      //                = 222 + 333
      //                = 555
      //Split at 2:
      //additional_cost = 222+333 + (3, 3, 0).additional_cost()+ (2, 1, 1).additional_cost()
      //                = 555 + 0 + 222
      //emplace_back(555, 1, 1)
      //
      //Table entry (3, 0, 1)
      //Split at 0:
      //additional_cost = 111 + (3, 1, 0).additional_cost() + (0, 0, 1).additional_cost()
      //                = 111 + 777 + 0
      //                = 888
      //Split at 1:
      //additional_cost = 111 + 222 + (3, 2, 0).additional_cost() + (1, 0, 1).additional_cost()
      //                = 333 + 333 + 111
      //                = 777
      //
      //Split at 2:
      //additional_cost = 111+222+333 + (3, 3, 0).additional_cost() + (2, 0, 1).additional_cost()
      //                = 666 + 0 + 333
      //                = 999
      //emplace_back(777, 1, 1)
      
      //Run the optimization algorithm with the constructor.
      binomial_checkpointing binomial_algorithm{subprograms_cost, 3, 0, 1};

      if(test.check_cell(binomial_algorithm.get_cell(0, 0, 0), 0, 0) &&
          test.check_cell(binomial_algorithm.get_cell(1, 1, 0), 0, 0) &&
          test.check_cell(binomial_algorithm.get_cell(1, 0, 0), 111, 0) &&
          test.check_cell(binomial_algorithm.get_cell(2, 2, 0), 0, 0) &&
          test.check_cell(binomial_algorithm.get_cell(2, 1, 0), 222, 0) &&
          test.check_cell(binomial_algorithm.get_cell(2, 0, 0), 444, 0) &&
          test.check_cell(binomial_algorithm.get_cell(3, 3, 0), 0, 0) &&
          test.check_cell(binomial_algorithm.get_cell(3, 2, 0), 333, 0) &&
          test.check_cell(binomial_algorithm.get_cell(3, 1, 0), 777, 0) &&
          test.check_cell(binomial_algorithm.get_cell(3, 0, 0), 1110, 0) &&
          test.check_cell(binomial_algorithm.get_cell(0, 0, 1), 0, 1) &&
          test.check_cell(binomial_algorithm.get_cell(1, 1, 1), 0, 1) &&
          test.check_cell(binomial_algorithm.get_cell(1, 0, 1), 111, 1) &&
          test.check_cell(binomial_algorithm.get_cell(2, 2, 1), 0, 1) &&
          test.check_cell(binomial_algorithm.get_cell(2, 1, 1), 222, 1, 1) &&
          test.check_cell(binomial_algorithm.get_cell(2, 0, 1), 333, 1, 0) &&
          test.check_cell(binomial_algorithm.get_cell(3, 3, 1), 0, 1) &&
          test.check_cell(binomial_algorithm.get_cell(3, 2, 1), 333, 1, 2) &&
          test.check_cell(binomial_algorithm.get_cell(3, 1, 1), 555, 1, 1) &&
          test.check_cell(binomial_algorithm.get_cell(3, 0, 1), 777, 1, 1)){

        test_algorithm_3_0_1 = true;
      }
      else{test_algorithm_3_0_1 = false;}
    }

    if(test_algorithm_2_0_1 && test_algorithm_4_2_1 && test_algorithm_3_0_1){
      test.set_test_algorithm(true);
    }
    else{test.set_test_algorithm(false);}
  }

  /* test.print_test_state(); */
}
