#include <vector>

#include "class_test_optimal_accumulation.hpp"
#include "jacobian.hpp"
#include "table.hpp"
#include "table_cell.hpp"
#include "optimal_accumulation.hpp"

//The optimal accumulation method does not check
//correctness of the accumulated cost value it simply
//assumes that the table data is correct and checks
//for the optimal problem partition.

bool test_jacobian(Jacobian* jac_0, Jacobian* jac_1){

  Table<Jacobian> table{2};

  table.emplace_back(jac_0);

  table.emplace_back(jac_1);

  table.emplace_back(11, 0);

  serial_accumulation_sequence<Jacobian> serial_accumulation{table, 2};

  //Test problem 1_0
  if(serial_accumulation[0].get_j() != 1 ||
      *serial_accumulation[0].get_i() != 0 ||
      serial_accumulation[0].operation()){

    return false;
  }

  return true;
}


bool test_jacobian(Jacobian* jac_0, Jacobian* jac_1, Jacobian* jac_2){

  Table<Jacobian> table{3};

  table.emplace_back(jac_0);
  
  table.emplace_back(jac_1);

  table.emplace_back(jac_2);

  table.emplace_back(23, 0);

  table.emplace_back(33, 1);

  table.emplace_back(44, 1);

  serial_accumulation_sequence<Jacobian> serial_accumulation{table, 3};

  //Test problem 2_0
  if(serial_accumulation[0].get_j() != 2 ||
      *serial_accumulation[0].get_i() != 0 ||
      serial_accumulation[0].operation()){

    return false;
  }

  //Test problem 1_0
  if(serial_accumulation[1].get_j() != 1 ||
      serial_accumulation[1].get_i() != 0 ||
      serial_accumulation[1].operation()){

    return false;
  }

  /* serial_accumulation.parser_sequence_2_graphviz_format(); */
  return true;
}

bool test_tangent_dense(Dense_Jacobian* jac_0, Dense_Jacobian* jac_1){

  Table<Dense_Jacobian> table{2};

  table.emplace_back(jac_0, 10, Operation::TANGENT);

  table.emplace_back(jac_1, 20, Operation::ADJOINT);

  table.emplace_back( 30, 0, Operation::TANGENT);

  serial_accumulation_sequence<Dense_Jacobian> serial_accumulation{table, 2};

  //Test problem 1_0
  if(serial_accumulation[0].get_j() != 1 ||
      *serial_accumulation[0].get_i() != 0 ||
      *serial_accumulation[0].operation() != Operation::TANGENT){

    return false;
  }

  //Test Subproblem 0
  if(serial_accumulation[1].get_j() != 0 ||
      serial_accumulation[1].get_i() ||
      *serial_accumulation[1].operation() != Operation::TANGENT){

    return false;
  }

  /* serial_accumulation.parser_sequence_2_graphviz_format(); */

return true;
}

bool test_adjoint_dense(Dense_Jacobian* jac_0, Dense_Jacobian* jac_1){

  Table<Dense_Jacobian> table{2};

  table.emplace_back(jac_0, 15, Operation::TANGENT);

  table.emplace_back(jac_1, 25, Operation::ADJOINT);

  table.emplace_back(32, 0, Operation::ADJOINT);

  serial_accumulation_sequence<Dense_Jacobian> serial_accumulation{table, 2};

  //Tests problem 1_0
  if(serial_accumulation[0].get_j() != 1 ||
       *serial_accumulation[0].get_i() != 0 ||
       *serial_accumulation[0].operation() != Operation::ADJOINT){

    return false;
  }

  //Test subproblem 1
  if(serial_accumulation[1].get_j() != 1 ||
       serial_accumulation[1].get_i() ||
       *serial_accumulation[1].operation() != Operation::ADJOINT){
  
      return false;
  }

  /* serial_accumulation.parser_sequence_2_graphviz_format(); */
  return true;
}

bool test_multiplication_dense(Dense_Jacobian* jac_0, Dense_Jacobian* jac_1){

  Table<Dense_Jacobian> table{2};

  table.emplace_back(jac_0, 17, Operation::TANGENT);

  table.emplace_back(jac_1, 27, Operation::ADJOINT);

  table.emplace_back(40, 0, Operation::MULTIPLICATION);

  serial_accumulation_sequence<Dense_Jacobian> serial_accumulation{table, 2};

  //Test problem 1_0
  if(serial_accumulation[0].get_j() != 1 ||
      *serial_accumulation[0].get_i() != 0 ||
      *serial_accumulation[0].operation() != Operation::MULTIPLICATION){

    return false;
  }

  //Test subproblem 0
  if(serial_accumulation[1].get_j() != 0 ||
      serial_accumulation[1].get_i() ||
      *serial_accumulation[1].operation() != Operation::TANGENT){

    return false;
  }

  //Test subproblem 1
  if(serial_accumulation[2].get_j() != 1 ||
      serial_accumulation[2].get_i() ||
      *serial_accumulation[2].operation() != Operation::ADJOINT){

    return false;
  }

  /* serial_accumulation.parser_sequence_2_graphviz_format(); */
  return true;
}

bool test_dense_jacobian(Dense_Jacobian* jac_0,
    Dense_Jacobian* jac_1, Dense_Jacobian* jac_2){

  Table<Dense_Jacobian> table{3};

  table.emplace_back(jac_0, 21, Operation::TANGENT);

  table.emplace_back(jac_1, 31, Operation::ADJOINT);

  table.emplace_back(jac_2, 41, Operation::TANGENT);

  table.emplace_back(55, 0, Operation::ADJOINT);

  table.emplace_back(66, 1, Operation::TANGENT);

  table.emplace_back(100, 0, Operation::MULTIPLICATION);

  serial_accumulation_sequence<Dense_Jacobian> serial_accumulation{table, 3};

  //Test problem 2_0
  if(serial_accumulation[0].get_j() != 2 ||
      *serial_accumulation[0].get_i() != 0 ||
      *serial_accumulation[0].operation() != Operation::MULTIPLICATION){

    return false;
  }

  //Test subproblem 0
  if(serial_accumulation[1].get_j() != 0 ||
      serial_accumulation[1].get_i() ||
      *serial_accumulation[1].operation() != Operation::TANGENT){

    return false;
  }

  //Test subproblem 2_1
  if(serial_accumulation[2].get_j() != 2 ||
      *serial_accumulation[2].get_i() != 1 ||
      *serial_accumulation[2].operation() != Operation::TANGENT){

    return false;
  }

  //Test subproblem 1
  if(serial_accumulation[3].get_j() != 1 ||
      serial_accumulation[3].get_i() ||
      *serial_accumulation[3].operation() != Operation::ADJOINT){

    return false;
  }

  /* serial_accumulation.parser_sequence_2_graphviz_format(); */
  return true;
}

void graph_test_0(Jacobian* jac_0){

  const std::size_t chain_length = 4;
  Table<Jacobian> table{chain_length};

  for(std::size_t i= 0; i < chain_length; i++){
    table.emplace_back(jac_0);
  }

  //[1,0]
  table.emplace_back(11, 0);
  //[2,1] does not matter
  table.emplace_back(13, 1);
  //[2,0] does not matter
  table.emplace_back(15, 1);
  //[3,2]
  table.emplace_back(25,2);
  //[3,1] does not matter
  table.emplace_back(35,1);
  //[3,0]
  table.emplace_back(60,1);

  serial_accumulation_sequence<Jacobian> serial_accumulation{table, chain_length};

  serial_accumulation.parser_sequence_2_graphviz_format();
}

void graph_test_0(Dense_Jacobian* jac_0){  

  const std::size_t chain_length = 3;
  Table<Dense_Jacobian> table{chain_length};

  //[0]
  table.emplace_back(jac_0, 11, Operation::ADJOINT);
  //[1]
  table.emplace_back(jac_0, 15, Operation::ADJOINT);
  //[2]
  table.emplace_back(jac_0, 19, Operation::TANGENT);
  //[1,0]
  table.emplace_back(21, 0, Operation::MULTIPLICATION);
  //[2,1] does not matter
  table.emplace_back(31, 1, Operation::ADJOINT);
  //[2,0]
  table.emplace_back(40, 1, Operation::MULTIPLICATION);

  serial_accumulation_sequence<Dense_Jacobian> serial_accumulation{table, chain_length};
  
  serial_accumulation.parser_sequence_2_graphviz_format();
}

void graph_test_1(Jacobian* jac_0){

  const std::size_t chain_length = 5;
  Table<Jacobian> table{chain_length};

  for(std::size_t i= 0; i < chain_length; i++){
    table.emplace_back(jac_0);
  }

  //[1,0] does not matter.
  table.emplace_back(12, 0);
  //[2,1]
  table.emplace_back(15, 1);
  //[2,0]
  table.emplace_back(28, 0);
  //[3,2] does not matter
  table.emplace_back(23, 2);
  //[3,1] does not matter
  table.emplace_back(45, 1);
  //[3,0] does not matter
  table.emplace_back(65, 1);
  //[4,3]
  table.emplace_back(11, 3);
  //[4,2] does not matter
  table.emplace_back(31, 3);
  //[4,1] does not matter
  table.emplace_back(45,1);
  //[4,0]
  table.emplace_back(60,2);

  serial_accumulation_sequence<Jacobian> serial_accumulation{table, chain_length};

  serial_accumulation.parser_sequence_2_graphviz_format();
}

void graph_test_1(Dense_Jacobian* jac_0){

  const std::size_t chain_length = 3;
  Table<Dense_Jacobian> table{chain_length};

  //[0]
  table.emplace_back(jac_0, 21, Operation::TANGENT);
  //[1]
  table.emplace_back(jac_0, 17, Operation::ADJOINT);
  //[2]
  table.emplace_back(jac_0, 11, Operation::ADJOINT);
  //[1,0] does not matter. 
  table.emplace_back(30, 0, Operation::TANGENT);
  //[2,1]
  table.emplace_back(45, 1, Operation::MULTIPLICATION);
  //[2,0]
  table.emplace_back(60, 0, Operation::MULTIPLICATION);

  serial_accumulation_sequence<Dense_Jacobian> serial_accumulation{table, chain_length};

  serial_accumulation.parser_sequence_2_graphviz_format();
}

void graph_test_2(Jacobian* jac_0){

  const std::size_t chain_length = 4;
  Table<Jacobian> table{chain_length};

  for(std::size_t idx = 0; idx < chain_length; idx++){

    table.emplace_back(jac_0);
  }

  //[1,0]
  table.emplace_back(13, 0);
  //[2,1]
  table.emplace_back(16, 1);
  //[2,0]
  table.emplace_back(23, 1);
  //[3,2] does not matter
  table.emplace_back(28, 2);
  //[3,1] does not matter
  table.emplace_back(44, 1);
  //[3,0]
  table.emplace_back(56, 2);

  serial_accumulation_sequence<Jacobian> serial_accumulation{table, chain_length};

  serial_accumulation.parser_sequence_2_graphviz_format();
}

void graph_test_2(Dense_Jacobian* jac_0){

  const std::size_t chain_length = 4;
  Table<Dense_Jacobian> table{chain_length};

  //[0]
  table.emplace_back(jac_0, 15, Operation::ADJOINT);
  //[1] does not matter.
  table.emplace_back(jac_0, 20, Operation::TANGENT);
  //[2]
  table.emplace_back(jac_0, 22, Operation::TANGENT);
  //[3]
  table.emplace_back(jac_0, 25, Operation::ADJOINT);
  //[1,0]
  table.emplace_back(42, 0, Operation::MULTIPLICATION);
  //[2,1] does not matter.
  table.emplace_back(53, 1, Operation::TANGENT);
  //[2,0] does not matter
  table.emplace_back(100, 0, Operation::ADJOINT);
  //[3,2]
  table.emplace_back(64, 2, Operation::MULTIPLICATION);
  //[3,1] does not matter
  table.emplace_back(111, 1, Operation::TANGENT);
  //[3,0]
  table.emplace_back(173, 1, Operation::MULTIPLICATION);

  serial_accumulation_sequence<Dense_Jacobian> serial_accumulation{table, chain_length};

  serial_accumulation.parser_sequence_2_graphviz_format();
}

void graph_test_3(Jacobian* jac_0){

  const std::size_t chain_length = 4;
  Table<Jacobian> table{chain_length};

  for(std::size_t idx = 0; idx < chain_length; idx++){

    table.emplace_back(jac_0);
  }

  //[1,0]
  table.emplace_back(12, 0);
  //[2,1]
  table.emplace_back(15, 1);
  //[2,0]
  table.emplace_back(23, 2);
  //[3,2]
  table.emplace_back(17, 2);
  //[3,1]
  table.emplace_back(33, 1);
  //[3,0]
  table.emplace_back(50, 0);

  serial_accumulation_sequence<Jacobian> serial_accumulation{table, chain_length};

  serial_accumulation.parser_sequence_2_graphviz_format();
}

void graph_test_3(Dense_Jacobian* jac_0){

  const std::size_t chain_length = 4;
  Table<Dense_Jacobian> table{chain_length};

  //[0]
  table.emplace_back(jac_0, 11, Operation::TANGENT);
  //[1]
  table.emplace_back(jac_0, 13, Operation::ADJOINT);
  //[2] does not matter
  table.emplace_back(jac_0, 17, Operation::ADJOINT);
  //[3]
  table.emplace_back(jac_0, 18, Operation::ADJOINT);
  //[1,0]
  table.emplace_back(25, 0, Operation::MULTIPLICATION);
  //[2,1] does not matter
  table.emplace_back(30, 1, Operation::TANGENT);
  //[2,0]
  table.emplace_back(53, 1, Operation::TANGENT);
  //[3,2] does not matter
  table.emplace_back(32, 2, Operation::ADJOINT);
  //[3,1] does not matter
  table.emplace_back(53, 1, Operation::ADJOINT);
  //[3,0]
  table.emplace_back(78, 2, Operation::MULTIPLICATION);

  serial_accumulation_sequence<Dense_Jacobian> serial_accumulation{table, chain_length};

  serial_accumulation.parser_sequence_2_graphviz_format();
}

int main(){
  //test object used to store test results and print test state.
  test_optimal_accumulation test_object;

  {
    Jacobian jac_0{4, 6};
    Jacobian jac_1{6, 5};
    Jacobian jac_2{5, 5};

    test_object.set_test_jacobian_small(test_jacobian(&jac_0, &jac_1));

    test_object.set_test_jacobian_medium(test_jacobian(&jac_0, &jac_1, &jac_2));

    /* graph_test_1(&jac_0); */
  }
  
  {

    //Base Jacobians
    Dense_Jacobian jac_0{3, 5, 25};
    Dense_Jacobian jac_1{5, 4, 40};
    Dense_Jacobian jac_2{4, 8, 55};

    test_object.set_test_tangent_dense_jacobian(
        test_tangent_dense(&jac_0, &jac_1));
    
    test_object.set_test_adjoint_dense_jacobian(
        test_adjoint_dense(&jac_0, &jac_1));

    test_object.set_test_multiplication_dense_jacobian(
        test_multiplication_dense(&jac_0, &jac_1));

    test_object.set_test_dense_jacobian_accumulation(
        test_dense_jacobian(&jac_0, &jac_1, &jac_2));

    /* graph_test_3(&jac_0); */
  }

  //Checks if all tests were successful.
  test_object.set_were_all_tests_successful();
  test_object.print_test_state();
  
  return 0;
}
