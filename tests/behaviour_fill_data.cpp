#include <iostream>
/* #include <memory> */
#include <cstdint>
#include "./../behaviour_build_chain.hpp"
#include "./../behaviour_fill_table.hpp"
#include "./../table.hpp"
#include "./../table_cell.hpp"
#include "./../data_file_to_vector.hpp"
#include "./../jacobian.hpp"

int main(int argc, char *v[]){
  assert(argc == 4);
  std::vector<std::vector<size_t>> problem_data;
  std::size_t chain_len;
  std::size_t num_cols = static_cast<std::size_t>(std::stoi(v[2]));
  std::size_t mem_lim= static_cast<std::size_t>(std::stoi(v[3]));
  data_file_vector(v[1], problem_data, chain_len, num_cols);

  std::cout<<"Building the Jacobian chain: \n";
  build_chain_dense<Dense_Jacobian> build_chain_obj;
  build_chain_obj.build_chain(problem_data);
  build_chain_obj.print();

  std::cout<<"Allocating memory for the table (DJCPB):\n";
  Table<cell_DJCPB> table {chain_len, (chain_len + 1) * chain_len / 2} ;

  //Storing the table
  auto jac_chain = build_chain_obj.get_chain();
  
  std::cout<<"Filling the dynamic programming table (DJCPB):\n"; 
  fill_DJCPB<cell_DJCPB, Dense_Jacobian> DJCPB_fill_beh;
  DJCPB_fill_beh.fill(table, jac_chain); 

  std::cout<<"Printing the dynamic table:\n";
  table.print();
  std::cout<< "Deallocating resources: \n";
  table.clear();

  /* std::cout<<"Reprinting the jacobian chain after filling the table using DJCPB.\n"; */
  /* build_chain_obj.print(); */

  std::cout<<"Allocating memory for the table (MFDJCPB):\n";
  Table<cell_MFDJCPB> table_ {chain_len, (chain_len + 1) * chain_len / 2};

  std::cout<<"Filling the dynamic programming table (MFDJCPB) without memory constrain:\n";
  // MFDJCPB_fill_beh{} sets the memory limit std::size_t max representable number.
  fill_MFDJCPB<cell_MFDJCPB, Dense_Jacobian> MFDJCPB_fill_beh {};
  MFDJCPB_fill_beh.fill(table_, jac_chain);

  std::cout<<"Printing the dynamic table MFDJCPB without memory constrain:\n";
  table_.print();

  std::cout<<"Printing last entry of the table. \n";
  table_.back().print();


  /* std::cout<<"Modifying the memory limit. \n"; */
  /* std::size_t new_mem_lim = table_.back().memory - 1; */

  std::cout<<"Deallocating resources: \n";
  table_.clear();
  std::cout<<"Reserving memory for running MFDJCPB with memory constrain. \n";
  table_.reserve((chain_len + 1) * chain_len / 2);

  std::cout<<"Memory limit: "<<mem_lim<<'\n';
  MFDJCPB_fill_beh.change_memory_limit(mem_lim);
  MFDJCPB_fill_beh.fill(table_, jac_chain);

  std::cout<<"Printing the dynamic table with memory constrain:\n";
  table_.print();

  //First try using jacobian.
  /* std::cout<<"Jacobian type: Jacobian.\n"; */
  /* std::unique_ptr<behaviour_build_chain<Jacobian>> p_build_chain; */
  /* p_build_chain = std::make_unique<build_chain_dense<Jacobian>>(); */
  /* p_build_chain-> build_chain(problem); */
  /* p_build_chain-> print_chain(); */

  //Second try using jacobian.
  /* std::cout<<"Jacobian type: Dense_Jacobian.\n"; */
  /* std::unique_ptr<behaviour_build_chain<Dense_Jacobian>> p_build_chain_2; */
  /* p_build_chain_2 = std::make_unique<build_chain_dense<Dense_Jacobian>>(); */
  /* p_build_chain_2-> build_chain(problem); */
  /* p_build_chain_2-> print_chain(); */

  return 0;
}
