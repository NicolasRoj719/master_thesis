#include <iostream>
/* #include <memory> */
#include <cstdint>
#include "./../behaviour_build_chain.hpp"
#include "./../behaviour_fill_table.hpp"
#include "./../table.hpp"
#include "./../table_cell.hpp"
#include "./../generator.hpp"
#include "./../jacobian.hpp"

int main(int argc, char *v[]){
  assert(argc == 7);
  // Obtaining data from the user via the terminal.
  std::size_t chain_len= static_cast<std::size_t>(std::stoi(v[1]));
  std::size_t dim_lb= static_cast<std::size_t>(std::stoi(v[2]));
  std::size_t dim_ub= static_cast<std::size_t>(std::stoi(v[3]));
  std::size_t n_E_lb= static_cast<std::size_t>(std::stoi(v[4]));
  std::size_t n_E_ub= static_cast<std::size_t>(std::stoi(v[5]));
  std::size_t mem_lim= static_cast<std::size_t>(std::stoi(v[6]));

  std::cout<<"Chain length: " <<chain_len<<'\n';
  std::cout<<"Dimension range: [ "<<dim_lb<<" , "<<dim_ub<<" ]\n";
  std::cout<<"Number of edges in the DAG range: [ "<<n_E_lb<<" , "<<n_E_ub<<" ]\n";
  std::cout<< "Memory limit: "<<mem_lim<<'\n';

  std::cout<<"Using constructor n_m_n_E_Generator(chain_len, dim_lb, dim_ub, n_E_lb, n_E_ub).\n";
  n_m_n_E_Generator gen{chain_len, dim_lb, dim_ub, n_E_lb, n_E_ub};

  std::cout<<"Building problem.\n";
  gen.build_problem();
  
  std::cout<<"Printing problem.\n";
  gen.print();

  std::cout<<"Building the Jacobian chain: \n";
  build_chain_dense<Dense_Jacobian> build_chain_obj;
  build_chain_obj.build_chain(gen.get_problem());
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

  std::cout<<"Filling the dynamic programming table (MFDJCPB):\n";
  fill_MFDJCPB<cell_MFDJCPB, Dense_Jacobian> MFDJCPB_fill_beh {mem_lim};
  MFDJCPB_fill_beh.fill(table_, jac_chain);

  std::cout<<"Printing the dynamic table:\n";
  table_.print();

  /* std::cout<<"Deallocating resources: \n"; */
  /* table_.clear(); */

  std::cout<<"Printing last entry of the dynamic programming table. \n";
  table_.back().print();


  std::cout<<"Modifying the memory limit. \n";
  std::size_t new_mem_lim = table_.back().memory - 1;

  std::cout<<"Deallocating resources: \n";
  table_.clear();

  std::cout<<"New memory limit: "<<new_mem_lim<<'\n';
  MFDJCPB_fill_beh.change_memory_limit(new_mem_lim);
  Table<cell_MFDJCPB> table_0 {chain_len, (chain_len + 1) * chain_len / 2};
  MFDJCPB_fill_beh.fill(table_0, jac_chain);

  std::cout<<"Printing the dynamic table:\n";
  table_0.print();

  

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
