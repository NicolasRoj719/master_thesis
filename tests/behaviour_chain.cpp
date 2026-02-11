#include <iostream>
#include <memory>
#include <cstdint>
#include "./../behaviour_build_chain.hpp"
#include "./../generator.hpp"
#include "./../jacobian.hpp"

int main(int argc, char *v[]){
  assert(argc == 6);
  // Obtaining data from the user via the terminal.
  std::size_t chain_len = static_cast<std::size_t>(std::stoi(v[1]));
  std::size_t dim_lb= static_cast<std::size_t>(std::stoi(v[2]));
  std::size_t dim_ub= static_cast<std::size_t>(std::stoi(v[3]));
  std::size_t n_E_lb= static_cast<std::size_t>(std::stoi(v[4]));
  std::size_t n_E_ub= static_cast<std::size_t>(std::stoi(v[5]));

  std::cout<<"Chain length: " <<chain_len<<'\n';
  std::cout<<"Dimension range: [ "<<dim_lb<<" , "<<dim_ub<<" ]\n";
  std::cout<<"Number of edges in the DAG range: [ "<<n_E_lb<<" , "<<n_E_ub<<" ]\n";

  std::cout<<"Using constructor n_m_n_E_Generator(chain_len, dim_lb, dim_ub, n_E_lb, n_E_ub).\n";
  n_m_n_E_Generator gen{chain_len, dim_lb, dim_ub, n_E_lb, n_E_ub};

  std::cout<<"Building problem.\n";
  gen.build_problem();
  
  std::cout<<"Printing problem.\n";
  gen.print();

  std::vector<std::vector<std::size_t>> problem = gen.get_problem();

  //First try using jacobian.
  std::cout<<"Jacobian type: Jacobian.\n";
  std::unique_ptr<behaviour_build_chain<Jacobian>> p_build_chain;
  p_build_chain = std::make_unique<build_chain_dense<Jacobian>>();
  p_build_chain-> build_chain(problem);
  p_build_chain-> print_chain();

  //Second try using jacobian.
  std::cout<<"Jacobian type: Dense_Jacobian.\n";
  std::unique_ptr<behaviour_build_chain<Dense_Jacobian>> p_build_chain_2;
  p_build_chain_2 = std::make_unique<build_chain_dense<Dense_Jacobian>>();
  p_build_chain_2-> build_chain(problem);
  p_build_chain_2-> print_chain();

  return 0;
}
