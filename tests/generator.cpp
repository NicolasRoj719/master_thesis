#include <cassert>
#include <cstdint>
#include "./../generator.hpp"

int main(int argc, char *v[]){
  assert(argc == 6);
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

  std::cout<<"Printing problem size: "<<gen.get_problem_size()<<'\n';

  std::cout<<"Getting problem: get_problem()\n";
  std::vector<std::vector<std::size_t>> problem = gen.get_problem();

  for(std::size_t i=0; i<gen.get_problem_size(); i++){
    std::cout<< "F'_"<<i<<" [ ";
    for(std::size_t j=0; j<problem[0].size(); j++){
      std::cout<<problem[i][j]<<' ';
    }
    std::cout<<"]\n";
  }

  /* std::size_t i = 0; */
  /* assert(i < chain_len); */

  /* std::vector<size_t> F_i_desc{gen.read_Fi_description(i)}; */
  /* std::cout<<"jac_chain_info["<< i <<"] = [ "; */

  /* for(size_t data: F_i_desc){ */
  /*   std::cout<< data<<' '; */
  /* } */
  /* std::cout<<"]\n"; */
  return 0;
}
