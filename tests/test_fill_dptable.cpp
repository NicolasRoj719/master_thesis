#include "./../generator.hpp"
#include <cassert>

int main(int argc, char *v[]){
  assert(argc == 6);
  std::size_t chain_len = static_cast<std::size_t>(std::stoi(v[1]));
  std::size_t dim_lb= static_cast<std::size_t>(std::stoi(v[2]));
  std::size_t dim_ub= static_cast<std::size_t>(std::stoi(v[3]));
  std::size_t n_E_lb= static_cast<std::size_t>(std::stoi(v[4]));
  std::size_t n_E_ub= static_cast<std::size_t>(std::stoi(v[5]));

  m_n_n_E_Generator gen{chain_len, dim_lb, dim_ub, n_E_lb, n_E_ub};
  gen.build_problem();
  gen.print_problem();

  std::cout<<"Jacobian chain length: "<<gen.chain_length<<'\n';

  std::size_t i = 0;
  assert(i < gen.chain_length);

  std::vector<size_t> F_i_desc{gen.read_Fi_description(i)};
  std::cout<<"jac_chain_info["<< i <<"] = [ ";

  for(size_t data: F_i_desc){
    std::cout<< data<<' ';
  }
  std::cout<<"]\n";

  return 0;
}
