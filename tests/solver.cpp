#include "./../solver.hpp"
#include <memory>
#include <cstdint>

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

  std::vector<std::vector<std::size_t>> problem = gen.get_problem();

  {
    std::cout<<"Running DJCPB solver.\n";
    std::cout<<"Cell type: cell_DJCPB, Jacobian type: dense.\n";
    DJCPB<cell_DJCPB, Dense_Jacobian> DJCPB_solver;
    DJCPB_solver.solve(problem);
  }

  {
    std::cout<<"Running DJCPB solver.\n";
    std::cout<<"Cell type: cell_MFDJCPB, Jacobian type: dense.\n";
    DJCPB<cell_MFDJCPB, Dense_Jacobian> DJCPB_solver;
    DJCPB_solver.solve(problem);
  }

  {
    std::cout<<"Running MFDJCPB solver without memory limit.\n";
    std::cout<<"Cell type: cell_MFDJCPB, Jacobian type: dense.\n";
    MFDJCPB<cell_MFDJCPB, Dense_Jacobian> MFDJCPB_solver{};
    MFDJCPB_solver.solve(problem);
  }

  {
    std::cout<<"Running MFDJCPB solver with memory limit ("<<mem_lim<<").\n";
    std::cout<<"Cell type: cell_MFDJCPB, Jacobian type: dense.\n";
    MFDJCPB<cell_MFDJCPB, Dense_Jacobian> MFDJCPB_solver{mem_lim};
    MFDJCPB_solver.solve(problem);
  }

  /* { */
  /*   std::cout<<"Running MFDJCPB solver"; */
  /*   std::cout<<"Cell type: cell_DJCPB, Jacobian type: dense."; */
  /*   MFDJCPB<cell_DJCPB, Dense_Jacobian> MFDJCPB_solver; */
  /*   MFDJCPB_solver.solve(problem); */
  /* } */

  return 0;
}
