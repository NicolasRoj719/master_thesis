#include <fstream>
#include <iostream>
#include "./../../generator.hpp"
#include "./../../util_structs.hpp"

void print_jacobian_information(std::ostream& out,
    const std::vector<Matrix_free_sparse_information>& basic_information){
  for(std::size_t jacobian_idx = 0; jacobian_idx < basic_information.size(); jacobian_idx++){
    out << basic_information[jacobian_idx].domain_dimension() << ' ';
    out << basic_information[jacobian_idx].codomain_dimension() << ' ';
    out << basic_information[jacobian_idx].number_of_edges() << ' ';
    out << basic_information[jacobian_idx].number_of_nonzeros() << '\n';
  }
}

void print_sparse_data(std::ostream& out,
    const std::vector<std::vector<NNZ>>& sparse_information){
  for(std::size_t jacobian_idx = 0; jacobian_idx < sparse_information.size(); jacobian_idx++){
    out << "#Sparse Jacobian: " << jacobian_idx << '\n';
    for(std::size_t nnz_idx = 0; nnz_idx < sparse_information[jacobian_idx].size(); nnz_idx++){
      out << sparse_information[jacobian_idx][nnz_idx].row() << ' '; 
      out << sparse_information[jacobian_idx][nnz_idx].col() << '\n';
    }
  }
}

int main(){
  const std::size_t chain_length = 3;
  const std::size_t dimension_lb = 2, dimension_ub = 5;
  const std::size_t number_edges_lb = 80, number_edges_ub = 160;
  const double density_lb = 0.2, density_ub = 0.5;
  const bool is_deterministic = true;
  const std::size_t seed = 26;
  Generator<Matrix_free_sparse_information>
    generator{chain_length, dimension_lb, dimension_ub, number_edges_lb, number_edges_ub,
    density_lb, density_ub, is_deterministic, seed};

  std::cout <<"#Generator constructor arguments:\n";
  std::cout <<"# chain_length dimension_lower_bound dimension_upper_bound\n";
  std::cout <<"# number_edges_lower_bound number_edges_upper_bound density_lower_bound\n";
  std::cout <<"# density_upper_bound deterministic seed\n";

  std::cout << chain_length << ' ' << dimension_lb << ' ' << dimension_ub << ' ';
  std::cout << number_edges_lb << ' ' << number_edges_ub << ' ' << density_lb << ' ';
  std::cout << density_ub << ' ';
  std::cout << std::boolalpha;
  std::cout << is_deterministic << ' ';
  std::cout << std::noboolalpha;
  std::cout << seed << '\n';

  std::cout << "Format: domain_dimension codomain_dimension number_of_edges number_of_nnz" << '\n';
  auto generator_data = generator.generate_data();

  print_jacobian_information(std::cout, generator_data.jacobian_information); 

  std::cout << "Sparsity information:\n";
  print_sparse_data(std::cout, generator_data.sparse_data); 

}
