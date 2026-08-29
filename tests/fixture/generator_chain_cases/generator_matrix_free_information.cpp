#include <fstream>
#include <iostream>
#include <stdexcept>
#include "./../../generator.hpp"
#include "./../../util_structs.hpp"

void print_jacobian_information(std::ostream& out,
    const std::vector<Matrix_free_information>& basic_information){
  for(std::size_t jacobian_idx = 0; jacobian_idx < basic_information.size(); jacobian_idx++){
    out << basic_information[jacobian_idx].domain_dimension() << ' ';
    out << basic_information[jacobian_idx].codomain_dimension() << ' ';
    out << basic_information[jacobian_idx].number_of_edges() << '\n';
  }
}


int main(){
  const std::size_t chain_length = 3;
  const std::size_t dimension_lb = 2, dimension_ub = 5;
  const std::size_t number_edges_lb = 50, number_edges_ub = 120;
  const bool is_deterministic = true;
  const std::size_t seed = 60;

  Generator<Matrix_free_information>
    generator{chain_length, dimension_lb, dimension_ub, number_edges_lb, number_edges_ub,
      is_deterministic, seed};

  std::cout << "#Generator constructor arguments:\n";
  std::cout << "# chain_length dimension_lower_bound dimension_upper_bound\n";
  std::cout << "# number_of_edges_lower_bound number_of_edges_upper_bound deterministic seed";
  std::cout << '\n';

  std::cout << chain_length << ' ' << dimension_lb << ' ' << dimension_ub << ' ';
  std::cout << number_edges_lb << ' ' << number_edges_ub << ' ';
  std::cout << std::boolalpha;
  std::cout << is_deterministic << ' ';
  std::cout << std::noboolalpha;
  std::cout << seed << '\n';

  std::cout << "#Format: domain_dimension codomain_dimension number_of_edges" << '\n';

  auto generated_jacobian_information = generator.generate_jacobian_information();

  print_jacobian_information(std::cout, generated_jacobian_information);

  //Generating validate data.
  const std::string file_path = "./validate/matrix_free_information";
  std::ofstream file(file_path);
  if(!file){
    throw std::runtime_error("Failed to open " + file_path + "used for writing validate data.");
  }
  print_jacobian_information(file, generated_jacobian_information); 


  return 0;
}
