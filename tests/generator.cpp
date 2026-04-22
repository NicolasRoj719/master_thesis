#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <stdexcept>
#include "./../generator.hpp"

enum class GeneratorType{
  N_M_N_E,
  N_M_N_E_NNZ,
  UNKOWN
};

GeneratorType parse_generator(const std::string& name){
  if (name == "n_m_n_E") return GeneratorType::N_M_N_E;
  if (name == "n_m_n_E_nnz") return GeneratorType::N_M_N_E_NNZ;
  return GeneratorType::UNKOWN;
}

int main(){

  std::string input;
  std::cout << "Enter generator type {n_m_n_E , n_m_n_E_nnz} \n";
  std::cin >> input;

  GeneratorType type = parse_generator(input);

  switch (type){
    case GeneratorType::N_M_N_E: {
      std::size_t chain_len, dim_lb, dim_ub, n_E_lb, n_E_ub, seed;
      bool is_deterministic, answer;
      std::cout<< "Enter the arguments for the constructor.\n";
      std::cout<< "chain_len dim_lb dim_ub n_E_lb n_E_ub is_deterministic{0,1} seed\n";
      std::cin >> chain_len >> dim_lb >> dim_ub >> n_E_lb;
      std::cin >> n_E_ub >> is_deterministic >> seed;

      n_m_n_E_Generator gen{chain_len, dim_lb, dim_ub, n_E_lb, n_E_ub, is_deterministic, seed};
      std::cout<< "Do you want to test build_problem? [0/1]\n";
      std::cin >> answer;
      if(answer){
        gen.build_problem();
        gen.print();
      }
      else{break;}
      
      std::cout<< "Do you want to test get_problem? [0/1]\n";
      std::cin >> answer;
      if(answer){
        auto problem = gen.get_problem();
        for(std::size_t i=0; i<gen.get_problem_size(); i++){
          std::cout<< "F'_"<<i<<" [ ";
          for(std::size_t j=0; j<problem[0].size(); j++){
            std::cout<<problem[i][j]<<' ';
          }
          std::cout<<"]\n";
        }
      }

      std::cout<< "Do you want to check if the generator is deterministic? [0/1]\n";
      std::cin >> answer;
      if(answer){
        std::cout<< "Calling build problem for the second time.\n";
        gen.build_problem();
        gen.print();
      }
      break;
    }
    
    case GeneratorType::N_M_N_E_NNZ: {
      std::size_t chain_len, dim_lb, dim_ub, n_E_lb, n_E_ub, seed;
      double den_lb, den_ub;
      bool is_deterministic, answer;
      std::cout<< "Enter the arguments for the constructor.\n";
      std::cout<< "chain_len dim_lb dim_ub n_E_lb n_E_ub den_lb den_ub";
      std::cout<< " is_deterministic{0,1} seed\n";
      std::cin >> chain_len >> dim_lb >> dim_ub >> n_E_lb;
      std::cin >> n_E_ub >> den_lb >> den_ub >>is_deterministic >> seed;

      std::cout<< "Calling the constructor.";
      n_m_n_E_nnz_Generator gen{chain_len, dim_lb, dim_ub, n_E_lb, n_E_ub, 
          den_lb, den_ub, is_deterministic, seed};

      std::cout<< "Do you want to test build_problem? [0/1]\n";
      std::cin >> answer;
      if(answer){
        gen.build_problem();
        gen.print();
        break;
      }
      else{break;}
      }

    case GeneratorType::UNKOWN: {
      std::cout<<"Invalid generator type.\n";
      break;
    }
  }
  return 0;
}
