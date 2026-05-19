//#include <iostream> included in jacobian.hpp
#include <iostream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <string>
#include "./../jacobian.hpp"
#include "./../generator.hpp"

enum class GeneratorType{
  N_M_N_E,
  N_M_N_E_NNZ,
  NO_GEN,
  UNKOWN
};

enum class JacobianType{
  JACOBIAN,
  DENSE_JACOBIAN,
  SPARSE_JACOBIAN,
  UNKOWN
};

GeneratorType parse_generator(const std::string& name){
  if (name == "n_m_n_E") return GeneratorType::N_M_N_E;
  if (name == "n_m_n_E_nnz") return GeneratorType::N_M_N_E_NNZ;
  if (name == "no_generator") return GeneratorType::NO_GEN;
  return GeneratorType::UNKOWN;
}

JacobianType parse_jacobian(const std::string& name){
  if (name == "Jacobian") return JacobianType::JACOBIAN;
  else if (name == "Dense_Jacobian") return JacobianType::DENSE_JACOBIAN;
  else if (name == "Sparse_Jacobian") return JacobianType::SPARSE_JACOBIAN;
  return JacobianType::UNKOWN;
}


int main(){

  std::string input_gen;
  std::cout << "Enter generator type {n_m_n_E, n_m_n_E_nnz, no_generator}\n";
  std::cin >> input_gen;

  GeneratorType gen_type = parse_generator(input_gen);
  std::vector<std::vector<std::size_t>> problem_data; 

  switch (gen_type){
    case GeneratorType::N_M_N_E: {
      std::size_t dim_lb, dim_ub, n_E_lb, n_E_ub, seed;
      bool is_deterministic;
      std::cout<< "Enter the arguments for the constructor.\n";
      std::cout<< "dim_lb dim_ub n_E_lb n_E_ub is_deterministic{0,1} seed\n";
      std::cin >> dim_lb >> dim_ub >> n_E_lb;
      std::cin >> n_E_ub >> is_deterministic >> seed;

      std::cout<< "Calling n_m_n_E_Generator constructor.\n";
      n_m_n_E_Generator gen{ 1, dim_lb, dim_ub, n_E_lb, n_E_ub, is_deterministic, seed};
      std::cout<< "Building the problem.\n";
      gen.build_problem();
      gen.print();
      std::cout<< "Copying problem data. \n";
      problem_data = gen.get_problem_copy(); 
      /* const auto problem_data = gen.get_problem_copy(); */ 
      break;
    }

    case GeneratorType::N_M_N_E_NNZ: {
      std::size_t dim_lb, dim_ub, n_E_lb, n_E_ub, seed;
      double den_lb, den_ub;
      bool is_deterministic;
      std::cout<< "Enter the arguments for the constructor.\n";
      std::cout<< "dim_lb dim_ub n_E_lb n_E_ub den_lb den_ub";
      std::cout<< " is_deterministic{0,1} seed\n";
      std::cin >> dim_lb >> dim_ub >> n_E_lb;
      std::cin >> n_E_ub >> den_lb >> den_ub >>is_deterministic >> seed;

      std::cout<< "Calling n_m_n_E_nnz_Generator constructor.\n";
      n_m_n_E_nnz_Generator gen{1, dim_lb, dim_ub, n_E_lb, n_E_ub, 
          den_lb, den_ub, is_deterministic, seed};
      std::cout<< "Building the problem.\n";
      gen.build_problem();
      gen.print();
      std::cout<< "Copying problem data. \n";
      problem_data = gen.get_problem_copy(); 
      break;
    }

    case GeneratorType::NO_GEN: {
      std::cout << "No generator is called. Data must be provided by the user. \n";
      break;
    }

    case GeneratorType::UNKOWN: {
      std::cout << "Invalid input.\n"; 
      break;
    }
  }

  std::string input_jac;
  std::string sparse_data_file_name;
  std::cout << "Enter Jacobian type {Jacobian, Dense_Jacobian, Sparse_Jacobian} \n";
  std::cin >> input_jac;
  JacobianType jac_type = parse_jacobian(input_jac);
  bool answer;

  switch (jac_type){
    case JacobianType::JACOBIAN: {
      if(gen_type == GeneratorType::NO_GEN){
        std::vector<size_t> problem_data;
        std::size_t n_m;
        std::cout << "Enter the the dimensions of the Jacobian.\n";
        std::cout << "Input dimension (n) = ";
        std::cin >> n_m;
        problem_data.push_back(n_m);
        std::cout << '\n';
        std::cout << "Output dimensions (m) = ";
        std::cin >> n_m;
        problem_data.push_back(n_m);
      }
      std::cout<<"Calling the constructor of the class Jacobian.\n";
      Jacobian jac{problem_data[0]}; 
      std::cout<<"Do you want to test print()? [0/1]\n";
      std::cin >> answer;
      if(answer){jac.print();}
      std::cout<<"Do you want to test n(), m()? [0/1]\n";
      std::cin >> answer;
      if(answer){
        std::cout<<"Calling n():" << jac.n()<<'\n';
        std::cout<<"Calling m():" << jac.m()<<'\n';
      }
      break;
    }
    case JacobianType::DENSE_JACOBIAN: {
      if(gen_type == GeneratorType::NO_GEN){
        std::vector<size_t> problem_data;
        std::size_t n_m;
        std::cout << "Enter the the dimensions of the Jacobian.\n";
        std::cout << "Input dimension (n) = ";
        std::cin >> n_m;
        problem_data.push_back(n_m);
        std::cout << '\n';
        std::cout << "Output dimensions (m) = ";
        std::cin >> n_m;
        problem_data.push_back(n_m);
        std::cout << "Enter the number of edges of the DAG representation of the program.\n";
        std::cout << "n_E = ";
        std::cin >> n_m;
        problem_data.push_back(n_m);
        std::cout << '\n';
      }
      std::cout<<"Calling the constructor of the class Dense_Jacobian.\n";
      Dense_Jacobian jac{problem_data[0]}; 
      std::cout<<"Do you want to test print()? [0/1]\n";
      std::cin >> answer;
      if(answer){jac.print();}
      std::cout<<"Do you want to test n(), m(), n_E()? [0/1]\n";
      std::cin >> answer;
      if(answer){
        std::cout<<"Calling n():" << jac.n()<<'\n';
        std::cout<<"Calling m():" << jac.m()<<'\n';
        std::cout<<"Calling n_E():" << jac.n_E() <<'\n';
      }
      break;
    }
    case JacobianType::SPARSE_JACOBIAN: {
      std::string data_file_name;
      if(gen_type == GeneratorType::NO_GEN){
        std::vector<size_t> problem_data;
        std::cout << "Please enter the following information.\n";
        std::size_t n, m, num_nnz, n_E;
        std::cout << "Enter the the dimensions of the Jacobian.\n";
        std::cout << "Input dimension (n) = ";
        std::cin >> n;
        problem_data.push_back(n);
        std::cout << "Output dimensions (m) = ";
        std::cin >> m;
        problem_data.push_back(m);
        std::cout << "Enter the number of edges of the DAG representation of the program.\n";
        std::cout << "n_E = ";
        std::cin >> n_E;
        problem_data.push_back(n_E);
        std::cout << "Number of non zero entries (num_nnz) = ";
        std::cin >> num_nnz;
        if(num_nnz < std::max(n,m)){
          std::cout << "The number of non zero entries must be greater than max(n,m).\n";
          std::cout << "Number of non zero entries (num_nnz) = ";
          std::cin >> num_nnz;
          if(num_nnz < std::max(n,m)){
            std::cout << "Invalid argument num_nnz \n";
            break;
          }
        }
        problem_data.push_back(num_nnz);
        std::cout << '\n';
        std::cout << "The data file should follow the following format.\n";
        std::cout << "First line: must begin with a hashtag #.\n";
        std::cout << "Second line on: row_index column_index. \n";
        std::cout << "No whitespace at the beginning but a white space is required between row and column index.\n";
        std::cout << "Example:\n";
        std::cout << "#\n";
        std::cout << "0 1\n";
        std::cout << "Enter the name of the data file.\n";
        std::cin >> data_file_name;
        std::cout << "Calling the constructor of the class Sparse_Jacobian.\n";
        Sparse_Jacobian jac{problem_data}; 
        std::cout << "Building CSR and CSC format for the given sparsity data.\n";
        jac.file_to_CSR_CSC(data_file_name);
        std::cout << "Printing CSR and CSC format.\n";
        jac.print_CSR();
        jac.print_CSC();
        std::cout << "Reprinting number of non zero entries:\n";
        std::cout << "nnz = " << jac.num_nnz() <<'\n'; 
        break;
      }

      else if(gen_type != GeneratorType::N_M_N_E_NNZ){
        std::cout << "Invalid generator type. Either no generator or n_m_n_E_nnz_Generator "<<
          "must be used to create a sparse jacobian object.\n";
        break;
      }

      else{
        data_file_name = "sparse_data";
      }
      std::cout << "Calling the constructor of the class Sparse_Jacobian.\n";
      Sparse_Jacobian jac{problem_data[0]}; 
      std::cout << "Do you want to test print()? [0/1]\n";
      std::cin >> answer;
      if(answer){
        jac.print();
      }
      std::cout << "Do you want to test n(), m(), n_E(), num_nnz()? [0/1]\n";
      std::cin >> answer;
      if(answer){
        std::cout << "Calling n(): " << jac.n() << '\n';
        std::cout << "Calling m(): " << jac.m() << '\n';
        std::cout << "Calling n_E(): " << jac.n_E() << '\n';
        std::cout << "Calling num_nnz(): " << jac.num_nnz() << '\n';
      }

      std::cout << "Do you want to test build_CSR_CSC_format? [0/1]\n";
      std::cin >> answer;
      if(answer){
        jac.file_to_CSR_CSC(data_file_name);
      }

      std::cout << "Do you want to test print_CSR() and print_CSC()? [0/1]\n";
      std::cin >> answer;
      if(answer){
        std::cout<< "Calling print_CSR(): \n";
        jac.print_CSR();
        std::cout<< "Calling print_CSC(): \n";
        jac.print_CSC();
      }
      break;
    }
    case JacobianType::UNKOWN: {
      std::cout << "Invalid input.\n"; 
    }
  }
}

