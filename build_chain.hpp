#include <cstdint>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include "util_structs.hpp"
#include "jacobian.hpp"

#ifndef BUILD_CHAIN_HPP
#define BUILD_CHAIN_HPP

template <class Jacobian_type>
class build_chain{
 public:
  const std::vector<Jacobian_type>& get_chain(){
    return jacobian_chain;
  }

  const std::vector<Jacobian_type> get_chain_copy(){
    return jacobian_chain;
  }

  void print(){
    for(std::size_t i=0; i<jacobian_chain.size(); i++){
      std::cout<<"F'_"<<i<<' ';
      jacobian_chain[i].print();
    }
  }

  const Jacobian_type get_jac(std::size_t jac_index){
    return jacobian_chain.at(jac_index);
  }

  std::size_t size(){
    return(jacobian_chain.size());
  }

 protected:
  build_chain() = default;
  std::vector<Jacobian_type> jacobian_chain;
};

template <class Jacobian_type>
class build_chain_implementation: public build_chain<Jacobian_type>{
 public:
  build_chain_implementation(const std::vector<std::vector<std::size_t>>& problem_data){
    bool dimension_mismatch = 0;
    std::size_t n = 0, m = 1;
    for(std::size_t i = 0; i<problem_data.size()-1; i++){
      if(problem_data[i][m] != problem_data[i+1][n]){
        dimension_mismatch = 1;
        i = problem_data.size();
      }
    }
    if(dimension_mismatch){
      throw std::invalid_argument("Matrix dimension mismatch.");
    }

    if(this -> jacobian_chain.capacity() < problem_data.size()){
      this -> jacobian_chain.reserve(problem_data.size());
    }
    for(std::size_t i=0; i<problem_data.size(); i++){
      this->jacobian_chain.emplace_back(problem_data[i]);
    }
  }
};

template<>
class build_chain_implementation<Sparse_Jacobian>: public build_chain<Sparse_Jacobian>{
 public:
  build_chain_implementation(const std::vector<std::vector<std::size_t>>& problem_data,
      const std::string& sparse_data_file_name){
    bool dimension_mismatch = 0;
    std::size_t n = 0, m = 1;
    for(std::size_t i = 0; i<problem_data.size()-1; i++){
      if(problem_data[i][m] != problem_data[i+1][n]){
        dimension_mismatch = 1;
        i = problem_data.size();
      }
    }
    if(dimension_mismatch){
      throw std::invalid_argument("Matrix dimension mismatch.");
    }

    std::ifstream file(sparse_data_file_name);

    if(!file){
      std::cerr<< "Error opening file " + sparse_data_file_name << ". \n";
    }

    if(jacobian_chain.capacity() < problem_data.size()){
      jacobian_chain.reserve(problem_data.size());
    }

    std::vector<NNZ>jac_sparse_data;
    std::size_t nnz = 3;
    std::size_t row, column;
    int line_counter = -1;
    std::string line;

    for(std::size_t i =0; i<problem_data.size(); i++){
      if(jac_sparse_data.capacity() < problem_data[i][nnz]){
        jac_sparse_data.reserve(problem_data[i][nnz]);
      }
      std::getline(file,line);
      line_counter++;
      if(line[0] == '#'){
        //Debugging purpose.
        /* std::cout << line << '\n'; */
      }
      else{std::runtime_error("# Jacobian header not found in line " + std::to_string(line_counter) + '.');}

      for(std::size_t nnz_idx = 0; nnz_idx < problem_data[i][nnz]; nnz_idx++){
        std::getline(file, line);
        line_counter++;
        std::istringstream iss(line);
        iss >> row >> column; 
        jac_sparse_data.emplace_back(row, column);
      }
      jacobian_chain.emplace_back(problem_data[i], jac_sparse_data);
      jac_sparse_data.clear();
    }
  }

  void print_CSR_CSC(){
    for(std::size_t i = 0; i < jacobian_chain.size(); i++){
      std::cout << "Sparse Jacobian " << i << '\n';
      std::cout << "CSR format:\n";
      jacobian_chain[i].print_CSR();
      std::cout << "CSC format:\n";
      jacobian_chain[i].print_CSC();
    }
  }
};
#endif
