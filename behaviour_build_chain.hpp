#include <cstdint>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include "jacobian.hpp"

template <class Jacobian_type>
class build_chain{
 public:
  std::vector<Jacobian_type> get_chain(){
    return jacobian_chain;
  }

  void print(){
    for(std::size_t i=0; i<jacobian_chain.size(); i++){
      std::cout<<"F'_"<<i<<' ';
      jacobian_chain[i].print();
    }
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
    std::ifstream file(sparse_data_file_name);

    if(!file){
      std::cerr<< "Error opening file " + sparse_data_file_name << ". \n";
    }

    if(jacobian_chain.capacity() < problem_data.size()){
      jacobian_chain.reserve(problem_data.size());
    }

    std::vector<std::vector<std::size_t>> jac_sparse_data;
    std::size_t nnz_row = 0, nnz_col = 1, nnz = 3;
    std::string line;
    std::size_t line_counter;

    for(std::size_t i =0; i<problem_data.size(); i++){
      if(jac_sparse_data.capacity() < problem_data[i][nnz]){
        jac_sparse_data.reserve(problem_data[i][nnz]);
      }
      std::getline(file,line);
      if(line[0] == '#'){
        //Debugging purpose.
        std::cout << line << '\n';
      }
      else{std::runtime_error("# matrix header not found in line " + std::to_string(line_counter) + '.');}
      
      line_counter++;
      for(std::size_t nnz_idx = 0; nnz_idx < problem_data[i][nnz]; nnz_idx++){
        std::getline(file, line);
        std::istringstream iss(line);
        iss >> jac_sparse_data[nnz_idx][nnz_row] >> jac_sparse_data[nnz_idx][nnz_col];
        line_counter++;
      }
      jacobian_chain.emplace_back(problem_data[i], jac_sparse_data);
      jac_sparse_data.clear();
    }
  }
};
