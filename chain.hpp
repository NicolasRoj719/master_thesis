#include <vector>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "util_structs.hpp"
#include "jacobian.hpp"

#ifndef CHAIN_HPP
#define CHAIN_HPP

template <class Jacobian_type>
class jacobian_chain{
 public:
  jacobian_chain(const std::vector<std::vector<std::size_t>>& jacobian_chain_information){
    static_assert(std::is_same_v<Jacobian_type, Jacobian> || 
        std::is_same_v<Jacobian_type, Dense_Jacobian>);
    generator_data_to_chain(jacobian_chain_information);
  }

  jacobian_chain(const std::string& file_name){
    static_assert(std::is_same_v<Jacobian_type, Jacobian> || 
        std::is_same_v<Jacobian_type, Dense_Jacobian>);
    file_to_chain(file_name);
  }

  const Jacobian_type& operator[](std::size_t index) const{
    return chain[index];
  }

  std::size_t size() const {return chain.size();}

 private:
  std::vector<Jacobian_type> chain;

  void generator_data_to_chain(const std::vector<std::vector<std::size_t>>& jac_chain_info){
    //n: dimension of the domain vector space (R^n)
    //m: dimension of the codomain vector space (R^m)
    const std::size_t n = 0, m = 1;
    bool dimensions_match = 1;
    
    for(std::size_t jac_idx = 0; jac_idx < jac_chain_info.size() - 1; jac_idx++){
      if(jac_chain_info[jac_idx][m] != jac_chain_info[jac_idx + 1][n]){
        dimensions_match = 0;
        jac_idx = jac_chain_info.size();
      }
    }

    if(!dimensions_match){
      throw std::runtime_error("Dimension mismatch found while executing "
          "Jacobian chain constructor.");
    }

    chain.reserve(jac_chain_info.size());

    for(std::size_t jac_idx = 0; jac_idx < jac_chain_info.size(); jac_idx++){
      chain.emplace_back(jac_chain_info[jac_idx]); 
    }
  }

  void file_to_chain(const std::string& file_name){
    std::ifstream file;
    file.open(file_name);
    if(!file.is_open()){
      throw std::runtime_error("There was a problem opening the file: " + 
          file_name);
    }

    std::string line;
    std::vector<std::vector<std::size_t>> jac_chain_info;
    
    std::getline(file, line);
    std::istringstream stream_first_line(line);
    std::string resize_data;
    while(stream_first_line >> resize_data){}
    std::size_t chain_len = std::stoull(resize_data);
    jac_chain_info.resize(chain_len);
    jac_chain_info[0].reserve(4);

    std::size_t data;
    std::size_t jac_idx= 0;
    std::istringstream iss;
    while(std::getline(file, line) && jac_idx < chain_len){
      if(std::isdigit(line[0])){
        iss.clear();
        iss.str(line);
        while(iss >> data){
          jac_chain_info[jac_idx].push_back(data);
        }
        jac_idx++;
      }
    }
    file.close();
    generator_data_to_chain(jac_chain_info);
  }
};

template<>
class jacobian_chain<Sparse_Jacobian>{
 public:
  jacobian_chain(const std::vector<std::vector<std::size_t>>& jacobian_chain_information,
      const std::vector<std::vector<NNZ>>& sparse_data){

    test_constructor_arguments(jacobian_chain_information, sparse_data);
    initialize_chain(jacobian_chain_information, sparse_data);
  }

  jacobian_chain(const std::vector<std::vector<std::size_t>>& jacobian_chain_information,
      const std::string& file_name){

    std::vector<std::vector<NNZ>> sparse_data;
    sparse_data.resize(jacobian_chain_information.size());
    const std::size_t idx_nnz = 3;

    for(std::size_t jac_idx = 0; jac_idx < jacobian_chain_information.size(); jac_idx++){
      sparse_data[jac_idx].reserve(jacobian_chain_information[jac_idx][idx_nnz]);
    }

    simple_sparse_file_to_sparse_data(sparse_data, file_name);
    jacobian_chain(jacobian_chain_information, sparse_data);
  }

  jacobian_chain(const std::string& file_name){
    std::vector<std::vector<std::size_t>> jacobian_chain_information;
    std::vector<std::vector<NNZ>> sparse_data;
    sparse_file_to_sparse_data(file_name, jacobian_chain_information, sparse_data);
    test_constructor_arguments(jacobian_chain_information, sparse_data);
    initialize_chain(jacobian_chain_information, sparse_data);
  }

  const Sparse_Jacobian& operator[](std::size_t index) const{
    return chain[index];
  }

  std::size_t size() const {return chain.size();}

 private:
  std::vector<Sparse_Jacobian> chain;

  void test_constructor_arguments(const std::vector<std::vector<std::size_t>>& jac_chain_info,
      const std::vector<std::vector<NNZ>>& sparse_data){
    if(jac_chain_info.size() != sparse_data.size()){
      throw std::runtime_error("Problem data and sparse data outer dimension do not match.\n"
          "Error found while executing Sparse Jacobian chain constructor.");
    }

    bool dimensions_match = 1;
    //n: dimension of the domain vector space (R^n)
    //m: dimension of the codomain vector space (R^m)
    const std::size_t n = 0, m = 1; 
    for(std::size_t jac_idx = 0; jac_idx < jac_chain_info.size() - 1; jac_idx++){
      if(jac_chain_info[jac_idx][m] != jac_chain_info[jac_idx + 1][n]){
        dimensions_match = 0;
        jac_idx = jac_chain_info.size();
      }
    }
    if(!dimensions_match){
      throw std::runtime_error("Dimension mismatch found while executing Sparse Jacobian "
          "chain constructor.");
    }
  }

  void initialize_chain(const std::vector<std::vector<std::size_t>>& jac_chain_info,
      const std::vector<std::vector<NNZ>>& sparse_data){

    chain.reserve(jac_chain_info.size());

    for(std::size_t jac_idx = 0; jac_idx < jac_chain_info.size(); jac_idx++){
      chain.emplace_back(jac_chain_info[jac_idx], sparse_data[jac_idx]);
    }
  }

  void simple_sparse_file_to_sparse_data(std::vector<std::vector<NNZ>>& sparse_data,
      const std::string& file_name){
    std::ifstream file;
    file.open(file_name);
    if(!file.is_open()){
      throw std::runtime_error("There was a problem opening the file: " + 
          file_name);
    }
    std::string line;
    std::size_t row, column;
    std::size_t jac_idx = 0;
    std::istringstream iss;

    while(std::getline(file, line)){
      if(line[0] == '#'){
        jac_idx++;
      }
      else if(std::isdigit(line[0])){
        iss.clear();
        iss.str(line);
        iss >> row >> column;
        sparse_data[jac_idx-1].emplace_back(row, column);
      }
    }
    file.close();
  }

  void sparse_file_to_sparse_data(const std::string& file_name,
      std::vector<std::vector<std::size_t>>& jac_chain_info,
      std::vector<std::vector<NNZ>>& sparse_data){

    jac_chain_info.clear();
    sparse_data.clear();

    std::ifstream file;
    file.open(file_name);
    if(!file.is_open()){
      throw std::runtime_error("There was a problem opening the file: " + 
          file_name);
    }
    std::string line;
    std::istringstream iss;
    std::string chain_size_string;
    std::size_t chain_len;
    std::getline(file, line);
    if(line[0] != '#'){
      throw std::runtime_error("Wrong format:  check " + file_name + " file.");
    }
    else{
      iss.str(line);
      while(iss >> chain_size_string){}
      chain_len = std::stoull(chain_size_string);
    }

    jac_chain_info.resize(chain_len);
    sparse_data.resize(chain_len);

    std::size_t jac_idx = 0;
    std::size_t jacobian_data;
    std::size_t row, column;
    while(std::getline(file, line)){
      if(line[0] == '#'){
        auto start = line.find('[') + 1;
        auto end = line.find(']');
        std::string jacobian_data_string = line.substr(start, end-start);
        jac_chain_info[jac_idx].reserve(4);
        iss.clear();
        iss.str(jacobian_data_string);
        while(iss >> jacobian_data){
          jac_chain_info[jac_idx].push_back(jacobian_data);
        }
        //The last value of jacobian_data_string correspond to the number of non-zero entries
        sparse_data[jac_idx].reserve(jacobian_data);
        jac_idx++;
      }
      else if(std::isdigit(line[0])){
        iss.clear();
        iss.str(line);
        iss >> row >> column;
        sparse_data[jac_idx-1].emplace_back(row, column);
      }
    }
  }
};

#endif
