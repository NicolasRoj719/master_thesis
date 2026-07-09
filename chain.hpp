#include <vector>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "util_structs.hpp"
#include "jacobian.hpp"

#ifndef CHAIN_HPP
#define CHAIN_HPP

template<class Jacobian_type, class Basic_information_type>
class jacobian_chain{
 public:
   jacobian_chain(std::vector<Basic_information_type> basic_information){
    if(!basic_information_dimension_check(basic_information)){
      throw std::runtime_error("Mismatching vector space dimension.\n"
          "Error encountered during jacobian chain initialization.");
    }
    chain_initializer(basic_information);
   }

   jacobian_chain(const std::string& file_name){
     file_to_chain(file_name);
   }

   const Jacobian_type& operator[](std::size_t index) const{
    return chain[index];
   } 

   const Jacobian_type& at(std::size_t index) const{
    return chain.at(index);
   }

   std::size_t size() const {return chain.size();}

 protected:
  std::vector<Jacobian_type> chain;

  bool basic_information_dimension_check
    (const std::vector<Basic_information_type>& basic_information) const{
      for(std::size_t jacobian_idx = 0; jacobian_idx < basic_information.size() - 1;
          jacobian_idx++){
       if(basic_information[jacobian_idx+1].domain_dimension() !=
           basic_information[jacobian_idx].codomain_dimension()){
        return false;
       } 
      }
      return true;
    }

  //Data ownership is transfer to the jacobian object.
  void chain_initializer(std::vector<Basic_information_type>& basic_information){
    chain.reserve(basic_information.size());
    for(std::size_t jacobian_idx = 0; jacobian_idx < basic_information.size();
        jacobian_idx++){
      chain.emplace_back(std::move(basic_information[jacobian_idx]));
    }
  }

  void file_to_chain(const std::string& file_name);
};

template<class Jacobian_type, class Basic_information_type>
void jacobian_chain<Jacobian_type, Basic_information_type>::file_to_chain(const std::string& file_name){

  std::ifstream file;
  file.open(file_name);
  if(!file.is_open()){
    throw std::runtime_error("There was a problem opening the file: " + 
        file_name);
  }

  std::string line;
  
  std::getline(file, line);
  std::istringstream iss(line);
  std::string word;
  //reading first line
  while(iss >> word){}
  std::size_t chain_len = std::stoull(word);
  chain.reserve(chain_len);

  std::size_t domain_dimension, codomain_dimension;
  std::size_t shared_dimension;
  std::size_t number_edges;
  //Format assumption: no empty line between the first line and first line with data.
  std::getline(file,line);
  iss.clear();
  iss.str(line);

  if constexpr(std::is_same_v<Basic_information_type, Jacobian_information>){
    iss >> domain_dimension >> codomain_dimension;
    chain.emplace_back(domain_dimension, codomain_dimension);
    shared_dimension = codomain_dimension;
    while(std::getline(file, line)){
      if(std::isdigit(line[0])){
        iss.clear();
        iss.str(line);
        iss >> domain_dimension >> codomain_dimension;
        if(shared_dimension != domain_dimension){
          throw std::runtime_error("Dimension mismatch found while "
              "chain initialization.\n"
              "Reading data from file: " + file_name);
        }
        chain.emplace_back(domain_dimension, codomain_dimension);
        shared_dimension = codomain_dimension;
      }
    }
  }

  else if constexpr(std::is_same_v<Basic_information_type, Matrix_free_information>){
    iss >> domain_dimension >> codomain_dimension >> number_edges;
    chain.emplace_back(domain_dimension, codomain_dimension, number_edges);
    shared_dimension = codomain_dimension;
    while(std::getline(file, line)){
      if(std::isdigit(line[0])){
        iss.clear();
        iss.str(line);
        iss >> domain_dimension >> codomain_dimension >> number_edges;
        if(shared_dimension != domain_dimension){
          throw std::runtime_error("Dimension mismatch found while " 
              "chain initialization.\n"
              "Reading data from file: " + file_name);
        }
        chain.emplace_back(domain_dimension, codomain_dimension, number_edges);
        shared_dimension = codomain_dimension;
      }
    }
  }

  file.close();
}

template<>
class jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>{
 public:
  jacobian_chain(Generator_data data){

    coherency_basic_information_sparse_data(
        data.jacobian_information, data.sparse_data);

    initialize_chain(data.jacobian_information, data.sparse_data);
  }

  jacobian_chain(const std::string& file_name){

    std::vector<Matrix_free_sparse_information> basic_information_;
    std::vector<std::vector<NNZ>> sparse_data_;

    file_to_sparse_data(file_name, basic_information_, sparse_data_);
    coherency_basic_information_sparse_data(basic_information_, sparse_data_);
    initialize_chain(basic_information_, sparse_data_);
    
  }

  const Sparse_Jacobian& operator[](std::size_t index) const{
    return chain[index];
  }

  const Sparse_Jacobian& at(std::size_t index) const{
    return chain.at(index);
  }

  std::size_t size() const {return chain.size();}

 protected:
  std::vector<Sparse_Jacobian> chain;
  bool basic_information_dimension_check(
      const std::vector<Matrix_free_sparse_information>& basic_information){

    for(std::size_t jacobian_idx = 0; jacobian_idx < basic_information.size() - 1; jacobian_idx++){
     if(basic_information[jacobian_idx+1].domain_dimension() !=
         basic_information[jacobian_idx].codomain_dimension()){
      return false;
     }
    }
    return true;
  }

  void coherency_basic_information_sparse_data(
      const std::vector<Matrix_free_sparse_information>& basic_information,
      const std::vector<std::vector<NNZ>>& sparse_data){
    
    if(!basic_information_dimension_check(basic_information)){
      throw std::runtime_error("Mismatching vector space dimension.\n"
          "Error encountered during jacobian chain initialization.");
    }

    if(basic_information.size() != sparse_data.size()){
      throw std::runtime_error("Error encountered during chain initialization.\n" 
          "Basic_information and Sparse data size do not match.");
    }

    for(std::size_t jacobian_idx =0; jacobian_idx < basic_information.size(); jacobian_idx++){
      if(sparse_data[jacobian_idx].size() != basic_information[jacobian_idx].number_of_nonzeros()){
        throw std::runtime_error("Error encountered during chain initialization.\n"
            "Inconsistency in the number of nonzero entries reported by the basic \n"
            "information array and the number of nonzero entries stored in sparse_data.");
      }
    }
  }

  void initialize_chain(std::vector<Matrix_free_sparse_information>& basic_information,
      std::vector<std::vector<NNZ>>& sparse_data){
    chain.reserve(basic_information.size());
    for(std::size_t jacobian_idx = 0; jacobian_idx < basic_information.size(); jacobian_idx++){
      chain.emplace_back(std::move(basic_information[jacobian_idx]), sparse_data[jacobian_idx]);
      sparse_data[jacobian_idx].clear();
    }
  }

  void file_to_sparse_data(const std::string& file_name,
      std::vector<Matrix_free_sparse_information>& basic_information,
      std::vector<std::vector<NNZ>>& sparse_data){

    basic_information.clear();
    sparse_data.clear();

    std::ifstream file;
    file.open(file_name);
    if(!file.is_open()){
      throw std::runtime_error("Error encountered during chain initialization.\n"
          "There was a problem opening the file: " + file_name);
    }
    std::string line;
    std::istringstream iss;
    std::string chain_size_string;
    std::size_t chain_len;
    std::getline(file, line);
    if(line[0] != '#'){
      throw std::runtime_error("Error encountered during chain initialization.\n"
          "Wrong format found in " + file_name + " file.");
    }

    iss.str(line);
    while(iss >> chain_size_string){}
    chain_len = std::stoull(chain_size_string);

    sparse_data.resize(chain_len);
    basic_information.reserve(chain_len);

    std::size_t jac_idx = 0;
    std::size_t domain_dimension, codomain_dimension, number_edges, number_nnz;
    std::size_t row, column;
    std::string basic_information_str;
    while(std::getline(file, line)){
      if(line[0] == '#'){
        auto start = line.find('[') + 1;
        auto end = line.find(']');
        basic_information_str = line.substr(start, end-start);
        iss.clear();
        iss.str(basic_information_str);
        iss >> domain_dimension >> codomain_dimension >> number_edges >> number_nnz;
        basic_information.emplace_back(domain_dimension, codomain_dimension,
            number_edges, number_nnz);
        sparse_data[jac_idx].reserve(number_nnz);
        jac_idx++;
      }
      else if(std::isdigit(line[0])){
        iss.clear();
        iss.str(line);
        iss >> row >> column;
        sparse_data[jac_idx - 1].emplace_back(row, column);
      }
    }
  }
};
#endif
