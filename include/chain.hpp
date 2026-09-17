/**
 * @file chain.hpp
 *
 * @brief Constructs and manages sequences (chain) of Dense, Split Dense, Sparse, and
 * Split Sparse Jacobian matrices from metadata structures of files.
 */

#ifndef CHAIN_HPP
#define CHAIN_HPP
#include <vector>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "util_structs.hpp"
#include "jacobian.hpp"


/**
 * @brief Generic container wrapping a sequence (chain) of Jacobian matrices.
 *
 * @tparam Jacobian_type stored in the chain (e.g., Jacobian, Dense_Jacobian).
 * @tparam Basic_information_type Metadata type holding dimensions, number of edges,
 * (e.g. Jacobian_information, Matrix_free_information).
 *
 */
template<class Jacobian_type, class Basic_information_type>
class Jacobian_chain{
 public:
  /**
   * @brief Initializes the Jacobian chain using an array of metadata objects.
   *
   * @param basic_information Array containing metadata for each Jacobian in the chain.
   *
   * @throws std::runtime_error If adjacent matrix dimensions are incompatible for multiplication.
   */
  Jacobian_chain(std::vector<Basic_information_type> basic_information){
    if(!basic_information_dimension_check(basic_information)){
      throw std::runtime_error("Mismatching vector space dimension.\n"
          "Error encountered during jacobian chain initialization.");
    }
    chain_initializer(basic_information);
  }

  /**
   * @brief Initializes the Jacobian chain from a data file. 
   *
   * @param file_name Path to file containing metadata for each Jacobian in the chain.
   * @throws std::runtime_error If file cannot be opened or parsed, or if adjacent matrix dimensions
   * are incompatible for multiplication.
   */
  Jacobian_chain(const std::string& file_name){
   file_to_chain(file_name);
  }

  ///Access element at index without bounds checking.
  const Jacobian_type& operator[](std::size_t index) const{
    return chain[index];
  } 

  /// Access element at index with bounds checking.
  const Jacobian_type& at(std::size_t index) const{
    return chain.at(index);
  }

  /// Return a deep copy of the Jacobian_type at the given index.
  Jacobian_type copy(std::size_t index) const{
    return chain[index];
  }

  /// Access a mutable element reference at index without bounds checking. 
  Jacobian_type& get(std::size_t index){
    return chain[index];
  }

  ///Gets the total number of Jacobians in the chain.
  std::size_t size() const {return chain.size();}

 protected:
  /// Storage sequence for Jacobian_type. objects.
  std::vector<Jacobian_type> chain;

  /// Dimension compatibility check for multiplication. 
  bool basic_information_dimension_check
    (const std::vector<Basic_information_type>& basic_information) const{

    for(std::size_t jacobian_idx = 0; jacobian_idx < basic_information.size() - 1;
          jacobian_idx++){

     if(basic_information[jacobian_idx+1].domain_dim() !=
           basic_information[jacobian_idx].codomain_dim()){
        return false;
     } 
    }

    return true;
  }

  /// Transfers data ownership from metadata into chain storage. 
  void chain_initializer(std::vector<Basic_information_type>& basic_information){
    chain.reserve(basic_information.size());
    for(std::size_t jacobian_idx = 0; jacobian_idx < basic_information.size();
        jacobian_idx++){
      chain.emplace_back(std::move(basic_information[jacobian_idx]));
    }
  }

/**
 * @brief Reads data from an input file and populates the internal chain.
 *
 * @param file_name Path to data file. 
 *
 * @throws std::runtime_error If file cannot be read or dimension compatibility check for 
 * multiplication fails.
 */
  void file_to_chain(const std::string& file_name);
};

template<class Jacobian_type, class Basic_information_type>
void Jacobian_chain<Jacobian_type, Basic_information_type>::file_to_chain(const std::string& file_name){

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


/**
 * @brief Specialization of Jacobian_chain for Sparse_Jacobian objects.
 */
template<>
class Jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>{
 public:
  /**
   * @brief Constructs Sparse Jacobian chain from a composite generator data structure. 
   *
   * @param data Generator structure containing metadata and non-zero patterns.
   * @thros std::runtime_error If non-zero metadata or dimensions are incoherent.
   */
  Jacobian_chain(Generator_data data){

    coherency_basic_information_sparse_data(
        data.jacobian_information, data.sparse_data);

    initialize_chain(data.jacobian_information, data.sparse_data);
  }

  /**
   * @brief Constructs Sparse Jacobian chain from a file.
   *
   * @param file_name Path to file containing sparse metadata and sparsity patterns.
   * @param std::runtime_error If file cannot be read or cannot be parsed.
   */
  Jacobian_chain(const std::string& file_name){

    std::vector<Matrix_free_sparse_information> basic_information_;
    std::vector<std::vector<NNZ>> sparse_data_;

    file_to_sparse_data(file_name, basic_information_, sparse_data_);
    coherency_basic_information_sparse_data(basic_information_, sparse_data_);
    initialize_chain(basic_information_, sparse_data_);
    
  }

  ///Access element at index without bounds checking.
  const Sparse_Jacobian& operator[](std::size_t index) const{
    return chain[index];
  }

  /// Access element at index with bounds checking.
  const Sparse_Jacobian& at(std::size_t index) const{
    return chain.at(index);
  }

  ///Gets the total number of Jacobians in the chain.
  std::size_t size() const {return chain.size();}

  //This will be used to test table_cell and table implementation.
  /// Return a deep copy of the Jacobian_type at the given index.
  Sparse_Jacobian copy(std::size_t index) const{
    return chain[index];
  }

  /// Access a mutable element reference at index without bounds checking. 
  Sparse_Jacobian& get(std::size_t index){
    return chain[index];
   }

 protected:
  /// Storage sequence for Sparse_Jacobian objects.
  std::vector<Sparse_Jacobian> chain;

  ///Dimension multiplication compatibility for adjacent objects in the chain.
  bool basic_information_dimension_check(
      const std::vector<Matrix_free_sparse_information>& basic_information){

    for(std::size_t jacobian_idx = 0; jacobian_idx < basic_information.size() - 1; jacobian_idx++){

     if(basic_information[jacobian_idx+1].domain_dim() !=
         basic_information[jacobian_idx].codomain_dim()){

      return false;
     }
    }
    return true;
  }

  /**
   * @brief testing consistency between sparse Jacobian metadata and the array with sparsity patterns. 
   *
   * @param basic_information Array with sparse jacobian metadata.
   * @param sparse_data Array with sparsity patterns.
   * 
   * @throw std::runtime_error If dimensionality condition for multiplication is not fulfilled.
   * @throw std::runtime_error If size mismatch between basic_information and sparse_data.
   * @throw std::runtime_error If promised number of non-zero elements differs with the number of 
   * non zeros in the sparsity pattern.
   */
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

      if(sparse_data[jacobian_idx].size() != basic_information[jacobian_idx].number_nnz()){
        throw std::runtime_error("Error encountered during chain initialization.\n"
            "Inconsistency in the number of nonzero entries reported by the basic \n"
            "information array and the number of nonzero entries stored in sparse_data.");
      }
    }
  }

  /**
   * @brief Instantiates Sparse_Jacobian objects directly into chain memory and clears input array 
   * containing sparsity data.
   * 
   * @param basic_information Sparse Jacobian metadata.
   * @param[in,out] Array with sparsity patterns. Emptied during construction.
   */
  void initialize_chain(std::vector<Matrix_free_sparse_information>& basic_information,
      std::vector<std::vector<NNZ>>& sparse_data){

    chain.reserve(basic_information.size());

    for(std::size_t jacobian_idx = 0; jacobian_idx < basic_information.size(); jacobian_idx++){

      chain.emplace_back(std::move(basic_information[jacobian_idx]), sparse_data[jacobian_idx]);
      sparse_data[jacobian_idx].clear();
    }

    sparse_data.shrink_to_fit();
  }


  /**
   * @brief Parses file contents into sparse metadata and raw coordinate non-zero collections 
   * (Array of sparsity patterns).
   *
   * @param file_name Path to formatted data input file.
   * @param[in, out] basic_information Array populated with parsed sparse metadata.
   * @param[in, out] sparse_data Array populated with sparsity patterns.
   *
   * @throws std::runtime_error If file cannot be read or contains invalid formatting.
   */
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
