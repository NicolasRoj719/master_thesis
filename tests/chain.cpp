#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <type_traits>
#include "./../util_structs.hpp"
#include "./../chain.hpp"
#include "./../jacobian.hpp"

// Test input text files are to be found under ./chain_test_cases/
// Validation files are to be found under ./chain_test_cases/validate/

//case_name e.g: case_0 or case_1
//jac_chain_info should be reserved to optimize code execution.
void parse_validation_non_sparse(std::vector<std::vector<std::size_t>>& jac_chain_info,
    const std::string& case_name){
  std::string complete_file_name = "./chain_test_cases/validate/" + case_name +
    "_jacobian_information";
  std::ifstream file;
  file.open(complete_file_name);
  if(!file.is_open()){
    throw std::runtime_error("There was a problem opening the file: " + case_name);
  }
  std::string line;
  std::istringstream iss;
  std::size_t jacobian_data;
  std::vector<std::size_t> jacobian_data_vector;
  //Depending on the jacobian type used 2, 3 or 4 entries are expected.
  jacobian_data_vector.reserve(4);
  
  while(std::getline(file, line)){
    iss.clear();
    iss.str(line);
    while(iss >> jacobian_data){
     jacobian_data_vector.push_back(jacobian_data); 
    }
    jac_chain_info.push_back(jacobian_data_vector);
    jacobian_data_vector.clear();
  }
  file.close();
}

void parse_sparse_format_line(std::vector<std::size_t>& outer_inner_array,
    std::ifstream& file, std::istringstream& iss, std::string& line){
  std::size_t entry;
  while(std::getline(file,line) && std::isdigit(line[0])){
    iss.clear();
    iss.str(line);
    while(iss >> entry){
      outer_inner_array.push_back(entry);
    }
  }
}

//file_name e.g: case_0_jacobian0
//main should reserve space for all arrays for a faster execution. 
void parse_jacobian_validation_sparse(const std::string& file_name,
    std::vector<std::size_t>& col_idx,
    std::vector<std::size_t>& row_ptr,
    std::vector<std::size_t>& row_idx,
    std::vector<std::size_t>& col_ptr){
  std::string complete_file_name = "./chain_test_cases/validate/" + file_name;
  std::ifstream file;
  file.open(complete_file_name);
  if(!file.is_open()){
    throw std::runtime_error("There was a problem opening the file: " + file_name);
  }
  std::string line;
  std::istringstream iss;
  while(std::getline(file, line)){
    if(line == "column_idx:"){
      parse_sparse_format_line(col_idx, file, iss, line);
    }
    if(line == "row_ptr:"){
      parse_sparse_format_line(row_ptr, file, iss, line);
    }
    if(line == "row_idx:"){
      parse_sparse_format_line(row_idx, file, iss, line);
    }
    if(line == "column_ptr:"){
      parse_sparse_format_line(col_ptr, file, iss, line);
    }
  }
}

bool are_vectors_equal(const std::vector<std::size_t>& vec_0,
    const std::vector<std::size_t>& vec_1){

  if(vec_0.size() != vec_1.size()){
    throw std::runtime_error("Error encountered while executing are_vectors_equal.\n"
        "vec_0 and vec_1 have different size.");
  }

  for(std::size_t i = 0; i<vec_0.size(); i++){
    if(vec_0[i] != vec_1[i]){
      return false;
    }
  }
  return true;
}

// read_vector contains all the jacobian information necessary to run all
// Jacobian types. Namely n, m, n_E, nnz.
bool are_vectors_equal_non_sparse(const std::vector<std::size_t>& read_vector,
    const std::vector<std::size_t>& obj_vector){
  for(std::size_t i = 0; i < obj_vector.size(); i++){
    if(read_vector[i] != obj_vector[i]){
      return false;
    }
  }
  return true;
}

bool is_sparse_jacobian_correctly_initialized(const Sparse_Jacobian& sparse_jacobian_obj,
    const std::vector<std::size_t>& col_idx,
    const std::vector<std::size_t>& row_ptr,
    const std::vector<std::size_t>& row_idx,
    const std::vector<std::size_t>& col_ptr){

  if(!are_vectors_equal(col_idx, sparse_jacobian_obj.get_col_idx())){
    return false;
  }
  if(!are_vectors_equal(row_ptr, sparse_jacobian_obj.get_row_ptr())){
    return false;
  }
  if(!are_vectors_equal(row_idx, sparse_jacobian_obj.get_row_idx())){
    return false;
  }
  if(!are_vectors_equal(col_ptr, sparse_jacobian_obj.get_col_ptr())){
    return false;
  }

  return true;
}

bool is_jacobian_information_correct(const Sparse_Jacobian& sparse_jacobian_obj,
    const std::vector<std::size_t>& col_idx,
    const std::vector<std::size_t>& row_ptr,
    const std::vector<std::size_t>& row_idx,
    const std::vector<std::size_t>& col_ptr){
  
  if(sparse_jacobian_obj.num_nnz() != col_idx.size()){return false;}
  if(sparse_jacobian_obj.num_nnz() != row_idx.size()){return false;}
  if(sparse_jacobian_obj.m() + 1 != row_ptr.size()){return false;}
  if(sparse_jacobian_obj.n() + 1 != col_ptr.size()){return false;}
  
  return true;
}

// The following test tests workflow 3. All the data is read from the input file.
// No jacobian_infomation vector of vectors is given.
//case_name e.g: case_0
bool is_sparse_chain_correctly_initialized_WF3(const std::string& case_name){

  std::string file_chain_sparse_data = "./chain_test_cases/" + case_name + "_sparse_WF3";
  std::string file_argument_name;
  std::vector<std::size_t> col_idx;
  std::vector<std::size_t> row_ptr;
  std::vector<std::size_t> row_idx;
  std::vector<std::size_t> col_ptr;
  jacobian_chain<Sparse_Jacobian> sparse_chain{file_chain_sparse_data};

  //n : domain real vector space dimension.
  //m : codomain real vector space dimension.
  //nnz : number of non zeros in the matrix representation.
  std::size_t n, m, nnz;
  for(std::size_t jac_idx = 0; jac_idx < sparse_chain.size(); jac_idx++){
    n = sparse_chain[jac_idx].n();
    m = sparse_chain[jac_idx].m();
    nnz = sparse_chain[jac_idx].num_nnz();
    //resources allocation
    col_idx.clear();
    col_idx.reserve(nnz);
    row_ptr.clear();
    row_ptr.reserve(m + 1);
    row_idx.clear();
    row_idx.reserve(nnz);
    col_ptr.clear();
    col_ptr.reserve(n + 1);

    file_argument_name = case_name + "_jacobian" + std::to_string(jac_idx);
    // Reading data from validation files. 
    parse_jacobian_validation_sparse(file_argument_name,
        col_idx, row_ptr, row_idx, col_ptr);

    if(!is_sparse_jacobian_correctly_initialized(sparse_chain[jac_idx],
          col_idx, row_ptr, row_idx, col_ptr)){
      return false;
    }  

    if(!is_jacobian_information_correct(sparse_chain[jac_idx],
          col_idx, row_ptr, row_idx, col_ptr)){
      return false;
    }
  }
  return true;
}

bool is_sparse_chain_correctly_initialized_WF2(const std::string& case_name){

  std::vector<std::vector<std::size_t>> jacobian_chain_information;
  std::string constructor_file_name = "./chain_test_cases/" + case_name +
    "_sparse_WF2";
  parse_validation_non_sparse(jacobian_chain_information, case_name);
  jacobian_chain<Sparse_Jacobian> sparse_chain{jacobian_chain_information,
    constructor_file_name};

  std::string file_argument_name;
  std::vector<std::size_t> col_idx;
  std::vector<std::size_t> row_ptr;
  std::vector<std::size_t> row_idx;
  std::vector<std::size_t> col_ptr;
  
  //n : domain real vector space dimension.
  //m : codomain real vector space dimension.
  //nnz : number of non zeros in the matrix representation.
  std::size_t n, m, nnz;
  for(std::size_t jac_idx = 0; jac_idx < sparse_chain.size(); jac_idx++){
    n = sparse_chain[jac_idx].n();
    m = sparse_chain[jac_idx].m();
    nnz = sparse_chain[jac_idx].num_nnz();
    //resources allocation
    col_idx.clear();
    col_idx.reserve(nnz);
    row_ptr.clear();
    row_ptr.reserve(m + 1);
    row_idx.clear();
    row_idx.reserve(nnz);
    col_ptr.clear();
    col_ptr.reserve(n + 1);

    file_argument_name = case_name + "_jacobian" + std::to_string(jac_idx);
    // Reading data from validation files. 
    parse_jacobian_validation_sparse(file_argument_name,
        col_idx, row_ptr, row_idx, col_ptr);

    if(!is_sparse_jacobian_correctly_initialized(sparse_chain[jac_idx],
          col_idx, row_ptr, row_idx, col_ptr)){
      return false;
    }  
    if(!is_jacobian_information_correct(sparse_chain[jac_idx],
          col_idx, row_ptr, row_idx, col_ptr)){
      return false;
    }
  }
  return true;
}

// case_name e.g: case_0
template <class Jacobian_type>
bool is_non_sparse_chain_correctly_initialized(const std::string& case_name){

  static_assert(std::is_same_v<Jacobian_type, Jacobian> || 
      std::is_same_v<Jacobian_type, Dense_Jacobian>);

  std::string chain_constructor_file_name = "./chain_test_cases/" + case_name + 
    "_non_sparse";

  jacobian_chain<Jacobian_type> non_sparse_chain{chain_constructor_file_name};
  std::vector<std::vector<std::size_t>> jacobian_chain_information;
  jacobian_chain_information.reserve(non_sparse_chain.size());
  parse_validation_non_sparse(jacobian_chain_information, case_name);

  for(std::size_t jac_idx = 0; jac_idx < non_sparse_chain.size(); jac_idx++){
    if(!are_vectors_equal_non_sparse(jacobian_chain_information[jac_idx],
        non_sparse_chain[jac_idx].jacobian_information())){
      return false;
    }
  }
  return true;
}

int main(){
  //Testing non sparse chain contructor using input file with Jacobian as Jacobian type.
  {
    const std::string case_name_ = "case_0";
    assert(is_non_sparse_chain_correctly_initialized<Jacobian>(case_name_));
    const std::string case_name_1 = "case_1";
    assert(is_non_sparse_chain_correctly_initialized<Jacobian>(case_name_1));
  } 

  //Testing non sparse chain contructor using input file with Dense_Jacobian as Jacobian type.
  {
    const std::string case_name_ = "case_0";
    assert(is_non_sparse_chain_correctly_initialized<Dense_Jacobian>(case_name_));
    const std::string case_name_1 = "case_1";
    assert(is_non_sparse_chain_correctly_initialized<Dense_Jacobian>(case_name_1));
  } 

  //Testing sparse chain constructor only with the input file data (Workflow 3)
  {
    const std::string case_name_ = "case_0";
    assert(is_sparse_chain_correctly_initialized_WF3(case_name_));
    const std::string case_name_1 = "case_1";
    assert(is_sparse_chain_correctly_initialized_WF3(case_name_1));
  }

  //Testing sparse chain constructor only with the input file data (Workflow 2)
  {
    const std::string case_name_ = "case_0";
    assert(is_sparse_chain_correctly_initialized_WF2(case_name_));
    const std::string case_name_1 = "case_1";
    assert(is_sparse_chain_correctly_initialized_WF2(case_name_1));
  }

  //Testing dimension mismatch error.
  {
    const std::string file_prefix = "./chain_test_cases/";
    const std::string file_suffix = "_dimension_mismatch";
    bool is_exception_thrown = 0;

    std::string case_name = "case_0_non_sparse";
    std::string complete_file_name = file_prefix + case_name + file_suffix;
    
    try{
      jacobian_chain<Jacobian> jac_chain{complete_file_name};
    }
    catch(const std::runtime_error& e){
      is_exception_thrown = 1;
    }
    assert(is_exception_thrown);
    is_exception_thrown = 0;

    case_name = "case_0_sparse_WF3";
    complete_file_name = file_prefix + case_name + file_suffix;
    try{
      jacobian_chain<Sparse_Jacobian> jac_chain{complete_file_name};
    }
    catch(const std::runtime_error& e){
      is_exception_thrown = 1;
    }
    assert(is_exception_thrown);
    is_exception_thrown = 0;

    case_name = "case_1_non_sparse";
    complete_file_name = file_prefix + case_name + file_suffix;
    try{
      jacobian_chain<Dense_Jacobian> jac_chain{complete_file_name};
    }
    catch(const std::runtime_error& e){
      is_exception_thrown = 1;
    }
    assert(is_exception_thrown);
    is_exception_thrown = 0;

    case_name = "case_1_sparse_WF3";
    complete_file_name = file_prefix + case_name + file_suffix;
    try{
      jacobian_chain<Sparse_Jacobian> jac_chain{complete_file_name};
    }
    catch(const std::runtime_error& e){
      is_exception_thrown = 1;
    }
    assert(is_exception_thrown);
    is_exception_thrown = 0;

  }
}
