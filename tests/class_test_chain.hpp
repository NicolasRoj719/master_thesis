#include <string>
#include <utility>
#include <vector>
#include "./../chain.hpp"
/* #include "./../test.hpp" */
#include "./../util_structs.hpp"
#include "./../jacobian.hpp"
#include "./../generator.hpp"

//Two validation methods are implemented.
//First: the chain object is created from an input text file.
//Second: the interaction between generator and chain is tested
// a generator is instantiated and from the generated data the chain
// object is constructed.

//To implement new test cases:
//First method: provide the path to data_file and validate_file that
//follows the format. Check the provided test cases to replicate the
//format.
//Second method: A directory ./generator_chain_cases/ provides an implementation
// .cpp that generates data and provides a validate file.
// For the simple case this is done automatically. The only modification required
// is below when the generator is instantiated. Generator instances must match.
// If the sparse case is to be tested then the validate files must be provided by
// the user.
template <class Jacobian_T, class Basic_info_T>
class test_chain{
 public:
  test_chain(const std::string& case_name, const std::string& validate_name){

    jacobian_chain<Jacobian_T, Basic_info_T> chain(case_to_path_with_data(case_name));

    test_constructor =
      is_data_from_file_equal_data_from_chain(case_to_validate_path(validate_name),
          chain);

    if(!test_constructor){
      print_test_state();
    }
  }

  test_chain(){
    if constexpr(std::is_same_v<Basic_info_T, Jacobian_information>){
      const std::size_t chain_length = 3;
      const std::size_t dimension_lower_bound = 2, dimension_upper_bound = 5;
      const bool is_deterministic = true;
      const std::size_t seed = 54;

      Generator<Jacobian_information> generator{chain_length, dimension_lower_bound,
      dimension_upper_bound, is_deterministic, seed};

      jacobian_chain<Jacobian_T, Basic_info_T> 
        chain{generator.generate_jacobian_information()};

      test_constructor =
        is_data_from_file_equal_data_from_chain(
            case_to_validate_path_generator(), chain);

      if(!test_constructor){
        print_test_state();
      }
    }

    else{
      const std::size_t chain_length = 3;
      const std::size_t dimension_lower_bound = 2, dimension_upper_bound = 5;
      const std::size_t number_edges_lower_bound = 50, number_edges_upper_bound = 120;
      const bool is_deterministic = true;
      const std::size_t seed = 60;

      Generator<Matrix_free_information> generator{chain_length, dimension_lower_bound,
      dimension_upper_bound, number_edges_lower_bound, number_edges_upper_bound,
      is_deterministic, seed};

      jacobian_chain<Jacobian_T, Basic_info_T> 
        chain{generator.generate_jacobian_information()};

      test_constructor =
        is_data_from_file_equal_data_from_chain(
            case_to_validate_path_generator(), chain);

      if(!test_constructor){
        print_test_state();
      }
    }
  }

  void print_test_state(){
    std::cout<<"State of jacobian chain test:\n";
    std::cout<<"test constructor: ";
    if(test_constructor){
      std::cout<<"successful.\n";
    }
    else{
      std::cout<<"failed.\n";
    }
    std::cout << '\n';
  }

 protected:
  bool test_constructor;

  std::string case_to_path_with_data(const std::string& case_name);

  std::string case_to_validate_path(const std::string& case_name);

  std::string case_to_validate_path_generator();

  bool is_data_from_file_equal_data_from_chain(const std::string& path_to_validate,
      const jacobian_chain<Jacobian_T, Basic_info_T>& chain);
};

template<class Jacobian_T,class Basic_info_T>
std::string test_chain<Jacobian_T, Basic_info_T>::case_to_path_with_data(
    const std::string& case_name){
  if(case_name == "case_0"){
    if constexpr (std::is_same_v<Basic_info_T, Jacobian_information>){
      return "./chain_test_cases/"+ case_name +
        "_non_sparse_jacobian_information";
    }

    else{
      return "./chain_test_cases/"+ case_name +
        "_non_sparse_matrix_free";
    }
  }

  if(case_name == "case_1"){
    if constexpr (std::is_same_v<Basic_info_T, Jacobian_information>){
      return "./chain_test_cases/"+ case_name +
        "_non_sparse_jacobian_information";
    }

    else{
      return "./chain_test_cases/"+ case_name +
        "_non_sparse_matrix_free";
    }
  }

  //This allows to use the test object with a new path.
  else{return case_name;}
}

template <class Jacobian_T, class Basic_info_T>
std::string test_chain<Jacobian_T, Basic_info_T>::case_to_validate_path(
    const std::string& case_name){

  if(case_name == "case_0"){
    if constexpr (std::is_same_v<Basic_info_T, Jacobian_information>){
      return "./chain_test_cases/validate/"+ case_name+
        "_non_sparse_jacobian_information";
    }

    else{
      return "./chain_test_cases/validate/"+ case_name+
        "_non_sparse_matrix_free";
    }
  }

  if(case_name == "case_1"){
    if constexpr (std::is_same_v<Basic_info_T, Jacobian_information>){
      return "./chain_test_cases/validate/"+ case_name+
        "_non_sparse_jacobian_information";
    }

    else{
      return "./chain_test_cases/validate/"+ case_name+
        "_non_sparse_matrix_free";
    }
  }

  //This allows to use the test object with a different path.
  else{return case_name;}
}

template <class Jacobian_T, class Basic_info_T>
std::string test_chain<Jacobian_T, Basic_info_T>::case_to_validate_path_generator(){

  if constexpr(std::is_same_v<Basic_info_T, Jacobian_information>){
    return "./generator_chain_cases/validate/jacobian_information";
  }

  else{ 
    return "./generator_chain_cases/validate/matrix_free_information";
  }

}

template <class Jacobian_T, class Basic_info_T>
bool test_chain<Jacobian_T, Basic_info_T>::is_data_from_file_equal_data_from_chain(
  const std::string& path_to_validate,
  const jacobian_chain<Jacobian_T, Basic_info_T>& chain){

  std::ifstream file;
  file.open(path_to_validate);
  if(!file.is_open()){
    throw std::runtime_error("There was a problem opening the file: "+
        path_to_validate);
  }

  std::string line;
  std::istringstream iss;
  std::size_t domain_dimension, codomain_dimension;
  std::size_t number_edges;
  std::size_t jacobian_index = 0;

  while(std::getline(file, line)){
    //Validate format flexibility: blank lines between lines with data are allowed.
    if(std::isdigit(line[0])){
      iss.clear();
      iss.str(line);
      if constexpr(std::is_same_v<Basic_info_T, Jacobian_information>){
        iss >> domain_dimension >> codomain_dimension;
        if(domain_dimension != chain.at(jacobian_index).domain_dim() ||
            codomain_dimension != chain.at(jacobian_index).codomain_dim()){
          return false;
        }
      }

      else{
        iss >> domain_dimension >> codomain_dimension >> number_edges;
        if(domain_dimension != chain.at(jacobian_index).domain_dim() ||
            codomain_dimension != chain.at(jacobian_index).codomain_dim() ||
            number_edges != chain.at(jacobian_index).number_edges()){
          return false;
        }
      }

      jacobian_index++;
    }
  }

  if(jacobian_index != chain.size()){
    throw std::runtime_error("Error encountered while testing the jacobian chain.\n"
      "Less data in validate file than the one stored in the jacobian chain.");
  }

  return true;
}

template<>
class test_chain<Sparse_Jacobian, Matrix_free_sparse_information>{
 public:
  test_chain(const std::string& case_name, const std::string& validate_name){
    jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>
      chain{case_to_path_with_data(case_name)};

    test_jacobian_information = 
      is_jacobian_information_correctly_initialized(
          validate_name, chain);

    test_compress_format =
      are_CSR_CSC_formats_correctly_initialized(
          validate_name, chain);

    if(test_jacobian_information && test_compress_format
        == true){
      were_all_tests_successful = true;
    }
    else{
      were_all_tests_successful = false;
      print_test_state();
    }
  }

  test_chain(){
    //Constructing generator.
    const std::size_t chain_length = 3;
    const std::size_t dimension_lb = 2, dimension_ub = 5;
    const std::size_t number_edges_lb = 80, number_edges_ub = 160;
    const double density_lb = 0.2, density_ub = 0.5;
    const bool is_deterministic = true;
    const std::size_t seed = 26;

    Generator<Matrix_free_sparse_information>
      generator{chain_length, dimension_lb, dimension_ub, number_edges_lb, number_edges_ub,
      density_lb, density_ub, is_deterministic, seed};

    jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information> 
      chain{generator.generate_data()};

    test_jacobian_information =
      is_jacobian_information_correctly_initialized_generator(chain);

    test_compress_format =
      are_CSR_CSC_formats_correctly_initialized_generator(chain);

    if(test_jacobian_information && test_compress_format
        == true){
      were_all_tests_successful = true;
    }
    else{
      were_all_tests_successful = false;
      print_test_state();
    }
  }

  void print_test_state(){
    std::cout << "State of jacobian chain test:\n";

    std::cout << "test_jacobian_information: ";
    if(test_jacobian_information){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_compress_format: ";
    if(test_compress_format){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << '\n';  
  }
 
 protected:
  std::string case_to_path_with_data(const std::string& case_name);

  std::string case_to_validate_path(const std::string& validate_name,
      std::size_t jacobian_index);

  std::string case_to_validate_generator(std::size_t jacobian_index);

  bool is_jacobian_information_correctly_initialized(
      const std::string& path_validate,
      const Sparse_Jacobian& sparse_jacobian);

  bool is_jacobian_information_correctly_initialized(
      const std::string& validate_name,
      const jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>& chain);

  bool is_jacobian_information_correctly_initialized_generator(
      const jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>& chain);


  bool is_array_data_equal_validate_data(
      std::ifstream& file, std::string& line,
      const std::vector<std::size_t>& array_);

  bool are_CSR_CSC_formats_correctly_initialized(
      const std::string& path_validate,
      const Sparse_Jacobian& sparse_jacobian);

  bool are_CSR_CSC_formats_correctly_initialized(
      const std::string& validate_name,
      const jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>& chain);

  bool are_CSR_CSC_formats_correctly_initialized_generator(
      const jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>& chain);

  bool test_jacobian_information;

  bool test_compress_format;

  bool were_all_tests_successful;
};

std::string test_chain<Sparse_Jacobian, Matrix_free_sparse_information>::
  case_to_path_with_data(const std::string& case_name){
  
  if(case_name == "case_0"){
    return "./chain_test_cases/case_0_sparse";
  }

  else if(case_name == "case_1"){
    return "./chain_test_cases/case_1_sparse";
  }

  else{return case_name;}
}

std::string test_chain<Sparse_Jacobian, Matrix_free_sparse_information>::
  case_to_validate_path(const std::string& validate_name,
      std::size_t jacobian_index){
  
  if(validate_name == "case_0"){
    return "./chain_test_cases/validate/case_0_sparse_jacobian"+
      std::to_string(jacobian_index);
  }

  else if(validate_name == "case_1"){
    return "./chain_test_cases/validate/case_1_sparse_jacobian"+
      std::to_string(jacobian_index);
  }

  else{return validate_name;}
}

std::string test_chain<Sparse_Jacobian, Matrix_free_sparse_information>::
  case_to_validate_generator(std::size_t jacobian_index){
  
  return "./generator_chain_cases/validate/sparse_" +
    std::to_string(jacobian_index);
}

bool test_chain<Sparse_Jacobian, Matrix_free_sparse_information>::
  is_jacobian_information_correctly_initialized(
      const std::string& path_validate,
      const Sparse_Jacobian& sparse_jacobian){
  
  std::ifstream file;
  file.open(path_validate);
  if(!file.is_open()){
    throw std::runtime_error("There was a problem opening the file: "+
        path_validate);
  }

  std::string line;
  std::istringstream iss;
  std::size_t domain_dimension, codomain_dimension;
  std::size_t number_edges, number_nnz;

  while(std::getline(file, line)){

    if(line == "Basic_information:"){
      
      std::getline(file, line);
      iss.clear();
      iss.str(line);

      iss >> domain_dimension >> codomain_dimension;
      iss >> number_edges >> number_nnz;

      if(sparse_jacobian.domain_dim() != domain_dimension ||
          sparse_jacobian.codomain_dim() != codomain_dimension ||
          sparse_jacobian.number_edges() != number_edges ||
          sparse_jacobian.number_nnz() != number_nnz){

        return false;
      }

      else{return true;}

    }
  
  }

  throw std::runtime_error("Error encountered while comparing "
    "the jacobian chain with validate data.\n"
    "No Basic_information section found in the validate file.");

  return false;
}

bool test_chain<Sparse_Jacobian, Matrix_free_sparse_information>::
  is_jacobian_information_correctly_initialized(
      const std::string& validate_name,
      const jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>& chain){

  for(std::size_t jacobian_idx = 0; jacobian_idx < chain.size();
      jacobian_idx++){
    
    if(!is_jacobian_information_correctly_initialized(
          case_to_validate_path(validate_name, jacobian_idx),
          chain[jacobian_idx])){
      
      return false;
    }

  }
  return true;
}

bool test_chain<Sparse_Jacobian, Matrix_free_sparse_information>::
  is_jacobian_information_correctly_initialized_generator(
    const jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>& chain){
  
  for(std::size_t jacobian_idx = 0; jacobian_idx < chain.size();
      jacobian_idx++){
    
    if(!is_jacobian_information_correctly_initialized(
          case_to_validate_generator(jacobian_idx),
          chain[jacobian_idx])){
      
      return false;
    }

  }
  return true;
}

bool test_chain<Sparse_Jacobian, Matrix_free_sparse_information>::
  is_array_data_equal_validate_data(
      std::ifstream& file, std::string& line,
      const std::vector<std::size_t>& array_){
  
  std::getline(file, line);
  std::istringstream iss(line);
  std::size_t data;
  std::size_t counter = 0;
  
  while(iss >> data){

    if(data != array_[counter]){
      return false;
    }

    counter++;
  }

  return true;
}

bool test_chain<Sparse_Jacobian, Matrix_free_sparse_information>::
  are_CSR_CSC_formats_correctly_initialized(
      const std::string& path_validate,
      const Sparse_Jacobian& sparse_jacobian){

  std::ifstream file;
  file.open(path_validate);
  if(!file.is_open()){
    throw std::runtime_error("There was a problem opening the file: "+
        path_validate);
  }

  std::string line;
  std::istringstream iss;

  while(std::getline(file, line)){
    if(line == "column_idx:"){
      
      if(!is_array_data_equal_validate_data(file, line,
          sparse_jacobian.get_column_idx_reference())){

        return false;
      }
    }

    else if(line == "row_ptr:"){

      if(!is_array_data_equal_validate_data(file, line,
          sparse_jacobian.get_row_pointer_reference())){

        return false;
      }

    }

    else if(line == "row_idx:"){

      if(!is_array_data_equal_validate_data(file, line,
          sparse_jacobian.get_row_idx_reference())){

        return false;
      }

    }

    else if(line == "column_ptr:"){

      if(!is_array_data_equal_validate_data(file, line,
          sparse_jacobian.get_column_pointer_reference())){

        return false;
      }

    }
  
  }
  return true;
}

bool test_chain<Sparse_Jacobian, Matrix_free_sparse_information>::
  are_CSR_CSC_formats_correctly_initialized(
      const std::string& validate_name,
      const jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>& chain){

  for(std::size_t jacobian_idx = 0; jacobian_idx < chain.size();
      jacobian_idx++){

    if(!are_CSR_CSC_formats_correctly_initialized(
       case_to_validate_path(validate_name, jacobian_idx),
        chain[jacobian_idx])){
      
      return false;
    }
  }

  return true;
}

bool test_chain<Sparse_Jacobian, Matrix_free_sparse_information>::
  are_CSR_CSC_formats_correctly_initialized_generator(
      const jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>& chain){

  for(std::size_t jacobian_idx = 0; jacobian_idx < chain.size();
      jacobian_idx++){

    if(!are_CSR_CSC_formats_correctly_initialized(
        case_to_validate_generator(jacobian_idx),
        chain[jacobian_idx])){
      
      return false;
    }
  }

  return true;
}
