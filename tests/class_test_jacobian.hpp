#include <string>
#include <cstdint>
#include <iostream>
#include <vector>
#include "./../jacobian.hpp"
#include "./../util_structs.hpp"

class test_Jacobian{
 public:
  test_Jacobian(std::size_t domain_dimension, std::size_t codomain_dimension):
    jacobian(domain_dimension, codomain_dimension){
      test_constructor = constructor_test(domain_dimension, codomain_dimension);
      if(!test_constructor){
        print_test_state();
      }
    }

  void print_test_state() const{
    std::cout << "State of test_Jacobian:\n";
    std::cout << "test_constructor: ";
    if(test_constructor){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}
    std::cout << '\n';
  }

 protected:
  Jacobian jacobian;
  bool test_constructor;
  bool constructor_test(std::size_t domain_dimension, std::size_t codomain_dimension){
    if(jacobian.domain_dim() != domain_dimension){
      return false;
    }
    if(jacobian.codomain_dim() != codomain_dimension){
      return false;
    }
    return true;
  }
};

class test_Dense_Jacobian{
 public:
  test_Dense_Jacobian(std::size_t domain_dimension, std::size_t codomain_dimension,
      std::size_t number_of_edges):
    jacobian(domain_dimension, codomain_dimension, number_of_edges){
      bool test_constructor =
        constructor_test(domain_dimension, codomain_dimension, number_of_edges);
      if(!test_constructor){
        print_test_state();
      }
    }

  void print_test_state() const{
    std::cout << "State of test_Jacobian:\n";
    std::cout << "test_constructor: ";
    if(test_constructor){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}
    std::cout << '\n';
  }

 protected:
  Dense_Jacobian jacobian;
  bool test_constructor;
  bool constructor_test(std::size_t domain_dimension, std::size_t codomain_dimension,
      std::size_t number_of_edges){
    if(jacobian.domain_dim() != domain_dimension){
      return false;
    }
    if(jacobian.codomain_dim() != codomain_dimension){
      return false;
    }
    if(jacobian.number_edges() != number_of_edges){
      return false;
    }
    return true;
  }
};

class test_Sparse_Jacobian{
 public:
  test_Sparse_Jacobian(const std::string& file_name):
    path_to_file(format_file_name(file_name)), 
    jacobian(Sparse_Jacobian::from_file(path_to_file + "_sparse_data")){

      test_compressed_format_invariants =
        compressed_format_invariants();

      std::vector<std::size_t> read_col_idx, read_row_ptr;
      std::vector<std::size_t> read_row_idx, read_col_ptr;

      std::vector<std::vector<std::size_t>> read_col_coloring;
      std::vector<std::vector<std::size_t>> read_row_coloring;

      std::size_t read_col_num_colors, read_row_num_colors;
      std::size_t read_max_nnz_row, read_max_nnz_col;

      read_validate_data(read_col_idx, read_row_ptr,
          read_row_idx, read_col_ptr, read_col_coloring,
          read_row_coloring, read_col_num_colors,
          read_row_num_colors, read_max_nnz_row,
          read_max_nnz_col);

      run_tests(read_col_idx, read_row_ptr,
          read_row_idx, read_col_ptr,
          read_col_coloring, read_row_coloring,
          read_col_num_colors, read_row_num_colors,
          read_max_nnz_row, read_max_nnz_col);
    }

  test_Sparse_Jacobian(const std::string& file_name,
      const Sparse_Jacobian& lhs, const Sparse_Jacobian& rhs):
    path_to_file(format_file_name(file_name)), jacobian(lhs*rhs){

      test_compressed_format_invariants =
        compressed_format_invariants();

      std::vector<std::size_t> read_col_idx, read_row_ptr;
      std::vector<std::size_t> read_row_idx, read_col_ptr;

      std::vector<std::vector<std::size_t>> read_col_coloring;
      std::vector<std::vector<std::size_t>> read_row_coloring;

      std::size_t read_col_num_colors, read_row_num_colors;
      std::size_t read_max_nnz_row, read_max_nnz_col;

      read_validate_data(read_col_idx, read_row_ptr,
          read_row_idx, read_col_ptr, read_col_coloring,
          read_row_coloring, read_col_num_colors,
          read_row_num_colors, read_max_nnz_row,
          read_max_nnz_col);

      run_tests(read_col_idx, read_row_ptr,
          read_row_idx, read_col_ptr,
          read_col_coloring, read_row_coloring,
          read_col_num_colors, read_row_num_colors,
          read_max_nnz_row, read_max_nnz_col);
    }


  void print_test_state() const{
    std::cout << "State of test_Sparse_Jacobian:\n";
    std::cout << "test_compressed_format_invariants: ";
    if(test_compressed_format_invariants){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_CSR_format: ";
    if(test_CSR_format){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_CSC_format: ";
    if(test_CSC_format){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_number_colors: ";
    if(test_number_colors){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_coloring: ";
    if(test_coloring){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "test_max_row_and_column: ";
    if(test_max_row_and_column){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}

    std::cout << "were_all_test_successful: ";
    if(were_all_test_successful){
      std::cout << "successful.\n";
    }
    else{std::cout << "failed.\n";}
    std::cout << '\n';
  }

  const Sparse_Jacobian& get_sparse_jacobian(){
    return jacobian;
  }

 protected:
  const std::string path_to_file;
  Sparse_Jacobian jacobian;
  bool test_compressed_format_invariants;
  bool test_CSR_format;
  bool test_CSC_format;
  bool test_number_colors;
  bool test_coloring;
  bool test_max_row_and_column;
  bool were_all_test_successful;

  static const std::string format_file_name(const std::string& file_name){
    if(file_name == "case_0" || file_name == "case_1" || file_name == "case_2" ||
        file_name == "case_0_MxM" || file_name == "case_1_MxM" || file_name == "case_2_MxM" ||
        file_name == "case_0_MxM_lhs" || file_name == "case_1_MxM_lhs" ||
        file_name == "case_2_MxM_lhs" || file_name == "case_0_MxM_rhs" ||
        file_name == "case_1_MxM_rhs" || file_name == "case_2_MxM_rhs"){
      std::string file_path = "./sparse_jacobian_sample_tests/" + file_name;
      return file_path;
    }
    return file_name;
  }

  bool is_ptr_array_increasing(const std::vector<size_t>& ptr_arr){
    for(std::size_t i = 0; i < ptr_arr.size() - 1; i++){
      if(ptr_arr[i+1] < ptr_arr[i]){
        return false;
      }
    }
    return true;
  }

  bool are_pointer_arrays_increasing(){
    if(is_ptr_array_increasing(jacobian.get_row_pointer_reference()) &&
        is_ptr_array_increasing(jacobian.get_column_pointer_reference())){

      return true;
    }

    return false;
  }

  bool compressed_format_invariants(){
    if(jacobian.column_idx_size() != jacobian.number_nnz()){
      return false;
    }
    if(jacobian.row_pointer_size() != jacobian.codomain_dim() + 1){
      return false;
    }
    if(jacobian.row_idx_size() != jacobian.number_nnz()){
      return false;
    }
    if(jacobian.column_pointer_size() != jacobian.domain_dim() + 1){
      return false;
    }
    if(!are_pointer_arrays_increasing()){
      return false;
    }
    return true;
  }

  void reading_outer_inner_array_data(std::vector<std::size_t>& outer_inner_array,
      std::ifstream& file, std::string& line){
    std::size_t entry;

    while(std::getline(file,line) && std::isdigit(line[0])){
      std::istringstream iss(line);
      while(iss >> entry){
        outer_inner_array.push_back(entry);
      }
    }
  }

  void reading_color_data(std::size_t& number_of_colors,
      std::ifstream&file, std::string& line){
    std::getline(file,line);
    number_of_colors = static_cast<std::size_t>(std::stoi(line));
  }

  void reading_color_array_data(std::vector<std::vector<std::size_t>>& color_arr,
      std::ifstream& file, std::string& line){
    std::size_t entry;
    std::size_t idx = 0;
    while(std::getline(file,line) && std::isdigit(line[0])){
      std::istringstream iss(line);
      while(iss >> entry){
        color_arr[idx].push_back(entry);
      }
      idx++;
    }
  }

  void read_validate_data(std::vector<std::size_t>& col_idx,
      std::vector<std::size_t>& row_ptr, std::vector<std::size_t>& row_idx,
      std::vector<std::size_t>& col_ptr,
      std::vector<std::vector<std::size_t>>& col_coloring,
      std::vector<std::vector<std::size_t>>& row_coloring,
      std::size_t& col_number_colors, std::size_t& row_number_colors,
      std::size_t& max_nnz_row, std::size_t& max_nnz_col){
    std::string path_to_validate_data = path_to_file + "_validate";

    std::ifstream file;
    file.open(path_to_validate_data);
    if(!file.is_open()){
      throw std::runtime_error("There was a problem opening "
          "the file: " + path_to_validate_data);
    }

    std::string line;
    while(std::getline(file,line)){
      if(line == "column_idx:"){reading_outer_inner_array_data(col_idx, file, line);}

      if(line == "row_ptr:"){reading_outer_inner_array_data(row_ptr, file, line);}

      if(line == "row_idx:"){reading_outer_inner_array_data(row_idx, file, line);}

      if(line == "column_ptr:"){reading_outer_inner_array_data(col_ptr, file, line);}

      if(line == "column_number_colors:"){
        reading_color_data(col_number_colors, file, line);
        col_coloring.resize(col_number_colors);
      }

      if(line == "row_number_colors:"){
        reading_color_data(row_number_colors, file, line);
        row_coloring.resize(row_number_colors);
      }

      if(line == "column_coloring:"){reading_color_array_data(col_coloring, file, line);}
      
      if(line == "row_coloring:"){reading_color_array_data(row_coloring, file, line);}

      if(line == "max_number_nnz_row:"){reading_color_data(max_nnz_row, file, line);}

      if(line == "max_number_nnz_column:"){reading_color_data(max_nnz_col, file, line);}
    }
  }

  bool are_vectors_equal(const std::vector<std::size_t>& arg_0,
      const std::vector<std::size_t>& arg_1){
    if(arg_0.size() != arg_1.size()){
      return 0;
    }

    for(std::size_t i = 0; i < arg_0.size(); i++){
      if(arg_0[i] != arg_1[i]){
        return false;
      }
    }
    return true;
  }

  //If colorings are equivalent then color assignation is equivalent up to order.
  // Example:
  // coloring_0: [0]: 0 1 [1]: 2
  // coloring_1: [0]: 1 0 [1]: 2
  // Coloring arrays are not equal but are indeed equivalent.
  bool are_colorings_equivalent(const std::vector<std::vector<std::size_t>>& arg_0,
      const std::vector<std::vector<std::size_t>>& arg_1){
    bool idx_found = false;
    if(arg_0.size() != arg_1.size()){
      return false;
    }
    for(std::size_t color_idx = 0; color_idx < arg_0.size(); color_idx++){
      if(arg_0[color_idx].size() != arg_1[color_idx].size()){
        return false;
      }

      for(std::size_t idx_0 = 0; idx_0 < arg_0[color_idx].size(); idx_0++){

        for(std::size_t idx_1 = 0; idx_1 < arg_1[color_idx].size(); idx_1++){

          if(arg_0[color_idx][idx_0] == arg_1[color_idx][idx_1]){
            idx_found = true;
            idx_1 = arg_1[color_idx].size();
          }

        }
        if(!idx_found){return false;}
        idx_found = false;
      }
    }
    return true;
  }

  void run_tests(const std::vector<std::size_t>& read_col_idx,
      const std::vector<std::size_t>& read_row_ptr,
      const std::vector<std::size_t>& read_row_idx,
      const std::vector<std::size_t>& read_col_ptr,
      const std::vector<std::vector<std::size_t>>& read_col_coloring,
      const std::vector<std::vector<std::size_t>>& read_row_coloring,
      const std::size_t read_col_num_colors, const std::size_t read_row_num_colors,
      const std::size_t read_max_nnz_row, const std::size_t read_max_nnz_col){

    if(are_vectors_equal(jacobian.get_column_idx_reference(),
          read_col_idx) &&
        are_vectors_equal(jacobian.get_row_pointer_reference(),
          read_row_ptr)){
      test_CSR_format = true;
    }
    else{test_CSR_format = false;}

    if(are_vectors_equal(jacobian.get_row_idx_reference(),
          read_row_idx) &&
        are_vectors_equal(jacobian.get_column_pointer_reference(),
          read_col_ptr)){
      test_CSC_format = true;
    }
    else{test_CSC_format = false;}

    if(read_row_num_colors == jacobian.get_row_number_colors() &&
        read_col_num_colors == jacobian.get_column_number_colors()){
      test_number_colors = true;
    }
    else{test_number_colors = false;}

    if(are_colorings_equivalent(read_row_coloring, jacobian.get_row_coloring()) &&
        are_colorings_equivalent(read_col_coloring, jacobian.get_column_coloring())){
      test_coloring = true;
    }
    else{test_coloring = false;}

    if(read_max_nnz_row == jacobian.get_max_number_nnz_row() &&
        read_max_nnz_col == jacobian.get_max_number_nnz_column()){
      test_max_row_and_column = true;
    }
    else{test_max_row_and_column = false;}

    if(test_compressed_format_invariants && test_CSR_format &&
        test_CSC_format && test_number_colors && test_coloring &&
        test_max_row_and_column){
      were_all_test_successful = true;
    }
    else{
      void print_test_state();
    }
  }

};
