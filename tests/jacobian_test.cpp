#include <cassert>
#include <cctype>
#include <cstdint>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include "./../jacobian.hpp"
#include "./../util_structs.hpp"

void file_to_sparse_data(std::vector<std::size_t>& jacobian_information,
    std::vector<NNZ>& sparse_data, std::ifstream& file){  
  std::size_t number_of_edges = 100;

  std::string line;
  std::getline(file, line);
  auto start = line.find('[') + 1;
  auto end = line.find(']');
  std::string numbers = line.substr(start, end-start);
  std::istringstream iss(numbers);
  std::size_t input_dim, output_dim, number_of_non_zero_elements;
  iss >> input_dim >> output_dim >> number_of_non_zero_elements;
  jacobian_information.push_back(input_dim);
  jacobian_information.push_back(output_dim);
  jacobian_information.push_back(number_of_edges);
  jacobian_information.push_back(number_of_non_zero_elements);
  
  std::size_t row, column;
  sparse_data.reserve(number_of_non_zero_elements);
  while(std::getline(file, line) && std::isdigit(line[0])){
    std::istringstream iss_data(line);
    iss_data >> row >> column;
    sparse_data.emplace_back(row, column);
  }
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

void file_to_sparse_format(std::vector<std::size_t>& col_idx,
    std::vector<std::size_t>& row_ptr, std::vector<std::size_t>& row_idx,
    std::vector<std::size_t>& col_ptr,
    std::vector<std::vector<std::size_t>>& col_coloring,
    std::vector<std::vector<std::size_t>>& row_coloring,
    std::size_t& col_num_colors, std::size_t& row_num_colors,
    std::ifstream& file){
  
  std::string line;
  while(std::getline(file,line)){
    if(line == "column_idx:"){reading_outer_inner_array_data(col_idx, file, line);}

    if(line == "row_ptr:"){reading_outer_inner_array_data(row_ptr, file, line);}

    if(line == "row_idx:"){reading_outer_inner_array_data(row_idx, file, line);}

    if(line == "column_ptr:"){reading_outer_inner_array_data(col_ptr, file, line);}

    if(line == "column_number_colors:"){
      reading_color_data(col_num_colors, file, line);
      col_coloring.resize(col_num_colors);
    }

    if(line == "row_number_colors:"){
      reading_color_data(row_num_colors, file, line);
      row_coloring.resize(row_num_colors);
    }

    if(line == "column_coloring:"){reading_color_array_data(col_coloring, file, line);}
    
    if(line == "row_coloring:"){reading_color_array_data(row_coloring, file, line);}
  }
}

std::ifstream file_name_to_ifstream(const std::string file_name){
  std::string complete_file_name = "./sparse_jacobian_sample_tests/" + file_name;
  std::ifstream file;

  file.open(complete_file_name);
  if(!file.is_open()){
    throw std::runtime_error("There was a problem opening the file: " + file_name);
  }
  return file;
}

void reading_data(std::vector<std::size_t>& jacobian_information,
    std::vector<NNZ>& sparse_data, const std::string& file_name,
    std::vector<std::size_t>& column_idx, std::vector<std::size_t>& row_ptr,
    std::vector<std::size_t>& row_idx, std::vector<std::size_t>& column_ptr,
    std::vector<std::vector<std::size_t>>& column_coloring,
    std::vector<std::vector<std::size_t>>& row_coloring,
    std::size_t& column_number_colors, std::size_t& row_number_colors){

  jacobian_information.reserve(4);
  std::ifstream file_stream = file_name_to_ifstream(file_name);
  file_to_sparse_data(jacobian_information, sparse_data, file_stream);
  const std::size_t n = 0, m = 1, num_nnz = 3;
  column_idx.reserve(jacobian_information[num_nnz]);
  row_ptr.reserve(jacobian_information[m] + 1);
  row_idx.reserve(jacobian_information[num_nnz]);
  column_ptr.reserve(jacobian_information[n] + 1);
  file_to_sparse_format(column_idx, row_ptr, row_idx, column_ptr,
      column_coloring, row_coloring, column_number_colors,
      row_number_colors, file_stream);
}

bool are_vectors_equal(const std::vector<std::size_t>& arg_0, const std::vector<std::size_t>& arg_1){
  if(arg_0.size() != arg_1.size()){
    throw std::runtime_error("The vectors have different size.");
    return 0;
  }

  for(std::size_t i = 0; i < arg_0.size(); i++){
    if(arg_0[i] != arg_1[i]){
      return false;
    }
  }
  return true;
}

bool are_colorings_equal(const std::vector<std::vector<std::size_t>>& arg_0,
    const std::vector<std::vector<std::size_t>>& arg_1){
  if(arg_0.size() != arg_1.size()){
    throw std::runtime_error("The coloring vectors have different outer size.");
    return 0;
  }
  for(std::size_t color_idx = 0; color_idx < arg_0.size(); color_idx++){
    if(arg_0[color_idx].size() != arg_1[color_idx].size()){
      throw std::runtime_error("The coloring vectors at entry " + std::to_string(color_idx) +
          " have different size.");
    }
    for(std::size_t idx = 0; idx < arg_0[color_idx].size(); idx++){
      if(arg_0[color_idx][idx] != arg_1[color_idx][idx]){
        return false;
      }
    }
  }
  return true;
}

void sparse_jacobian_test(const std::string file_name){
  std::vector<std::size_t> jacobian_info;
  std::vector<NNZ> jacobian_sparse_data;
  std::vector<std::size_t> col_idx, row_ptr;
  std::vector<std::size_t> row_idx, col_ptr;
  std::vector<std::vector<std::size_t>> read_col_coloring;
  std::vector<std::vector<std::size_t>> read_row_coloring;
  std::size_t read_col_num_colors, read_row_num_colors;
  reading_data(jacobian_info, jacobian_sparse_data, file_name,
      col_idx, row_ptr, row_idx, col_ptr, read_col_coloring, read_row_coloring,
      read_col_num_colors, read_row_num_colors);
  Sparse_Jacobian sparse_jacobian{jacobian_info, jacobian_sparse_data};
  if(!are_vectors_equal(col_idx, sparse_jacobian.get_col_idx())){
    throw std::runtime_error("Entry mismatch in col_idx array.");
  }
  if(!are_vectors_equal(row_ptr, sparse_jacobian.get_row_ptr())){
    throw std::runtime_error("Entry mismatch in row_ptr array.");
  }
  if(!are_vectors_equal(row_idx, sparse_jacobian.get_row_idx())){
    throw std::runtime_error("Entry mismatch in row_idx array.");
  }
  if(!are_vectors_equal(col_ptr, sparse_jacobian.get_col_ptr())){
    throw std::runtime_error("Entry mismatch in col_ptr array.");
  }
  if(!are_colorings_equal(read_col_coloring, sparse_jacobian.get_column_coloring())){
    throw std::runtime_error("Entry mismatch in column coloring array.");
  }
  if(!are_colorings_equal(read_row_coloring, sparse_jacobian.get_row_coloring())){
    throw std::runtime_error("Entry mismatch in row coloring array.");
  }
  assert(read_col_num_colors == sparse_jacobian.col_num_colors());
  assert(read_row_num_colors == sparse_jacobian.row_num_colors());
}

void sparse_jacobian_test(Sparse_Jacobian sparse_jacobian, const std::string& file_name){
  std::vector<std::size_t> jacobian_info;
  std::vector<NNZ> jacobian_sparse_data;
  std::vector<std::size_t> col_idx, row_ptr;
  std::vector<std::size_t> row_idx, col_ptr;
  std::vector<std::vector<std::size_t>> read_col_coloring;
  std::vector<std::vector<std::size_t>> read_row_coloring;
  std::size_t read_col_num_colors, read_row_num_colors;
  reading_data(jacobian_info, jacobian_sparse_data, file_name,
      col_idx, row_ptr, row_idx, col_ptr, read_col_coloring, read_row_coloring,
      read_col_num_colors, read_row_num_colors);
  if(!are_vectors_equal(col_idx, sparse_jacobian.get_col_idx())){
    throw std::runtime_error("Entry mismatch in col_idx array.");
  }
  if(!are_vectors_equal(row_ptr, sparse_jacobian.get_row_ptr())){
    throw std::runtime_error("Entry mismatch in row_ptr array.");
  }
  if(!are_vectors_equal(row_idx, sparse_jacobian.get_row_idx())){
    throw std::runtime_error("Entry mismatch in row_idx array.");
  }
  if(!are_vectors_equal(col_ptr, sparse_jacobian.get_col_ptr())){
    throw std::runtime_error("Entry mismatch in col_ptr array.");
  }
  if(!are_colorings_equal(read_col_coloring, sparse_jacobian.get_column_coloring())){
    throw std::runtime_error("Entry mismatch in column coloring array.");
  }
  if(!are_colorings_equal(read_row_coloring, sparse_jacobian.get_row_coloring())){
    throw std::runtime_error("Entry mismatch in row coloring array.");
  }
  assert(read_col_num_colors == sparse_jacobian.col_num_colors());
  assert(read_row_num_colors == sparse_jacobian.row_num_colors());
}

void print_vector(std::vector<std::size_t>& arg_vector){
  for(std::size_t i=0; i < arg_vector.size(); i++){
    std::cout<< arg_vector[i] << ' ';
  }
  std::cout << '\n';
}

void print_coloring(const std::vector<std::vector<std::size_t>>& coloring){
  for(std::size_t color_idx = 0; color_idx < coloring.size(); color_idx++){
    std::cout << '[' << color_idx << "]: ";
    for(std::size_t idx = 0; idx < coloring[color_idx].size(); idx++)
      std::cout << idx << ' ';
  }
  std::cout << '\n';
}

int main(){
  {
    std::size_t input_dim = 5, output_dim = 10;
    std::vector<std::size_t> jac_initializer{input_dim, output_dim};
    Jacobian jac{jac_initializer};
    assert(jac.n() == input_dim);
    assert(jac.m() == output_dim);
  }

  {
    std::size_t single_argument = 4;
    std::vector<std::size_t> jac_initializer{single_argument};
    bool invalid_arg = 0;
    try{
      Jacobian jac{jac_initializer};
    }
    catch(const std::invalid_argument&){
      invalid_arg = 1;
    }
    assert(invalid_arg);
  }

  {
    std::size_t input_dim = 4, output_dim = 8, number_of_edges = 150;
    std::vector<std::size_t> jac_initializer{input_dim, output_dim, number_of_edges};
    Dense_Jacobian jac{jac_initializer};
    assert(jac.n_E() == 150);
  }

  {
    std::size_t first_argument = 10, second_argument = 15;
    std::vector<std::size_t> jac_initializer{first_argument, second_argument};
    bool invalid_arg =0;
    try{
      Dense_Jacobian ja{jac_initializer};
    }
    catch(const std::invalid_argument&){
      invalid_arg = 1;
    }
    assert(invalid_arg);
  }

  {
    const std::string file_name = "case_1_MxM";
    sparse_jacobian_test(file_name);
  }

  //Sparse_Jacobian multiplication test
  {
    std::vector<std::size_t> jac_info_lhs, jac_info_rhs;
    std::vector<NNZ> sparse_data_lhs, sparse_data_rhs;

    const std::string file_name_lhs = "case_2_MxM_lhs";
    std::ifstream lhs_file_stream = file_name_to_ifstream(file_name_lhs);
    const std::string file_name_rhs = "case_2_MxM_rhs";
    std::ifstream rhs_file_stream = file_name_to_ifstream(file_name_rhs);
    const std::string file_name_product = "case_2_MxM";

    file_to_sparse_data(jac_info_lhs, sparse_data_lhs, lhs_file_stream);
    Sparse_Jacobian sparse_jac_lhs{jac_info_lhs, sparse_data_lhs};

    file_to_sparse_data(jac_info_rhs, sparse_data_rhs, rhs_file_stream);
    Sparse_Jacobian sparse_jac_rhs{jac_info_rhs, sparse_data_rhs};

    Sparse_Jacobian sparse_product = sparse_jac_lhs * sparse_jac_rhs;
    sparse_jacobian_test(sparse_product, file_name_product);
  }
  
}
