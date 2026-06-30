#include <cstdint>
#include <iostream>
#include <vector>
#include <cassert>
#include <string>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <algorithm>
#include <numeric>
#include "util_structs.hpp"

#ifndef JACOBIAN_HPP
#define JACOBIAN_HPP

class Jacobian{
 public:
  Jacobian(Jacobian_information jacobian_obj):
    jacobian_basic_data(std::move(jacobian_obj)){}

  Jacobian(std::size_t domain_dimension, std::size_t codomain_dimension):
    jacobian_basic_data(domain_dimension, codomain_dimension){}

  std::size_t domain_dim() const {
    return jacobian_basic_data.domain_dimension();
  }
  std::size_t codomain_dim() const {
    return jacobian_basic_data.codomain_dimension();
  }

 protected:
  Jacobian_information jacobian_basic_data;
};

class Dense_Jacobian{
 public:
  Dense_Jacobian(Matrix_free_infomation jacobian_obj):
    jacobian_basic_data(std::move(jacobian_obj)){}

  Dense_Jacobian(std::size_t domain_dimension, std::size_t codomain_dimension,
      std::size_t number_of_edges):
    jacobian_basic_data(domain_dimension, codomain_dimension, number_of_edges){}

  std::size_t domain_dim() const {
    return jacobian_basic_data.domain_dimension();
  }
  std::size_t codomain_dim() const {
    return jacobian_basic_data.codomain_dimension();
  }
  std::size_t number_edges() const {
    return jacobian_basic_data.number_of_edges();
  }

 protected:
  Matrix_free_infomation jacobian_basic_data;
};

class Sparse_Jacobian{
 public:
  Sparse_Jacobian(Matrix_free_sparse_information jacobian_obj,
      std::vector<NNZ>& sparse_data):
    jacobian_basic_data(std::move(jacobian_obj)){

      if(number_nnz() != sparse_data.size()){
        throw std::invalid_argument("Number of non zero entries and "
            " sparse data size do not match.");
      }

      from_sparse_data_initializer(sparse_data);
    }

  Sparse_Jacobian(std::size_t domain_dimension, std::size_t codomain_dimension,
      std::size_t number_edges, std::size_t number_nnz_,
      std::vector<NNZ>& sparse_data):
    jacobian_basic_data{domain_dimension, codomain_dimension, number_edges, number_nnz_}{

      if(number_nnz() != sparse_data.size()){
        throw std::invalid_argument("Number of non zero entries and "
            " sparse data size do not match.");
      }

      from_sparse_data_initializer(sparse_data);
    }


  Sparse_Jacobian(std::size_t domain_dimension, std::size_t codomain_dimension,
      std::size_t number_of_edges, std::size_t number_nonzeros,
      std::vector<std::size_t> col_idx, std::vector<std::size_t> row_ptr,
      std::vector<std::size_t> row_idx_, std::vector<std::size_t> col_ptr):
    jacobian_basic_data{domain_dimension, codomain_dimension, number_of_edges, number_nonzeros},
    column_idx(std::move(col_idx)), row_pointer(std::move(row_ptr)),
    row_idx(std::move(row_idx_)), column_pointer(std::move(col_ptr)){

      if((column_idx.size() != number_nnz()) ||
          (row_idx.size() != number_nnz())){
        throw std::invalid_argument("Dimension mismatch.\n"
        "column_idx and/or row_idx have less entries than number of zeros\n"
        "promised by the jacobian information.");
      }

      if(column_pointer.size() != (domain_dim() + 1)){
        throw std::invalid_argument("Dimension mismatch.\n"
        "column_pointer size is not equal to the dimension of the domain "
        "plus one.");
      }

      if(row_pointer.size() != (codomain_dim() + 1)){
        throw std::invalid_argument("Dimension mismatch.\n"
        "column_pointer size is not equal to the dimension of the codomain "
        "plus one.");
      }

      if((!is_ptr_array_increasing(row_pointer)) ||
          (!is_ptr_array_increasing(column_pointer))){
        throw std::invalid_argument("row_pointer and/or column "
            "pointers are non increasing.");
      }

      increasing_index_array(column_idx, row_pointer);

      increasing_index_array(row_idx, column_pointer);

      coloring_algorithm(column_idx, row_pointer, column_coloring,
          domain_dim(), column_number_colors);

      coloring_algorithm(row_idx, column_pointer, row_coloring,
          codomain_dim(), row_number_colors);

      max_number_nnz_row = max_nnz_row_or_col(column_pointer);
      max_number_nnz_column = max_nnz_row_or_col(row_pointer);
    }
  
  std::size_t domain_dim() const {
    return jacobian_basic_data.domain_dimension();
  }
  std::size_t codomain_dim() const {
    return jacobian_basic_data.codomain_dimension();
  }
  std::size_t number_edges() const {
    return jacobian_basic_data.number_of_edges();
  }
  std::size_t number_nnz() const {
    return jacobian_basic_data.number_of_nonzeros();
  }

  std::size_t column_idx_size() const{
    return column_idx.size();
  }

  std::size_t row_pointer_size() const{
    return row_pointer.size();
  }

  std::size_t row_idx_size() const{
    return row_idx.size();
  }

  std::size_t column_pointer_size() const{
    return column_pointer.size();
  }

  const std::vector<std::size_t>& get_column_idx_reference() const{
    return column_idx;
  }
  
  const std::vector<std::size_t>& get_row_pointer_reference() const{
    return row_pointer;
  }

  const std::vector<std::size_t>& get_row_idx_reference() const{
    return row_idx;
  }

  const std::vector<std::size_t>& get_column_pointer_reference() const{
    return column_pointer;
  }

  const std::vector<std::vector<std::size_t>>& get_column_coloring() const{
    return column_coloring;
  }

  const std::vector<std::vector<std::size_t>>& get_row_coloring() const{
    return row_coloring;
  }

  std::size_t get_column_number_colors() const{
    return column_number_colors;
  }

  std::size_t get_row_number_colors() const{
    return row_number_colors;
  }

  std::size_t get_max_number_nnz_row() const {
    return max_number_nnz_row;
  }

  std::size_t get_max_number_nnz_column() const {
    return max_number_nnz_column;
  }


  static Sparse_Jacobian from_file(const std::string& file_name){
    std::size_t domain_dim, codomain_dim;
    std::size_t number_edges, number_nnz;
    std::vector<NNZ> sparse_data;
    sparse_file_to_sparse_data(file_name, domain_dim, codomain_dim,
        number_edges, number_nnz, sparse_data);
    return Sparse_Jacobian(domain_dim, codomain_dim, number_edges,
        number_nnz, sparse_data);
  }

  friend Sparse_Jacobian operator*(const Sparse_Jacobian& lhs, const Sparse_Jacobian& rhs);

  friend void mul_CSR_CSC_2_CSR(const Sparse_Jacobian& lhs, const Sparse_Jacobian& rhs, std::vector<size_t>& col_idx,
      std::vector<std::size_t>& row_ptr);

  friend void mul_CSR_CSC_2_CSC(const Sparse_Jacobian& lhs, const Sparse_Jacobian& rhs, std::vector<size_t>& row_idx,
      std::vector<std::size_t>& col_ptr);

 protected:
  Matrix_free_sparse_information jacobian_basic_data;
  std::vector<std::size_t> column_idx, row_pointer;
  std::vector<std::size_t> row_idx, column_pointer;
  std::vector<std::vector<std::size_t>> column_coloring;
  std::vector<std::vector<std::size_t>> row_coloring;

  std::size_t column_number_colors;
  std::size_t row_number_colors;

  std::size_t max_number_nnz_row;
  std::size_t max_number_nnz_column;

  void sparse_data_non_repeated_entries(
      std::vector<NNZ>& sparse_data) const{
    std::sort(sparse_data.begin(), sparse_data.end());
    sparse_data.erase(
        std::unique(sparse_data.begin(), sparse_data.end()),
        sparse_data.end());
  }

  void build_CSR_CSC_format(const std::vector<NNZ>& sparse_data){
    row_pointer.assign(codomain_dim() + 1, 0);
    column_pointer.assign(domain_dim() + 1, 0);

    //Counting
    for(std::size_t nnz_idx = 0; nnz_idx < number_nnz();
        nnz_idx++){
      column_pointer[sparse_data[nnz_idx].col() + 1]++;
      row_pointer[sparse_data[nnz_idx].row() + 1]++;
    }
    //Accumulate row pointer array
    for(std::size_t i = 0; i < codomain_dim(); i++){
      row_pointer[i+1] = row_pointer[i] + row_pointer[i+1];
    }

    //Accumulate column pointer array
    for(std::size_t i = 0; i < domain_dim(); i++){
      column_pointer[i+1] = column_pointer[i] + column_pointer[i+1];
    }

    row_idx.resize(number_nnz());
    column_idx.resize(number_nnz());

    auto aux_column_pointer = column_pointer;
    auto aux_row_pointer = row_pointer;

    for(std::size_t nnz_idx = 0; nnz_idx < number_nnz();
        nnz_idx++){
      row_idx[aux_column_pointer[sparse_data[nnz_idx].col()]] = 
        sparse_data[nnz_idx].row();

      column_idx[aux_row_pointer[sparse_data[nnz_idx].row()]] =
        sparse_data[nnz_idx].col();

      aux_column_pointer[sparse_data[nnz_idx].col()]++;
      aux_row_pointer[sparse_data[nnz_idx].row()]++;
    }
  }

  //Assumption: for indeces within a same line or column
  // they are stored in increasing order.
  //Build graph:
  //Vertices: indexed columns or rows.
  //Edges: Pairs of non orthogonal columns or rows.
  std::vector<std::vector<std::size_t>> build_cig_from_compressed(
      const std::vector<std::size_t>& idx_arr, const std::vector<std::size_t>& ptr_arr,
      const std::size_t domain_or_codomain_dim) const{
    std::vector<std::pair<std::size_t, std::size_t>> edges;

    std::size_t idx_start, idx_end;
    for(std::size_t i = 0; i < ptr_arr.size() - 1; i++){
      idx_start = ptr_arr[i];
      idx_end = ptr_arr[i+1];
      for(std::size_t p_idx = idx_start; p_idx < idx_end; p_idx++){
        for(std::size_t q_idx = p_idx+1; q_idx < idx_end; q_idx++){
          edges.emplace_back(idx_arr[p_idx], idx_arr[q_idx]);
        }
      }
    }

    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    std::vector<std::vector<std::size_t>> graph(domain_or_codomain_dim);
    std::vector<std::size_t> degree(domain_or_codomain_dim, 0);
    for(auto [u,v] : edges){
      ++degree[u];
      ++degree[v];
    }

    for(std::size_t i = 0; i < domain_or_codomain_dim; i++){
      graph[i].reserve(degree[i]);
    }

    for(auto [u,v]: edges){
      graph[u].push_back(v);
      graph[v].push_back(u);
    }
    return graph;
  } 
  
  // Greedy graph coloring
  std::vector<int> color_graph(const std::vector<std::vector<std::size_t>>& graph,
      std::size_t& max_color) const{

   const std::size_t domain_or_codomain_dim = graph.size();
   max_color = 0;
   std::vector<std::size_t> order(domain_or_codomain_dim);
   std::iota(order.begin(), order.end(),0);

   //Sort the vertices by degree
   std::sort(order.begin(), order.end(),
       [&] (std::size_t a, std::size_t b){return graph[a].size() > graph[b].size();});

   std::vector<int> color(domain_or_codomain_dim, -1);
   std::vector<char> forbidden;
   forbidden.reserve(domain_or_codomain_dim);
   std::size_t color_u;

   for(std::size_t u : order){
    forbidden.assign(domain_or_codomain_dim,0);
    for(std::size_t v : graph[u]){
      if(color[v] >= 0){
        forbidden[color[v]] = 1;
      }
    }
    color_u = 0;
    while(color_u < domain_or_codomain_dim && forbidden[color_u]){
      color_u++;
    }
    if(max_color < color_u){
      max_color = color_u;
    }
    color[u] = color_u;
   }
   max_color++;
   return color;
  }

  // coloring stores for every row or color its assigned
  // color.
  // new_coloring_format groups rows or columns by 
  // assigned color.
  void coloring_formatting(const std::vector<int>& coloring, 
      std::vector<std::vector<std::size_t>>& new_coloring_format,
      const std::size_t& number_of_colors){

    new_coloring_format.resize(number_of_colors);
    std::vector<std::size_t> number_elements_per_color(number_of_colors, 0);

    for(std::size_t i = 0; i < coloring.size(); i++){
      number_elements_per_color[coloring[i]]++;
    }

    for(std::size_t i = 0; i < number_of_colors; i++){
      new_coloring_format[i].reserve(number_elements_per_color[i]);
    }

    for(std::size_t i = 0; i < coloring.size(); i++){
      new_coloring_format[coloring[i]].push_back(i);
    }
  } 

  void coloring_algorithm(const std::vector<std::size_t>& idx_arr,
      const std::vector<std::size_t>& ptr_arr,
      std::vector<std::vector<std::size_t>>& color_arr,
      std::size_t domain_or_codomain_dim, std::size_t& number_of_colors){
    auto graph = 
      build_cig_from_compressed(idx_arr, ptr_arr, domain_or_codomain_dim);
    auto coloring = color_graph(graph, number_of_colors);
    coloring_formatting(coloring, color_arr, number_of_colors);
  }

  std::size_t max_nnz_row_or_col(const std::vector<std::size_t>& ptr_arr){
    std::size_t max_nnz = 0;
    std::size_t local_nnz;
    for(std::size_t i = 0; i < ptr_arr.size() - 1; i++){
      local_nnz = ptr_arr[i+1] - ptr_arr[i];
      if(max_nnz < local_nnz){
        max_nnz = local_nnz;
      }
    }
    return max_nnz;
  }

  void increasing_index_array(std::vector<std::size_t>& idx_arr,
      const std::vector<std::size_t>& ptr_arr){
    
    for(std::size_t idx = 0; idx < ptr_arr.size() - 1; idx++){
      std::sort(idx_arr.begin() + ptr_arr[idx],
          idx_arr.begin() + ptr_arr[idx + 1]);
    }
  }
  
  bool is_ptr_array_increasing(const std::vector<size_t>& ptr_arr){
    for(std::size_t i = 0; i < ptr_arr.size() - 1; i++){
      if(ptr_arr[i+1] < ptr_arr[i]){
        return false;
      }
    }
    return true;
  }

  static void sparse_file_to_sparse_data(const std::string& file_name,
      std::size_t& domain_dim, std::size_t& codomain_dim,
      std::size_t& number_of_edges, std::size_t& number_nnz,
      std::vector<NNZ>& sparse_data){
    sparse_data.clear();
    domain_dim = 0; codomain_dim = 0;
    number_of_edges = 0; number_nnz = 0;

    std::ifstream file;
    file.open(file_name);
    if(!file.is_open()){
      throw std::runtime_error("There was a problem opening "
          "the file: " + file_name);
    }

    std::string line;
    std::istringstream iss;
    std::size_t row, column;
    while(std::getline(file, line)){
      if(line[0] == '#'){
        auto start = line.find('[') + 1;
        auto end = line.find(']');
        std::string jacobian_data = line.substr(start, end-start);
        iss.clear();
        iss.str(jacobian_data);

        iss >> domain_dim >> codomain_dim;
        iss >> number_of_edges >> number_nnz;

        if(!(domain_dim && codomain_dim &&
              number_of_edges && number_nnz)){
          throw std::runtime_error("Invalid file format.\n"
              "File name: " + file_name);
        }
        sparse_data.reserve(number_nnz);
      }
      
      else if(std::isdigit(line[0])){
        if(!number_nnz){
          throw std::runtime_error("Invalid file format.\n"
              "File name: " + file_name);
        }
        iss.clear();
        iss.str(line);
        iss >> row >> column;
        sparse_data.emplace_back(row, column);
      }
    }

  }

  void from_sparse_data_initializer(std::vector<NNZ>& sparse_data){
    sparse_data_non_repeated_entries(sparse_data);

    build_CSR_CSC_format(sparse_data);

    coloring_algorithm(column_idx, row_pointer, column_coloring,
        domain_dim(), column_number_colors);

    coloring_algorithm(row_idx, column_pointer, row_coloring,
        codomain_dim(), row_number_colors);

    max_number_nnz_row = max_nnz_row_or_col(row_pointer);
    max_number_nnz_column = max_nnz_row_or_col(column_pointer);
  }

};

void mul_CSR_CSC_2_CSR(const Sparse_Jacobian& lhs, const Sparse_Jacobian& rhs,
    std::vector<size_t>& col_idx, std::vector<size_t>& row_ptr){
  std::size_t first_nnz_row, first_nnz_next_row;
  std::size_t first_nnz_col, first_nnz_next_col;
  std::size_t col_lhs, row_rhs;
  std::size_t counter = 0, col_idx_val = 0;

  row_ptr.push_back(0);
  for(std::size_t ptr_row = 0; ptr_row < lhs.codomain_dim(); ptr_row++){
    first_nnz_row = lhs.row_pointer[ptr_row];
    first_nnz_next_row = lhs.row_pointer[ptr_row + 1];
    for(std::size_t ptr_col = 0; ptr_col < rhs.domain_dim(); ptr_col++){
      first_nnz_col = rhs.column_pointer[ptr_col];
      first_nnz_next_col = rhs.column_pointer[ptr_col + 1];
      for(std::size_t col_idx_lhs = first_nnz_row; col_idx_lhs < first_nnz_next_row; col_idx_lhs++){
          col_lhs = lhs.column_idx[col_idx_lhs];
        for(std::size_t row_idx_rhs = first_nnz_col; row_idx_rhs < first_nnz_next_col; row_idx_rhs++){
            row_rhs = rhs.row_idx[row_idx_rhs];
            if(col_lhs == row_rhs){
              col_idx.push_back(col_idx_val);
              counter++;
              row_idx_rhs = first_nnz_next_col;
              col_idx_lhs = first_nnz_next_row;
            }
        }
      }
      col_idx_val++;
    }
    row_ptr.push_back(counter);
    col_idx_val = 0;
  }
}

void mul_CSR_CSC_2_CSC(const Sparse_Jacobian& lhs, const Sparse_Jacobian& rhs,
    std::vector<std::size_t>& row_idx, std::vector<std::size_t>& col_ptr){
  std::size_t first_nnz_col, first_nnz_next_col;
  std::size_t first_nnz_row, first_nnz_next_row;
  std::size_t row_rhs, col_lhs;
  std::size_t counter = 0, row_idx_val =0;

  col_ptr.push_back(0);
  for(std::size_t ptr_col = 0; ptr_col < rhs.domain_dim(); ptr_col++){
    first_nnz_col = rhs.column_pointer[ptr_col];
    first_nnz_next_col = rhs.column_pointer[ptr_col + 1];
    for(std::size_t ptr_row = 0; ptr_row < lhs.codomain_dim(); ptr_row++){
      first_nnz_row = lhs.row_pointer[ptr_row];
      first_nnz_next_row = lhs.row_pointer[ptr_row+1];
      for(std::size_t col_idx_lhs = first_nnz_row; col_idx_lhs < first_nnz_next_row; col_idx_lhs++){
        col_lhs = lhs.column_idx[col_idx_lhs];
        for(std::size_t row_idx_rhs = first_nnz_col; row_idx_rhs < first_nnz_next_col; row_idx_rhs++){
          row_rhs = rhs.row_idx[row_idx_rhs];
          if(row_rhs == col_lhs){
            row_idx.push_back(row_idx_val);
            counter++;
            col_idx_lhs = first_nnz_next_row;
            row_idx_rhs = first_nnz_next_col;
          }
        }
      }
      row_idx_val++;
    }
    col_ptr.push_back(counter);
    row_idx_val = 0;
  }
}

Sparse_Jacobian operator*(const Sparse_Jacobian& lhs, const Sparse_Jacobian& rhs){
  if(rhs.codomain_dim() != lhs.domain_dim()){
    throw std::invalid_argument(
      "Matrix multiplication dimension mismatch: rhs.codomain_dim() "
      " must equal to lhs.domain_dim()");
  }

  std::vector<size_t> col_idx, row_ptr;
  std::vector<size_t> row_idx, col_ptr;
  std::vector<size_t> data;
  col_ptr.reserve(rhs.domain_dim() + 1);
  row_ptr.reserve(lhs.codomain_dim() + 1);
  mul_CSR_CSC_2_CSR(lhs, rhs, col_idx, row_ptr);
  mul_CSR_CSC_2_CSC(lhs, rhs, row_idx, col_ptr);
  std::size_t num_nnz = col_idx.size();
  return Sparse_Jacobian(rhs.domain_dim(), lhs.codomain_dim(),
      lhs.number_edges() + rhs.number_edges(), num_nnz,
      std::move(col_idx), std::move(row_ptr),
      std::move(row_idx), std::move(col_ptr));
}
#endif
