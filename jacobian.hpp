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
  Jacobian(const std::vector<size_t>& jac_info){
    /* assert(1<jac_info.size()); */
    if(jac_info.size()<2){
      throw std::invalid_argument("Jacobian information must have at least two elements.");
    }
    n_ = jac_info[0]; m_ = jac_info[1];
  }

  //Print function
  void print() const{
    std::cout<<"[ "<< n_ <<' '<< m_ <<" ]\n";
  }

  std::size_t n() const noexcept {return n_;}
  std::size_t m() const noexcept {return m_;}

  std::vector<std::size_t> jacobian_information() const{
    std::vector<std::size_t> jac_info;
    jac_info.reserve(2);
    jac_info.push_back(n_);
    jac_info.push_back(m_);
    return jac_info;
  }

 protected:
  //Input dimension R^n
  std::size_t n_;
  //Output dimension R^m
  std::size_t m_;
};

class Dense_Jacobian: public Jacobian{
 public:
  Dense_Jacobian(const std::vector<size_t>& jac_info): Jacobian(jac_info){
    if(jac_info.size()<3){
      throw std::invalid_argument("Jacobian information must have at least three elements.");
    }
    n_E_ = jac_info[2];
  }

  void print() const{
    std::cout<<"[ "<< n_ <<' '<< m_ <<' '<<
      n_E_ <<" ]\n";
  }
 
  std::size_t n_E() const noexcept {return n_E_;}

  std::vector<std::size_t> jacobian_information() const{
    std::vector<std::size_t> jac_info;
    jac_info.reserve(3);
    jac_info.push_back(n_);
    jac_info.push_back(m_);
    jac_info.push_back(n_E_);
    return jac_info;
  }
  
 protected:
  //Number of edges in the DAG representation
  std::size_t n_E_;
};

class Sparse_Jacobian: public Dense_Jacobian{
 public:
  Sparse_Jacobian(const std::vector<size_t>& jac_info,
      const std::vector<NNZ>& sparse_data):
    Dense_Jacobian(jac_info){
      if(jac_info.size() < 4){
        throw std::invalid_argument("Jacobian information must have at least four elements.");
      }
      if(jac_info[3] != sparse_data.size()){
        throw std::invalid_argument("Number of non zero entries and sparse data size do not match.");
      }
      num_nnz_ = jac_info[3];
      build_CSR_CSC_format(sparse_data);
      coloring_algorithm(row_ptr, col_idx, column_coloring, n_, col_heu_num_colors);
      coloring_algorithm(col_ptr, row_idx, row_coloring, m_, row_heu_num_colors);
      rhs_inner_dim_ = max_nnz_row_or_col(col_ptr);
      lhs_inner_dim_ = max_nnz_row_or_col(row_ptr);
    }

  Sparse_Jacobian(const std::vector<size_t>& jac_info): Dense_Jacobian(jac_info){
      if(jac_info.size() < 4){
        throw std::invalid_argument("Jacobian information must have at least four elements.");
      }
      num_nnz_ = jac_info[3];
  }

  Sparse_Jacobian(const std::vector<size_t>& jac_info,
      std::vector<std::size_t> row_idx_, std::vector<std::size_t> col_ptr_,
      std::vector<std::size_t> col_idx_, std::vector<std::size_t> row_ptr_): Dense_Jacobian(jac_info),
    row_idx(std::move(row_idx_)),
    col_ptr(std::move(col_ptr_)),
    col_idx(std::move(col_idx_)),
    row_ptr(std::move(row_ptr_)){
      if(jac_info.size() < 4){
        throw std::invalid_argument("Jacobian information must have at least four elements.");
      }
      if(col_ptr.size() != n_ + 1){
        throw std::invalid_argument("There is a mismatch between col_ptr.size() and n.");
      }
      if(row_ptr.size() != m_ + 1){
        throw std::invalid_argument("There is a mismatch between row_ptr.size() and m.");
      }
      if((col_idx.size() != row_idx.size())){
        throw std::invalid_argument("col_idx and row_idx have different sizes.");
      }
      if(col_idx.size() != jac_info[3]){
        throw std::invalid_argument("The number of non zero entries does not match "
            "the size of the index arrays.");
      }
      if(!is_increasing(row_ptr)){
        throw std::invalid_argument("row_ptr is not an increasing array.");
      }
      if(!is_increasing(col_ptr)){
        throw std::invalid_argument("col_ptr is not an increasing array.");
      }

      num_nnz_ = jac_info[3];
      handling_repeated_entries(row_ptr, col_idx);
      handling_repeated_entries(col_ptr, row_idx);
      coloring_algorithm(row_ptr, col_idx, column_coloring, n_, col_heu_num_colors);
      coloring_algorithm(col_ptr, row_idx, row_coloring, m_, row_heu_num_colors);
      rhs_inner_dim_ = max_nnz_row_or_col(col_ptr);
      lhs_inner_dim_ = max_nnz_row_or_col(row_ptr);
    }

  void print_CSR() const{
    std::cout << "col_idx: \n";
    for(std::size_t i = 0; i < num_nnz_; i++){
      std::cout << col_idx[i] << ' ';
    }
    std::cout << '\n';

    std::cout << "row_ptr: \n";
    for(std::size_t i = 0; i < m_ + 1; i++){
      std::cout << row_ptr[i] << ' ';
    }
    std::cout << '\n';
  }

  void print_CSC() const{
    std::cout << "row_idx: \n";
    for(std::size_t i = 0; i < num_nnz_; i++){
      std::cout << row_idx[i] << ' ';
    }
    std::cout << '\n';

    std::cout << "col_ptr: \n";
    for(std::size_t i = 0; i < n_ + 1; i++){
      std::cout << col_ptr[i] << ' ';
    }
    std::cout << '\n';
  }

  void print_CSR_CSC() const{
    std::cout << "CSR: \n";
    print_CSR();
    std::cout << "CSC: \n";
    print_CSC();
  }

  void print() const{
    std::cout<<"[ "<< n_ <<' '<< m_ <<' '<<
      n_E_<< ' ' << num_nnz_ << " ]\n";
  }

  void file_to_CSR_CSC(const std::string& sparse_data_file_name){
    std::ifstream file(sparse_data_file_name);
    if(!file){
      std::cerr<< "Error opening file " + sparse_data_file_name << ". \n";
    }

    std::vector<NNZ> jac_sparse_data;
    std::string line;
    std::size_t row, column;
    std::getline(file, line);
    if(line[0] != '#'){
      std::runtime_error("Invalid format.");
    }

    jac_sparse_data.reserve(num_nnz_);
    for(std::size_t i = 0; i< num_nnz_; i++){
      std::getline(file, line);
      std::istringstream iss(line);
      iss >> row >> column;
      jac_sparse_data.emplace_back(row, column);
    }
    build_CSR_CSC_format(jac_sparse_data);
  }

  std::size_t row_or_col_num_colors(const std::vector<std::size_t> ptr_arr,
      const std::vector<std::size_t> idx_arr, std::size_t n_or_m){
    auto graph = build_cig_from_compressed(ptr_arr, idx_arr, n_or_m);
    std::size_t heuristic_color_num;
    auto coloring = color_graph(graph, heuristic_color_num);
    assert(are_all_rows_or_columns_colored(coloring));
    /* auto itr_max = std::max_element(coloring.begin(), coloring.end()); */
    /* return static_cast<std::size_t>(*itr_max) + 1; */
    return heuristic_color_num;
  }

  void coloring_algorithm(const std::vector<std::size_t>& ptr_arr,
      const std::vector<std::size_t>& idx_arr,
      std::vector<std::vector<std::size_t>>& color_arr,
      std::size_t n_or_m, std::size_t& number_of_colors){
    auto graph = build_cig_from_compressed(ptr_arr, idx_arr, n_or_m);
    auto coloring = color_graph(graph, number_of_colors);
    coloring_formatting(coloring, color_arr, number_of_colors);
    assert(is_coloring_valid(color_arr, idx_arr, ptr_arr));
  }

  std::size_t num_nnz() const{return num_nnz_;}

  std::vector<std::size_t> jacobian_information() const{
    std::vector<std::size_t> jac_info;
    jac_info.reserve(4);
    jac_info.push_back(n_);
    jac_info.push_back(m_);
    jac_info.push_back(n_E_);
    jac_info.push_back(num_nnz_);
    return jac_info;
  }

  std::size_t col_num_colors() const{return col_heu_num_colors;}
  std::size_t row_num_colors() const{return row_heu_num_colors;}

  std::size_t rhs_inner_dim() const{return rhs_inner_dim_;}
  std::size_t lhs_inner_dim() const{return lhs_inner_dim_;}

  const std::vector<std::size_t>& get_row_idx() const{return row_idx;}
  std::vector<std::size_t> get_row_idx_copy(){return row_idx;}
  
  const std::vector<std::size_t>& get_col_ptr() const{return col_ptr;}
  std::vector<std::size_t> get_col_ptr_copy(){return col_ptr;}

  const std::vector<std::size_t>& get_col_idx() const{return col_idx;}
  std::vector<std::size_t> get_col_idx_copy(){return col_idx;}

  const std::vector<std::size_t>& get_row_ptr() const{return row_ptr;}
  std::vector<std::size_t> get_row_ptr_copy(){return row_ptr;}

  const std::vector<std::vector<std::size_t>>& get_column_coloring() const{
    return column_coloring;
  }

  const std::vector<std::vector<std::size_t>>& get_row_coloring() const{
    return row_coloring;
  }

  friend Sparse_Jacobian operator*(const Sparse_Jacobian& lhs, const Sparse_Jacobian& rhs);
  friend void mul_CSR_CSC_2_CSR(const Sparse_Jacobian& lhs, const Sparse_Jacobian& rhs, std::vector<size_t>& col_idx,
      std::vector<std::size_t>& row_ptr);
  friend void mul_CSR_CSC_2_CSC(const Sparse_Jacobian& lhs, const Sparse_Jacobian& rhs, std::vector<size_t>& row_idx,
      std::vector<std::size_t>& col_ptr);

 protected:
  std::vector<std::size_t> row_idx, col_ptr;
  std::vector<std::size_t> col_idx, row_ptr;
  std::vector<std::vector<std::size_t>> column_coloring, row_coloring;
  std::size_t num_nnz_;
  std::size_t col_heu_num_colors, row_heu_num_colors;
  // Upper bound for the cost of matrix matrix mul.
  std::size_t rhs_inner_dim_, lhs_inner_dim_;

 private:
  bool is_increasing(const std::vector<size_t>& ptr_arr){
    bool is_increasing_v = 1;
    for(std::size_t i = 0; i < ptr_arr.size() - 1; i++){
      if(ptr_arr[i+1] < ptr_arr[i]){
        is_increasing_v = 0;
        i = ptr_arr.size() - 1;
      }
    }
    return is_increasing_v;
  }

  // Eliminate repeated entries and modifies ptr_arr and idx_arr accordingly.
  // After calling the method idx_arr is monotonically increasing inside the range
  // [ptr_arr[i], ptr_arr[i+1]) and modifies the data member num_nnz_.
  void handling_repeated_entries(std::vector<std::size_t>& ptr_arr,
      std::vector<std::size_t>& idx_arr){
    auto first_iter = idx_arr.begin();
    auto end_iter = idx_arr.begin();
    std::vector<size_t> aux_idx_arr, aux_ptr_arr;
    aux_idx_arr.reserve(idx_arr.size());
    aux_ptr_arr.reserve(ptr_arr.size());
    aux_ptr_arr.push_back(0);
    std::size_t repeated_entries = 0;
    bool once = 0;
    for(std::size_t i = 0; i < ptr_arr.size() - 1; i++){
      first_iter = end_iter;
      end_iter += ptr_arr[i+1] - ptr_arr[i];
      std::sort(first_iter, end_iter);
      auto new_end_iter = std::unique(first_iter, end_iter);
      repeated_entries += static_cast<std::size_t>(end_iter - new_end_iter);
      if(0 < repeated_entries && !once){
        std::cout << "Repeated entries in sparse data were found.\n";
        once = 1; 
      }
      aux_ptr_arr.push_back(repeated_entries);
      /* ptr_arr[i+1] -= repeated_entries; */
      aux_idx_arr.insert(aux_idx_arr.end(), first_iter, new_end_iter);
    }
    //Modifying ptr_arr accordingly.
    if(once){
      for(std::size_t i = 1; i < ptr_arr.size(); i++){
        ptr_arr[i] = ptr_arr[i] - aux_ptr_arr[i];
      }
    }
    num_nnz_ = aux_idx_arr.size();
    idx_arr = std::move(aux_idx_arr);
  }

  void build_CSR_CSC_format(const std::vector<NNZ>& sparse_data){
    row_ptr.assign(m_ + 1, 0);
    col_ptr.assign(n_ + 1, 0);
    //Count
    for(std::size_t nnz_idx = 0; nnz_idx < num_nnz_; nnz_idx++){
      col_ptr[sparse_data[nnz_idx].col() + 1]++;
      row_ptr[sparse_data[nnz_idx].row() + 1]++;
    }
    //Accumulate row pointer array
    for(std::size_t i = 0; i < m_; i++){
      row_ptr[i+1] = row_ptr[i] + row_ptr[i+1];
    }

    //Accumulate column pointer array
    for(std::size_t i = 0; i < n_; i++){
      col_ptr[i+1] = col_ptr[i] + col_ptr[i+1];
    }

    row_idx.resize(num_nnz_);
    col_idx.resize(num_nnz_);

    auto aux_col_ptr = col_ptr;
    auto aux_row_ptr = row_ptr;

    for(std::size_t nnz_idx = 0; nnz_idx < num_nnz_; nnz_idx++){
      row_idx[aux_col_ptr[sparse_data[nnz_idx].col()]] = sparse_data[nnz_idx].row();
      col_idx[aux_row_ptr[sparse_data[nnz_idx].row()]] = sparse_data[nnz_idx].col();
      aux_col_ptr[sparse_data[nnz_idx].col()]++;
      aux_row_ptr[sparse_data[nnz_idx].row()]++;
    }
    handling_repeated_entries(row_ptr, col_idx);
    handling_repeated_entries(col_ptr, row_idx);
  }

  // Build column-interface graph.
  // The method assumes that idx_arr is within a same row or column monotonically increasing.
  // The method is essentially taken from the code developed by TODO.
  std::vector<std::vector<std::size_t>> build_cig_from_compressed(
      const std::vector<std::size_t>& ptr_arr, const std::vector<std::size_t>& idx_arr,
      const std::size_t n_or_m){
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

    std::vector<std::vector<std::size_t>> graph(n_or_m);
    std::vector<std::size_t> degree(n_or_m, 0);
    for(auto [u,v] : edges){
      ++degree[u];
      ++degree[v];
    }

    for(std::size_t i = 0; i < n_or_m; i++){
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
      std::size_t& max_color){

   const std::size_t n_or_m = graph.size();
   max_color = 0;
   std::vector<std::size_t> order(n_or_m);
   std::iota(order.begin(), order.end(),0);

   //Sort the vertices by degree
   std::sort(order.begin(), order.end(),
       [&] (std::size_t a, std::size_t b){return graph[a].size() > graph[b].size();});

   std::vector<int> color(n_or_m, -1);
   std::vector<char> forbidden;
   forbidden.reserve(n_or_m);
   std::size_t color_u;

   for(std::size_t u : order){
    forbidden.assign(n_or_m,0);
    for(std::size_t v : graph[u]){
      if(color[v] >= 0){
        forbidden[color[v]] = 1;
      }
    }
    color_u = 0;
    while(color_u < n_or_m && forbidden[color_u]){
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


  bool are_all_rows_or_columns_colored(const std::vector<int>& coloring){
    bool are_all_colored = 1;
    for(std::size_t i = 0; i < coloring.size(); i++){
      if(coloring[i] == -1){
        are_all_colored = 0;
      }
    }
    return are_all_colored;
  }

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

  // idx_arr must be sorted.
  // color should follow the format given by coloring_formatting.
  bool is_coloring_valid(const std::vector<std::vector<std::size_t>>& color,
      const std::vector<std::size_t>& idx_arr, const std::vector<std::size_t>& ptr_arr){
    bool is_coloring_valid_ = 1;
    for(std::size_t i = 0; i < ptr_arr.size() - 1; i++){
      for(std::size_t color_idx = 0; color_idx < color.size(); color_idx++){
        for(std::size_t color_i = 0; color_i < color[color_idx].size(); color_i++){

          if(std::binary_search(idx_arr.begin() + ptr_arr[i], idx_arr.begin() + ptr_arr[i+1],
                color[color_idx][color_i])){

            for(std::size_t color_j = color_i + 1; color_j < color[color_idx].size(); color_j++){
              if(std::binary_search(idx_arr.begin() + ptr_arr[i], idx_arr.begin() + ptr_arr[i+1],
                    color[color_idx][color_j])){is_coloring_valid_ = 0;}
              
            }
          }

        }
      }
    }
    return is_coloring_valid_;
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
};

void mul_CSR_CSC_2_CSR(const Sparse_Jacobian& lhs, const Sparse_Jacobian& rhs, std::vector<size_t>& col_idx,
    std::vector<size_t>& row_ptr){
  std::size_t first_nnz_row, first_nnz_next_row;
  std::size_t first_nnz_col, first_nnz_next_col;
  std::size_t col_lhs, row_rhs;
  std::size_t counter = 0, col_idx_val = 0;

  row_ptr.push_back(0);
  for(std::size_t ptr_row = 0; ptr_row < lhs.m_; ptr_row++){
    first_nnz_row = lhs.row_ptr[ptr_row];
    first_nnz_next_row = lhs.row_ptr[ptr_row + 1];
    for(std::size_t ptr_col = 0; ptr_col < rhs.n_; ptr_col++){
      first_nnz_col = rhs.col_ptr[ptr_col];
      first_nnz_next_col = rhs.col_ptr[ptr_col + 1];
      for(std::size_t col_idx_lhs = first_nnz_row; col_idx_lhs < first_nnz_next_row; col_idx_lhs++){
          col_lhs = lhs.col_idx[col_idx_lhs];
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
  for(std::size_t ptr_col = 0; ptr_col < rhs.n_; ptr_col++){
    first_nnz_col = rhs.col_ptr[ptr_col];
    first_nnz_next_col = rhs.col_ptr[ptr_col + 1];
    for(std::size_t ptr_row = 0; ptr_row < lhs.m_; ptr_row++){
      first_nnz_row = lhs.row_ptr[ptr_row];
      first_nnz_next_row = lhs.row_ptr[ptr_row+1];
      for(std::size_t col_idx_lhs = first_nnz_row; col_idx_lhs < first_nnz_next_row; col_idx_lhs++){
        col_lhs = lhs.col_idx[col_idx_lhs];
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
  if(rhs.m_ != lhs.n_){
    throw std::invalid_argument(
      "Matrix multiplication dimension mismatch: rhs.m_ must equal lhs.n_."
      );
  }
  std::vector<size_t> col_idx, row_ptr;
  std::vector<size_t> row_idx, col_ptr;
  std::vector<size_t> data;
  std::size_t n_ = rhs.n_;
  std::size_t m_ = lhs.m_;
  std::size_t n_E_ = rhs.n_E_ + lhs.n_E_;
  data.push_back(n_); data.push_back(m_); data.push_back(n_E_);
  col_ptr.reserve(m_ + 1);
  row_ptr.reserve(n_ + 1);
  mul_CSR_CSC_2_CSR(lhs, rhs, col_idx, row_ptr);
  mul_CSR_CSC_2_CSC(lhs, rhs, row_idx, col_ptr);
  assert(col_idx.size() == row_idx.size());
  std::size_t num_nnz_ = col_idx.size();
  data.push_back(num_nnz_);
  return Sparse_Jacobian(data, std::move(row_idx), std::move(col_ptr),
      std::move(col_idx), std::move(row_ptr));
}

#endif
