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
  void print(){
    std::cout<<"[ "<< n_ <<' '<< m_ <<" ]\n";
  }

  std::size_t n() const noexcept {return n_;}
  std::size_t m() const noexcept {return m_;}

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

  void print(){
    std::cout<<"[ "<< n_ <<' '<< m_ <<' '<<
      n_E_ <<" ]\n";
  }
 
  std::size_t n_E() const noexcept {return n_E_;}
  
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
      num_nnz_ = jac_info[3];
      build_CSR_CSC_format(sparse_data);
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
    }

  void print_CSR(){
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

  void print_CSC(){
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

  void print_CSR_CSC(){
    std::cout << "CSR: \n";
    print_CSR();
    std::cout << "CSC: \n";
    print_CSC();
  }

  void print(){
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

  std::size_t num_nnz(){return num_nnz_;}

  const std::vector<size_t>& get_row_idx(){return row_idx;}
  std::vector<size_t> get_row_idx_copy(){return row_idx;}
  
  const std::vector<size_t>& get_col_ptr(){return col_ptr;}
  std::vector<size_t> get_col_ptr_copy(){return col_ptr;}

  const std::vector<size_t>& get_col_idx(){return col_idx;}
  std::vector<size_t> get_col_idx_copy(){return col_idx;}

  const std::vector<size_t>& get_row_ptr(){return row_ptr;}
  std::vector<size_t> get_row_ptr_copy(){return row_ptr;}

  friend Sparse_Jacobian operator*(const Sparse_Jacobian& lhs, const Sparse_Jacobian& rhs);
  friend void mul_CSR_CSC_2_CSR(const Sparse_Jacobian& lhs, const Sparse_Jacobian& rhs, std::vector<size_t>& col_idx,
      std::vector<size_t>& row_ptr);
  friend void mul_CSR_CSC_2_CSC(const Sparse_Jacobian& lhs, const Sparse_Jacobian& rhs, std::vector<size_t>& row_idx,
      std::vector<size_t>& col_ptr);

 protected:
  std::vector<size_t> row_idx, col_ptr;
  std::vector<size_t> col_idx, row_ptr;
  std::size_t num_nnz_, col_colors, row_colors;

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
  // [ptr_arr[i], ptr_arr[i+1]).
  void handling_repeated_entries(std::vector<size_t>& ptr_arr, std::vector<size_t>& idx_arr){
    auto first_iter = idx_arr.begin();
    auto end_iter = idx_arr.begin();
    std::vector<size_t> aux_idx_arr, aux_ptr_arr;
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
    num_nnz_ = col_idx.size();
  }

  // Build column-interface graph.
  // The method assumes that idx_arr is within a same row or column monotonically increasing.
  // The method is essentially taken from the code developed by TODO.
  std::vector<std::vector<std::size_t>> build_cig_from_compressed(
      const std::vector<size_t>& ptr_arr, const std::vector<size_t>& idx_arr,
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
  std::vector<int> color_graph(const std::vector<std::vector<std::size_t>>& graph){
   const std::size_t n_or_m = graph.size();
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
    color[u] = color_u;
   }
   return color;
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

void mul_CSR_CSC_2_CSC(const Sparse_Jacobian& lhs, const Sparse_Jacobian& rhs, std::vector<size_t>& row_idx,
    std::vector<size_t>& col_ptr){
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
