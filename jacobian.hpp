#include <cstdint>
#include <iostream>
#include <vector>
#include <cassert>
#include <stdexcept>

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
    /* assert(2<jac_info.size()); */
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
      const std::vector<std::vector<std::size_t>>& sparse_data):
    Dense_Jacobian(jac_info){
      if(jac_info.size() < 4){
        throw std::invalid_argument("Jacobian information must have at least four elements.");
      }
      nnz = jac_info[3];
      build_CSR_CSC_format(sparse_data);
    }

  void build_CSR_CSC_format(const std::vector<std::vector<std::size_t>>& sparse_data){
    col_ptr.assign(m_ + 1, 0);
    row_ptr.assign(n_ + 1, 0);
    std::size_t row = 0, column = 1;
    //Count
    for(std::size_t nnz_idx = 0; nnz_idx < nnz; nnz_idx++){
      col_ptr[sparse_data[nnz_idx][column] + 1]++;
      row_ptr[sparse_data[nnz_idx][row] + 1]++;
    }
    //Accumulate row pointer array
    for(std::size_t i = 0; i < m_; i++){
      row_ptr[i+1] = row_ptr[i] + row_ptr[i+1];
    }

    //Accumulate column pointer array
    for(std::size_t i = 0; i < n_; i++){
      col_ptr[i+1] = col_ptr[i] + col_ptr[i+1];
    }

    row_idx.reserve(nnz);
    col_idx.reserve(nnz);

    auto aux_col_ptr = col_ptr;
    auto aux_row_ptr = row_ptr;

    for(std::size_t nnz_idx = 0; nnz_idx < nnz; nnz_idx++){
      row_idx[aux_col_ptr[sparse_data[nnz_idx][column]]] = sparse_data[nnz_idx][row];
      col_idx[aux_row_ptr[sparse_data[nnz_idx][row]]] = sparse_data[nnz_idx][column];
      ++aux_col_ptr[sparse_data[nnz_idx][column]];
      ++aux_row_ptr[sparse_data[nnz_idx][row]];
    }
  }

  void print_CSR(){
    std::cout << "col_idx: \n";
    for(std::size_t i = 0; i < col_idx.size(); i++){
      std::cout << col_idx[i] << ' ';
    }
    std::cout << '\n';

    std::cout << "row_ptr: \n";
    for(std::size_t i = 0; i < row_ptr.size(); i++){
      std::cout << row_ptr[i] << ' ';
    }
    std::cout << '\n';
  }

  void print_CSC(){
    std::cout << "row_idx: \n";
    for(std::size_t i = 0; i < row_idx.size(); i++){
      std::cout << row_idx[i] << ' ';
    }
    std::cout << '\n';

    std::cout << "col_ptr: \n";
    for(std::size_t i = 0; i < col_ptr.size(); i++){
      std::cout << col_ptr[i] << ' ';
    }
    std::cout << '\n';
  }

 protected:
  std::vector<size_t> row_idx, col_ptr;
  std::vector<size_t> col_idx, row_ptr;
  std::size_t nnz, col_colors, row_colors;
};


#endif
