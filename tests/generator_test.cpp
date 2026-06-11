#include <algorithm>
#include <cstdint>
#include <cassert>
#include <fstream>
#include <sstream>
#include <vector>
#include "./../generator.hpp"
#include "./../util_structs.hpp"

void sparse_data_to_vector(std::vector<std::vector<NNZ>>& sparse_data,
    const std::string& file_name){
  std::ifstream file(file_name);
  if(!file){
    std::cerr << "Error opening file " + file_name << ".\n";
  }
  int jac_idx = -1;
  std::string line;
  std::size_t row, column;
  while(std::getline(file,line)){
   if(line[0] == '#'){jac_idx++;}
   else{
     std::istringstream iss(line);
     iss >> row >> column;
     sparse_data[jac_idx].emplace_back(row,column);
   }
  }
}

void are_col_row_index_in_range(const std::vector<std::vector<NNZ>>& sparse_data,
    const std::vector<std::vector<std::size_t>>& chain_info){
  std::size_t n= 0, m=1;
  for(std::size_t jac_idx = 0; jac_idx < sparse_data.size(); jac_idx++){
    for(std::size_t nnz_idx = 0; nnz_idx < sparse_data[jac_idx].size(); nnz_idx++){
      assert(sparse_data[jac_idx][nnz_idx].row() <= chain_info[jac_idx][m]);
      assert(sparse_data[jac_idx][nnz_idx].col() <= chain_info[jac_idx][n]);
    }
  } 
}

void are_repeated_entries(std::vector<std::vector<NNZ>>& sparse_data){
  for(std::size_t jac_idx= 0; jac_idx < sparse_data.size(); jac_idx++){
    std::sort(sparse_data[jac_idx].begin(), sparse_data[jac_idx].end());
    auto it = std::unique(sparse_data[jac_idx].begin(), sparse_data[jac_idx].end());
    assert(it == sparse_data[jac_idx].end());
  }
}

//The method assumes that are_repeated_entries was called before.
void all_rows_and_columns_have_nnz_entry(const std::vector<std::vector<NNZ>>& sparse_data,
    const std::vector<std::vector<std::size_t>>& chain_info){
  const std::size_t n = 0, m = 1;
  std::size_t idx_last_i_found = 0;
  bool nnz_in_col_j = 0, nnz_in_row_i = 0;

  for(std::size_t jac_idx = 0; jac_idx < sparse_data.size(); jac_idx++){

    for(std::size_t i = 0; i < chain_info[jac_idx][m]; i++){
      for(std::size_t nnz_idx = idx_last_i_found; nnz_idx < sparse_data[jac_idx].size(); nnz_idx++){
        if(sparse_data[jac_idx][nnz_idx].row() == i){
          idx_last_i_found = nnz_idx + 1;
          nnz_idx = sparse_data[jac_idx].size();
          nnz_in_row_i = 1;
        }
      }
      assert(nnz_in_row_i);
      nnz_in_row_i = 0;
    }
    idx_last_i_found = 0;

    for(std::size_t j = 0; j < chain_info[jac_idx][n]; j++){
      for(std::size_t nnz_idx = 0; nnz_idx < sparse_data[jac_idx].size(); nnz_idx++){
        if(sparse_data[jac_idx][nnz_idx].col() == j){
          nnz_idx = sparse_data[jac_idx].size();
          nnz_in_col_j = 1;
        }
      }
      assert(nnz_in_col_j);
      nnz_in_col_j = 0;
    }
  }
}


int main(){
  {
    std::size_t const chain_len = 3, dimension_lower_bound = 3, dimension_upper_bound = 8;
    std::size_t const number_edges_lower_bound = 50, number_edges_upper_bound = 150;
    bool const is_deterministic= 1;
    std::size_t const seed = 25;
    n_m_n_E_Generator gen{chain_len, dimension_lower_bound, dimension_upper_bound,
    number_edges_lower_bound, number_edges_upper_bound, is_deterministic, seed};
    gen.build_problem();

    std::size_t const n= 0, m= 1, n_E = 3;
    for(std::size_t i=0; i<gen.get_problem_size(); i++){
      auto jac_info = gen.get_jacobian_info(i);
      assert(jac_info[n] >= dimension_lower_bound);
      assert(jac_info[m] >= dimension_lower_bound);
      assert(jac_info[n_E] >= number_edges_lower_bound);
      assert(jac_info[n] <= dimension_upper_bound);
      assert(jac_info[m] <= dimension_upper_bound);
      assert(jac_info[n_E] <= number_edges_upper_bound);
    }

    for(std::size_t i = 0; i < gen.get_problem_size() - 1; i++){
      auto const jac_info_i = gen.get_jacobian_info(i);
      auto const jac_info_j = gen.get_jacobian_info(i+1);
      assert(jac_info_i[m] == jac_info_j[n]);
    }

    //Testing reproducibility.
    auto jac_chain_info = gen.get_problem_copy();
    gen.build_problem();
    auto const jac_chain_info_2 = gen.get_problem_copy();
    for(std::size_t i = 0; i<gen.get_problem_size(); i++){
      for(std::size_t j = 0; j < 3; j++){
        assert(jac_chain_info[i][j] == jac_chain_info_2[i][j]);
      }
    }
  }

  {
    std::size_t const chain_len = 4, dimension_lower_bound = 4, dimension_upper_bound = 9;
    std::size_t const number_edges_lower_bound = 50, number_edges_upper_bound = 150;
    double const density_lower_bound = 0.1, density_upper_bound = 0.5;
    bool is_deterministic = 1;
    std::size_t const seed = 40;

    n_m_n_E_nnz_Generator gen{chain_len, dimension_lower_bound, dimension_upper_bound,
    number_edges_lower_bound, number_edges_upper_bound, density_lower_bound,
    density_upper_bound, is_deterministic, seed};

    gen.build_sparse_problem();
    std::string const file_name{"sparse_data"};
    gen.build_sparse_structure(file_name);
    auto const jac_chain_info = gen.get_problem_copy();
    std::size_t const n =0, m=1, nnz=3;
    std::size_t max_n_m;

    for(std::size_t jac_idx = 0; jac_idx < gen.get_problem_size(); jac_idx++){
      if(jac_chain_info[jac_idx][n] <= jac_chain_info[jac_idx][m]){
        max_n_m = jac_chain_info[jac_idx][m];
      }
      else{max_n_m = jac_chain_info[jac_idx][n];}
      assert(max_n_m <= jac_chain_info[jac_idx][nnz]);
    }

    std::vector<std::vector<NNZ>> sparse_data(jac_chain_info.size());
    for(std::size_t i = 0; i < jac_chain_info.size(); i++){
      sparse_data[i].reserve(jac_chain_info[i][nnz]);
    }

    sparse_data_to_vector(sparse_data, file_name);
    are_col_row_index_in_range(sparse_data, jac_chain_info);
    are_repeated_entries(sparse_data);
    all_rows_and_columns_have_nnz_entry(sparse_data, jac_chain_info);

    for(std::size_t jac_idx = 0; jac_idx < sparse_data.size(); jac_idx++){
      assert(sparse_data[jac_idx].size() == jac_chain_info[jac_idx][nnz]);
    }
  }
  return 0;
}
