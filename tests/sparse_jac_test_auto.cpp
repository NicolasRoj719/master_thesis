#include <iostream>
#include <cstdint>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Sparse>
#include "./../generator.hpp"
#include "./../jacobian.hpp"
#include "./../build_chain.hpp"

typedef Eigen::Triplet<std::size_t> Triplet;

void file_2_triplet(std::vector<Triplet>& tri_vector, const std::string& data_file_name,
    std::size_t nnz){
  std::ifstream file(data_file_name);
  if(!file){
    std::cerr << "Error opening file " + data_file_name << ". \n";
  }
  std::string line;
  std::size_t row, col;
  std::getline(file,line);
  if(line[0]!='#'){
    std::runtime_error("Invalid format in file " + data_file_name +
      ": first line mist start with #");
  }

  for(std::size_t nnz_i = 0; nnz_i < nnz; nnz_i++){
    std::getline(file,line);
    std::istringstream iss(line);
    iss >> row >> col;
    tri_vector.emplace_back(Triplet(row, col, 1));
  }
}

void file_2_triplet(std::vector<Triplet>& rhs, std::vector<Triplet>& lhs,
    const std::string& data_file_name, std::size_t nnz_rhs, std::size_t nnz_lhs){
  std::ifstream file(data_file_name);
  if(!file){
    std::cerr << "Error opening file " + data_file_name << ". \n";
  }
  std::string line;
  std::size_t row, col;
  std::size_t hash_tag_counter = 0;

  std::getline(file,line);
  if(line[0]=='#'){
    hash_tag_counter++;
    //Debugging
    /* std::cout << line << '\n'; */
  }
  else{
    std::runtime_error("Invalid format in file " + data_file_name +
      ": first line must start with #");
  }
  rhs.reserve(nnz_rhs);
  lhs.reserve(nnz_lhs);
  std::vector<std::size_t> nnz_vec{nnz_rhs, nnz_lhs};
  for(std::size_t matrix_i = 0; matrix_i<2; matrix_i++){
    for(std::size_t nnz_i = 0; nnz_i < nnz_vec[matrix_i]; nnz_i++){
      std::getline(file, line);
      std::istringstream iss(line);
      iss >> row >> col;
      if(matrix_i == 0){
        rhs.emplace_back(Triplet(row, col, 1));
      }
      else{lhs.emplace_back(Triplet(row, col, 1));}
    }
    if(std::getline(file, line)){
      if(line[0] != '#'){
        std::runtime_error("The matrix separator '#' was not found.\n" 
            "This could be caused by an erronous value of the number of non zero entries "
            "or the absecne of the matrix separator starting with #.\n");
      }
    }
  }
}

void print_Eigen_vector(const int* ptr_first_element, std::size_t size){
  for(std::size_t i = 0; i<size; i++){
    std::cout << *ptr_first_element << ' ';
    ptr_first_element++;
  }
  std::cout<<'\n';
}

bool ptr_array_test(const int* ptr_first_element, const std::vector<size_t>& ptr_array,
    std::size_t num_rows_cols){
  bool are_equal = 1;
  std::size_t idx_error = num_rows_cols;
  for(std::size_t i = 0; i < num_rows_cols + 1; i++){
    if(static_cast<std::size_t>(*ptr_first_element) != ptr_array[i]){
      are_equal = 0;
      idx_error = i;
      i = num_rows_cols + 1;
    }
    ptr_first_element++;
  }
  // Prints the first non matching entry.
  if(!are_equal){
    std::cout << "The entry " << idx_error << " is different" << '\n';
  }
  return are_equal;
}

bool idx_array_test(const std::vector<size_t>& ptr_array,
    const int* ptr_first_e_idx, const std::vector<size_t>& idx_array,
    std::size_t num_rows_cols){
  bool is_idx_found = 0;
  for(std::size_t i = 0; i < num_rows_cols; i++){
    for(std::size_t idx = ptr_array[i]; idx < ptr_array[i+1]; idx++){
      for(std::size_t idx_it = ptr_array[i]; idx_it < ptr_array[i+1]; idx_it++){
        if(static_cast<std::size_t>(*ptr_first_e_idx) == idx_array[idx_it]){
          is_idx_found = 1;
          idx_it = ptr_array[i+1];
        }
      }
      if(!is_idx_found){
        std::cout << "Index " << *ptr_first_e_idx << " not found in row/column " << i << ".\n";
        return 0;
      }
      ptr_first_e_idx++;
      is_idx_found = 0;
    }
  }
  return 1;
}

template<typename T>
bool parse_arg(char* text, T& value){
  std::istringstream iss(text);
  return !(iss >> value);
}

int main(int argc, char* argv[]){
  
  if(argc < 8){
    std::cout << "Usage: \n";
    std::cout << "./sparse_jac_auto CSR_CSC_verbose dim_lb dim_ub den_lb den_ub is_deterministic seed\n";
    std::cout << "./sparse_jac_auto CSR_CSC dim_lb dim_ub den_lb den_ub is_deterministic seed\n";
    std::cout << "./sparse_jac_auto Sparse_mul_verbose dim_lb dim_ub den_lb den_ub is_deterministic seed\n";
    std::cout << "./sparse_jac_auto Sparse_mul dim_lb dim_ub den_lb den_ub is_deterministic seed\n";
    return 0;
  }
  std::string mode = argv[1];

  if((mode != "CSR_CSC_verbose") && (mode != "CSR_CSC") 
      && (mode != "Sparse_mul_verbose") && (mode != "Sparse_mul")){
    std::cout << "Wrong console argument. Rerun the executable with one of the following arguments:\n";
    std::cout << "CSR_CSC_verbose\n";
    std::cout << "CSR_CSC\n";
    std::cout << "Sparse_mul_verbose\n";
    std::cout << "Sparse_mul\n";
    return 1;
  }

  std::size_t dim_lb, dim_ub, seed;
  double den_lb, den_ub;
  bool is_deterministic;

  if(parse_arg(argv[2], dim_lb)){
    std::cout << "Wrong input.\n";
    std::cout << "The argument: " << argv[2] << " is not convertible to std::size_t\n";
    return 1;
  }
  if(parse_arg(argv[3], dim_ub)){
    std::cout << "Wrong input.\n";
    std::cout << "The argument: " << argv[3] << " is not convertible to std::size_t\n";
    return 1;
  }
  if(parse_arg(argv[4], den_lb)){
    std::cout << "Wrong input.\n";
    std::cout << "The argument: " << argv[4] << " is not convertible to double\n";
    return 1;
  }
  if(parse_arg(argv[5], den_ub)){
    std::cout << "Wrong input.\n";
    std::cout << "The argument: " << argv[5] << " is not convertible to double\n";
    return 1;
  }
  if(parse_arg(argv[6], is_deterministic)){
    std::cout << "Wrong input.\n";
    std::cout << "The argument: " << argv[6] << " is not convertible to bool\n";
    return 1;
  }
  if(parse_arg(argv[7], seed)){
    std::cout << "Wrong input.\n";
    std::cout << "The argument: " << argv[7] << " is not convertible to std::size_t.\n";
    return 1;
  }

  if(mode == "CSR_CSC_verbose" || mode == "CSR_CSC"){
    std::cout << "Generating data.\n";
    n_m_n_E_nnz_Generator gen{1, dim_lb, dim_ub, 50, 60,
    den_lb, den_ub, is_deterministic, seed};  
    gen.build_problem();
    auto problem_data = gen.get_problem();
    const std::string file_data_name = "sparse_data";
    gen.print();
    std::size_t n = 0, m = 1, nnz = 3; 
    //EIGEN
    Eigen::SparseMatrix<std::size_t, Eigen::RowMajor> eigen_row_jac(problem_data[0][m], problem_data[0][n]);
    Eigen::SparseMatrix<std::size_t, Eigen::ColMajor> eigen_col_jac(problem_data[0][m], problem_data[0][n]);
    std::vector<Triplet> sparse_data_eigen;
    file_2_triplet(sparse_data_eigen, file_data_name, problem_data[0][nnz]);

    eigen_row_jac.setFromTriplets(sparse_data_eigen.begin(), sparse_data_eigen.end());
    eigen_row_jac.makeCompressed();

    eigen_col_jac.setFromTriplets(sparse_data_eigen.begin(), sparse_data_eigen.end());
    eigen_col_jac.makeCompressed();
    
    // CODE IMPLEMENTATION
    Sparse_Jacobian jac{problem_data[0]};
    jac.file_to_CSR_CSC(file_data_name);

    if(mode == "CSR_CSC_verbose"){
      std::cout << "Printing Eigen data:\n";
      std::cout << "CSR format Eigen:\n";
      std::cout << "col_idx:\n";
      print_Eigen_vector(eigen_row_jac.innerIndexPtr(), eigen_row_jac.nonZeros());
      std::cout << "row_ptr:\n";
      print_Eigen_vector(eigen_row_jac.outerIndexPtr(), eigen_row_jac.rows() + 1); 
      std::cout << "CSC format Eigen:\n";
      std::cout << "row_idx:\n";
      print_Eigen_vector(eigen_col_jac.innerIndexPtr(), eigen_col_jac.nonZeros());
      std::cout << "col_ptr:\n";
      print_Eigen_vector(eigen_col_jac.outerIndexPtr(), eigen_col_jac.cols() + 1);
      std::cout << "Printing Sparse_Jacobian data:\n";
      jac.print_CSR();
      jac.print_CSC();
    }

    //TESTING
    if(!ptr_array_test(eigen_row_jac.outerIndexPtr(), jac.get_row_ptr(), problem_data[0][m])){
      std::cout << "CSR pointer arrays are not equal.\n";
      return 1;
    }

    if(!ptr_array_test(eigen_col_jac.outerIndexPtr(), jac.get_col_ptr(), problem_data[0][n])){
      std::cout << "CSC pointer arrays are not equal.\n";
      return 1;
    }

    if(!idx_array_test(jac.get_row_ptr(), eigen_row_jac.innerIndexPtr(),
          jac.get_col_idx(), problem_data[0][m])){
      std::cout << "CSR index arrays contain different sparse information.\n";
      return 1;
    }

    if(!idx_array_test(jac.get_col_ptr(), eigen_col_jac.innerIndexPtr(),
          jac.get_row_idx(), problem_data[0][n])){
      std::cout << "CSC index arrays contain different sparse information.\n";
      return 1;
    }

    std::cout << "The tests were successful.\n";
    std::cout << "The compressed formats are equivalent.\n";
    return 0;
  }

  else if(mode == "Sparse_mul_verbose" || mode == "Sparse_mul"){
    std::cout << "Generating data.\n";
    n_m_n_E_nnz_Generator gen{2, dim_lb, dim_ub, 50, 60,
    den_lb, den_ub, is_deterministic, seed};  
    gen.build_problem();
    auto problem_data = gen.get_problem();
    const std::string file_data_name = "sparse_data";
    gen.print();
    std::size_t n = 0, m = 1, nnz = 3; 

    //EIGEN
    Eigen::SparseMatrix<std::size_t, Eigen::RowMajor> eigen_row_rhs(problem_data[0][m], problem_data[0][n]);
    Eigen::SparseMatrix<std::size_t, Eigen::ColMajor> eigen_col_rhs(problem_data[0][m], problem_data[0][n]);

    Eigen::SparseMatrix<std::size_t, Eigen::RowMajor> eigen_row_lhs(problem_data[1][m], problem_data[1][n]);
    Eigen::SparseMatrix<std::size_t, Eigen::ColMajor> eigen_col_lhs(problem_data[1][m], problem_data[1][n]);

    std::vector<Triplet> rhs_data, lhs_data;
    file_2_triplet(rhs_data, lhs_data, file_data_name, problem_data[0][nnz], problem_data[1][nnz]);

    eigen_row_rhs.setFromTriplets(rhs_data.begin(), rhs_data.end());
    eigen_row_rhs.makeCompressed();

    eigen_col_rhs.setFromTriplets(rhs_data.begin(), rhs_data.end());
    eigen_col_rhs.makeCompressed();

    eigen_row_lhs.setFromTriplets(lhs_data.begin(), lhs_data.end());
    eigen_row_lhs.makeCompressed();

    eigen_col_lhs.setFromTriplets(lhs_data.begin(), lhs_data.end());
    eigen_col_lhs.makeCompressed();

    Eigen::SparseMatrix<std::size_t, Eigen::RowMajor> eigen_row_product = eigen_row_lhs * eigen_row_rhs;
    Eigen::SparseMatrix<std::size_t, Eigen::ColMajor> eigen_col_product = eigen_col_lhs * eigen_col_rhs;
    
    // CODE IMPLEMENTATION
    build_chain_implementation<Sparse_Jacobian> build_obj{problem_data, file_data_name};
    auto jac_chain = build_obj.get_chain();
    Sparse_Jacobian jac_product = jac_chain[1] * jac_chain[0];

    if(mode == "Sparse_mul_verbose"){
      std::cout << "Printing product Eigen data:\n";
      std::cout << "CSR format Eigen:\n";
      std::cout << "col_idx:\n";
      print_Eigen_vector(eigen_row_product.innerIndexPtr(), eigen_row_product.nonZeros());
      std::cout << "row_ptr:\n";
      print_Eigen_vector(eigen_row_product.outerIndexPtr(), eigen_row_product.rows() + 1); 
      std::cout << "CSC format Eigen:\n";
      std::cout << "row_idx:\n";
      print_Eigen_vector(eigen_col_product.innerIndexPtr(), eigen_col_product.nonZeros());
      std::cout << "col_ptr:\n";
      print_Eigen_vector(eigen_col_product.outerIndexPtr(), eigen_col_product.cols() + 1);
      std::cout << "Printing Sparse_Jacobian product data:\n";
      jac_product.print_CSR();
      jac_product.print_CSC();
    }
    //TESTING
    if(!ptr_array_test(eigen_row_product.outerIndexPtr(), jac_product.get_row_ptr(), problem_data[1][m])){
      std::cout << "CSR pointer arrays are not equal.\n";
      return 1;
    }

    if(!ptr_array_test(eigen_col_product.outerIndexPtr(), jac_product.get_col_ptr(), problem_data[0][n])){
      std::cout << "CSC pointer arrays are not equal.\n";
      return 1;
    }

    if(!idx_array_test(jac_product.get_row_ptr(), eigen_row_product.innerIndexPtr(),
          jac_product.get_col_idx(), problem_data[1][m])){
      std::cout << "CSR index arrays contain different sparse information.\n";
      return 1;
    }

    if(!idx_array_test(jac_product.get_col_ptr(), eigen_col_product.innerIndexPtr(),
          jac_product.get_row_idx(), problem_data[0][n])){
      std::cout << "CSC index arrays contain different sparse information.\n";
      return 1;
    }

    std::cout << "The tests were successful.\n";
    std::cout << "the compressed formats are equivalent.\n";
    return 0;
    }
}
