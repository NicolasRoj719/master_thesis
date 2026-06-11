#include <random>
#include <cmath>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <string>

#ifndef GENERATOR_HPP
#define GENERATOR_HPP

class Generator{
 public:
  Generator(std::size_t chain_length_, std::size_t inner_size, std::size_t dim_lb_,
      std::size_t dim_ub_, std::size_t n_E_lb_, std::size_t n_E_ub_, bool is_deterministic_ = 0,
      std::size_t seed_ = 0):
    chain_length(chain_length_),
    jac_chain_info(chain_length_, std::vector<std::size_t>(inner_size)),
    dim_lb(dim_lb_), dim_ub(dim_ub_), n_E_lb(n_E_lb_), n_E_ub(n_E_ub_),
    is_deterministic(is_deterministic_), seed(seed_){
      if(chain_length == 0){
        throw std::invalid_argument("Chain length must be greater than 0.");
      }
      if(dim_ub < dim_lb){
        throw std::invalid_argument("Dimension lower bound must be smaller or equal to the upper bound.");
      }
      if(n_E_ub < n_E_lb){
        throw std::invalid_argument("The lower bound for the number of edges must be smaller or equal to the upper bound.");
      }
    }

  virtual ~Generator() = default;
 
 public:
  void print(){
    print_format();
    for(std::size_t i=0; i<chain_length; i++){
      std::cout<<"F'_"<<i<<" [ ";
      for(std::size_t j=0; j<jac_chain_info[i].size(); j++){
        std::cout<<jac_chain_info[i][j]<<' ';
      }
      std::cout<<"]\n";
    }
  };


  void build_problem();

  std::size_t get_problem_size(){return jac_chain_info.size();}

  const std::vector<std::vector<std::size_t>>& get_problem(){
    return jac_chain_info;
  }

  std::vector<std::vector<std::size_t>> get_problem_copy() const{
    return jac_chain_info;
  }

  std::vector<std::size_t> get_jacobian_info(std::size_t index){
    return jac_chain_info.at(index);
  }

  /* //This method will be used to build the elemental jacobians array. */
  /* std::vector<size_t> read_Fi_description(std::size_t i){ */
  /*   return(jac_chain_info[i]); */
  /* } */

 protected:
  std::size_t const chain_length;
  //vector to store n,m,n_E
  std::vector<std::vector<std::size_t>> jac_chain_info;
  //input/output dimensions lower and upper bouds
  std::size_t dim_lb, dim_ub;
  //lower and upper on the number of edges in the DAG
  std::size_t n_E_lb, n_E_ub;
  //Flag is_deterministic
  bool is_deterministic;
  //seed
  std::size_t seed;
  virtual void print_format() = 0;
};

// m_n_n_E_Generator populates jac_chain_info data structure
// following the following format
// n_0 m_0 n_E_0
// with n,m and n_E taken randomely form a uniform distribution.
class n_m_n_E_Generator: public Generator{
 public:
  //Inner size is set to 3, corresponding to n,m,n_E.
  n_m_n_E_Generator(std::size_t chain_length_, std::size_t dim_lb_,
    std::size_t dim_ub_, std::size_t n_E_lb_, std::size_t n_E_ub_,
    bool is_deterministic_ = 0, std::size_t seed_ = 0):
    Generator(chain_length_, 3, dim_lb_, dim_ub_, n_E_lb_, n_E_ub_,
        is_deterministic_, seed_){}

  void build_problem(){
    std::default_random_engine g;
    
    if(!is_deterministic){
      std::random_device r;
      g.seed(r());
    }

    else{
      g.seed(seed);
    }

    /* std::random_device r; */
    /* std::default_random_engine g(r()); */

    std::uniform_int_distribution<std::size_t> d_dim_in_out(dim_lb, dim_ub);
    std::uniform_int_distribution<std::size_t> d_n_E(n_E_lb, n_E_ub);

    std::size_t jac_index=0;
    const std::size_t n=0, m=1, n_E=2;

    jac_chain_info[jac_index][n] = d_dim_in_out(g);
    jac_chain_info[jac_index][m] = d_dim_in_out(g);
    jac_chain_info[jac_index][n_E] = d_n_E(g);
    jac_index++;

    while(jac_index < chain_length){
      jac_chain_info[jac_index][n] = jac_chain_info[jac_index-1][m];
      jac_chain_info[jac_index][m] = d_dim_in_out(g);
      jac_chain_info[jac_index][n_E] = d_n_E(g);
      jac_index++;
    }
  }
 protected:
  void print_format() override{
    std::cout<<"Jacobian information: F'_i: [ n_i m_i n_E_i ]\n";
  }
};

// m_n_n_E_nnz_Generator populates jac_chain_info data structure
// following the following format. Inner size equals 4.
// n_0 m_0 n_E_0 nnz_0
// with n,m, n_E, nnz taken randomly form a uniform distribution.
// Additionally the m_n_n_E_nnz_Generator generates non zero
// coordinate file following the format
// #sparse matrix structure 0 [n_0 m_0 nnz_0]
// r_0 c_0
// ...
// r_last c_last
// #sparse matrix structure 1 [n_1 m_1 nnz_1]
// r_0 c_0
// ...
//
class n_m_n_E_nnz_Generator: public Generator{
 public:
  n_m_n_E_nnz_Generator(std::size_t chain_length_, std::size_t dim_lb_,
    std::size_t dim_ub_, std::size_t n_E_lb_, std::size_t n_E_ub_,
    double den_lb, double den_ub, bool is_deterministic_ = 0, std::size_t seed_ = 0):
    Generator(chain_length_, 4, dim_lb_, dim_ub_, n_E_lb_, n_E_ub_,
        is_deterministic_, seed_), density_lb(den_lb), density_ub(den_ub){
      if(1 < den_ub || 1 < den_lb || den_lb < 0 || density_ub < 0){
        throw std::invalid_argument("Density must be an element of (0,1).");
      }
      if(den_ub < den_lb){
        throw std::invalid_argument("Density lower bound must be smaller or equal to the upper bound.");
      }
    }

  void build_sparse_problem(){
    std::default_random_engine g;
    
    if(!is_deterministic){
      std::random_device r;
      g.seed(r());
    }

    else{
      g.seed(seed);
    }


    std::uniform_int_distribution<std::size_t> d_dim_in_out(dim_lb, dim_ub);
    std::uniform_int_distribution<std::size_t> d_n_E(n_E_lb, n_E_ub);
    std::uniform_int_distribution<std::size_t> d_nnz;

    std::size_t jac_index=0;
    const std::size_t n=0, m=1, n_E=2, nnz=3;

    jac_chain_info[jac_index][m] = d_dim_in_out(g);
    jac_chain_info[jac_index][n] = d_dim_in_out(g);
    jac_chain_info[jac_index][n_E] = d_n_E(g);
    
    std::size_t nnz_lb, nnz_ub, nm;
    nm = jac_chain_info[jac_index][n] * jac_chain_info[jac_index][m];

    // At least a single non zero entry is guaranteed at every row and column.
    if(jac_chain_info[jac_index][n] <= jac_chain_info[jac_index][m]){
      if(jac_chain_info[jac_index][n] < static_cast<std::size_t>(1/density_lb)){
        nnz_lb = jac_chain_info[jac_index][m];
      }
      else{
        nnz_lb = static_cast<std::size_t>(density_lb * nm);
      }
    }

    else{
      if(jac_chain_info[jac_index][m] < static_cast<std::size_t>(1/density_lb)){
        nnz_lb = jac_chain_info[jac_index][n];
      }
      else{
        nnz_lb = static_cast<std::size_t>(density_lb * nm);
      }
    }

    nnz_ub = static_cast<std::size_t>(density_ub * nm);

    std::uniform_int_distribution<std::size_t>::param_type p(nnz_lb, nnz_ub);
    jac_chain_info[jac_index][nnz] = d_nnz(g, p);
    jac_index++;

    while(jac_index < chain_length){
      jac_chain_info[jac_index][n] = jac_chain_info[jac_index-1][m];
      jac_chain_info[jac_index][m] = d_dim_in_out(g);
      jac_chain_info[jac_index][n_E] = d_n_E(g);

      nm = jac_chain_info[jac_index][n] * jac_chain_info[jac_index][m];
      // At least a single non zero entry is guaranteed at every row and column.
      if(jac_chain_info[jac_index][n] <= jac_chain_info[jac_index][m]){
        if(jac_chain_info[jac_index][n] < static_cast<std::size_t>(1/density_lb)){
          nnz_lb = jac_chain_info[jac_index][m];
        }
        else{
          nnz_lb = static_cast<std::size_t>(density_lb * nm);
        }
      }

      else{
        if(jac_chain_info[jac_index][m] < static_cast<std::size_t>(1/density_lb)){
          nnz_lb = jac_chain_info[jac_index][n];
        }
        else{
          nnz_lb = static_cast<std::size_t>(density_lb * nm);
        }
      }
      nnz_ub = static_cast<std::size_t>(density_ub * nm);

      std::uniform_int_distribution<std::size_t>::param_type p(nnz_lb, nnz_ub);
      jac_chain_info[jac_index][nnz] = d_nnz(g, p);
      jac_index++;
    }
  }

  void build_sparse_structure(const std::string& filename = "sparse_data"){
    if(jac_chain_info.empty()){
      throw std::logic_error("Precondition: build_sparse_problem needs to be called first.");
    }

    std::ofstream outFile(filename);
    if(!outFile){
      std::cerr << "Error opening file."<< std::endl;
    }

    std::default_random_engine g;

    if(!is_deterministic){
      std::random_device r;
      g.seed(r());
    }

    else{
      g.seed(seed);
    }

    std::size_t matrix_idx;
    std::size_t n = 0, m = 1, nnz = 3;
    std::size_t max_m_n, min_m_n, aux, aux_0;
    bool is_new_coordinate = 1;
    std::vector<std::vector<std::size_t>> used_v;
    std::vector<std::size_t> min_m_n_vector;
    std::uniform_int_distribution<std::size_t> d_min_m_n;
    std::uniform_int_distribution<std::size_t> d_max_m_n;

    for(matrix_idx = 0; matrix_idx < jac_chain_info.size(); matrix_idx++){
      outFile << "#sparse matrix structure "<< matrix_idx << ' ';
      outFile << "[ " << jac_chain_info[matrix_idx][n] << ' ' << jac_chain_info[matrix_idx][m];
      outFile << ' ' << jac_chain_info[matrix_idx][nnz] << " ]\n";

      //Jacobians with equal or more rows than columns.
      if(jac_chain_info[matrix_idx][n] <= jac_chain_info[matrix_idx][m]){
        max_m_n = jac_chain_info[matrix_idx][m];
        min_m_n = jac_chain_info[matrix_idx][n];
        used_v.resize(max_m_n);
        min_m_n_vector.resize(min_m_n);
        std::iota(min_m_n_vector.begin(), min_m_n_vector.end(), 0);
        std::shuffle(min_m_n_vector.begin(), min_m_n_vector.end(), g);
        std::uniform_int_distribution<std::size_t>::param_type p_min_m_n(0, min_m_n - 1);
        std::uniform_int_distribution<std::size_t>::param_type p_max_m_n(0, max_m_n - 1);
        for(std::size_t i = 0; i < min_m_n; i++){
          outFile << i << ' ' << min_m_n_vector[i] << '\n';
          used_v[i].push_back(min_m_n_vector[i]);
        }
        for(std::size_t i = min_m_n; i < max_m_n; i++){
          aux = d_min_m_n(g, p_min_m_n);
          outFile << i << ' ' << aux << '\n';
          used_v[i].push_back(aux);
        }
        for(std::size_t i = max_m_n; i < jac_chain_info[matrix_idx][nnz]; i++){
          do{
            is_new_coordinate = 1;
            aux = d_min_m_n(g, p_min_m_n);
            aux_0 = d_max_m_n(g, p_max_m_n);
            for(std::size_t i = 0; i < used_v[aux_0].size(); i++){
              if(used_v[aux_0][i] == aux){is_new_coordinate = 0;}
            }
          }
          while(!is_new_coordinate);
          outFile << aux_0 << ' ' << aux << '\n';
          used_v[aux_0].push_back(aux);
        }
      }
      // Jacobians with more columns than rows
      else{
        max_m_n = jac_chain_info[matrix_idx][n];
        min_m_n = jac_chain_info[matrix_idx][m];
        used_v.resize(max_m_n);
        min_m_n_vector.resize(min_m_n);
        std::iota(min_m_n_vector.begin(), min_m_n_vector.end(), 0);
        std::shuffle(min_m_n_vector.begin(), min_m_n_vector.end(), g);
        std::uniform_int_distribution<std::size_t>::param_type p_min_m_n(0, min_m_n-1);
        std::uniform_int_distribution<std::size_t>::param_type p_max_m_n(0, max_m_n-1);
        for(std::size_t i = 0; i < min_m_n; i++){
          outFile << min_m_n_vector[i] << ' ' << i << '\n';
          used_v[i].push_back(min_m_n_vector[i]);
        }
        for(std::size_t i = min_m_n; i < max_m_n; i++){
          aux = d_min_m_n(g, p_min_m_n);
          outFile << aux << ' ' << i << '\n';
          used_v[i].push_back(aux);
        }
        for(std::size_t i = max_m_n; i < jac_chain_info[matrix_idx][nnz]; i++){
          do{
            is_new_coordinate = 1;
            aux = d_min_m_n(g, p_min_m_n);
            aux_0 = d_max_m_n(g, p_max_m_n);
            for(std::size_t i = 0; i < used_v[aux_0].size(); i++){
              if(used_v[aux_0][i] == aux){is_new_coordinate = 0;}
            }
          }
          while(!is_new_coordinate);
          outFile << aux << ' ' << aux_0 << '\n';
          used_v[aux_0].push_back(aux);
        }
      }

    }
    outFile.close();
  }

  void build_problem(){
    build_sparse_problem();
    build_sparse_structure();
  }

 protected:
  double density_lb;
  double density_ub;
  void print_format() override{
    std::cout<<"Jacobian information: F'_i: [ n_i m_i n_E_i nnz ]\n";
  }
};

#endif
