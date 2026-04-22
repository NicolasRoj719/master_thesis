#include <random>
#include <cmath>
#include <iostream>
#include <vector>
#include <stdexcept>

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

  virtual void print_format() = 0;

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

  virtual ~Generator() = default;

  virtual void build_problem() = 0;

  std::size_t get_problem_size(){return jac_chain_info.size();}

  const std::vector<std::vector<std::size_t>>& get_problem(){
    return jac_chain_info;
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

    jac_chain_info[jac_index][m] = d_dim_in_out(g);
    jac_chain_info[jac_index][n] = d_dim_in_out(g);
    jac_chain_info[jac_index][n_E] = d_n_E(g);
    jac_index++;

    while(jac_index < chain_length){
      jac_chain_info[jac_index][n] = jac_chain_info[jac_index-1][m];
      jac_chain_info[jac_index][m] = d_dim_in_out(g);
      jac_chain_info[jac_index][n_E] = d_n_E(g);
      jac_index++;
    }
  }

  void print_format() override{
    std::cout<<"Jacobian information: F'_i: [ n_i m_i n_E_i ]\n";
  }
};

// m_n_n_E_nnz_Generator populates jac_chain_info data structure
// following the following format. Inner size equals 4.
// n_0 m_0 n_E_0 nnz_0
// with n,m and n_E taken randomely form a uniform distribution.
// Additionally the m_n_n_E_nnz_Generator generates non zero
// coordinate file following the format
// #matrix 1
// r_0 c_0
// ...
// r_last c_last
// #matrix 2
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

  void build_problem(){
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
    
    //Minimum density of a single non zero element per row.
    std::size_t nnz_lb, nnz_ub, nm;
    nm = jac_chain_info[jac_index][n] * jac_chain_info[jac_index][m];

    if(jac_chain_info[jac_index][m] < static_cast<std::size_t>(1/density_lb)){
      nnz_lb = jac_chain_info[jac_index][n];
    }
    else{
      nnz_lb = static_cast<std::size_t>(density_lb * nm);
    }
    nnz_ub = static_cast<std::size_t>(density_ub * nm);

    std::uniform_int_distribution<std::size_t>::param_type p(nnz_lb, nnz_lb);
    jac_chain_info[jac_index][nnz] = d_nnz(g, p);
    jac_index++;

    while(jac_index < chain_length){
      jac_chain_info[jac_index][n] = jac_chain_info[jac_index-1][m];
      jac_chain_info[jac_index][m] = d_dim_in_out(g);
      jac_chain_info[jac_index][n_E] = d_n_E(g);

      nm = jac_chain_info[jac_index][n] * jac_chain_info[jac_index][m];
      if(jac_chain_info[jac_index][m] < static_cast<std::size_t>(1/density_lb)){
        nnz_lb = jac_chain_info[jac_index][n];
      }
      else{
        nnz_lb = static_cast<std::size_t>(density_lb * nm);
      }
      nnz_ub = static_cast<std::size_t>(density_ub * nm);

      std::uniform_int_distribution<std::size_t>::param_type p(nnz_lb, nnz_ub);
      jac_chain_info[jac_index][nnz] = d_nnz(g, p);
      jac_index++;
    }
  }

  void print_format() override{
    std::cout<<"Jacobian information: F'_i: [ n_i m_i n_E_i nnz ]\n";
  }

 private:
  double density_lb;
  double density_ub;
};

#endif
