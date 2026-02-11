#include <random>
#include <cmath>
#include <iostream>
#include <vector>

#ifndef GENERATOR_HPP
#define GENERATOR_HPP

class Generator{
 public:
  Generator(std::size_t chain_length_, std::size_t inner_size, std::size_t dim_lb_,
      std::size_t dim_ub_, std::size_t n_E_lb_, std::size_t n_E_ub_):
    chain_length(chain_length_),
    jac_chain_info(chain_length_, std::vector<std::size_t>(inner_size)),
    dim_lb(dim_lb_), dim_ub(dim_ub_), n_E_lb(n_E_lb_), n_E_ub(n_E_ub_){}

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

  const std::vector<std::vector<std::size_t>> get_problem(){
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
};

// m_n_n_E_Generator populates jac_chain_info data structure
// following the following format
// n_0 m_0 n_E_0
// ...
// n_chain_length-1 m_chain_length-1 n_E_chain_length-1
// with n,m and n_E taken randomely form a uniform distribution.
class n_m_n_E_Generator: public Generator{
 public:
  //Inner size is set to 3, corresponding to n,m,n_E.
  n_m_n_E_Generator(std::size_t chain_length_, std::size_t dim_lb_,
    std::size_t dim_ub_, std::size_t n_E_lb_, std::size_t n_E_ub_):
    Generator(chain_length_, 3, dim_lb_, dim_ub_, n_E_lb_, n_E_ub_){}

  void build_problem(){

    std::random_device r;
    std::default_random_engine g(r());

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
    std::cout<<"Jacobian chain information: F'_i: [ n_i m_i n_E_i ]\n";
  }
};

#endif
