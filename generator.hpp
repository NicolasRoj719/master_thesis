#include <random>
#include <cmath>
#include <iostream>
#include <vector>

class Generator{
 public:
  Generator(std::size_t chain_length_): chain_length(chain_length_){}

  virtual void print_problem() = 0;

  virtual ~Generator() = default;

  std::size_t const chain_length;
};

//m_n_n_E_Generator derived class populates a data structure chain_length 
//with the elemental jacobian essential information (m, n, n_E).
//Given the length of the Jacobian chain and lower and upper bound for 
//input and output dimensions and the number of edges. Corresponding random
//numbers are obtained following a uniform distribution.
class m_n_n_E_Generator: public Generator{
 public:
  m_n_n_E_Generator(std::size_t chain_length, std::size_t dim_lower_bound, std::size_t dim_upper_bound,
      std::size_t n_E_lower_bound, std::size_t n_E_upper_bound): Generator(chain_length),
  jac_chain_info(chain_length, std::vector<std::size_t>(3)){
    std::random_device r;
    std::default_random_engine g(r());

    std::uniform_int_distribution<std::size_t> d_dim_in_out(dim_lower_bound, dim_upper_bound);
    std::uniform_int_distribution<std::size_t> d_n_E(n_E_lower_bound, n_E_upper_bound);

    std::size_t jac_index=0;
    const std::size_t m=0, n=1, n_E=2;

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

  void print_problem() override{
    std::size_t jac_index = 0;

    std::cout<<"Jacobian chain information: F'_i: [ m_i n_i n_E_i ]\n";

    for(std::vector<std::size_t> F_i_info: jac_chain_info){
      std::cout<<"F'_"<<jac_index<<": [ ";
      for(std::size_t data: F_i_info){
        std::cout<<data<<' ';
      }
      std::cout<<"]\n";
      jac_index++;
    }
  }

  //This method will be used to build the elemental jacobians array.
  std::vector<size_t> read_Fi_description(std::size_t i){
    return(jac_chain_info[i]);
  }

 private:
   //vector to store m,n,n_E
   std::vector<std::vector<std::size_t>> jac_chain_info;
};
