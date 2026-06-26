#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include "./util_structs.hpp"

#ifndef GENERATOR_HPP
#define GENERATOR_HPP

template <class Matrix_information_type>
class Generator{};

template<>
class Generator<Jacobian_information>{
 public:
  Generator(std::size_t chain_length, std::size_t dimension_lb, std::size_t dimension_ub,
      bool is_deterministic_, std::size_t seed = 45):
  jacobian_chain_lenght(chain_length), dimension_lower_bound(dimension_lb),
  dimension_upper_bound(dimension_ub), is_deterministic(is_deterministic_){

    if(chain_length == 0){
      throw std::invalid_argument("Chain length must be greater than 0.");
    }
    if(dimension_ub < dimension_lb){
      throw std::invalid_argument("Dimension lower bound must be smaller or equal to the upper bound.");
    }
    jacobian_information.reserve(chain_length);
    generate_jacobian_information(seed);
  }

  std::size_t jacobian_information_size() const {return jacobian_information.size();}

  std::vector<Jacobian_information> get_jacobian_information_copy() const{
    return jacobian_information;  
  }

  const std::vector<Jacobian_information>& get_jacobian_information_reference() const{
    return jacobian_information;
  } 

 protected:
  std::size_t jacobian_chain_lenght;
  std::size_t dimension_lower_bound, dimension_upper_bound;
  std::vector<Jacobian_information> jacobian_information;
  bool is_deterministic;
  
  void generate_jacobian_information(std::size_t seed){
    std::default_random_engine g;
    
    if(!is_deterministic){
      std::random_device r;
      g.seed(r());
    }

    else{
      g.seed(seed);
    }

    std::uniform_int_distribution<std::size_t> 
      dimension_probability_distribution(dimension_lower_bound, dimension_upper_bound);

    std::size_t dimension_shared = dimension_probability_distribution(g);
    std::size_t dimension_non_shared = dimension_probability_distribution(g);
    jacobian_information.emplace_back(dimension_non_shared, dimension_shared);
    
    std::size_t counter = 1;
    while(counter < jacobian_chain_lenght){
      dimension_non_shared = dimension_probability_distribution(g);
      jacobian_information.emplace_back(dimension_shared, dimension_non_shared);
      dimension_shared = dimension_non_shared;
      counter++;
    }
  }
};

template<>
class Generator<Matrix_free_infomation>{
 public:
  Generator(std::size_t chain_length, std::size_t dimension_lb, std::size_t dimension_ub,
      std::size_t n_E_lb, std::size_t n_E_ub, bool is_deterministic_, std::size_t seed = 45):
    jacobian_chain_lenght(chain_length), dimension_lower_bound(dimension_lb),
    dimension_upper_bound(dimension_ub), number_of_edges_lower_bound(n_E_lb),
    number_of_edges_upper_bound(n_E_ub), is_deterministic(is_deterministic_){

      if(chain_length == 0){
        throw std::invalid_argument("Chain length must be greater than 0.");
      }
      if(dimension_ub < dimension_lb){
      throw std::invalid_argument("Dimension lower bound must be smaller or " 
          "equal its upper bound.");
      }
      if(n_E_ub < n_E_lb){
      throw std::invalid_argument("Number of edges lower bound must be smaller " 
          "or equal to its upper bound.");
      }
      jacobian_information.reserve(chain_length);
      generate_jacobian_information(seed);
    }

  std::size_t jacobian_information_size() const{return jacobian_information.size();}

  std::vector<Matrix_free_infomation> get_jacobian_information_copy() const{
    return jacobian_information;  
  }

  const std::vector<Matrix_free_infomation>& get_jacobian_information_reference() const{
    return jacobian_information;
  } 

 protected:
  std::size_t jacobian_chain_lenght;
  std::size_t dimension_lower_bound, dimension_upper_bound;
  std::size_t number_of_edges_lower_bound, number_of_edges_upper_bound;
  std::vector<Matrix_free_infomation> jacobian_information;
  bool is_deterministic;

  void generate_jacobian_information(std::size_t seed){
    std::default_random_engine g;
    
    if(!is_deterministic){
      std::random_device r;
      g.seed(r());
    }

    else{
      g.seed(seed);
    }

    std::uniform_int_distribution<std::size_t> 
      dimension_probability_distribution(dimension_lower_bound, dimension_upper_bound);
    std::uniform_int_distribution<std::size_t>
      number_edges_probability_distribution(number_of_edges_lower_bound, number_of_edges_upper_bound);

    std::size_t dimension_shared = dimension_probability_distribution(g);
    std::size_t dimension_non_shared = dimension_probability_distribution(g);
    std::size_t number_of_edges = number_edges_probability_distribution(g);
    jacobian_information.emplace_back(dimension_non_shared, dimension_shared, number_of_edges);
    
    std::size_t counter = 1;
    while(counter < jacobian_chain_lenght){
      dimension_non_shared = dimension_probability_distribution(g);
      number_of_edges = number_edges_probability_distribution(g);
      jacobian_information.emplace_back(dimension_shared, dimension_non_shared, number_of_edges);
      dimension_shared = dimension_non_shared;
      counter++;
    }
  }
};

template<>
class Generator<Matrix_free_sparse_information>{
 public:
  Generator(std::size_t chain_length, std::size_t dimension_lb, std::size_t dimension_ub,
      std::size_t n_E_lb, std::size_t n_E_ub, double density_lb, double density_ub,
      bool is_deterministic_, std::size_t seed):
    jacobian_chain_lenght(chain_length), dimension_lower_bound(dimension_lb),
    dimension_upper_bound(dimension_ub), number_of_edges_lower_bound(n_E_lb),
    number_of_edges_upper_bound(n_E_ub), density_lower_bound(density_lb),
    density_upper_bound(density_ub), is_deterministic(is_deterministic_){

      if(chain_length == 0){
        throw std::invalid_argument("Chain length must be greater than 0.");
      }
      if(dimension_ub < dimension_lb){
      throw std::invalid_argument("Dimension lower bound must be smaller or " 
          "equal its upper bound.");
      }
      if(n_E_ub < n_E_lb){
      throw std::invalid_argument("Number of edges lower bound must be smaller " 
          "or equal to its upper bound.");
      }

      jacobian_information.reserve(chain_length);
      generate_jacobian_information(seed);
      sparse_data.reserve(chain_length);
      generate_sparse_data(seed);
    }

  std::size_t jacobian_information_size() const {return jacobian_information.size();}
  std::size_t sparse_data_size() const {return sparse_data.size();}

  std::vector<Matrix_free_sparse_information> get_jacobian_information_copy() const{
    return jacobian_information;  
  }

  const std::vector<Matrix_free_sparse_information>& get_jacobian_information_reference() const{
    return jacobian_information;
  } 

  std::vector<std::vector<NNZ>> get_sparse_data_copy() const{
    return sparse_data;
  }

  const std::vector<std::vector<NNZ>>& get_sparse_data_reference() const{
    return sparse_data;
  }

 protected:
  std::size_t jacobian_chain_lenght;
  std::size_t dimension_lower_bound, dimension_upper_bound;
  std::size_t number_of_edges_lower_bound, number_of_edges_upper_bound;
  double density_lower_bound, density_upper_bound;

  std::vector<Matrix_free_sparse_information> jacobian_information;
  std::vector<std::vector<NNZ>> sparse_data;
  bool is_deterministic;

  void set_number_of_zeros_upper_and_lower_bound(std::size_t domain_dimension,
      std::size_t codomain_dimension, std::size_t& number_of_zeros_lower_bound,
      std::size_t& number_of_zeros_upper_bound){

    std::size_t max_domain_codomain = 
      (domain_dimension > codomain_dimension)? domain_dimension : codomain_dimension;

    number_of_zeros_lower_bound = 
    static_cast<std::size_t>(std::ceil(density_lower_bound * domain_dimension * codomain_dimension / max_domain_codomain));

    //The lower limit should be set such that the condition:
    //all columns and rows have at least one non zero element.
    if(number_of_edges_lower_bound < max_domain_codomain){
      number_of_edges_lower_bound = max_domain_codomain;
    }

    number_of_zeros_upper_bound =
    static_cast<std::size_t>(std::floor(density_upper_bound * domain_dimension * codomain_dimension / max_domain_codomain));

    if(number_of_zeros_upper_bound < max_domain_codomain){
      number_of_edges_upper_bound = max_domain_codomain;
    }
  }

  void generate_jacobian_information(std::size_t seed){
    std::default_random_engine g;
    
    if(!is_deterministic){
      std::random_device r;
      g.seed(r());
    }

    else{
      g.seed(seed);
    }

    std::uniform_int_distribution<std::size_t> 
      dimension_probability_distribution(dimension_lower_bound, dimension_upper_bound);
    std::uniform_int_distribution<std::size_t>
      number_edges_probability_distribution(number_of_edges_lower_bound, number_of_edges_upper_bound);
    std::uniform_int_distribution<std::size_t>
      number_of_nonzeros_distribution;

    std::size_t dimension_shared = dimension_probability_distribution(g);
    std::size_t dimension_non_shared = dimension_probability_distribution(g);
    std::size_t number_of_edges = number_edges_probability_distribution(g);

    set_number_of_zeros_upper_and_lower_bound(dimension_non_shared, dimension_shared,
        number_of_edges_lower_bound, number_of_edges_upper_bound);
    std::uniform_int_distribution<std::size_t>::param_type 
      num_nonzeros_param(number_of_edges_lower_bound, number_of_edges_upper_bound);
    std::size_t number_nonzeros = number_of_nonzeros_distribution(g, num_nonzeros_param); 

    jacobian_information.emplace_back(dimension_non_shared, dimension_shared,
        number_of_edges, number_nonzeros);
    
    std::size_t counter = 1;
    while(counter < jacobian_chain_lenght){
      dimension_non_shared = dimension_probability_distribution(g);
      number_of_edges = number_edges_probability_distribution(g);
      set_number_of_zeros_upper_and_lower_bound(dimension_non_shared, dimension_shared,
          number_of_edges_lower_bound, number_of_edges_upper_bound);
      std::uniform_int_distribution<std::size_t>::param_type 
        num_nonzeros_param(number_of_edges_lower_bound, number_of_edges_upper_bound);
      number_nonzeros = number_of_nonzeros_distribution(g, num_nonzeros_param); 
      jacobian_information.emplace_back(dimension_shared, dimension_non_shared,
          number_of_edges, number_nonzeros);
      dimension_shared = dimension_non_shared;
      counter++;
    }
  }

  void generate_data_square_submatrix(std::vector<NNZ>& single_sparse_data,
      std::default_random_engine& random_engine, std::size_t min_domain_codomain){

    std::vector<std::size_t> index_vector(min_domain_codomain);
    std::iota(index_vector.begin(), index_vector.end(), 0);
    std::shuffle(index_vector.begin(), index_vector.end(), random_engine);
    
    for(std::size_t i = 0; i < min_domain_codomain; i++){
      single_sparse_data.emplace_back(i, index_vector[i]);
    }
  }

  void generate_sparse_data(std::size_t seed){

    std::default_random_engine g;

    if(!is_deterministic){
      std::random_device r;
      g.seed(r());
    }

    else{
      g.seed(seed);
    }

    std::vector<NNZ> sparse_data_per_matrix;
    std::size_t domain_dimension, codomain_dimension;
    std::size_t min_domain_codomain;
    std::size_t number_nonzeros;
    std::uniform_int_distribution<std::size_t> non_zero_distribution;
    std::uniform_int_distribution<std::size_t> non_zero_distribution_aux;
    std::size_t row, column;

    for(std::size_t matrix_idx = 0; matrix_idx < jacobian_chain_lenght; matrix_idx++){
      number_nonzeros = jacobian_information[matrix_idx].number_of_nonzeros();
      sparse_data_per_matrix.reserve(number_nonzeros);
      domain_dimension = jacobian_information[matrix_idx].domain_dimension();
      codomain_dimension = jacobian_information[matrix_idx].codomain_dimension();
      
      min_domain_codomain = 
        (domain_dimension < codomain_dimension)? domain_dimension : codomain_dimension;

      generate_data_square_submatrix(sparse_data_per_matrix, g, min_domain_codomain);
      
      if(min_domain_codomain == domain_dimension){

        std::uniform_int_distribution<std::size_t>::param_type param(0, domain_dimension - 1);
        for(std::size_t i = min_domain_codomain; i < codomain_dimension; i++){
          sparse_data_per_matrix.emplace_back(i, non_zero_distribution(g, param));
        }

        for(std::size_t i = codomain_dimension; i < number_nonzeros; i++){
          std::uniform_int_distribution<std::size_t>::param_type param_aux(0, codomain_dimension - 1);
          row = non_zero_distribution(g, param_aux);
          column = non_zero_distribution_aux(g, param);

          while(!(std::find(sparse_data_per_matrix.begin(), sparse_data_per_matrix.end(),
                NNZ {row, column}) == sparse_data_per_matrix.end())){
            row = non_zero_distribution(g, param_aux);
            column = non_zero_distribution_aux(g, param);
          }

          sparse_data_per_matrix.emplace_back(row, column);
        }
      }

      else{

        std::uniform_int_distribution<std::size_t>::param_type param(0, codomain_dimension - 1);
        for(std::size_t i = min_domain_codomain; i < domain_dimension; i++){
          sparse_data_per_matrix.emplace_back(non_zero_distribution(g, param), i);
        }

        for(std::size_t i = domain_dimension; i < number_nonzeros; i++){
          std::uniform_int_distribution<std::size_t>::param_type param_aux(0, domain_dimension - 1);
          row = non_zero_distribution(g, param);
          column = non_zero_distribution_aux(g, param_aux);

          while(!(std::find(sparse_data_per_matrix.begin(), sparse_data_per_matrix.end(),
                NNZ {row, column}) == sparse_data_per_matrix.end())){
            row = non_zero_distribution(g, param);
            column = non_zero_distribution_aux(g, param_aux);
          }

          sparse_data_per_matrix.emplace_back(row, column);
        }
      }
      sparse_data.emplace_back(sparse_data_per_matrix);
      sparse_data_per_matrix.clear();
    }
  }

};
#endif
