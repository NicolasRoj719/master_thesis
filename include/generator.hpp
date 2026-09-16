/**
 * @file generator.hpp
 * @brief Type-specific Jacobian generator classes. 
 */
#ifndef GENERATOR_HPP
#define GENERATOR_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include "./util_structs.hpp"


/**
 * @brief Primary template for synthetic Jacobian generator classes.
 *
 * @tparam Matrix_information_type. The metadata type driving generation
 *         (e.g., Jacobian_information, Matrix_free_information, or Matrix_free_sparse_information).
 */
template <class Matrix_information_type>
class Generator{};

/**
 * @brief Generator specialization for standard Jacobian_information metadata.
 */
template<>
class Generator<Jacobian_information>{
 public:
  /**
  * @brief Constructs a Jacobian_information generator.
  *
  * @param chain_length Number of elemental jacobians in the chain.
  * @param dimension_lb Lower bound for domain/codomain dimensions.
  * @param dimension_ub Upper bound for domain/codomain dimensions.
  * @param is_deterministic_ Enables deterministic generation when true.
  * @param seed_ Seed value passed to the random engine.
  *
  * @throws std::invalid_argument If 'chain_length' is zero.
  * @throws std::invalid_argument If 'dimension_ub < dimension_lb'
  */
  Generator(std::size_t chain_length, std::size_t dimension_lb, std::size_t dimension_ub,
      bool is_deterministic_, std::size_t seed_):
  jacobian_chain_length(chain_length), dimension_lower_bound(dimension_lb),
  dimension_upper_bound(dimension_ub), is_deterministic(is_deterministic_),
  seed(seed_){

    if(chain_length == 0){
      throw std::invalid_argument("Chain length must be greater than 0.");
    }
    if(dimension_ub < dimension_lb){
      throw std::invalid_argument("Dimension lower bound must be smaller or equal to the upper bound.");
    }
  }

  /**
   * @brief Generates a chain of Jacobian_information metadata objects.
   * @return Vector of randomly generated Jacobian_information elements.
   */
  std::vector<Jacobian_information> generate_jacobian_information(){
    std::default_random_engine g;
    
    if(!is_deterministic){
      std::random_device r;
      g.seed(r());
    }

    else{
      g.seed(seed);
    }

    // Metadata vector initialization.
    std::vector<Jacobian_information> jacobian_information;
    jacobian_information.reserve(jacobian_chain_length);

    // Initialization of random uniform distributed integer generator
    // within the closed interval [dimension_lower_bound, dimension_lower_bound]
    std::uniform_int_distribution<std::size_t> 
      dimension_probability_distribution(dimension_lower_bound, dimension_upper_bound);

    std::size_t dimension_shared = dimension_probability_distribution(g);
    std::size_t dimension_non_shared = dimension_probability_distribution(g);
    jacobian_information.emplace_back(dimension_non_shared, dimension_shared);
    
    std::size_t counter = 1;
    while(counter < jacobian_chain_length){
      dimension_non_shared = dimension_probability_distribution(g);
      jacobian_information.emplace_back(dimension_shared, dimension_non_shared);
      dimension_shared = dimension_non_shared;
      counter++;
    }
    
    return jacobian_information;
  }

 protected:
  /// Number of elemental Jacobians in the chain.
  std::size_t jacobian_chain_length;
  /// Lower and upper bounds for domain and codomain dimensions.
  std::size_t dimension_lower_bound, dimension_upper_bound;
  /// Flag indicating wether random generation is deterministic.
  bool is_deterministic;
  /// Seed ensuring reproducible generation when is_deterministic is true.
  std::size_t seed;
};

/**
 * @brief Generator specialization for Matrix_free_information metadata.
 */
template<>
class Generator<Matrix_free_information>{
 public:
  /**
   * @brief Constructs a Matrix_free_information generator.
   *
   * @param chain_length Number of elemental jacobians in the chain.
   * @param dimension_lb Lower bound for domain/codomain dimensions.
   * @param dimension_ub Upper bound for domain/codomain dimensions.
   * @param n_E_lb Lower bound for the number of edges in the graph representation. 
   * @param n_E_ub Upper bound for the number of edges in the graph representation. 
   * @param is_deterministic_ Enables deterministic generation when true.
   * @param seed_ Seed value passed to the random engine.
   *
   * @throw std::invalid_argument If 'chain_length' is zero.
   * @throw std::invalid_argument If 'dimension_ub < dimension_lb'.
   * @throw std::invalid_argument If 'n_E_ub < n_E_lb'
   */
  Generator(std::size_t chain_length, std::size_t dimension_lb, std::size_t dimension_ub,
      std::size_t n_E_lb, std::size_t n_E_ub, bool is_deterministic_, std::size_t seed_):
    jacobian_chain_length(chain_length), dimension_lower_bound(dimension_lb),
    dimension_upper_bound(dimension_ub), number_of_edges_lower_bound(n_E_lb),
    number_of_edges_upper_bound(n_E_ub), is_deterministic(is_deterministic_),
    seed(seed_){

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
    }

  /**
   * @brief Generates a chain of Matrix_free_information metadata objects.
   * @return Vector of randomly generated Matrix_free_information elements.
   */
  std::vector<Matrix_free_information> generate_jacobian_information(){
    std::default_random_engine g;
    
    if(!is_deterministic){
      std::random_device r;
      g.seed(r());
    }

    else{
      g.seed(seed);
    }

    // Metadata vector initialization.
    std::vector<Matrix_free_information> jacobian_information;
    jacobian_information.reserve(jacobian_chain_length);

    // Initialization of random uniform distributed integer generator
    // within the closed interval [dimension_lower_bound, dimension_lower_bound]
    std::uniform_int_distribution<std::size_t> 
      dimension_probability_distribution(dimension_lower_bound, dimension_upper_bound);

    // Initialization of random uniform distributed integer generator
    // within the closed interval [number_of_edges_lower_bound, number_of_edges_upper_bound]
    std::uniform_int_distribution<std::size_t>
      number_edges_probability_distribution(number_of_edges_lower_bound, number_of_edges_upper_bound);

    std::size_t dimension_shared = dimension_probability_distribution(g);
    std::size_t dimension_non_shared = dimension_probability_distribution(g);
    std::size_t number_of_edges = number_edges_probability_distribution(g);
    jacobian_information.emplace_back(dimension_non_shared, dimension_shared, number_of_edges);
    
    std::size_t counter = 1;
    while(counter < jacobian_chain_length){
      dimension_non_shared = dimension_probability_distribution(g);
      number_of_edges = number_edges_probability_distribution(g);
      jacobian_information.emplace_back(dimension_shared, dimension_non_shared, number_of_edges);
      dimension_shared = dimension_non_shared;
      counter++;
    }

    return jacobian_information;
  }

 protected:
  /// Number of elemental Jacobians in the chain.
  std::size_t jacobian_chain_length;
  /// Lower and upper bounds for domain and codomain dimensions.
  std::size_t dimension_lower_bound, dimension_upper_bound;
  /// Lower and upper bounds for computational graph edge counts.
  std::size_t number_of_edges_lower_bound, number_of_edges_upper_bound;
  /// Flag indicating wether random generation is deterministic.
  bool is_deterministic;
  /// Seed ensuring reproducible generation when is_deterministic is true.
  std::size_t seed;
};

/**
 * @brief Generator specialization for Matrix_free_sparse_information metadata and sparsity patterns. 
 */
template<>
class Generator<Matrix_free_sparse_information>{
 public:
  /**
   * @brief Constructs a Matrix_free_sparse_information generator.
   *
   * @param chain_length Number of elemental Jacobians in the chain.
   * @param dimension_lb Lower bound for domain/codomain dimensions.
   * @param dimension_ub Upper bound for domain/codomain dimensions.
   * @param n_E_lb Lower bound for the number of edges in computational graph representation. 
   * @param n_E_ub Upper bound for the number of edges in computation graph representation.
   * @param density_lb Suggested density lower bound for elemental jacobians.
   * @param density_ub Suggested density upper bound for elemental jacobians.
   * @param is_deterministic_ Enables deterministc generation when true.
   * @param seed_ Seed value passed to the random engine.
   *
   * @throws std::invalid_argument If 'chain_length' is zero.
   * @throws std::invalid_argument If 'dimension_ub < dimension_lb'
   * @throw std::invalid_argument If 'n_E_ub < n_E_lb'
   */
  Generator(std::size_t chain_length, std::size_t dimension_lb, std::size_t dimension_ub,
      std::size_t n_E_lb, std::size_t n_E_ub, double density_lb, double density_ub,
      bool is_deterministic_, std::size_t seed_):
    jacobian_chain_length(chain_length), dimension_lower_bound(dimension_lb),
    dimension_upper_bound(dimension_ub), number_of_edges_lower_bound(n_E_lb),
    number_of_edges_upper_bound(n_E_ub), density_lower_bound(density_lb),
    density_upper_bound(density_ub), is_deterministic(is_deterministic_),
    seed(seed_){

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
    }

  /**
   * @brief Generates sparse Jacobian metadata for each matrix in the chain.
   * @return Vector of Matrix_free_sparse_information objects.
   */
  std::vector<Matrix_free_sparse_information> generate_jacobian_information(){
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

    std::size_t number_nonzeros_lower_bound, number_nonzeros_upper_bound;

    //Transform density lower and upper bound to number_nonzeros_lower_bound and 
    // number_nonzeros_upper_bound.
    // Notice that this depends on the domain and codomain dimension of the elemental jacobian.
    set_number_of_zeros_upper_and_lower_bound(dimension_non_shared, dimension_shared,
        number_nonzeros_lower_bound, number_nonzeros_upper_bound);

    std::uniform_int_distribution<std::size_t>::param_type 
      num_nonzeros_param(number_nonzeros_lower_bound, number_nonzeros_upper_bound);

    std::size_t number_nonzeros = number_of_nonzeros_distribution(g, num_nonzeros_param); 

    std::vector<Matrix_free_sparse_information> jacobian_information;
    jacobian_information.reserve(jacobian_chain_length);

    jacobian_information.emplace_back(dimension_non_shared, dimension_shared,
        number_of_edges, number_nonzeros);
    
    std::size_t counter = 1;
    while(counter < jacobian_chain_length){
      dimension_non_shared = dimension_probability_distribution(g);
      number_of_edges = number_edges_probability_distribution(g);

      set_number_of_zeros_upper_and_lower_bound(dimension_non_shared, dimension_shared,
          number_nonzeros_lower_bound, number_nonzeros_upper_bound);

      std::uniform_int_distribution<std::size_t>::param_type 
        num_nonzeros_param(number_nonzeros_lower_bound, number_nonzeros_upper_bound);

      number_nonzeros = number_of_nonzeros_distribution(g, num_nonzeros_param); 

      jacobian_information.emplace_back(dimension_shared, dimension_non_shared,
          number_of_edges, number_nonzeros);

      dimension_shared = dimension_non_shared;

      counter++;
    }
    return jacobian_information;
  }


  /**
   * @brief Constructs sparsity patterns (non-zero entries) for each Jacobian in the chain.
   *
   * @param[in] jacobian_information Sparse Jacobian metadata vector for each matrix in the chain. 
   * @param Nested vector containing non-zero (NNZ) index pairs for each elemental Jacobian.
   */
  std::vector<std::vector<NNZ>> generate_sparsity_patterns(
      const std::vector<Matrix_free_sparse_information>& jacobian_information){

    std::default_random_engine g;

    if(!is_deterministic){
      std::random_device r;
      g.seed(r());
    }

    else{
      g.seed(seed);
    }

    //Temporary sparsity pattern holder.
    std::vector<NNZ> sparse_data_per_matrix;
    std::size_t domain_dimension, codomain_dimension;
    std::size_t min_domain_codomain;
    std::size_t number_nonzeros;
    std::uniform_int_distribution<std::size_t> non_zero_distribution;
    std::uniform_int_distribution<std::size_t> non_zero_distribution_aux;
    std::size_t row, column;

    //Chain sparsity pattern vector.
    std::vector<std::vector<NNZ>> sparse_data;
    sparse_data.reserve(jacobian_chain_length);

    //Loop over elemental jacobians
    for(std::size_t matrix_idx = 0; matrix_idx < jacobian_chain_length; matrix_idx++){
      number_nonzeros = jacobian_information[matrix_idx].number_nnz();
      sparse_data_per_matrix.reserve(number_nonzeros);
      domain_dimension = jacobian_information[matrix_idx].domain_dim();
      codomain_dimension = jacobian_information[matrix_idx].codomain_dim();
      
      min_domain_codomain = 
        (domain_dimension < codomain_dimension)? domain_dimension : codomain_dimension;

      //Populate the biggest square matrice within the elemental jacobian with a shuffled
      // version of the identity matrix.
      generate_data_square_submatrix(sparse_data_per_matrix, g, min_domain_codomain);
      
      //More rows than columns.
      if(min_domain_codomain == domain_dimension){

        std::uniform_int_distribution<std::size_t>::param_type param(0, domain_dimension - 1);

        //Populating the remaining rows one element per row.
        for(std::size_t i = min_domain_codomain; i < codomain_dimension; i++){
          sparse_data_per_matrix.emplace_back(i, non_zero_distribution(g, param));
        }

        //Placing missing non zero entries randomly inside the matrix.
        for(std::size_t i = codomain_dimension; i < number_nonzeros; i++){
          std::uniform_int_distribution<std::size_t>::param_type param_aux(0, codomain_dimension - 1);
          row = non_zero_distribution(g, param_aux);
          column = non_zero_distribution_aux(g, param);

          //Check if the randomly generated non zero entry is new to the current sparsity pattern. 
          while(!(std::find(sparse_data_per_matrix.begin(), sparse_data_per_matrix.end(),
                NNZ {row, column}) == sparse_data_per_matrix.end())){
            row = non_zero_distribution(g, param_aux);
            column = non_zero_distribution_aux(g, param);
          }

          sparse_data_per_matrix.emplace_back(row, column);
        }
      }

      // Number of columns equals numbers of rows or there are more columns than rows.
      else{

        std::uniform_int_distribution<std::size_t>::param_type param(0, codomain_dimension - 1);

        // Populating the remaining columns one element per column.
        for(std::size_t i = min_domain_codomain; i < domain_dimension; i++){
          sparse_data_per_matrix.emplace_back(non_zero_distribution(g, param), i);
        }

        // Placing missing non zero entries randomly inside the matrix.
        for(std::size_t i = domain_dimension; i < number_nonzeros; i++){
          std::uniform_int_distribution<std::size_t>::param_type param_aux(0, domain_dimension - 1);
          row = non_zero_distribution(g, param);
          column = non_zero_distribution_aux(g, param_aux);

          // Check if the randomly generated non zero entry is new to the current sparsity patterns.
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

    return sparse_data;
  }

  
  /**
   * @brief Generates complete synthetic data containing sparse medata and explicit sparsity patterns. 
   * 
   * @return Generator_data container with generated metadata and sparsity patterns.
   */
  Generator_data generate_data(){

    auto sparse_jacobian_information = generate_jacobian_information();

    return Generator_data{sparse_jacobian_information,
            generate_sparsity_patterns(sparse_jacobian_information)};
  }

 protected:
  /// Number of elemental Jacobians in the chain. 
  std::size_t jacobian_chain_length;
  /// Lower and upper bounds for domain and codomain dimensions.
  std::size_t dimension_lower_bound, dimension_upper_bound;
  /// Lower and upper bound for computational graph edge counts.
  std::size_t number_of_edges_lower_bound, number_of_edges_upper_bound;
  /// Suggested lower and upper bounds for the elemental Jacobian density in the chain. 
  double density_lower_bound, density_upper_bound;
  /// Flag indicating wether random generation is deterministic.
  bool is_deterministic;
  /// Seed ensuring reproducible generation when is_deterministic is true.
  std::size_t seed;


  /**
   * @brief Computes bounds for the number of non-zero entries based on density limits.
   *
   * Ensured condition: \f$ \text{NNZ}_{\text{min}} \ge \max(\text{domain}, \text{codomain}) \f$
   * so every row and column has at least one non-zero entry.
   *
   * @param[in] domain_dimension Domain dimension of the Jacobian.
   * @param[in] codomain_dimension Codomain dimension of the Jacobian.
   * @param[out] number_of_zeros_lower_bound Lower bound on the non-zero element count.
   * @paramp[out] number_of_zeros_upper_bound Upper bound on the non-zero element count.
   */
  void set_number_of_zeros_upper_and_lower_bound(std::size_t domain_dimension,
      std::size_t codomain_dimension, std::size_t& number_of_zeros_lower_bound,
      std::size_t& number_of_zeros_upper_bound){

    std::size_t max_domain_codomain = 
      (domain_dimension > codomain_dimension)? domain_dimension : codomain_dimension;

    number_of_zeros_lower_bound = 
    static_cast<std::size_t>(std::ceil(density_lower_bound * domain_dimension * codomain_dimension / max_domain_codomain));

    // Lower bound for the number of edges must be at least equal to the maximum 
    // between domain and codomain dimension.
    if(number_of_zeros_lower_bound < max_domain_codomain){
      number_of_zeros_lower_bound = max_domain_codomain;
    }

    number_of_zeros_upper_bound =
    static_cast<std::size_t>(std::floor(density_upper_bound * domain_dimension * codomain_dimension / max_domain_codomain));

    // Lower bound for the number of edges must be at least equal to the maximum 
    // between domain and codomain dimension.
    if(number_of_zeros_upper_bound < max_domain_codomain){
      number_of_zeros_upper_bound = max_domain_codomain;
    }
  }

  /**
   * @brief Initializes a square submatrix with a shuffled identity matrix pattern.
   *
   * @param[in,out] single_sparse_data Empty sparsity pattern.
   * @param[in] random_engine Reuse the random engine initialized inside generate_data method.
   * @param min_domain_codomain minimum between the dimension of the domain and codomain.
   *
   */
  void generate_data_square_submatrix(std::vector<NNZ>& single_sparse_data,
      std::default_random_engine& random_engine, std::size_t min_domain_codomain){

    std::vector<std::size_t> index_vector(min_domain_codomain);
    std::iota(index_vector.begin(), index_vector.end(), 0);
    std::shuffle(index_vector.begin(), index_vector.end(), random_engine);
    
    for(std::size_t i = 0; i < min_domain_codomain; i++){
      single_sparse_data.emplace_back(i, index_vector[i]);
    }
  }

};
#endif
