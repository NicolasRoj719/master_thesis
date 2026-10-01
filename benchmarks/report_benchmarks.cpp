#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <format>
#include <iomanip>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "generator.hpp"
#include "chain.hpp"
#include "fill_table.hpp"
#include "optimal_accumulation.hpp"
#include "util_structs.hpp"
#include "binomial_checkpointing.hpp"


// Helper macros to force preprocessor stringification
/* #define STRINGIFY(x) #x */
/* #define TOSTRING(x) STRINGIFY(x) */

/**
 * @brief a pseudo random number generator is used to produce a given number of seeds given a 
 * master seed. 
 *
 * @param master_seed this seed will garanty that the seeds used for the results are 
 * reproducible.
 * @param number_seeds number of seeds to produce. This equals to the number of experiments to 
 * be executed.
 * @return vector with number_seeds pseudo random seeds. 
 */
std::vector<uint64_t> generate_reproducible_seeds(uint64_t master_seed, std::size_t number_seeds){

  std::seed_seq seq{

    static_cast<uint32_t>(master_seed),
    static_cast<uint32_t>(master_seed >> 32),
  };

  std::vector<uint64_t> seeds(number_seeds);

  if constexpr (sizeof(std::size_t) >= 8) {
      // 64-bit systems: Use 64-bit Mersenne Twister
      std::mt19937_64 rng(seq);

      for (std::size_t i = 0; i < number_seeds; ++i) {

        seeds[i] = static_cast<std::size_t>(rng());
      }
  } 
  else {
      // 32-bit systems: Use 32-bit Mersenne Twister
      std::mt19937 rng(seq);

      for (std::size_t i = 0; i < number_seeds; ++i) {

        seeds[i] = static_cast<std::size_t>(rng());
    }
  }

  return seeds;
}

/**
 * @brief Uses the generated data for the Sparse case to create a vector with elements 
 * of type Matrix_free_information.
 *
 * @param jacobian_information The vector of Matrix_free_sparse_information is created inside the 
 * function and its information its used to initialize jac_info.
 * @return A vector with Matrix_free_information that will then be used to initialize the Dense 
 * Jacobian chain.
 */
std::vector<Matrix_free_information> jacobian_information_slicing(
      std::vector<Matrix_free_sparse_information> jacobian_information){
  
  std::vector<Matrix_free_information> jac_info;
  jac_info.reserve(jacobian_information.size());

  for(std::size_t idx = 0; idx < jacobian_information.size(); idx++){

   jac_info.push_back(static_cast<Matrix_free_information>(jacobian_information[idx])); 
  }

  return jac_info;
}

/**
 * @brief A Jacobian_information vector is initialized from a vector of Matrix_free_information 
 * elements via object slicing.
 *
 * @param[in, out] jacobian_information The vector is used to initialize jac_info vector. After 
 * initialization of vector jac_info the memory allocated for jacobian_information vector is 
 * deallocated.
 * @return A vector with Jacobian_information elements that sill then be used to initialize the 
 * Jacobian chain.
 */
std::vector<Jacobian_information> jacobian_information_slicing(
                                    std::vector<Matrix_free_information>& jacobian_information){

  std::vector<Jacobian_information> jac_info;
  jac_info.reserve(jacobian_information.size());

  for(std::size_t idx = 0; idx < jacobian_information.size(); idx++){

    jac_info.push_back(static_cast<Jacobian_information>(jacobian_information[idx]));
  }

  jacobian_information.clear();
  jacobian_information.shrink_to_fit();

  return jac_info;
}

/**
 * @brief Creates a vector with values looking like a v shape with minimum value equal to 
 * minimum values and increases monotonically with slope 'slope' to both sides.
 *
 * @pre number_values must be an odd number.
 * @param minimum_value Mininum value to be found at the middle point of the array.
 * @param number_values Odd number of values to generate.
 * @param 'slope' Slope.
 * @return subprograms_cost Array with subprograms cost.
 */
std::vector<std::size_t> v_shape(std::size_t minimum_value, int number_values,
                                  std::size_t slope){
  
  std::vector<std::size_t> subprograms_cost;
  subprograms_cost.reserve(number_values);


  for(int idx = -(number_values - 1)/2; idx <= (number_values - 1)/2; idx++){
    
    subprograms_cost.push_back(minimum_value + std::abs(idx) * slope);
  }
  return subprograms_cost;
}

/**
 * @brief Creates a vector with values looking like a ^ shape with maximum value equal to 
 * maximum values and decreases monotonically with slope 'slope' to both sides.
 *
 * @pre number_values must be an odd number.
 * @param maximum_value Maximum value to be found at the middle point of the array.
 * @param number_values Odd number of values to generate.
 * @param slope Slope.
 * @param[out] subprograms_cost Array with subprograms cost.
 */
std::vector<std::size_t> n_shape(std::size_t maximum_value, int number_values,
                                  std::size_t slope){
  
  std::vector<std::size_t> subprograms_cost; 
  subprograms_cost.reserve(number_values);

  for(int idx = -(number_values - 1)/2; idx <= (number_values - 1)/2; idx++){
    
    subprograms_cost.push_back(maximum_value - std::abs(idx) * slope);
  }
  return subprograms_cost;
}


/**
 * @brief Produces either a increasing or decreasing linear array.
 *
 * @param initial_value.
 * @param number_values.
 * @param slope.
 * @param[out] subprograms_cost Array with subprograms execution cost data.
 */
std::vector<std::size_t> linear_array(std::size_t initial_value, std::size_t number_values,
                                        int slope){

  std::vector<std::size_t> subprograms_cost; 
  subprograms_cost.reserve(number_values);

  for(std::size_t idx = 0; idx < number_values; idx++){

    subprograms_cost.push_back(initial_value + slope * idx);
  }
  return subprograms_cost;
}

/**
 * @brief Executes all three available dynamic programming algorithms and stores data into three 
 * different files:
 * 1)chain_information: This file contains the following information.
 * chain_length input_dim out_dim number_edges number_nnz column_number_colors max_number_nnz_row
 * row_number_colors max_number_nnz_column
 * 2)visualization: DOT file with the three representation for each optimal solution.
 * 3)optimal_costs: sparse_optimal_cost dense_optimal_cost ratio jacobian_optimal_cost.
 *
 * The data to be found inside ./results/sparse_vs_dense/density_0.05_0.1 and 
 * ./results/sparse_vs_dense/density_0.1_0.15 was generated with the following arguments
 *
 * chain_length = 5
 * dimension_lb = 5, dimension_ub = 20
 * number_edges_lb = 100, number_edges_ub = 300
 * density_lb = 0.05, density_ub = 0.1
 *
 * 
 * chain_length = 5
 * dimension_lb = 5, dimension_ub = 20
 * number_edges_lb = 100, number_edges_ub = 300
 * density_lb = 0.1, density_ub = 0.15 
 *
 * seeds_arr array data was generated using as master seeds 2012 and 1942 respectively, with a 
 * total of 3 seeds generated.
 */

void general_execution(const std::vector<std::size_t>& seeds_arr, std::size_t chain_length,
                              std::size_t dimension_lb, std::size_t dimension_ub,
                              std::size_t number_edges_lb, std::size_t number_edges_ub,
                              double density_lb, double density_ub){

  std::size_t test_counter = 0;
  std::string file_path_fixed = std::string(RESULTS_DIR) + 
                                std::format("density_{}_{}/", density_lb, density_ub);
  std::string file_path;
  std::ofstream out_file;

  while(test_counter < seeds_arr.size()){

    //Generator instance to create synthetic data.
    Generator<Matrix_free_sparse_information> gen{chain_length, dimension_lb, dimension_ub,
                                                    number_edges_lb, number_edges_ub,
                                                    density_lb, density_ub, true,
                                                    seeds_arr[test_counter]};

    std::string test_counter_string = std::to_string(test_counter);
    std::size_t optimal_cost_sparse;
    {
      Jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information> chain{gen.generate_data()};

      file_path = file_path_fixed + "chain_information_" + test_counter_string;
      out_file.open(file_path);

      if(out_file.is_open()){

        chain.print(out_file);
        out_file.close();
      }
      else{

        std::cerr << "Error: Unable to open file " << file_path << " for writing.\n";
      }

      //Construct solver and fill loop up table.
      Fill_table<Sparse_Jacobian, Matrix_free_sparse_information> solver{chain};
      optimal_cost_sparse = solver.get_optimal_cost();

      file_path = file_path_fixed + "visualizations/sparse_" + test_counter_string;
      out_file.open(file_path);

      if(out_file.is_open()){

        //Generate Graphviz accumulation tree
        std::vector<Node_matrix_free> optimal_accumulation_sequence =
          operation_sequence_accumulation<Node_matrix_free>(solver.get_table(), chain.size(),
                                                              &out_file);
        out_file.close();
      }
      else{

        std::cerr << "Error: Unable to open file " << file_path << " for writing.\n";
      }
    }

    std::size_t optimal_cost_dense;
    //Generates a vector of Matrix_free_information from a vector of Matrix_free_sparse_information 
    //via object slicing.
    std::vector<Matrix_free_information> jacobian_info_dense = 
          jacobian_information_slicing(gen.generate_jacobian_information());
    {
      Jacobian_chain<Dense_Jacobian, Matrix_free_information> chain{jacobian_info_dense};

      //Construct solver and fill look up table.
      Fill_table<Dense_Jacobian, Matrix_free_information> solver{chain};
      optimal_cost_dense = solver.get_optimal_cost();

      file_path = file_path_fixed + "visualizations/dense_" + test_counter_string;
      out_file.open(file_path);

      if(out_file.is_open()){

        //Generate Graphviz accumulation tree
        std::vector<Node_matrix_free> optimal_accumulation_sequence = 
          operation_sequence_accumulation<Node_matrix_free>(solver.get_table(), chain.size(),
                                                              &out_file);

        out_file.close();
      }
      else{

        std::cerr << "Error: Unable to open file " << file_path << " for writing.\n";
      }
    }

    std::size_t optimal_cost_jacobian;
    //Generates a vector of Jacobian_information from a vector of Matrix_free_information via 
    //object slicing.
    std::vector<Jacobian_information> jacobian_info =
            jacobian_information_slicing(jacobian_info_dense);

    {
      Jacobian_chain<Jacobian, Jacobian_information> chain{jacobian_info};

      //Construct solver and fill look up table.
      Fill_table<Jacobian, Jacobian_information> solver{chain};
      optimal_cost_jacobian = solver.get_optimal_cost();

      file_path = file_path_fixed + "visualizations/jacobian_" + test_counter_string;
      out_file.open(file_path);

      if(out_file.is_open()){

        std::vector<Node_jacobian> optimal_accumulation_sequence = 
          operation_sequence_accumulation<Node_jacobian>(solver.get_table(), chain.size(),
                                                          &out_file);
        out_file.close();
      }
      else{

        std::cerr << "Error: Unable to open file " << file_path << " for writing.\n";
      }
    }

    file_path = file_path_fixed + "optimal_costs_" + test_counter_string;
    out_file.open(file_path);

    if(out_file.is_open()){

      out_file << optimal_cost_sparse << ' ' << optimal_cost_dense << ' ';
      out_file << std::fixed << std::setprecision(2) << 
        static_cast<double>(optimal_cost_sparse)/optimal_cost_dense << ' ';
      out_file << optimal_cost_jacobian << '\n';

     out_file.close();
    }
    else{

      std::cerr << "Error: Unable to open file " << file_path << " for writing.\n";
    }

    test_counter++;
  }
}

/**
 * @brief This method was designed to be used recursively to obtain the three 
 * quartiles from a data set.
 *
 * @param data Data set.
 * @param[in, out] first_idx First index of the data to be taken into account. After execution
 * it is assigned the index of the element next to the median.
 * @param[in, out] last_idx Last index of the data to be taken into account. After execution 
 * it is assigned the index of the element before the median.
 * @return return the median of the data set containing range of data [first_idx, last_idx].
 */

double median(const std::vector<double>& data,
                    std::size_t& first_idx, std::size_t& last_idx){

  std::size_t N = last_idx - first_idx + 1;
  double median_value;

  //Even
  if(N%2 == 0){

    median_value = (data[first_idx + N/2 - 1] + data[first_idx + N/2]) / 2;
    //last index lhs
    last_idx = N/2 - 1;
    //first index rhs
    first_idx = N/2;
  }
  else{

    median_value = data[first_idx + (N+1)/2 - 1];
    //last index lhs
    last_idx = (N-1)/2 - 1;
    //first index rhs
    first_idx = (N+3)/2 - 1;
  }

  return median_value;
}

/**
 * @brief This method uses the method median to obtain the three main quartiles of a 
 * given data set.
 *
 * @param data Data set. 
 * @param return A vector with the three main quartiles. Ordered as follows [0]: Q1, 
 * [1]: Q2, [2] Q3.
 */
std::vector<double> obtain_quartiles(const std::vector<double>& data){

  std::vector<double> quartiles;
  quartiles.reserve(3);

  std::size_t first_idx = 0, last_idx = data.size() - 1;

  double quartile_2 = median(data, first_idx, last_idx);

  std::size_t first_idx_ = 0;

  double quartile_1 = median(data, first_idx_, last_idx);

  std::size_t last_idx_ = data.size() - 1;

  double quartile_3 = median(data, first_idx, last_idx_);

  quartiles.push_back(quartile_1);
  quartiles.push_back(quartile_2);
  quartiles.push_back(quartile_3);

  return quartiles;
}

/**
 * @brief From a given data set the method calculates the lower limit, Q1, Q2, Q3 and upper limit.
 * Additionally outliers are stored in the vector passed by referece. The outliers are ordered from 
 * smallest to greatest.
 *
 * @param data Data set.
 * @param[out] outliers Vector with outliers.
 * @param[out] box_data vector containing all the data necessary to produce a box plot of the 
 * data, excepting the outliers, these are stored in the vector passed by referece.
 */
void box_whisker_data(std::vector<double>& data, std::vector<double>& outliers,
                                      std::vector<double>& box_data){

  std::sort(data.begin(), data.end());

  std::vector<double> quartiles = obtain_quartiles(data);

  double iqr = quartiles[2] - quartiles[0];
  double lower_lim = quartiles[0] - 1.5 * iqr;
  double upper_lim = quartiles[2] + 1.5 * iqr;

  //Clearing box_data before populating it with new data.
  box_data.clear();
  box_data.reserve(5);
  box_data.push_back(lower_lim);
  box_data.push_back(quartiles[0]);
  box_data.push_back(quartiles[1]);
  box_data.push_back(quartiles[2]);
  box_data.push_back(upper_lim);

  //Deallocating data from quartiles.
  quartiles.clear();
  quartiles.shrink_to_fit();

  //Returns an iterator to the position of the first element greater or equal to lower_lim.
  auto it_first = std::lower_bound(data.begin(), data.end(), lower_lim);

  //Returns an iterator to the position of the first element greater than upper_bound.
  auto it_last = std::upper_bound(data.begin(), data.end(), upper_lim);

  std::size_t total_number_outliers = static_cast<std::size_t>(it_first - data.begin()) + 
    static_cast<std::size_t>(data.end() - it_last + 1);

  //Clearing outliers before populating it with new data.
  outliers.clear();
  outliers.reserve(total_number_outliers);

  for(auto iter = data.begin(); iter != it_first; iter++){

    outliers.push_back(*iter);
  }

  for(auto iter = it_last; iter != data.end(); iter++){

    outliers.push_back(*iter);
  }
}

void run_experiments(std::size_t chain_length,
                                  std::size_t dimension_lb, std::size_t dimension_ub,
                                  std::size_t number_edges_lb, std::size_t number_edges_ub,
                                  double density_lb, double density_ub,
                                  const std::vector<std::size_t>& seeds_arr,
                                  std::size_t number_experiments,
                                  std::vector<double>& ratio_sparse_dense){

  //Clears input vector before allocating new data.
  ratio_sparse_dense.clear();
  ratio_sparse_dense.reserve(number_experiments);

  std::size_t counter = 0;

  std::size_t optimal_cost_sparse, optimal_cost_dense;

  while(counter < number_experiments){

    //Generator instance to create synthetic data.
    Generator<Matrix_free_sparse_information> gen{chain_length, dimension_lb, dimension_ub,
                                            number_edges_lb, number_edges_ub,
                                            density_lb, density_ub, true, seeds_arr[counter]};

    {
      Jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information> chain{gen.generate_data()};

      //Construct sparse solver and fill look up table.
      Fill_table<Sparse_Jacobian, Matrix_free_sparse_information> solver{chain};
      optimal_cost_sparse = solver.get_optimal_cost();
    }

    {
      //The array of Matrix_free_sparse_information is sliced entry by entry to obtain 
      //an array of Matrix_free_information and then used to initialize the Dense Jacobian chain.
      Jacobian_chain<Dense_Jacobian, Matrix_free_information> 
        chain{jacobian_information_slicing(gen.generate_jacobian_information())};

      //Construct sparse solver and fill loop up table.
      Fill_table<Dense_Jacobian, Matrix_free_information> solver{chain};
      optimal_cost_dense = solver.get_optimal_cost();
    }

    //The ratio between optimal_cost_sparse and optimal_cost_dense is stored ratio_sparse_dense
    ratio_sparse_dense.push_back(static_cast<double>(optimal_cost_sparse)/optimal_cost_dense);

    counter++;
  }
}

void run_experiments_sparse(std::size_t chain_length,
                        std::size_t dimension_lb, std::size_t dimension_ub,
                        std::size_t number_edges_lb, std::size_t number_edges_ub,
                        double density_lb, double density_ub,
                        const std::vector<std::size_t>& seeds_arr,
                        std::size_t number_experiments,
                        std::vector<double>& resulting_jacobian_density){

  resulting_jacobian_density.clear();
  resulting_jacobian_density.reserve(number_experiments);

  std::size_t counter = 0;

  while(counter < number_experiments){

    Generator<Matrix_free_sparse_information> gen{chain_length, dimension_lb, dimension_ub,
                                                    number_edges_lb, number_edges_ub,
                                                    density_lb, density_ub, true, seeds_arr[counter]};

    Jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information> chain{gen.generate_data()};

    //Construct sparse solver and fill look up table.
    Fill_table<Sparse_Jacobian, Matrix_free_sparse_information> solver{chain};

    //Access last cell of the look up table. This cell contains the Sparse Jacobian of the 
    //accumulated Jacobian.
    auto last_jacobian_cell = solver.back();

    resulting_jacobian_density.push_back(
        static_cast<double>(last_jacobian_cell.number_nnz())/
                            (last_jacobian_cell.domain_dim() * last_jacobian_cell.codomain_dim()));
    counter++;
  }
}

/**
 * @brief From box_plot_data and outliers write a JSON file to be late used  to 
 * generate the plots with a python script.
 *
 * @param filename Path to file.
 * @param label Title
 * @param box_plot_data vector constaining the quartiles and bounds for the plot.
 * @param outliers vector with outliers.
 */
using json = nlohmann::json;

void write_box_plot_json(const std::string& file_name, 
                                const std::string& label,
                                const std::vector<double>& box_plot_data, 
                                const std::vector<double>& outliers) 
{
  if (box_plot_data.size() < 5){

    std::cerr << "Error: box_plot_data must contain 5 elements." << '\n';
    return;
  }

  json stats_json;
  stats_json["label"] = label;
  stats_json["whislo"] = box_plot_data[0];
  stats_json["q1"] = box_plot_data[1];
  stats_json["med"] = box_plot_data[2];
  stats_json["q3"] = box_plot_data[3];
  stats_json["whishi"] = box_plot_data[4];

  stats_json["fliers"] = outliers;

  std::ofstream out_file(file_name);

  if(out_file.is_open()){

    out_file << stats_json.dump(4);
    out_file.close();
  }
  else{

    std::cerr << "Error: Unable to open file " << file_name << " for writing.\n";
  }
}

/**
 * @brief Provided an array with densities, number_experiments_per_test are executing for each
 * given density. The other arguments to the generator are kept constant.
 *
 * Box plots were generated with the following arguments:
 * densities {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4}.
 * number_experiments_per_test = 100
 * master_seed = 2026
 * chain_length = 5
 * dimension_lb = 20, dimension_ub = 40
 * number_edges_lb = 100, number_edges_ub = 200
 *
 * @param[in] densities to execute the experiments.
 * @param number_experiments_per_test 
 * @param master_seed 
 */
void run_experiments_varying_density(const std::vector<double>& densities,
                                      std::size_t number_experiments_per_test,
                                      const uint64_t master_seed,
                                      std::size_t chain_length,
                                      std::size_t dimension_lb, std::size_t dimension_ub,
                                      std::size_t number_edges_lb, std::size_t number_edges_ub){

  std::size_t test_counter = 0;

  //Seed generation using master_seed
  std::vector<std::size_t> seeds_arr = 
    generate_reproducible_seeds(master_seed, number_experiments_per_test);

  std::vector<double> ratio_values;
  std::vector<double> outliers;
  std::vector<double> box_plot_data;

  std::string file_name;
  std::string file_path_fixed = std::string(RESULTS_DIR) + "sparse_vs_dense/density_results/";
  std::string file_path;
  std::string label;

  while(test_counter < densities.size() - 1){

    run_experiments(chain_length, dimension_lb, dimension_ub, number_edges_lb, number_edges_ub,
                                  densities[test_counter], densities[test_counter + 1],
                                  seeds_arr, number_experiments_per_test, ratio_values);

    box_whisker_data(ratio_values, outliers, box_plot_data);

    file_name = std::format("density_{}_{}.json", densities[test_counter],
                              densities[test_counter + 1]);

    file_path = file_path_fixed + file_name;

    label = std::format("[{}, {}]", densities[test_counter],
                          densities[test_counter + 1]);

    //Generating JSON file with data.
    write_box_plot_json(file_path, label, box_plot_data, outliers);

    test_counter++;
  }
}

/**
 * @brief 
 *
 * Box plots were generated with the following arguments:
 * chain_lengths{5, 10, 15, 20, 25, 30}
 * number_experiments_per_test = 100
 * master_seed = 2233
 * dimension_lb = 20, dimension_ub = 40
 * number_edges_lb = 100, number_edges_ub = 200
 * density_lb = 0.05, density_ub = 0.1
 */
void run_experiments_varying_chain_length(const std::vector<std::size_t>& chain_lengths,
                                          std::size_t number_experiments_per_test,
                                          const uint64_t master_seed,
                                          std::size_t dimension_lb, std::size_t dimension_ub,
                                          std::size_t number_edges_lb, std::size_t number_edges_ub,
                                          double density_lb, double density_ub){

  std::size_t test_counter = 0;

  std::vector<std::size_t> seeds_arr = 
    generate_reproducible_seeds(master_seed, number_experiments_per_test);

  std::vector<double> ratio_values;
  std::vector<double> outliers;
  std::vector<double> box_plot_data;

  std::string file_name;
  std::string file_path_fixed = std::string(RESULTS_DIR) + "sparse_vs_dense/chain_length_results/";
  std::string file_path;
  std::string label;

  while(test_counter < chain_lengths.size()){
    
    run_experiments(chain_lengths[test_counter], dimension_lb, dimension_ub,
                      number_edges_lb, number_edges_ub, density_lb, density_ub,
                      seeds_arr, number_experiments_per_test, ratio_values);

    box_whisker_data(ratio_values, outliers, box_plot_data);

    file_name = std::format("length_{}.json", chain_lengths[test_counter]);

    file_path = file_path_fixed + file_name;

    label = std::format("{}", chain_lengths[test_counter]);

    //Generating JSON file with data.
    write_box_plot_json(file_path, label, box_plot_data, outliers);

    test_counter++;
  }
}

/**
 * @brief
 *
 * Box plots were generated with the following arguments:
 * dimensions {20, 40, 60, 80, 100}
 * number_experiments_per_test = 100
 * master_seed = 1984
 * chain_length = 5
 * number_edges_lb = 100, number_edges_ub = 200
 * density_lb = 0.05, density_ub = 0.1
 */
void run_experiments_varying_dimensions(const std::vector<std::size_t>& dimensions,
                                          std::size_t number_experiments_per_test,
                                          const uint64_t master_seed,
                                          std::size_t chain_length,
                                          std::size_t number_edges_lb, std::size_t number_edges_ub,
                                          double density_lb, double density_ub){

  std::size_t test_counter = 0;

  std::vector<std::size_t> seeds_arr = 
    generate_reproducible_seeds(master_seed, number_experiments_per_test);

  std::vector<double> ratio_values;
  std::vector<double> outliers;
  std::vector<double> box_plot_data;

  std::string file_name;
  std::string file_path_fixed = std::string(RESULTS_DIR) + "sparse_vs_dense/dimension_results/";
  std::string file_path;
  std::string label;

  while(test_counter < dimensions.size() - 1){
    
    run_experiments(chain_length, dimensions[test_counter], dimensions[test_counter + 1],
                      number_edges_lb, number_edges_ub, density_lb, density_ub,
                      seeds_arr, number_experiments_per_test, ratio_values);

    box_whisker_data(ratio_values, outliers, box_plot_data);

    file_name = std::format("dimensions_{}_{}.json", dimensions[test_counter], 
                              dimensions[test_counter + 1]);

    file_path = file_path_fixed + file_name;

    label = std::format("[{}, {}]", dimensions[test_counter],
                          dimensions[test_counter + 1]);

    //Generating JSON file with data.
    write_box_plot_json(file_path, label, box_plot_data, outliers);

    test_counter++;
  }
}

void run_experiments_varying_number_edges(const std::vector<std::size_t>& number_edges,
                                            std::size_t number_experiments_per_test,
                                            const uint64_t master_seed,
                                            std::size_t chain_length,
                                            std::size_t dimension_lb, std::size_t dimension_ub,
                                            double density_lb, double density_ub){
  std::size_t test_counter = 0;

  std::vector<std::size_t> seeds_arr =
    generate_reproducible_seeds(master_seed, number_experiments_per_test);

  std::vector<double> ratio_values;
  std::vector<double> outliers;
  std::vector<double> box_plot_data;

  std::string file_name;
  std::string file_path_fixed = std::string(RESULTS_DIR) + "sparse_vs_dense/number_edges_results/";
  std::string file_path;
  std::string label;

  while(test_counter < number_edges.size() - 1){

    run_experiments(chain_length, dimension_lb, dimension_ub,
                      number_edges[test_counter], number_edges[test_counter+1],
                      density_lb, density_ub, seeds_arr, number_experiments_per_test,
                      ratio_values);

    box_whisker_data(ratio_values, outliers, box_plot_data);

    file_name = std::format("number_edges_{}_{}.json", number_edges[test_counter],
                              number_edges[test_counter + 1]);

    file_path = file_path_fixed + file_name;

    label = std::format("[{}, {}]", number_edges[test_counter], number_edges[test_counter + 1]);

    //Generate JSON file with box plot data.
    write_box_plot_json(file_path, label, box_plot_data, outliers);
    
    test_counter++;
  }
}

void run_experiment_sparsity_behaviour(const std::vector<std::size_t>& chain_lengths,
                                          std::size_t number_experiments_per_test,
                                          const uint64_t master_seed,
                                          std::size_t dimension_lb, std::size_t dimension_ub,
                                          std::size_t number_edges_lb, std::size_t number_edges_ub,
                                          double density_lb, double density_ub){

  std::size_t test_counter = 0;

  std::vector<std::size_t> seeds_arr = 
    generate_reproducible_seeds(master_seed, number_experiments_per_test);

  std::vector<double> resulting_jacobian_density_values;
  std::vector<double> outliers;
  std::vector<double> box_plot_data;

  std::string file_name;
  std::string file_path_fixed = std::string(RESULTS_DIR) + 
                                  "sparse_vs_dense/sparsity_behaviour_results/";
  std::string file_path;
  std::string label;

  while(test_counter < chain_lengths.size()){

    run_experiments_sparse(chain_lengths[test_counter], dimension_lb, dimension_ub, 
                            number_edges_lb, number_edges_ub, density_lb, density_ub, 
                            seeds_arr, number_experiments_per_test,
                            resulting_jacobian_density_values);

    box_whisker_data(resulting_jacobian_density_values, outliers, box_plot_data);

    file_name = std::format("chain_length_{}.json", chain_lengths[test_counter]);

    file_path = file_path_fixed + file_name;
    label = std::format("{}", chain_lengths[test_counter]);

    //Generating JSON file with data.
    write_box_plot_json(file_path, label, box_plot_data, outliers);

    test_counter++;
  }
}

void run_experiment_sparsity_behaviour(const std::vector<std::size_t>& dimensions,
                                          std::size_t number_experiments_per_test,
                                          uint64_t master_seed,
                                          std::size_t chain_length,
                                          std::size_t number_edges_lb, std::size_t number_edges_ub,
                                          double density_lb, double density_ub){

  std::size_t test_counter = 0;

  std::vector<std::size_t> seeds_arr = 
    generate_reproducible_seeds(master_seed, number_experiments_per_test);

  std::vector<double> resulting_jacobian_density_values;
  std::vector<double> outliers;
  std::vector<double> box_plot_data;

  std::string file_name;
  std::string file_path_fixed = std::string(RESULTS_DIR) + 
                                  "sparse_vs_dense/sparsity_behaviour_results/";
  std::string file_path;
  std::string label;

  while(test_counter < dimensions.size() - 1){

    run_experiments_sparse(chain_length, dimensions[test_counter], dimensions[test_counter + 1],
                              number_edges_lb, number_edges_ub, density_lb, density_ub,
                              seeds_arr, number_experiments_per_test,
                              resulting_jacobian_density_values);

    box_whisker_data(resulting_jacobian_density_values, outliers, box_plot_data);

    file_name = std::format("dimensions_{}_{}.json", dimensions[test_counter],
                              dimensions[test_counter + 1]);

    file_path = file_path_fixed + file_name;
    label = std::format("[{}, {}]", dimensions[test_counter], dimensions[test_counter + 1]);

    //Generating JSON file.
    write_box_plot_json(file_path, label, box_plot_data, outliers);

    test_counter++;
  }
}

/**
 * @brief Data is sent to the file_out stream. The data is stored in a single column.
 * 
 * The data shown in the report is stored inside ./results/binomial/additional_costs 
 * This was done way > (redirecting)
 *
 * @param file_out out stream to file.
 * @param data vector with data.
 */
void print_subprograms_cost(std::ostream& file_out, const std::vector<std::size_t>& data){

  for(std::size_t idx = 0; idx < data.size(); idx++){
    
    file_out << data[idx] << '\n';
  }
}

/**
 * @brief Data will be collected with subprograms execution cost varying in four different ways.
 * v shape, ^ (n) shape, linear ascending and descending.
 *
 * @pre number of subprograms must be odd and care must be taken to guarantee that the produced 
 * execution costs never go below zero.
 * @pre slope_linear is assumed to be positive. 
 *
 * Data used in the report was generated using the arguments:
 * number_of_subprograms = 7
 * minimum_value_v_shape = 30
 * maximum_value_n_shape = 90
 * slope_v_n_shape = 20
 * initial_value_linear_increase = 30
 * initial_value_linear_decrease = 90
 * slope_linear = 10
 * number_checkpoints{1, 2, 3, 4}
 */
void run_experiments_binomial_checkpointing(std::size_t number_subprograms,
                                              std::size_t minimum_value_v_shape,
                                              std::size_t maximum_value_n_shape,
                                              std::size_t slope_v_n_shape,
                                              std::size_t initial_value_linear_increase,
                                              std::size_t initial_value_linear_decrease,
                                              int slope_linear,
                                              const std::vector<std::size_t>& number_checkpoints){

  std::size_t test_counter = 0;

  std::string fixed_path = std::string(RESULTS_DIR) + "binomial/";
  std::string file_name;
  std::string file_path;
  std::ofstream out_file;
  std::string test_counter_string;

  //Generating and storing subprograms cost.
  std::vector<std::size_t> v_shape_subprograms_cost = 
    v_shape(minimum_value_v_shape, number_subprograms, slope_v_n_shape); 

  file_name = "v_shape/subprograms_cost";
  file_path = fixed_path + file_name;
  
  out_file.open(file_path);
  if(out_file.is_open()){

    print_subprograms_cost(out_file, v_shape_subprograms_cost);
    out_file.close();
  }
  else{

    std::cerr << "Error: Unable to open file " << file_path << " for writing.\n";
  }

  std::vector<std::size_t> n_shape_subprograms_cost =
    n_shape(maximum_value_n_shape, number_subprograms, slope_v_n_shape);

  file_name = "n_shape/subprograms_cost";
  file_path = fixed_path + file_name;
  
  out_file.open(file_path);
  if(out_file.is_open()){

    print_subprograms_cost(out_file, n_shape_subprograms_cost);
    out_file.close();
  }
  else{

    std::cerr << "Error: Unable to open file " << file_path << " for writing.\n";
  }

  std::vector<std::size_t> linear_increasing_subprograms_cost =
    linear_array(initial_value_linear_increase, number_subprograms, slope_linear);

  file_name = "linear_increasing/subprograms_cost";
  file_path = fixed_path + file_name;
  
  out_file.open(file_path);
  if(out_file.is_open()){

    print_subprograms_cost(out_file, linear_increasing_subprograms_cost);
    out_file.close();
  }
  else{

    std::cerr << "Error: Unable to open file " << file_path << " for writing.\n";
  }

  std::vector<std::size_t> linear_decreasing_subprograms_cost =
    linear_array(initial_value_linear_decrease, number_subprograms, (-1) * slope_linear);


  file_name = "linear_decreasing/subprograms_cost";
  file_path = fixed_path + file_name;
  
  out_file.open(file_path);
  if(out_file.is_open()){

    print_subprograms_cost(out_file, linear_decreasing_subprograms_cost);
    out_file.close();
  }
  else{

    std::cerr << "Error: Unable to open file " << file_path << " for writing.\n";
  }

  //Declaring cost variables.
  std::size_t v_shape_additional_cost, n_shape_additional_cost;
  std::size_t linear_increasing_additional_cost, linear_decreasing_additional_cost;
  while(test_counter < number_checkpoints.size()){

    test_counter_string = std::to_string(number_checkpoints[test_counter]) + "_checkpoints";

    //v shape subprograms execution costs
    {
      Binomial_checkpointing algorithm{v_shape_subprograms_cost, number_checkpoints[test_counter]};

      v_shape_additional_cost = algorithm.get_additional_cost();

      file_name = "v_shape/visualizations/" + test_counter_string;
      file_path = fixed_path + file_name;
      out_file.open(file_path);

      if(out_file.is_open()){

        std::vector<Node_binomial_checkpointing> accumulation_sequence =
          subproblem_sequence_accumulation(algorithm.get_table(), number_subprograms,
                                            number_checkpoints[test_counter], &out_file);
        out_file.close();
      }
      else{

        std::cerr << "Error: Unable to open file " << file_path << " for writing.\n";
      }

    }

    //n shape subprograms execution costs
    {
      Binomial_checkpointing algorithm{n_shape_subprograms_cost, number_checkpoints[test_counter]};

      n_shape_additional_cost = algorithm.get_additional_cost();

      file_name = "n_shape/visualizations/" + test_counter_string;
      file_path = fixed_path + file_name;
      
      out_file.open(file_path);
      if(out_file.is_open()){

        std::vector<Node_binomial_checkpointing> accumulation_sequence =
          subproblem_sequence_accumulation(algorithm.get_table(), number_subprograms,
                                            number_checkpoints[test_counter], &out_file);
        out_file.close();
      }
      else{

        std::cerr << "Error: Unable to open file " << file_path << " for writing.\n";
      }
    }

    // linear increasing execution costs
    {
      Binomial_checkpointing algorithm{linear_increasing_subprograms_cost,
                                          number_checkpoints[test_counter]};

      linear_increasing_additional_cost = algorithm.get_additional_cost();

      file_name = "linear_increasing/visualizations/" + test_counter_string;
      file_path = fixed_path + file_name;

      out_file.open(file_path);
      if(out_file.is_open()){

        std::vector<Node_binomial_checkpointing> accumulation_sequence =
          subproblem_sequence_accumulation(algorithm.get_table(), number_subprograms,
                                            number_checkpoints[test_counter], &out_file);
        out_file.close();
      }
      else{

        std::cerr << "Error: Unable to open file " << file_path << " for writing.\n";
      }
    }

    //linear decreasing execution costs
    {
      Binomial_checkpointing algorithm{linear_decreasing_subprograms_cost, 
                                        number_checkpoints[test_counter]};

      linear_decreasing_additional_cost= algorithm.get_additional_cost();

      file_name = "linear_decreasing/visualizations/" + test_counter_string;
      file_path = fixed_path + file_name;

      out_file.open(file_path);
      if(out_file.is_open()){

        std::vector<Node_binomial_checkpointing> accumulation_sequence =
          subproblem_sequence_accumulation(algorithm.get_table(), number_subprograms,
                                            number_checkpoints[test_counter], &out_file);
        out_file.close();
      }
      else{

        std::cerr << "Error: Unable to open file " << file_path << " for writing.\n";
      }
    }
    //Print additional cost data to std::cout and then redirect the output to a file inside 
    // binomial/ called additional_costs.
    std::cout << v_shape_additional_cost << ' ' << n_shape_additional_cost << ' ';
    std::cout << linear_increasing_additional_cost << ' ' << linear_decreasing_additional_cost << '\n';

    test_counter++;
  }
}

void post_processing_study_case(){

  //Problem data:
  std::vector<std::size_t> dimension{27, 23, 20, 17, 15, 5, 20, 18, 25};
  std::vector<std::size_t> number_edges{180, 185, 140, 150, 255, 199, 234, 241};

  //Create Matrix_free_information vector
  std::vector<Matrix_free_information> problem_data;
  problem_data.reserve(number_edges.size());

  for(std::size_t idx = 0; idx < number_edges.size(); idx++){
    
    problem_data.emplace_back(dimension[idx], dimension[idx + 1], number_edges[idx]);
  }

  //Creating Jacobian Chain from problem data
  Jacobian_chain<Dense_Jacobian, Matrix_free_information> chain{problem_data};

  //Execute dynamic programming algoritm (Matrix Free Dense Jacobian Chain Product Bracketing)
  Fill_table<Dense_Jacobian, Matrix_free_information> solver{chain};

  std::size_t optimal_cost = solver.get_optimal_cost();

  //Store problem and solution data.
  std::string fixed_path = std::string(RESULTS_DIR) + "post_processing_case_study/";
  
  //Store problem data.
  std::string file_name = "problem_data";
  std::string file_path = fixed_path + file_name;
  std::ofstream out_file;

  out_file.open(file_path);
  if(out_file.is_open()){
    
    chain.print(out_file);
    out_file.close();
  }
  else{
    std::cerr << "Error: Unable to open file " << file_path << "for writting.\n";
  }

  //Generate optimal solution visualization.
  file_name = "matrix_free_bracketing_visualization";
  file_path = fixed_path + file_name;
  out_file.open(file_path);
  if(out_file.is_open()){
    
    std::vector<Node_matrix_free> optimal_accumulation_sequence = 
      operation_sequence_accumulation<Node_matrix_free>(solver.get_table(), chain.size(),
                                                          &out_file);
    out_file.close();
  }
  else{
    std::cerr << "Error: Unable to open file " << file_path << "for writting.\n";
  }

  //Post processing: check the visualization before reading the following lines.
  //The optimal accumulation procedure suggests to accumulate the subchain (5,0) using 
  //adjoint mode.
  std::size_t memory_limit = 300;


  //Checks if all subprograms individually have less number of edges than the memory limit. 
  if(!tool_box_split::is_split_reversable(solver.get_table(), 5, 0, memory_limit)){

    std::cerr << "Error the subchain is not split reversable.";
  }

  /* //Cost model subprogram cost equal number of edges. */
  std::vector<std::size_t> subprograms_cost_subchain{number_edges[0], number_edges[1],
                                                      number_edges[2], number_edges[3],
                                                      number_edges[4]};

  //Partitions the subchain into groups that the sum of their edges is less than the memory 
  //limit.
  std::vector<std::size_t> partition = 
    tool_box_split::adjoint_chain_partition(solver.get_table(), 4, 0, memory_limit);
  
  //Adds subprograms costs of subprograms belonging to the same partition.
  std::vector<std::size_t> binomial_problem =
      tool_box_split::build_subchain_execution_costs_array(subprograms_cost_subchain, partition);
  

  //Store binomial_problem data
  file_name = "adjoint_subchain_partitioning";
  file_path = fixed_path + file_name;
  out_file.open(file_path);

  if(out_file.is_open()){

    tool_box_split::subchain_partition_parser(partition, &out_file);
    out_file.close();
  }
  else{
    std::cerr << "Error: Unable to open file " << file_path << "for writting.\n";
  }

  file_name = "binomial_subprograms_cost";
  file_path = fixed_path + file_name;
  out_file.open(file_path);

  if(out_file.is_open()){

    for(std::size_t idx = 0; idx < binomial_problem.size(); idx++){

      out_file << binomial_problem[idx] << '\n';
    }
    out_file.close();
  }
  else{
    std::cerr << "Error: Unable to open file " << file_path << "for writting.\n";
  }

  //Solving binomial checkpointing problem
  std::size_t available_checkpoints = 2;
  Binomial_checkpointing binomial_solver{binomial_problem, available_checkpoints};
  std::size_t additional_cost = binomial_solver.get_additional_cost();
  
  //Store visualization data
  file_name = "binomial_checkpointing_visualization";
  file_path = fixed_path + file_name;
  out_file.open(file_path);

  if(out_file.is_open()){

    std::vector<Node_binomial_checkpointing> accumulation_sequence =
      subproblem_sequence_accumulation(binomial_solver.get_table(), binomial_problem.size(),
                                        available_checkpoints, &out_file);
    out_file.close();
  }
  else{
    std::cerr << "Error: Unable to open file " << file_path << "for writting.\n";
  }

  //Store optimal bracketing cost, additional cost and their sum.
  file_name = "costs";
  file_path = fixed_path + file_name;
  out_file.open(file_path);
  if(out_file.is_open()){

    out_file << optimal_cost << '\n' << additional_cost << '\n' << 
      optimal_cost + additional_cost << '\n';
    out_file.close();
  }
  else{
    std::cerr << "Error: Unable to open file " << file_path << "for writting.\n";
  }
}

int main(){

  //--------------- Experiments with varying density -------------------------
  {
    std::vector<double> densities{0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4};
    std::size_t number_experiments_per_test = 100;
    const uint64_t master_seed = 2026;
    std::size_t chain_length = 5;
    std::size_t dimension_lb = 20, dimension_ub = 40;
    std::size_t number_edges_lb = 100, number_edges_ub = 200; 

    /* run_experiments_varying_density(densities, number_experiments_per_test, master_seed, */
    /*                                   chain_length, dimension_lb, dimension_ub, */
    /*                                   number_edges_lb, number_edges_ub); */
  }

  //-----------Experiments with varying chain length -----------------------
  {
    std::vector<std::size_t> chain_lengths{5, 10, 15, 20, 25, 30};
    std::size_t number_experiments_per_test = 100;
    const uint64_t master_seed = 2233;
    std::size_t dimension_lb = 20, dimension_ub = 40;
    std::size_t number_edges_lb = 100, number_edges_ub = 200;
    double density_lb = 0.05, density_ub = 0.1;

    /* run_experiments_varying_chain_length(chain_lengths, number_experiments_per_test, */
    /*                                       master_seed, dimension_lb, dimension_ub, */
    /*                                       number_edges_lb, number_edges_ub, */
    /*                                       density_lb, density_ub); */
  }

  //-----------Experiments with varying dimensions ---------------
  {
    std::vector<std::size_t> dimensions{20, 40, 60, 80, 100};
    std::size_t number_experiments_per_test = 100;
    const uint64_t master_seed = 1984;
    std::size_t chain_length = 5;
    std::size_t number_edges_lb = 100, number_edges_ub = 200;
    double density_lb = 0.05, density_ub = 0.1;

    /* run_experiments_varying_dimensions(dimensions, number_experiments_per_test, master_seed, */
    /*                                     chain_length, number_edges_lb, number_edges_ub, */
    /*                                     density_lb, density_ub); */
  }

  //----------Experiments with varying number edges---------------
  {
    std::vector<std::size_t> number_edges{100, 200, 300, 400, 500};
    std::size_t number_experiments_per_test = 100;
    uint64_t master_seed = 1234;
    std::size_t chain_length = 5;
    std::size_t dimension_lb = 20, dimension_ub = 40;
    double density_lb = 0.05, density_ub = 0.1;

    /* run_experiments_varying_number_edges(number_edges, number_experiments_per_test, */
    /*                                       master_seed, chain_length, dimension_lb, dimension_ub, */
    /*                                       density_lb, density_ub); */

  }

  //-----------Sample experiments with complete data recording-----
  {
    //Generator arguments
    std::size_t chain_length = 5;
    std::size_t dimension_lb = 5, dimension_ub = 20;
    std::size_t number_edges_lb = 100, number_edges_ub = 300;
    
    //First sample experiment
    {
      const uint64_t master_seed = 2012;
      std::size_t number_experiments = 3;
      /* std::vector<std::size_t> seeds_arr = */
      /*     generate_reproducible_seeds(master_seed, number_experiments); */

      double density_lb = 0.05, density_ub = 0.1;

      /* general_execution(seeds_arr, chain_length, dimension_lb, dimension_ub, */
      /*                   number_edges_lb, number_edges_ub, density_lb, density_ub); */

    }

    //Second sample experiment
    {
      const uint64_t master_seed = 1942;
      std::size_t number_experiments = 3;
      /* std::vector<std::size_t> seeds_arr = */
      /*     generate_reproducible_seeds(master_seed, number_experiments); */

      double density_lb = 0.1, density_ub = 0.15;

      /* general_execution(seeds_arr, chain_length, dimension_lb, dimension_ub, */
      /*                   number_edges_lb, number_edges_ub, density_lb, density_ub); */
    }
  }

  //------------Binomial Checkpointing experiments ----------------------------
  {
    std::size_t number_subprograms = 7;
    std::size_t minimum_value_v_shape = 30, maximum_value_n_shape = 90;
    std::size_t slope_v_n_shape = 20;
    std::size_t initial_value_linear_increase = 30, initial_value_linear_decrease = 90;
    int slope_linear = 10;
    std::vector<std::size_t> number_checkpoints{1, 2, 3, 4};
    /* run_experiments_binomial_checkpointing(number_subprograms, minimum_value_v_shape, */
    /*                                         maximum_value_n_shape, slope_v_n_shape, */
    /*                                         initial_value_linear_increase, initial_value_linear_decrease, */
    /*                                         slope_linear, number_checkpoints); */
  }

  //------Resultant Jacobian Sparsity behaviour with varying chain lenghts ---------------
  //IMPORTANT: USE THE SAME MASTER SEED AS IN Experiments with varying chain length.
  {
    //With chain sizes greater than 15 the resulting Jacobian has almost always a density of 1.
    std::vector<std::size_t> chain_lengths{5, 10, 15, 20};
    std::size_t number_experiments_per_test = 100;
    const uint64_t master_seed = 2233;
    std::size_t dimension_lb = 20, dimension_ub = 40;
    std::size_t number_edges_lb = 100, number_edges_ub = 200;
    double density_lb = 0.05, density_ub = 0.1;

    /* run_experiment_sparsity_behaviour(chain_lengths, number_experiments_per_test, master_seed, */
    /*                                     dimension_lb, dimension_ub, */
    /*                                     number_edges_lb, number_edges_ub, */
    /*                                     density_lb, density_ub); */
  }
  
  //-----Resultant Jacobian Sparsity behaviour with varying dimensions-------------------
  //IMPORTANT: USE THE SAME MASTER SEED AS IN Experiments with varying dimensions
  {
    std::vector<std::size_t> dimensions{20, 40, 60, 80, 100};
    std::size_t number_experiments_per_test = 100;
    const uint64_t master_seed = 1984;
    std::size_t chain_length = 5;
    std::size_t number_edges_lb = 100, number_edges_ub = 200;
    double density_lb = 0.05, density_ub = 0.1;

    /* run_experiment_sparsity_behaviour(dimensions, number_experiments_per_test, master_seed, */
    /*                                     chain_length, number_edges_lb, number_edges_ub, */
    /*                                     density_lb, density_ub); */

  }
  //-------Post-processing study case---------------
  {
    //post_processing_study_case();
  }


  return 0;
}
