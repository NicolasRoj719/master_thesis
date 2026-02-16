#include <cassert>
#include <iostream>
#include <vector>
#include <cstdint>
#include "./../data_file_to_vector.hpp"

int main(int argc, char* v[]){
  assert(argc==3);

  std::size_t length;
  std::vector<std::vector<size_t>> problem_data;
  std::size_t num_cols = static_cast<std::size_t>(std::stoi(v[2]));

  //problem_data and length are given to the function by reference.
  data_file_vector(v[1], problem_data, length, num_cols); 

  std::cout<<length<<'\n';

  for(std::size_t i= 0; i<problem_data.size(); i++){
    for(std::size_t j=0; j<problem_data[0].size(); j++){
      std::cout<<problem_data[i][j]<<' ';
    }
    std::cout<<'\n';
  }

  return 0;
}
