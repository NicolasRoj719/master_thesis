#include <fstream>
#include <vector>
#include <iostream>
#include <cstdint>

#ifndef DATA_FILE_TO_VECTOR_HPP
#define DATA_FILE_TO_VECTOR_HPP

void data_file_vector(char* file_name, std::vector<std::vector<std::size_t>>& data,
    std::size_t& length, std::size_t cols){
  std::ifstream in(file_name);

  // Number of rows in the file. Each row contains relevant information about
  // the jacobian. For example n (input dimension), m (output dimension),
  // n_E (number of edges of the DAG representation of the underlying function).
  in >> length;

  data.reserve(length);
  std::vector<size_t> data_row(cols);
  for(std::size_t i=0; i<length; i++){
    for(std::size_t j=0; j<cols; j++){
      in >> data_row[j];
    }
    data.push_back(data_row);
  }

}
#endif
