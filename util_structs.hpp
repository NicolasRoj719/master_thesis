#include <cstdint>

#ifndef UTIL_STRUCTS_HPP
#define UTIL_STRUCTS_HPP

struct NNZ{
  NNZ(std::size_t row, std::size_t col): row_(row), col_(col){}
  std::size_t row_;
  std::size_t col_;
  std::size_t row() const {return row_;}
  std::size_t col() const {return col_;}
};
#endif
