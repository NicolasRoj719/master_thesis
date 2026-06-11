#include <cstdint>

#ifndef UTIL_STRUCTS_HPP
#define UTIL_STRUCTS_HPP

class NNZ{
 public: 
  NNZ(std::size_t row, std::size_t col): row_(row), col_(col){}
  std::size_t row() const {return row_;}
  std::size_t col() const {return col_;}
  bool operator<(NNZ const& rhs) const{
    if(row_ < rhs.row()){
      return true;
    }
    if(row_ == rhs.row()){
      if(col_ < rhs.col()){
        return true;
      }
    }
    return false;
  }

  bool operator==(NNZ const& rhs) const{
    if(row_ == rhs.row() && col_ == rhs.col()){
      return true;
    }
    return false;
  }

 private:
  std::size_t row_;
  std::size_t col_;
};
#endif
