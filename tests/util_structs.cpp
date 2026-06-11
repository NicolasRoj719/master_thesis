#include <cstdint>
#include <cassert>
#include "./../util_structs.hpp"

int main(){
  {
    std::size_t row = 3, column = 4;
    NNZ nnz_entry{row, column};
    assert(nnz_entry.row() == row);
    assert(nnz_entry.col() == column);
  }

  {
    std::size_t row = 3, column = 2;
    std::size_t row_rhs = 4, column_rhs = 0;
    NNZ nnz_lhs{row, column};
    NNZ nnz_rhs{row_rhs, column_rhs};
    assert(nnz_lhs < nnz_rhs);
  }

  {
    std::size_t row = 3, column = 2;
    std::size_t row_rhs = 3, column_rhs = 3;
    NNZ nnz_lhs{row, column};
    NNZ nnz_rhs{row_rhs, column_rhs};
    assert(nnz_lhs < nnz_rhs);
  }

  {
    std::size_t row = 3, column = 2;
    std::size_t row_rhs = 3, column_rhs = 2;
    NNZ nnz_lhs{row, column};
    NNZ nnz_rhs{row_rhs, column_rhs};
    assert(nnz_lhs == nnz_rhs);
  }

  {
    std::size_t row = 3, column = 2;
    std::size_t row_rhs = 2, column_rhs = 2;
    NNZ nnz_lhs{row, column};
    NNZ nnz_rhs{row_rhs, column_rhs};
    assert(!(nnz_lhs < nnz_rhs));
  }
  return 0;
}

