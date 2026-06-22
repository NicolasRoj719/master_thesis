#include <cstdint>
#include <optional>
#include <string>
#include "./../chain.hpp"
#include "./../jacobian.hpp"
#include "./../table_cell.hpp"

int main(){
  //Test cell<Jacobian>
  {
    const std::size_t cost = 100, split_position =3;
    cell<Jacobian> cell_obj{cost, split_position};

    assert(cell_obj.split_position() == split_position);
    assert(cell_obj.accumulated_cost() == cost);
  }
  
  //Test cell<Dense_Jacobian> with out considering memory use.
  {
    const std::size_t cost=150, split_position =2;
    Operation op = Operation::MULTIPLICATION;
    cell<Dense_Jacobian> cell_obj{cost, split_position, op};
    assert(!cell_obj.accumulated_memory_use());
  }
  
  //Test cell<Dense_Jacobian> considering memory use.
  {
    const std::size_t cost=150, split_position =2;
    const std::size_t memory = 80;
    Operation op = Operation::MULTIPLICATION;
    cell<Dense_Jacobian> cell_obj{cost, split_position, op, memory};
    assert(cell_obj.accumulated_memory_use());
    assert(cell_obj.accumulated_memory_use().value() == memory);
  }

  //Test cell<Sparse_Jacobian>
  //The chain interface is easier to work with.
  {
    std::string file_chain_sparse_data = "./chain_test_cases/case_0_sparse_WF3";
    jacobian_chain<Sparse_Jacobian> sparse_chain{file_chain_sparse_data};

    const std::size_t cost=250, split_position =3;
    Operation op = Operation::TANGENT;

    //Test cell<Sparse_Jacobian> without memory restriction.
    {
      const Sparse_Jacobian &test_jacobian = sparse_chain[0];
      cell<Sparse_Jacobian> cell_obj{test_jacobian, cost, split_position, op};
      assert(cell_obj.num_nnz() == 4);
      assert(cell_obj.col_number_colors() == 2);
      assert(cell_obj.row_number_colors() == 2);
      assert(cell_obj.rhs_inner_dimension() == 2);
      assert(cell_obj.lhs_inner_dimension() == 2);
    }

    //Test cell<Sparse_Jacobian> without memory restriction.
    {
      std::size_t memory= 50;
      const Sparse_Jacobian &test_jacobian = sparse_chain[1];
      cell<Sparse_Jacobian> cell_obj{test_jacobian, cost, split_position, op, memory};
      assert(cell_obj.num_nnz() == 3);
      assert(cell_obj.col_number_colors() == 1);
      assert(cell_obj.row_number_colors() == 1);
      assert(cell_obj.rhs_inner_dimension() == 1);
      assert(cell_obj.lhs_inner_dimension() == 1);
    }

    //Test cell_with_pointer with and without memory restriction
    {
      cell_with_pointer<Sparse_Jacobian> cell_obj{&sparse_chain[1], cost, split_position, op};
      assert(cell_obj.num_nnz() == 3);
      assert(cell_obj.col_number_colors() == 1);
      assert(cell_obj.row_number_colors() == 1);
      assert(cell_obj.rhs_inner_dimension() == 1);
      assert(cell_obj.lhs_inner_dimension() == 1);

      std::size_t memory= 60;
      cell_with_pointer<Sparse_Jacobian> cell_obj_mem{&sparse_chain[1], cost, split_position, op, memory};
      assert(cell_obj_mem.num_nnz() == 3);
      assert(cell_obj_mem.col_number_colors() == 1);
      assert(cell_obj_mem.row_number_colors() == 1);
      assert(cell_obj_mem.rhs_inner_dimension() == 1);
      assert(cell_obj_mem.lhs_inner_dimension() == 1);
    }

  }

  // Test cell_with_pointer<Dense_Jacobian>
  {
    std::string file_chain_non_sparse_data = "./chain_test_cases/case_1_non_sparse";
    jacobian_chain<Dense_Jacobian> dense_chain{file_chain_non_sparse_data};

    const std::size_t cost = 120, split_position = 0;
    Operation op = Operation::ADJOINT;
    
    {
      cell_with_pointer<Dense_Jacobian> cell_obj{&dense_chain[0], cost, split_position, op};
      assert(cell_obj.n() == 4);
      assert(cell_obj.m() == 2);
      assert(cell_obj.n_E() == 100);
    }
  }

  return 0;
}
