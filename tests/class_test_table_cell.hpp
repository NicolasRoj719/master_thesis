#include <string>
#include <vector>
#include <type_traits>
#include "./../jacobian.hpp"
#include "./../table_cell.hpp"

class test_table_cell{
 public:

  test_table_cell():
    test_jacobian_cell(false), test_dense_cell(false), test_sparse_cell(false),
    test_jacobian_cell_with_pointer(false), test_dense_cell_with_pointer(false),
    test_sparse_cell_with_pointer(false){}

  void set_test_jacobian_cell(bool argument){
  
    test_jacobian_cell = argument;
  }

  void set_test_dense_cell(bool argument){

    test_dense_cell = argument;
  }

  void set_test_sparse_cell(bool argument){

    test_sparse_cell = argument;
  }

  void set_test_jacobian_cell_with_pointer(bool argument){

    test_jacobian_cell_with_pointer = argument;
  }

  void set_test_dense_cell_with_pointer(bool argument){
    
    test_dense_cell_with_pointer = argument;
  }

  void set_test_sparse_cell_with_pointer(bool argument){

    test_sparse_cell_with_pointer = argument;
  }

  void  print_test_state(){
    std::cout << "State of table cell test:\n";

    std::cout << "test_jacobian_cell: ";
    if(test_jacobian_cell){

      std::cout << "successful.\n";
    }
    else{

      std::cout << "failed.\n";
    }

    std::cout << "test_dense_cell: ";
    if(test_dense_cell){

      std::cout << "successful.\n";
    }
    else{

      std::cout << "failed.\n";
    }

    std::cout << "test_sparse_cell: ";
    if(test_sparse_cell){

      std::cout << "successful.\n";
    }
    else{

      std::cout << "failed.\n";
    }

    std::cout << "test_jacobian_cell_with_pointer: ";
    if(test_jacobian_cell_with_pointer){

      std::cout << "successful.\n";
    }
    else{

      std::cout << "failed.\n";
    }

    std::cout << "test_dense_cell_with_pointer: ";
    if(test_dense_cell_with_pointer){

      std::cout << "successful.\n";
    }
    else{

      std::cout << "failed.\n";
    }

    std::cout << "test_sparse_cell_with_pointer: ";
    if(test_sparse_cell_with_pointer){

      std::cout << "successful.\n";
    }
    else{

      std::cout << "failed.\n";
    }

    std::cout << "\n";
  }

  void set_were_all_tests_successful(){

    if(test_jacobian_cell && test_dense_cell && test_sparse_cell &&
        test_jacobian_cell_with_pointer && test_dense_cell_with_pointer &&
        test_sparse_cell_with_pointer){
      
      were_all_tests_successful = true;
    }

    else{ were_all_tests_successful = false;}
  }

  void check_final_state_of_the_tests(){
    
    if(!were_all_tests_successful){
      print_test_state();
    }
  }

  bool check_cell_initialization(const cell<Jacobian>& cell,
      std::size_t cost, std::size_t split_position);

  bool check_cell_initialization(const cell<Dense_Jacobian>& cell,
      std::size_t cost, std::size_t split_position, Operation operation);

  bool check_cell_initialization(const cell<Sparse_Jacobian>& cell,
        const Sparse_Jacobian& jacobian, std::size_t cost,
        std::size_t split_position, Operation operation);

  bool check_cell_initialization(const cell_with_pointer<Jacobian>& cell,
        const Jacobian& jacobian);

  bool check_cell_initialization(const cell_with_pointer<Dense_Jacobian>& cell,
        const Dense_Jacobian& jacobian, std::size_t cost,
        Operation operation);

  bool check_cell_initialization(const cell_with_pointer<Sparse_Jacobian>& cell,
        const Sparse_Jacobian& jacobian, std::size_t cost,
        Operation operation);

 protected:
  bool test_jacobian_cell;

  bool test_dense_cell;

  bool test_sparse_cell;

  bool test_jacobian_cell_with_pointer;

  bool test_dense_cell_with_pointer;

  bool test_sparse_cell_with_pointer;

  bool were_all_tests_successful;

  template <class Cell_type>
  bool check_cell_data(const Cell_type& cell,
      std::size_t cost, std::size_t split_position);

  template<class Cell_type>
  bool check_cell_data(const Cell_type& cell,
      std::size_t cost, std::size_t split_position, Operation operation);

  template<class Cell_type>
  bool check_cell_data(const Cell_type& cell,
      std::size_t cost, Operation operation);

  template <class Cell_type, class Jacobian_T>
  bool check_cell_jacobian_information(const Cell_type& cell,
          const Jacobian_T& jacobian);

  template <class Cell_type>
  bool check_cell_sparse_data(const Cell_type& cell,
          const Sparse_Jacobian& jacobian);


  bool check_cell_sparse_basic_information(const cell<Sparse_Jacobian>& cell,
        const Sparse_Jacobian& jacobian);


  template<class Jacobian_T>
    bool check_cell_jacobian_basic_information(const cell_with_pointer<Jacobian_T>& cell,
        const Jacobian& jacobian);

  bool are_arrays_equal(const std::vector<std::size_t>& array_0,
          const std::vector<std::size_t>& array_1);
};

bool test_table_cell::are_arrays_equal(const std::vector<std::size_t>& array_0,
        const std::vector<std::size_t>& array_1){

    if(array_0.size() != array_1.size()){
        
        return false;
    }

    else{

        for(std::size_t i = 0; i < array_0.size(); i++){

            if(array_0[i] != array_1[i]){

                return false;
            }
        }
    }

    return true;
}

template <class Cell_type>
bool test_table_cell::check_cell_data(const Cell_type& cell,
    std::size_t cost, std::size_t split_position){

    if(cell.accumulated_cost() != cost ||
        cell.split_position() != split_position){

        return false;
    }

    return true;
}

template <class Cell_type>
bool test_table_cell::check_cell_data(const Cell_type& cell,
    std::size_t cost, std::size_t split_position,
    Operation operation){

    if(cell.accumulated_cost() != cost ||
        cell.split_position() != split_position ||
        cell.operation() != operation){

        return false;
    }

    return true;
}

template<class Cell_type>
bool test_table_cell::check_cell_data(const Cell_type& cell,
      std::size_t cost, Operation operation){

  if(cell.accumulated_cost() != cost ||
      cell.operation() != operation){
    
    return false;
  }

  return true;
}

template <class Cell_type, class Jacobian_T>
bool test_table_cell::check_cell_jacobian_information(const Cell_type& cell,
            const Jacobian_T& jacobian){

    if constexpr(std::is_same_v<Cell_type, cell_with_pointer<Jacobian>> &&
                    std::is_same_v<Jacobian_T, Jacobian>){

        if(cell.domain_dim() != jacobian.domain_dim() ||
            cell.codomain_dim() != jacobian.codomain_dim()){
            
            return false;
        }

    }

    else if constexpr(std::is_same_v<Cell_type, cell_with_pointer<Dense_Jacobian>> &&
                        std::is_same_v<Jacobian_T, Dense_Jacobian>){

        if(cell.domain_dim() != jacobian.domain_dim() ||
            cell.codomain_dim() != jacobian.codomain_dim() ||
            cell.number_edges() != jacobian.number_edges()){

            return false;
        }

    }

    //Cell_type = cell<Sparse_Jacobian> or cell_with_pointer<Sparse_Jacobian>
    //Jacobian_T = Sparse_Jacobian
    else{

        if(cell.domain_dim() != jacobian.domain_dim() ||
            cell.codomain_dim() != jacobian.codomain_dim() ||
            cell.number_edges() != jacobian.number_edges() ||
            cell.number_nnz() != jacobian.number_nnz()){

            return false;
        }
    }

    return true;
}

template <class Cell_type>
bool test_table_cell::check_cell_sparse_data(const Cell_type& cell,
    const Sparse_Jacobian& jacobian){

   if(!are_arrays_equal(cell.get_column_idx(), jacobian.get_column_idx_reference()) ||
        !are_arrays_equal(cell.get_row_pointer(), jacobian.get_row_pointer_reference())){

        return false;
   }

   if(cell.column_number_colors() != jacobian.get_column_number_colors() || 
        cell.row_number_colors() != jacobian.get_row_number_colors() ||
        cell.max_number_nnz_row() != jacobian.get_max_number_nnz_row() ||
        cell.max_number_nnz_column() != jacobian.get_max_number_nnz_column()){

        return false;
   }

   return true;
}

// Implementation of check_cell_initialization for cells without pointer
bool test_table_cell::check_cell_initialization(const cell<Jacobian>& cell,
        std::size_t cost, std::size_t split_position){

    return check_cell_data(cell, cost, split_position);
}

bool test_table_cell::check_cell_initialization(const cell<Dense_Jacobian>& cell,
            std::size_t cost, std::size_t split_position,
            Operation operation){

    return(check_cell_data(cell, cost, split_position, operation));
}

bool test_table_cell::check_cell_initialization(const cell<Sparse_Jacobian>& cell,
        const Sparse_Jacobian& jacobian, std::size_t cost,
        std::size_t split_position, Operation operation){

    if(!check_cell_data(cell, cost, split_position, operation)){

        return false;
    }

    if(!check_cell_jacobian_information(cell, jacobian)){

        return false;
    }

    if(!check_cell_sparse_data(cell, jacobian)){

        return false;
    }

    return true;
}

//Implementation of check_cell_initialization for cells with pointers
bool test_table_cell::check_cell_initialization(const cell_with_pointer<Jacobian>& cell,
            const Jacobian& jacobian){

    return check_cell_jacobian_information(cell, jacobian);
}

bool test_table_cell::check_cell_initialization(const cell_with_pointer<Dense_Jacobian>& cell,
            const Dense_Jacobian& jacobian, std::size_t cost,
            Operation operation){

    if(!check_cell_data(cell, cost, operation)){

        return false;
    }

    if(!check_cell_jacobian_information(cell, jacobian)){

        return false;
    }

    return true;
}

bool test_table_cell::check_cell_initialization(const cell_with_pointer<Sparse_Jacobian>& cell,
            const Sparse_Jacobian& jacobian, std::size_t cost,
            Operation operation){

    if(!check_cell_data(cell, cost, operation)){

        return false;
    }

    if(!check_cell_jacobian_information(cell, jacobian)){

        return false;
    }

    if(!check_cell_sparse_data(cell, jacobian)){

        return false;
    }

    return true;
}
