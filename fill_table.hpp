#include <algorithm>
#include <optional>
#include <limits>
#include <type_traits>
#include <vector>
#include "chain.hpp"
#include "table_cell.hpp"
#include "table.hpp"
#include "jacobian.hpp"

#ifndef FILL_DPTABLE
#define FILL_DPTABLE

template<class Jacobian_T, class Jacobian_info_T>
class fill_table{
 public:
   fill_table(const jacobian_chain<Jacobian_T, Jacobian_info_T>& chain):
      table(chain.size()){

      fill(chain);
   }

   fill_table(const jacobian_chain<Jacobian_T, Jacobian_info_T>& chain,
               std::size_t memory_bound_):
      table(chain.size()), memory_bound(memory_bound_){

      fill(chain, memory_bound_);
   }

   //testing purposes
   fill_table(Table<Jacobian_T> table_):
      table(std::move(table_)){}

   //testing purposes
   fill_table(): table(0){}

  //Dynamic Programming Methods
  // Mathematical notation:
  // Subchain : F'_j ... F'_i.
  // Subchain with split position k: F'_j ... F'_(k+1) F'_(k)...F'_(i)
  // j: last elemental function considered within the subchain.
  // i: first elemental function considered within the subchain.
  // k: split position. k element of {i, ..., j-1}

  std::size_t accumulate_number_of_edges(
        std::size_t j_or_k, std::size_t k_plus_1_or_i);

  std::size_t tangent_cost(const Jacobian_T* elemental_jacobian_ptr);

  std::size_t tangent_cost(std::size_t j, std::size_t k, std::size_t i);

  std::size_t tangent_cost(std::size_t j, std::size_t k, std::size_t i,
                           std::size_t j_i_column_number_colors);

  std::size_t adjoint_cost(const Jacobian_T* elemental_jacobian_ptr);
  
  std::size_t adjoint_cost(std::size_t j, std::size_t k, std::size_t i);

  std::size_t adjoint_cost(std::size_t j, std::size_t k, std::size_t i,
                           std::size_t j_i_row_number_colors);

  std::size_t multiplication_cost(std::size_t j, std::size_t k, std::size_t i);

  std::size_t multiplication_cost(std::size_t j, std::size_t k,
        std::size_t i, std::size_t number_nnz);

  std::size_t seed_memory_load(std::size_t j, std::size_t k, std::size_t i,
        Operation operation);

  std::optional<std::size_t> get_memory_bound() const{

   return memory_bound; 
  }

  //Debugging purposes
  const cell_with_pointer<Jacobian_T>& get_cell(std::size_t index){

     return table.get_cell(index);
  }

  //Debugging purposes
  const cell<Jacobian_T>& get_cell(std::size_t index_j, std::size_t index_i){

      return table.get_cell(index_j, index_i);
  }

  const Sparse_Jacobian propagate_sparsity_pattern(std::size_t index_j, std::size_t index_i){
     if(index_j == index_i + 1){
         return table.get_cell(index_j).get_jacobian() *
                  table.get_cell(index_i).get_jacobian();
     }

     else{
         return table.get_cell(index_j, index_i+1).get_jacobian() *
                  table.get_cell(index_i).get_jacobian();
     }
  }

 protected:
  Table<Jacobian_T> table;

  std::optional<std::size_t> memory_bound;
  
  void fill(const jacobian_chain<Jacobian_T, Jacobian_info_T>& chain);

  void fill(const jacobian_chain<Jacobian_T, Jacobian_info_T>& chain,
        std::size_t memory_bound);
};

//Dynamic programming methods implementation

// Elemental Jacobian  accumulation methods:
template <class Jacobian_T, class Jacobian_info_T>
std::size_t fill_table<Jacobian_T, Jacobian_info_T>::tangent_cost(
      const Jacobian_T* elemental_jacobian_ptr){

   if constexpr(std::is_same_v<Jacobian_T, Dense_Jacobian>){

      return elemental_jacobian_ptr -> domain_dim() * 
         elemental_jacobian_ptr -> number_edges();
   }

   else if constexpr(std::is_same_v<Jacobian_T, Sparse_Jacobian>){

      return elemental_jacobian_ptr -> get_column_number_colors() *
         elemental_jacobian_ptr -> number_edges();
   }

   return 0;
}

template <class Jacobian_T, class Jacobian_info_T>
std::size_t fill_table<Jacobian_T, Jacobian_info_T>::adjoint_cost(
      const Jacobian_T* elemental_jacobian_ptr){

   if constexpr(std::is_same_v<Jacobian_T, Dense_Jacobian>){

      return elemental_jacobian_ptr -> codomain_dim() * 
         elemental_jacobian_ptr -> number_edges();
   }

   else if constexpr(std::is_same_v<Jacobian_T, Sparse_Jacobian>){

      return elemental_jacobian_ptr -> get_row_number_colors() *
         elemental_jacobian_ptr -> number_edges();
   }

   return 0;
}

template <class Jacobian_T, class Jacobian_info_T>
std::size_t fill_table<Jacobian_T, Jacobian_info_T>::accumulate_number_of_edges(
      std::size_t j_or_k, std::size_t k_plus_1_or_i){

   if constexpr(std::is_same_v<Jacobian_T, Sparse_Jacobian>){

      if(j_or_k == k_plus_1_or_i){

         return table.get_cell(j_or_k).number_edges();
      }

      else{

         return table.get_cell(j_or_k, k_plus_1_or_i).number_edges();
      }
   }

   else{

      std::size_t sum_of_edges = 0;
      for(std::size_t jacobian_idx = k_plus_1_or_i; jacobian_idx < 
            j_or_k + 1; jacobian_idx++){

         sum_of_edges += table.get_cell(jacobian_idx).number_edges();
      }
      
      return sum_of_edges;
   }
}



template <class Jacobian_T, class Jacobian_info_T>
std::size_t fill_table<Jacobian_T, Jacobian_info_T>::tangent_cost(
      std::size_t j, std::size_t k, std::size_t i){

  if(k == i){

     return table.get_cell(i).accumulated_cost() + 
        table.get_cell(i).domain_dim() * accumulate_number_of_edges(j, k+1);
  } 

  else{

   return table.get_cell(k, i).accumulated_cost() +
      table.get_cell(i).domain_dim() * accumulate_number_of_edges(j, k+1);

  }
}

template <class Jacobian_T, class Jacobian_info_T>
std::size_t fill_table<Jacobian_T, Jacobian_info_T>::tangent_cost(
      std::size_t j, std::size_t k, std::size_t i,
      std::size_t j_i_column_number_colors){

   if(k == i){

      return table.get_cell(i).accumulated_cost() +
         j_i_column_number_colors * accumulate_number_of_edges(j, k+1);
   }

   else{

      return table.get_cell(k,i).accumulated_cost() +
         j_i_column_number_colors * accumulate_number_of_edges(j, k+1);
   }
}

template <class Jacobian_T, class Jacobian_info_T>
std::size_t fill_table<Jacobian_T, Jacobian_info_T>::adjoint_cost(
      std::size_t j, std::size_t k, std::size_t i){

   if(j == k+1){

      return table.get_cell(j).accumulated_cost() +
         table.get_cell(j).codomain_dim() * accumulate_number_of_edges(k, i);
   }

   else{

      return table.get_cell(j, k+1).accumulated_cost() +
         table.get_cell(j).codomain_dim() * accumulate_number_of_edges(k, i);
   }
}

template <class Jacobian_T, class Jacobian_info_T>
std::size_t fill_table<Jacobian_T, Jacobian_info_T>::adjoint_cost(
      std::size_t j, std::size_t k, std::size_t i,
      std::size_t j_i_row_number_colors){

   if(j == k+1){

      return table.get_cell(j).accumulated_cost() +
         j_i_row_number_colors * accumulate_number_of_edges(k, i);
   }

   else{

      return table.get_cell(j,k+1).accumulated_cost() +
         j_i_row_number_colors * accumulate_number_of_edges(k,i);
   }
}

template <class Jacobian_T, class Jacobian_info_T>
std::size_t fill_table<Jacobian_T, Jacobian_info_T>::multiplication_cost(
      std::size_t j, std::size_t k, std::size_t i){

   std::size_t accumulated_cost_lhs, accumulated_cost_rhs;

   if(k == i){

      if constexpr(std::is_same_v<Jacobian_T, Jacobian>){

         accumulated_cost_rhs = 0;
      }

      else{

         accumulated_cost_rhs = table.get_cell(i).accumulated_cost();   
      }
   }

   else{

      accumulated_cost_rhs = table.get_cell(k, i).accumulated_cost();
   }

   if(j == k+1){

      if constexpr(std::is_same_v<Jacobian_T, Jacobian>){

         accumulated_cost_lhs = 0;
      }

      else{

         accumulated_cost_lhs = table.get_cell(j).accumulated_cost();
      }

   }

   else{

      accumulated_cost_lhs = table.get_cell(j, k+1).accumulated_cost();
   }

   return accumulated_cost_lhs +
            accumulated_cost_rhs +
            table.get_cell(j).codomain_dim() *
            table.get_cell(i).domain_dim() *
            table.get_cell(k).codomain_dim();
}

template <class Jacobian_T, class Jacobian_info_T>
std::size_t fill_table<Jacobian_T, Jacobian_info_T>::multiplication_cost(
      std::size_t j, std::size_t k, std::size_t i, std::size_t number_nnz){

   std::size_t accumulated_cost_lhs, accumulated_cost_rhs;
   std::size_t max_number_nnz_row, max_number_nnz_column;

   if(k == i){
     
      accumulated_cost_rhs = table.get_cell(i).accumulated_cost();   
      max_number_nnz_column = table.get_cell(i).max_number_nnz_column();
   }

   else{

      accumulated_cost_rhs = table.get_cell(k, i).accumulated_cost();
      max_number_nnz_column = table.get_cell(k,i).max_number_nnz_column();
   }

   if(j == k+1){

      accumulated_cost_lhs = table.get_cell(j).accumulated_cost();
      max_number_nnz_row = table.get_cell(j).max_number_nnz_row();
   }

   else{

      accumulated_cost_lhs = table.get_cell(j, k+1).accumulated_cost();
      max_number_nnz_row = table.get_cell(j, k+1).max_number_nnz_row();
   }

   return accumulated_cost_lhs +
            accumulated_cost_rhs+
            number_nnz *
            std::min(max_number_nnz_row, max_number_nnz_column);
}

template <class Jacobian_T, class Jacobian_info_T>
std::size_t fill_table<Jacobian_T, Jacobian_info_T>::seed_memory_load(
      std::size_t j, std::size_t k, std::size_t i, Operation operation){
   if(operation == Operation::TANGENT){

      if(k == i){

         return table.get_cell(i).accumulated_memory().value();
      }

      else{

         return table.get_cell(k, i).accumulated_memory().value();
      }
   }

   else if(operation == Operation::ADJOINT){

      if(j == k+1){

         return table.get_cell(j).accumulated_memory().value();
      }

      else{

         return table.get_cell(j, k+1).accumulated_memory().value();
      }
   }

   else{

      return seed_memory_load(j, k, i, Operation::ADJOINT) +
            seed_memory_load(j, k, i, Operation::TANGENT);
   }
}
//Ideas:
//Memory bound should be an optional value.
//Cell accumulate is makes the notation and implementation less readable.

template <class Jacobian_T, class Jacobian_info_T>
void fill_table<Jacobian_T, Jacobian_info_T>::fill(
      const jacobian_chain<Jacobian_T, Jacobian_info_T>& chain){

   //Initializing cells with pointer.
   std::size_t minimum_cost;
   std::size_t minimum_cost_j_k_i;
   std::size_t split_position;
   std::size_t split_position_j_k_i;
   Operation operation;
   Operation operation_j_k_i;

   for(std::size_t elemental_idx = 0; elemental_idx < chain.size(); elemental_idx++){

      if(chain[elemental_idx].domain_dim() < chain[elemental_idx].codomain_dim()){

         table.emplace_back(&chain[elemental_idx], tangent_cost(&chain[elemental_idx]),
               Operation::TANGENT);
      }

      else{
         
         table.emplace_back(&chain[elemental_idx], adjoint_cost(&chain[elemental_idx]),
                  Operation::ADJOINT);
      }
   }


   std::size_t index_i;
   std::size_t tangent_cost_j_k_i;
   std::size_t adjoint_cost_j_k_i;

   for(std::size_t index_j = 1; index_j < chain.size(); index_j++){


      for(std::size_t index_s = 1; index_s < index_j + 1; index_s++){

         index_i = index_j - index_s;
         minimum_cost= std::numeric_limits<std::size_t>::max();

         if constexpr(std::is_same_v<Jacobian_T, Dense_Jacobian>){
            for(std::size_t index_k = index_i; index_k < index_j; index_k++){
            
               minimum_cost_j_k_i = multiplication_cost(index_j, index_k, index_i); 

               split_position_j_k_i = index_k;
               operation_j_k_i = Operation::MULTIPLICATION;

               tangent_cost_j_k_i = tangent_cost(index_j, index_k, index_i);
               adjoint_cost_j_k_i = adjoint_cost(index_j, index_k, index_i);

               if(tangent_cost_j_k_i < minimum_cost_j_k_i){

                  minimum_cost_j_k_i = tangent_cost_j_k_i;
                  operation_j_k_i = Operation::TANGENT;
               }

               if(adjoint_cost_j_k_i < minimum_cost_j_k_i){

                  minimum_cost_j_k_i = adjoint_cost_j_k_i;
                  operation_j_k_i = Operation::ADJOINT;
               }

               if(minimum_cost_j_k_i < minimum_cost){

                  minimum_cost = minimum_cost_j_k_i;
                  split_position = split_position_j_k_i;
                  operation = operation_j_k_i;
               }
            }
            table.emplace_back(minimum_cost, split_position, operation);
         }

         else if constexpr(std::is_same_v<Jacobian_T, Sparse_Jacobian>){

            Sparse_Jacobian sparse_jacobian_j_i =
               propagate_sparsity_pattern(index_j, index_i);

            if(index_j == index_i+1){

            }
            for(std::size_t index_k = index_i; index_k < index_j; index_k++){

               minimum_cost_j_k_i = multiplication_cost(index_j, index_k,
                    index_i, sparse_jacobian_j_i.number_nnz());

               split_position_j_k_i = index_k;
               operation_j_k_i = Operation::MULTIPLICATION;

               tangent_cost_j_k_i = tangent_cost(index_j, index_k, index_i,
                     sparse_jacobian_j_i.get_column_number_colors());

               adjoint_cost_j_k_i = adjoint_cost(index_j, index_k, index_i,
                     sparse_jacobian_j_i.get_row_number_colors());


               if(tangent_cost_j_k_i < minimum_cost_j_k_i){

                  minimum_cost_j_k_i = tangent_cost_j_k_i;
                  operation_j_k_i = Operation::TANGENT;
               }

               if(adjoint_cost_j_k_i < minimum_cost_j_k_i){

                  minimum_cost_j_k_i = adjoint_cost_j_k_i;
                  operation_j_k_i = Operation::ADJOINT;
               }

               if(minimum_cost_j_k_i < minimum_cost){

                  minimum_cost = minimum_cost_j_k_i;
                  split_position = split_position_j_k_i;
                  operation = operation_j_k_i;
                  
               }
            }

            table.emplace_back(std::move(sparse_jacobian_j_i), minimum_cost,
                  split_position, operation);
         }
      }
   }
}

template <class Jacobian_T, class Jacobian_info_T>
void fill_table<Jacobian_T, Jacobian_info_T>::fill(
      const jacobian_chain<Jacobian_T, Jacobian_info_T>& chain,
      std::size_t memory_bound){

   //Initializing cells with pointer.
   std::size_t minimum_cost;
   std::size_t minimum_cost_j_k_i;
   std::size_t split_position;
   std::size_t split_position_j_k_i;
   std::size_t operation_memory_load;
   std::size_t operation_memory_load_j_k_i;
   std::size_t accumulated_memory_load;
   Operation operation;
   Operation operation_j_k_i;

   for(std::size_t elemental_idx = 0; elemental_idx < chain.size(); elemental_idx++){

      if(chain[elemental_idx].codomain_dim() < chain[elemental_idx].domain_dim() &&
            chain[elemental_idx].number_edges() <= memory_bound){

         operation_memory_load = chain[elemental_idx].number_edges();

         table.emplace_back(&chain[elemental_idx], adjoint_cost(&chain[elemental_idx]),
                  Operation::ADJOINT, operation_memory_load);
      }

      else{

         operation_memory_load = 0;

         table.emplace_back(&chain[elemental_idx], tangent_cost(&chain[elemental_idx]),
               Operation::TANGENT, operation_memory_load);
      }
   }


   std::size_t index_i;
   std::size_t tangent_cost_j_k_i, adjoint_cost_j_k_i;

   for(std::size_t index_j = 1; index_j < chain.size(); index_j++){


      for(std::size_t index_s = 1; index_s < index_j + 1; index_s++){

         index_i = index_j - index_s;
         minimum_cost= std::numeric_limits<std::size_t>::max();

         if constexpr(std::is_same_v<Jacobian_T, Dense_Jacobian>){

            for(std::size_t index_k = index_i; index_k < index_j; index_k++){

               minimum_cost_j_k_i = multiplication_cost(index_j, index_k, index_i); 
               split_position_j_k_i = index_k;
               operation_j_k_i = Operation::MULTIPLICATION;
               operation_memory_load_j_k_i = 0;

               tangent_cost_j_k_i = tangent_cost(index_j, index_k, index_i);
               adjoint_cost_j_k_i = adjoint_cost(index_j, index_k, index_i);

               if(tangent_cost_j_k_i < minimum_cost_j_k_i){

                  operation_memory_load_j_k_i = 0;
                  minimum_cost_j_k_i = tangent_cost_j_k_i;
                  operation_j_k_i = Operation::TANGENT;
               }

               if(adjoint_cost_j_k_i < minimum_cost_j_k_i){

                  operation_memory_load_j_k_i = accumulate_number_of_edges(index_k, index_i);

                  accumulated_memory_load = operation_memory_load_j_k_i +
                     seed_memory_load(index_j, index_k, index_i, Operation::ADJOINT);

                  if(accumulated_memory_load <= memory_bound){

                     minimum_cost_j_k_i = adjoint_cost_j_k_i;
                     operation_j_k_i = Operation::ADJOINT;
                  }
               }

               if(minimum_cost_j_k_i < minimum_cost){

                  minimum_cost = minimum_cost_j_k_i;
                  split_position = split_position_j_k_i;
                  operation = operation_j_k_i;
                  operation_memory_load = operation_memory_load_j_k_i; 
               }
            }

            table.emplace_back(minimum_cost, split_position,
                  operation, operation_memory_load +
                  seed_memory_load(index_j, split_position, index_i, operation));
         }

         if constexpr(std::is_same_v<Jacobian_T, Sparse_Jacobian>){

            Sparse_Jacobian sparse_jacobian_j_i =
               propagate_sparsity_pattern(index_j, index_i);

            for(std::size_t index_k = index_i; index_k < index_j; index_k++){

               operation_memory_load_j_k_i = 0;
               minimum_cost_j_k_i = multiplication_cost(index_j, index_k,
                    index_i, sparse_jacobian_j_i.number_nnz());

               split_position_j_k_i = index_k;
               operation_j_k_i = Operation::MULTIPLICATION;

               tangent_cost_j_k_i = tangent_cost(index_j, index_k, index_i,
                     sparse_jacobian_j_i.get_column_number_colors());

               adjoint_cost_j_k_i = adjoint_cost(index_j, index_k, index_i,
                     sparse_jacobian_j_i.get_row_number_colors());

               if(tangent_cost_j_k_i < minimum_cost_j_k_i){

                  operation_memory_load_j_k_i = 0;
                  minimum_cost_j_k_i = tangent_cost_j_k_i;
                  operation_j_k_i = Operation::TANGENT;
               }

               if(adjoint_cost_j_k_i < minimum_cost_j_k_i){

                  operation_memory_load_j_k_i = accumulate_number_of_edges(index_k, index_i);

                  accumulated_memory_load = operation_memory_load_j_k_i +
                     seed_memory_load(index_j, index_k, index_i, Operation::ADJOINT);

                  if(accumulated_memory_load <= memory_bound){

                     minimum_cost_j_k_i = adjoint_cost_j_k_i;
                     operation_j_k_i = Operation::ADJOINT;
                  }
               }

               if(minimum_cost_j_k_i < minimum_cost){

                  minimum_cost = minimum_cost_j_k_i;
                  split_position = split_position_j_k_i;
                  operation = operation_j_k_i;
                  operation_memory_load = operation_memory_load_j_k_i;
                  
               }
            }

            table.emplace_back(std::move(sparse_jacobian_j_i), minimum_cost,
                  split_position, operation, operation_memory_load +
                  seed_memory_load(index_j, split_position, index_i, operation));
         }

      }
   }
}

template <>
void fill_table<Jacobian, Jacobian_information>::fill(
      const jacobian_chain<Jacobian, Jacobian_information>& chain){

   for(std::size_t elemental_idx = 0; elemental_idx < chain.size(); elemental_idx++){

      table.emplace_back(&chain[elemental_idx]);
   }

   std::size_t minimum_cost;
   std::size_t index_i;
   std::size_t split_position;

   for(std::size_t index_j = 1; index_j < chain.size(); index_j++){

      for(std::size_t index_s = 1; index_s < index_j + 1; index_s++){

         index_i = index_j - index_s;
         minimum_cost= std::numeric_limits<std::size_t>::max();
         for(std::size_t index_k = index_i; index_k < index_j; index_k++){

            if(multiplication_cost(index_j, index_k, index_i) < minimum_cost){
               
               minimum_cost = multiplication_cost(index_j, index_k, index_i);
               split_position = index_k;
            }
         }

         table.emplace_back(minimum_cost, split_position);
      }
   }
}
#endif
