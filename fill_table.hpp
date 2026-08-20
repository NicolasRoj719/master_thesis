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
class fill_table{};

template<>
class fill_table<Jacobian, Jacobian_information>{
 public:
   fill_table(const jacobian_chain<Jacobian, Jacobian_information>& chain):
      table(chain.size()){

      fill(chain);
   }

   //testing purposes
   fill_table(Table<Jacobian> table_):
      table(std::move(table_)){}

   //Dense multiplication cost.
   std::size_t multiplication_cost(std::size_t j, std::size_t k, std::size_t i){

      std::size_t accumulation_cost_lhs, accumulation_cost_rhs;
      //Access cell_with_pointer
      if(k == i){
         //Jacobian dynamic programming assumes that the matrix representation
         //of the elemental jacobians is given.
         accumulation_cost_rhs = 0;
      }
      else{
         
         accumulation_cost_rhs = table.get_cell(k, i).accumulated_cost();
      }

      //Access cell_with_pointer
      if(j == k+1){
         
         //Jacobian dynamic programming assumes that the matrix representation
         //of the elemental jacobians is given.
         accumulation_cost_lhs = 0;
      }
      else{

         accumulation_cost_lhs = table.get_cell(j, k+1).accumulated_cost();
      }

      return accumulation_cost_lhs + accumulation_cost_rhs +
               table.get_cell(j).codomain_dim() * 
               table.get_cell(i).domain_dim() *
               table.get_cell(k).codomain_dim();
   }

   //Testing purposes
   const cell_with_pointer<Jacobian>& get_cell(std::size_t index){

      return table.get_cell(index);
   }

   const cell<Jacobian>& get_cell(std::size_t j, std::size_t i){

      return table.get_cell(j,i);
   }
 
 private:
   Table<Jacobian> table;

   void fill(const jacobian_chain<Jacobian, Jacobian_information>& chain);
};

void fill_table<Jacobian, Jacobian_information>::fill(
      const jacobian_chain<Jacobian, Jacobian_information>& chain){

   //Filling cells with pointers
   for(std::size_t elemental_idx = 0; elemental_idx < chain.size(); elemental_idx++){

      table.emplace_back(&chain[elemental_idx]);
   }

   std::size_t minimum_cost, i_index, split_position;
   std::size_t multiplication_cost_j_k_i;

   for(std::size_t j_index = 1; j_index < chain.size(); j_index++){

      for(std::size_t aux_var = 1; aux_var < j_index + 1; aux_var++){

         i_index = j_index - aux_var;
         minimum_cost = std::numeric_limits<std::size_t>::max();
         //Problem instance (j,i)
         for(std::size_t k_index = i_index; k_index < j_index; k_index++){
            

            multiplication_cost_j_k_i = multiplication_cost(j_index, k_index, i_index);
            if(multiplication_cost_j_k_i < minimum_cost){

               minimum_cost = multiplication_cost_j_k_i;
               split_position = k_index;
            }
         }

         table.emplace_back(minimum_cost, split_position);
      }
   }
}

template<>
class fill_table<Dense_Jacobian, Matrix_free_information>{
 public:
   fill_table(const jacobian_chain<Dense_Jacobian, Matrix_free_information>& chain):
      table(chain.size()){

      fill(chain);
   }

   fill_table(const jacobian_chain<Dense_Jacobian, Matrix_free_information>& chain,
               std::size_t memory_bound_):
      table(chain.size()), memory_bound(memory_bound_){

      fill(chain);
   }

   //testing purposes
   fill_table(Table<Dense_Jacobian> table_):
      table(std::move(table_)){}
   
   std::size_t accumulate_edges(std::size_t j_or_k, std::size_t k_plus_1_or_i) const{
      
      std::size_t sum_of_edges = 0;
      for(std::size_t idx = k_plus_1_or_i; idx < j_or_k + 1; idx++){
         
         sum_of_edges += table.get_cell(idx).number_edges();
      }

      return sum_of_edges;
   }

   //Tangent and adjoint accumulation cost for a elemental jacobian.
   static std::size_t tangent_cost(const Dense_Jacobian* elemental_jacobian_ptr) {
     
      return elemental_jacobian_ptr -> domain_dim() *
               elemental_jacobian_ptr -> number_edges();
   }

   //Assumption the memory bound is greater than the memory required
   //to preaccumulate elemental jacobians.
   static std::size_t adjoint_cost(const Dense_Jacobian* elemental_jacobian_ptr) {

      return elemental_jacobian_ptr -> codomain_dim() *
               elemental_jacobian_ptr -> number_edges();
   }

   //Tangent general accumulation formula
   std::size_t tangent_cost(std::size_t j, std::size_t k, std::size_t i) const{

      //Access a cell_with_pointer
      if(k == i){
         
         return table.get_cell(i).accumulated_cost() +
            table.get_cell(i).domain_dim() * accumulate_edges(j, k+1);
      }

      else{
         
         return table.get_cell(k,i).accumulated_cost() +
            table.get_cell(i).domain_dim() * accumulate_edges(j, k+1);
      }
   }

   //Adjoint general accumulation formula
   std::optional<std::size_t> adjoint_cost(std::size_t j, std::size_t k, std::size_t i) const{

      std::size_t total_edges = accumulate_edges(k,i);
      //Access a cell_with_pointer
      if(j == k+1){
         
         //If memory bound is set
         if(memory_bound){
            
            //If memory bound is not surpassed 
            if(total_edges <= *memory_bound){

               return table.get_cell(j).accumulated_cost() +
                  table.get_cell(j).codomain_dim() * total_edges;
            }

            //If memory bound is surpassed.
            return {};
         }
         //Access a normal cell
         else{
            
            return table.get_cell(j).accumulated_cost() +
               table.get_cell(j).codomain_dim() * total_edges; 
         }
      }
      //Access normal cell
      else{
         
         if(memory_bound){

            if(total_edges <= *memory_bound){

               return table.get_cell(j, k+1).accumulated_cost() +
                  table.get_cell(j).codomain_dim() * total_edges;
            }

            //Memory bound not fulfilled
            return {};
         }

         else{
            
            return table.get_cell(j, k+1).accumulated_cost() +
               table.get_cell(j).codomain_dim() * total_edges;
         }
      }
   }

   std::size_t multiplication_cost(std::size_t j, std::size_t k, std::size_t i) const{

      std::size_t accumulated_cost_lhs, accumulated_cost_rhs;
      //Access cell_with_pointer
      if(k == i){
         
         accumulated_cost_rhs = table.get_cell(i).accumulated_cost();
      }
      else{

         accumulated_cost_rhs = table.get_cell(k,i).accumulated_cost();
      }

      //Access cell_with_pointer
      if(j == k+1){
         
         accumulated_cost_lhs = table.get_cell(j).accumulated_cost();
      }
      else{
         
         accumulated_cost_lhs = table.get_cell(j, k+1).accumulated_cost();
      }

      return accumulated_cost_lhs + accumulated_cost_rhs +
               table.get_cell(j).codomain_dim() * table.get_cell(i).domain_dim() *
               table.get_cell(k).codomain_dim();
   }

   std::optional<std::size_t> get_memory_bound() const{

      return memory_bound;
   }

   //Testing purposes
   const cell_with_pointer<Dense_Jacobian>& get_cell(std::size_t index){
      
      return table.get_cell(index);
   }

   const cell<Dense_Jacobian>& get_cell(std::size_t j, std::size_t i){

      return table.get_cell(j,i);
   }

 private:
   Table<Dense_Jacobian> table;
   std::optional<std::size_t> memory_bound;

   void fill(const jacobian_chain<Dense_Jacobian, Matrix_free_information>& chain);
};

void fill_table<Dense_Jacobian, Matrix_free_information>::fill(
      const jacobian_chain<Dense_Jacobian, Matrix_free_information>& chain){

   table.clear();

   //Initializing cells with pointer.
   for(std::size_t elemental_idx = 0; elemental_idx < chain.size(); elemental_idx++){

      if(chain[elemental_idx].domain_dim() <= chain[elemental_idx].codomain_dim()){

         table.emplace_back(&chain[elemental_idx], tangent_cost(&chain[elemental_idx]),
               Operation::TANGENT);
      }

      else{

         if(memory_bound){

            if(chain[elemental_idx].number_edges() <= *memory_bound){
               
               table.emplace_back(&chain[elemental_idx], adjoint_cost(&chain[elemental_idx]),
                     Operation::ADJOINT);
            }
            //Memory bound is exceeded then the elemental jacobian is accumulated using tangent
            //mode.
            else{

               table.emplace_back(&chain[elemental_idx], tangent_cost(&chain[elemental_idx]),
                     Operation::TANGENT);
            }
         }

         else{

            table.emplace_back(&chain[elemental_idx], adjoint_cost(&chain[elemental_idx]),
                  Operation::ADJOINT);
         }

      }
   }

   std::size_t minimum_cost, minimum_cost_j_k_i, split_position;
   std::size_t split_position_j_k_i;
   std::size_t i_index, tangent_cost_j_k_i;
   std::optional<std::size_t> adjoint_cost_j_k_i;
   Operation operation, operation_j_k_i;

   for(std::size_t j_index = 1; j_index < chain.size(); j_index++){
   
      for(std::size_t aux_var = 1; aux_var < j_index + 1; aux_var++){

         i_index = j_index - aux_var;
         minimum_cost = std::numeric_limits<std::size_t>::max();

         //k_index split position.
         //Problem instance (j_index, i_index)
         for(std::size_t k_index = i_index; k_index < j_index; k_index++){ 
            
            minimum_cost_j_k_i = multiplication_cost(j_index, k_index, i_index);
            split_position_j_k_i = k_index;
            operation_j_k_i = Operation::MULTIPLICATION;

            //Minimum over the different accumulation methods.
            tangent_cost_j_k_i = tangent_cost(j_index, k_index, i_index);
            if(tangent_cost_j_k_i < minimum_cost_j_k_i){

               minimum_cost_j_k_i = tangent_cost_j_k_i;
               operation_j_k_i = Operation::TANGENT;
            }

            adjoint_cost_j_k_i = adjoint_cost(j_index, k_index, i_index);
            if(adjoint_cost_j_k_i){

               if(*adjoint_cost_j_k_i < minimum_cost_j_k_i){

                  minimum_cost_j_k_i = *adjoint_cost_j_k_i;
                  operation_j_k_i = Operation::ADJOINT;
               }
            }

            //Compare the minimum cost over all accumulation methods
            //for split position k_index with the current minimum for
            // the problem instance (j_index, i_index)
            if(minimum_cost_j_k_i < minimum_cost){

               minimum_cost = minimum_cost_j_k_i;
               split_position = split_position_j_k_i;
               operation = operation_j_k_i;
            }
         }
         table.emplace_back(minimum_cost, split_position, operation);
      }
   }
}

template<>
class fill_table<Sparse_Jacobian, Matrix_free_sparse_information>{
 public:
   fill_table(const jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>& chain):
      table(chain.size()){
      
      fill(chain);
   }

   fill_table(const jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>& chain,
                  std::size_t memory_bound_):
      table(chain.size()), memory_bound(memory_bound_){

      fill(chain);
   }

   //testing purposes
   fill_table(Table<Sparse_Jacobian> table_):
      table(std::move(table_)){}
   
   //Sparsity patterns are propagated throughout the table.
   //accumulate_edges simplifies to reading a cell from the table and extracting
   // the number of edges.
   std::size_t accumulate_edges(std::size_t j_or_k, std::size_t k_plus_1_or_i) const{

      //Access cell with pointer
      if(j_or_k == k_plus_1_or_i){

         return table.get_cell(j_or_k).number_edges();
      }
      else{
         
         return table.get_cell(j_or_k, k_plus_1_or_i).number_edges();
      }
   }

   //Tangent and adjoint accumulation cost for a elemental jacobian.
   static std::size_t tangent_cost(const Sparse_Jacobian* elemental_jacobian_ptr) {

      return elemental_jacobian_ptr -> get_column_number_colors() *
               elemental_jacobian_ptr -> number_edges();
   }

   //Assumption the memory bound is greater than the memory required
   //to preaccumulate elemental jacobians.
   static std::size_t adjoint_cost(const Sparse_Jacobian* elemental_jacobian_ptr) {

      return elemental_jacobian_ptr -> get_row_number_colors() *
               elemental_jacobian_ptr -> number_edges();
   }

   //Tangent general accumulation formula
   std::size_t tangent_cost(std::size_t j, std::size_t k, std::size_t i,
                              std::size_t j_i_column_number_colors) const{

      //Access cell with pointer
     if(k == i){

        return table.get_cell(i).accumulated_cost() +
               j_i_column_number_colors * accumulate_edges(j, k+1);
     }
     //Access normal cell
     else{

         return table.get_cell(k, i).accumulated_cost() +
               j_i_column_number_colors * accumulate_edges(j, k+1);
     }
   }

   //Adjoint general accumulation formula
   std::optional<std::size_t> adjoint_cost(std::size_t j, std::size_t k, std::size_t i,
                                             std::size_t j_i_row_number_colors) const{
      
      std::size_t total_edges = accumulate_edges(k, i);
      //Access a cell_with_pointer
      if(j == k+1){

         //If memory bound is set
         if(memory_bound){

            //If memory bound is not surpassed. 
            if(total_edges <= *memory_bound){

               return table.get_cell(j).accumulated_cost() +
                        j_i_row_number_colors * total_edges;
            }

            //If memory bound is surpassed
            return {};
         }

         return table.get_cell(j).accumulated_cost() +
                  j_i_row_number_colors * total_edges;
      }
      //Access a normal cell
      else{

         //If memory bound is set.
         if(memory_bound){
            
            if(total_edges <= *memory_bound){

               return table.get_cell(j, k+1).accumulated_cost() +
                        j_i_row_number_colors * total_edges;
            }

            return {};
         }

         return table.get_cell(j, k+1).accumulated_cost() +
                  j_i_row_number_colors * total_edges;
      }
   }

   std::size_t multiplication_cost(std::size_t j, std::size_t k, std::size_t i,
                                    std::size_t number_nnz) const{

      std::size_t accumulated_cost_lhs, accumulated_cost_rhs;
      std::size_t max_number_nnz_row, max_number_nnz_column;
      //Access cell_with_pointer
      if(k == i){

         accumulated_cost_rhs = table.get_cell(i).accumulated_cost();
         max_number_nnz_column = table.get_cell(i).max_number_nnz_column();
      }
      else{
         
         accumulated_cost_rhs = table.get_cell(k, i).accumulated_cost();
         max_number_nnz_column = table.get_cell(k, i).max_number_nnz_column();
      }

      //Access cell_with_pointer
      if(j == k+1){

         accumulated_cost_lhs = table.get_cell(j).accumulated_cost();
         max_number_nnz_row = table.get_cell(j).max_number_nnz_row();
      }
      else{
         
         accumulated_cost_lhs = table.get_cell(j, k+1).accumulated_cost();
         max_number_nnz_row = table.get_cell(j, k+1).max_number_nnz_row();
      }

      return accumulated_cost_rhs + accumulated_cost_lhs +
               number_nnz * 
               std::min(max_number_nnz_column, max_number_nnz_row);
   }

   const Sparse_Jacobian propagate_sparsity_pattern(std::size_t j, std::size_t i) const{

      //Accessing cells with pointer
      if(j == i + 1){

         return table.get_cell(j).get_jacobian() *
                  table.get_cell(i).get_jacobian();
      }
      
      else{
         
         return table.get_cell(j, i+1).get_jacobian() *
                  table.get_cell(i).get_jacobian();
      }

   }

   std::optional<std::size_t> get_memory_bound() const{
      
      return memory_bound;
   }

   //Debugging purposes
   const cell_with_pointer<Sparse_Jacobian>& get_cell(std::size_t index){
      
      return table.get_cell(index);
   }

   const cell<Sparse_Jacobian>& get_cell(std::size_t j, std::size_t i){

      return table.get_cell(j,i);
   }

 private:
   Table<Sparse_Jacobian> table;

   std::optional<std::size_t> memory_bound;

   void fill(const jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>& chain);
};

void fill_table<Sparse_Jacobian, Matrix_free_sparse_information>::fill(
      const jacobian_chain<Sparse_Jacobian, Matrix_free_sparse_information>& chain){

   table.clear();

   //Initializing cells with pointer.
   for(std::size_t elemental_idx = 0; elemental_idx < chain.size(); elemental_idx++){

      if(chain[elemental_idx].get_column_number_colors() <=
            chain[elemental_idx].get_row_number_colors()){

         table.emplace_back(&chain[elemental_idx], tangent_cost(&chain[elemental_idx]),
                              Operation::TANGENT);
      }

      else{

         if(memory_bound){


            if(chain[elemental_idx].number_edges() <= *memory_bound){
               
               table.emplace_back(&chain[elemental_idx], adjoint_cost(&chain[elemental_idx]),
                        Operation::ADJOINT);
            }
            //Memory bound is exceeded. The elemental jacobian is accumulated using tangent mode.
            else{

               table.emplace_back(&chain[elemental_idx], tangent_cost(&chain[elemental_idx]),
                     Operation::TANGENT);
            }
         }

         else{

            table.emplace_back(&chain[elemental_idx], adjoint_cost(&chain[elemental_idx]),
                  Operation::ADJOINT);
         }
      }
   }

   std::size_t minimum_cost, minimum_cost_j_k_i, split_position;
   std::size_t split_position_j_k_i;
   std::size_t i_index, tangent_cost_j_k_i;
   std::optional<std::size_t> adjoint_cost_j_k_i;
   Operation operation, operation_j_k_i;

   for(std::size_t j_index = 1; j_index < chain.size(); j_index++){

      for(std::size_t aux_var = 1; aux_var < j_index + 1; aux_var++){

         i_index = j_index - aux_var;
         minimum_cost = std::numeric_limits<std::size_t>::max();
         
         //Propagating sparsity pattern
         Sparse_Jacobian sparse_jacobian_j_i{propagate_sparsity_pattern(j_index, i_index)};
         //k_index split_position.
         //Problem instance (j_index, i_index)
         for(std::size_t k_index = i_index; k_index < j_index; k_index++){

            minimum_cost_j_k_i = multiplication_cost(j_index, k_index,
                                 i_index, sparse_jacobian_j_i.number_nnz());

            split_position_j_k_i = k_index;
            operation_j_k_i = Operation::MULTIPLICATION;

            //Minimum over the different accumulation methods.
            tangent_cost_j_k_i = tangent_cost(j_index, k_index,
                                 i_index, sparse_jacobian_j_i.get_column_number_colors());

            if(tangent_cost_j_k_i < minimum_cost_j_k_i){

               minimum_cost_j_k_i = tangent_cost_j_k_i;
               operation_j_k_i = Operation::TANGENT;
            }

            adjoint_cost_j_k_i = adjoint_cost(j_index, k_index,
                                 i_index, sparse_jacobian_j_i.get_row_number_colors());

            if(adjoint_cost_j_k_i){

               if(*adjoint_cost_j_k_i < minimum_cost_j_k_i){

                  minimum_cost_j_k_i = *adjoint_cost_j_k_i;
                  operation_j_k_i = Operation::ADJOINT;
               }
            }

            //Compare the minimum cost over all accumulation methods
            //for split position k_index with the current minimum for
            // the problem instance (j_index, i_index)
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

#endif
