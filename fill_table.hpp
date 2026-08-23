#include <algorithm>
#include <optional>
#include <limits>
#include <type_traits>
#include <vector>
#include "binomial_checkpointing.hpp"
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

//Shared methods between fill_table<Dense_Jacobian, Matrix_free_information>
//and fill_table<Split_dense_Jacobian, Split_reversal_dense_information>
namespace tool_box_dense{

   template<class Jacobian_T>
   std::size_t tangent_cost(const Jacobian_T* elemental_jacobian_ptr){

      
      return elemental_jacobian_ptr -> domain_dim() *
               elemental_jacobian_ptr -> number_edges();
   }

   template<class Jacobian_T>
   std::size_t adjoint_cost(const Jacobian_T* elemental_jacobian_ptr){

      return elemental_jacobian_ptr -> codomain_dim() *
               elemental_jacobian_ptr -> number_edges();
   }

   template<class Jacobian_T>
   std::size_t accumulate_edges(const Table<Jacobian_T>& table, std::size_t j_or_k,
                                 std::size_t k_plus_1_or_i){

      std::size_t sum_of_edges = 0;
      for(std::size_t idx = k_plus_1_or_i; idx < j_or_k + 1; idx++){
         
         sum_of_edges += table.get_cell(idx).number_edges();
      }

      return sum_of_edges;
   }

   template<class Jacobian_T>
   std::size_t tangent_cost(const Table<Jacobian_T>& table, std::size_t j, std::size_t k,
                              std::size_t i){
      //Access a cell_with_pointer
      if(k == i){
         
         return table.get_cell(i).accumulated_cost() +
            table.get_cell(i).domain_dim() * accumulate_edges(table, j, k+1);
      }

      else{
         
         return table.get_cell(k,i).accumulated_cost() +
            table.get_cell(i).domain_dim() * accumulate_edges(table, j, k+1);
      }
   }

   template<class Jacobian_T>
   std::size_t multiplication_cost(const Table<Jacobian_T>& table, std::size_t j, std::size_t k,
                                    std::size_t i){

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
      
      return tool_box_dense::accumulate_edges(table, j_or_k, k_plus_1_or_i);
   }

   //Tangent and adjoint accumulation cost for a elemental jacobian.
   static std::size_t tangent_cost(const Dense_Jacobian* elemental_jacobian_ptr) {
     
      return tool_box_dense::tangent_cost(elemental_jacobian_ptr);
   }

   //Assumption the memory bound is greater than the memory required
   //to preaccumulate elemental jacobians.
   static std::size_t adjoint_cost(const Dense_Jacobian* elemental_jacobian_ptr) {

      return tool_box_dense::adjoint_cost(elemental_jacobian_ptr);
   }

   //Tangent general accumulation formula
   std::size_t tangent_cost(std::size_t j, std::size_t k, std::size_t i) const{

      return tool_box_dense::tangent_cost(table, j, k, i);
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

            //If memory bound is exceeded.
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

      return tool_box_dense::multiplication_cost(table, j, k, i);
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
   std::size_t i_index, k_index, tangent_cost_j_k_i;
   std::optional<std::size_t> adjoint_cost_j_k_i;
   Operation operation, operation_j_k_i;

   for(std::size_t j_index = 1; j_index < chain.size(); j_index++){
   
      for(std::size_t aux_var_i = 1; aux_var_i < j_index + 1; aux_var_i++){

         i_index = j_index - aux_var_i;
         minimum_cost = std::numeric_limits<std::size_t>::max();

         //k_index split position.
         //Problem instance (j_index, i_index)
         for(std::size_t aux_var_k = 1; aux_var_k <= j_index - i_index; aux_var_k++){ 
            //Decreasing order. From j_index -1 to i_index
            k_index = j_index - aux_var_k;
            
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

namespace tool_box_split{

   template<class Split_T>
   std::size_t accumulate_function_cost(const Table<Split_T>& table, std::size_t j, std::size_t i){

      std::size_t sum_function_cost = 0;
      for(std::size_t idx = i; idx < j + 1; idx++){

         sum_function_cost += table.get_cell(idx).function_cost();
      }

      return sum_function_cost;
   }

   //To apply binomial chekpoiting algorithm every single elemental jacobian 
   //within the chain must be split reversable. Meaning there is enough memory 
   //to apply adjoint mode to the elemental jacobians within the given range [i,j].
   template<class Split_T>
   bool is_split_reversable(const Table<Split_T>& table, std::size_t j, std::size_t i,
                              std::size_t memory_bound){

      for(std::size_t idx = i; idx < j + 1; idx++){

         if(memory_bound < table.get_cell(idx).number_edges()){

            return false;
         }
      }

      return true;
   }

   template<class Split_T>
   std::size_t split_reversed_chain(const Table<Split_T>& table, std::size_t i,
                                    std::size_t accumulated_edges_j_i, std::size_t memory_bound){

      std::size_t counter = 0;
      while(memory_bound < accumulated_edges_j_i){

         accumulated_edges_j_i -= table.get_cell(i + counter).number_edges();
         counter ++;
      }

      counter--;

      // k = i + counter.
      // The Binomial_checkpointing algorithm will run over the chain
      // (i, k)(k+1,j)
      // (i,k) is the subchain of chain within the range [i,j]
      // (k+1, j) corresponds to F'_(j,k+1).
      return i + counter;
   }


}


template<>
class fill_table<Split_dense_Jacobian, Split_reversal_dense_information>{
 public:
   fill_table(const jacobian_chain<Split_dense_Jacobian, Split_reversal_dense_information>& chain_,
                  std::size_t memory_bound_, std::size_t number_checkpoints_):
      table(chain_.size()), chain(chain_), memory_bound(memory_bound_),
      number_checkpoints(number_checkpoints_){

      fill();
   }

   std::size_t accumulate_edges(std::size_t j_or_k, std::size_t k_plus_1_or_i) const{

      return tool_box_dense::accumulate_edges(table, j_or_k, k_plus_1_or_i);
   }

   static std::size_t tangent_cost(const Split_dense_Jacobian* elemental_jacobian_ptr){

      return tool_box_dense::tangent_cost(elemental_jacobian_ptr);
   }

   static std::size_t adjoint_cost(const Split_dense_Jacobian* elemental_jacobian_ptr){

      return tool_box_dense::adjoint_cost(elemental_jacobian_ptr);
   }

   std::size_t tangent_cost(std::size_t j, std::size_t k, std::size_t i){

      return tool_box_dense::tangent_cost(table, j, k, i);
   }

   std::size_t accumulate_function_cost(std::size_t j, std::size_t i){

      return tool_box_split::accumulate_function_cost(table, j, i);
   }

   bool is_split_reversable(std::size_t j, std::size_t i){
      
      return tool_box_split::is_split_reversable(table, j, i, memory_bound);
   }

   std::size_t split_reversed_chain(std::size_t i, std::size_t accumulated_edges_j_i){
      
      return tool_box_split::split_reversed_chain(table, i, accumulated_edges_j_i, memory_bound);
   }


   Split_dense_Jacobian joint_jacobian(std::size_t j, std::size_t i){

      return Split_dense_Jacobian{table.get_cell(i).domain_dim(),
                                    table.get_cell(j).codomain_dim(),
                                    accumulate_edges(j,i),
                                    accumulate_function_cost(j,i)};
   }

   //Joint adjoint cost assuming no memory bound.
   //Executing adjoint_cost could be expensive if binomial checkpointg needs to be 
   //executed.
   //One would like to avoid calling adjoint at least if its potentially the optimal 
   //accumulation method.
   std::size_t pre_adjoint_cost(std::size_t j, std::size_t k, std::size_t i){
      //Access a cell with pointer
      if(j == k+1){
         
         return table.get_cell(j).accumulated_cost() +
                  table.get_cell(j).codomain_dim() * accumulate_edges(k,i); 
      }
      //Access a normal cell
      else{

         return table.get_cell(j, k+1).accumulated_cost() +
                  table.get_cell(j).codomain_dim() * accumulate_edges(k,i);
      }
   }

   std::optional<std::size_t> adjoint_cost(std::size_t j, std::size_t k, std::size_t i,
                                             bool& is_adjoint_split_optimal){

      std::size_t number_edges_k_i = accumulate_edges(k,i);

      //Is the subchain k,i joint reversable?
      if(number_edges_k_i <= memory_bound){

         //Access a cell_with_pointer
         if(j == k+1){

            return table.get_cell(j).accumulated_cost() +
                     table.get_cell(j).codomain_dim() * number_edges_k_i;
         }
         //Access a normal cell
         else{
            return table.get_cell(j, k+1).accumulated_cost() +
                     table.get_cell(j).codomain_dim() * number_edges_k_i;
         }
      }
      //subchain k,i is not joint reversable
      {
         //Is the subchain k,i split reversable? Said in other words
         //does s_number_edges <= memory_bound for s in i, i+1, ..., k
         if(is_split_reversable(k,i)){

            is_adjoint_split_optimal = true;
            
            //Task: from subchain the subchain k,i find least possible k_tilde such that
            // the subchain k,k_tilde number of edges <= memory_bound. k_hat is obtained 
            // using split_reversed_chain method.
            std::size_t k_tilde = split_reversed_chain(i, number_edges_k_i);

            //Create split_dense_jacobian (k,k_tilde + 1)
            Split_dense_Jacobian split_dense_k_k_tilde_plus_one = joint_jacobian(k, k_tilde + 1);

            //Running binomial checkpointg algorithm on 
            // split_dense_k_k_tilde_plus_one subchain k_tilde,i.
            // A checkpoint is stored at z_(i-1) before executing the algorithm.
            // Therefore for the algorithm one has only number_checkpoints - 1 available 
            // checkpoints.
            binomial_checkpointing 
               binomial_algortihm{chain, k_tilde, i, number_checkpoints - 1,
                                    &split_dense_k_k_tilde_plus_one};

            //Adjoint cost calculation
            //Access a cell with pointer
            if(j == k+1){

               return table.get_cell(j).accumulated_cost() +
                        table.get_cell(j).codomain_dim() * number_edges_k_i +
                        binomial_algortihm.get_additional_cost();
            }
            //Access a normal cell
            else{

               return table.get_cell(j, k+1).accumulated_cost() +
                        table.get_cell(j).codomain_dim() * number_edges_k_i +
                        binomial_algortihm.get_additional_cost();
            }
         }
         //Not split reversable
         else{return {};}
      }
   }

   std::size_t multiplication_cost(std::size_t j, std::size_t k, std::size_t i){

      return tool_box_dense::multiplication_cost(table, j, k, i);
   }

 private:
   Table<Split_dense_Jacobian> table;
   const jacobian_chain<Split_dense_Jacobian, Split_reversal_dense_information>& chain;
   std::size_t memory_bound;
   std::size_t number_checkpoints;

   void fill();
};

void fill_table<Split_dense_Jacobian, Split_reversal_dense_information>::fill(){

   table.clear();

   //Initializing cells with pointer.
   for(std::size_t elemental_idx = 0; elemental_idx < chain.size(); elemental_idx++){

      if(chain[elemental_idx].domain_dim() <= chain[elemental_idx].codomain_dim()){

         table.emplace_back(&chain[elemental_idx], tangent_cost(&chain[elemental_idx]),
               Operation::TANGENT);
      }

      else{
         if(chain[elemental_idx].number_edges() <= memory_bound){
            
            table.emplace_back(&chain[elemental_idx], adjoint_cost(&chain[elemental_idx]),
                  Operation::ADJOINT);
         }
         //Memory bound is exceeded then the elemental jacobian is accumulated using
         // tangent mode.
         else{

            table.emplace_back(&chain[elemental_idx], tangent_cost(&chain[elemental_idx]),
                  Operation::TANGENT);
         }
      }
   }

   std::size_t minimum_cost, minimum_cost_j_k_i, split_position;
   std::size_t split_position_j_k_i;
   std::size_t i_index, k_index, tangent_cost_j_k_i;
   std::optional<std::size_t> adjoint_cost_j_k_i;
   bool is_adjoint_split_optimal = false;
   Operation operation, operation_j_k_i;

   for(std::size_t j_index = 1; j_index < chain.size(); j_index++){

      for(std::size_t aux_var_i = 1; aux_var_i < j_index + 1; aux_var_i++){

         i_index = j_index - aux_var_i;
         minimum_cost = std::numeric_limits<std::size_t>::max();

         //k_index split position
         //Problem instance (j_index, i_index)
         for(std::size_t aux_var_k = 1; aux_var_k <= j_index - i_index; aux_var_k++){
            //Decreasing order. From j_index - 1 to i_index
            k_index = j_index - aux_var_k;

            minimum_cost_j_k_i = multiplication_cost(j_index, k_index, i_index);
            split_position_j_k_i = k_index;
            operation_j_k_i = Operation::MULTIPLICATION;

            //Minimum over the different accumulation methods
            tangent_cost_j_k_i = tangent_cost(j_index, k_index, i_index);
            if(tangent_cost_j_k_i < minimum_cost_j_k_i){

               minimum_cost_j_k_i = tangent_cost_j_k_i;
               operation_j_k_i = Operation::TANGENT;
            }

            //Assuming no memory bound: is adjoint mode potentially the optimal
            //accumulation method for the subproblem j,i?
            if(pre_adjoint_cost(j_index, k_index, i_index) < minimum_cost_j_k_i){

               adjoint_cost_j_k_i = adjoint_cost(j_index, k_index, i_index,
                                                   is_adjoint_split_optimal);

               if(adjoint_cost_j_k_i){
                  
                  if(*adjoint_cost_j_k_i < minimum_cost_j_k_i){

                     minimum_cost_j_k_i = *adjoint_cost_j_k_i;
                     if(is_adjoint_split_optimal){
                        
                        operation_j_k_i = Operation::ADJOINT_SPLIT;
                     }
                     else{
                        
                        operation_j_k_i = Operation::ADJOINT;
                     }
                  }
               }

               is_adjoint_split_optimal = false;
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

namespace tool_box_sparse{

   template<class Jacobian_T>
   std::size_t tangent_cost(const Jacobian_T* elemental_jacobian_ptr){
      
      return elemental_jacobian_ptr -> get_column_number_colors() *
               elemental_jacobian_ptr -> number_edges();
   }

   template<class Jacobian_T>
   std::size_t adjoint_cost(const Jacobian_T* elemental_jacobian_ptr){

      return elemental_jacobian_ptr -> get_row_number_colors() *
               elemental_jacobian_ptr -> number_edges();
   }

   template<class Jacobian_T>
   std::size_t accumulate_edges(const Table<Jacobian_T>& table, std::size_t j_or_k,
                                 std::size_t k_plus_1_or_i){
      //Access cell with pointer
      if(j_or_k == k_plus_1_or_i){

         return table.get_cell(j_or_k).number_edges();
      }
      else{
         
         return table.get_cell(j_or_k, k_plus_1_or_i).number_edges();
      }
   }

   template<class Jacobian_T>
   std::size_t tangent_cost(const Table<Jacobian_T>& table, std::size_t j, std::size_t k,
                              std::size_t i, std::size_t j_i_column_number_colors){
      //Access cell with pointer
     if(k == i){

        return table.get_cell(i).accumulated_cost() +
               j_i_column_number_colors * accumulate_edges(table, j, k+1);
     }
     //Access normal cell
     else{

         return table.get_cell(k, i).accumulated_cost() +
               j_i_column_number_colors * accumulate_edges(table, j, k+1);
     }

   }

   template<class Jacobian_T>
   std::size_t multiplication_cost(const Table<Jacobian_T>& table, std::size_t j, std::size_t k,
                                    std::size_t i, std::size_t number_nnz){

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

   template<class Jacobian_T>
   const Jacobian_T propagate_sparsity_pattern(const Table<Jacobian_T>& table, std::size_t j,
                                                std::size_t i){
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

      return tool_box_sparse::accumulate_edges(table, j_or_k, k_plus_1_or_i);
   }

   //Tangent and adjoint accumulation cost for a elemental jacobian.
   static std::size_t tangent_cost(const Sparse_Jacobian* elemental_jacobian_ptr) {

      return tool_box_sparse::tangent_cost(elemental_jacobian_ptr);
   }

   //Assumption the memory bound is greater than the memory required
   //to preaccumulate elemental jacobians.
   static std::size_t adjoint_cost(const Sparse_Jacobian* elemental_jacobian_ptr) {

      return tool_box_sparse::adjoint_cost(elemental_jacobian_ptr);
   }

   //Tangent general accumulation formula
   std::size_t tangent_cost(std::size_t j, std::size_t k, std::size_t i,
                              std::size_t j_i_column_number_colors) const{

      return tool_box_sparse::tangent_cost(table, j, k, i, j_i_column_number_colors);
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

      return tool_box_sparse::multiplication_cost(table, j, k, i, number_nnz);
   }

   const Sparse_Jacobian propagate_sparsity_pattern(std::size_t j, std::size_t i) const{

      return tool_box_sparse::propagate_sparsity_pattern(table, j, i);
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
   std::size_t i_index, k_index, tangent_cost_j_k_i;
   std::optional<std::size_t> adjoint_cost_j_k_i;
   Operation operation, operation_j_k_i;

   for(std::size_t j_index = 1; j_index < chain.size(); j_index++){

      for(std::size_t aux_var_i = 1; aux_var_i < j_index + 1; aux_var_i++){

         i_index = j_index - aux_var_i;
         minimum_cost = std::numeric_limits<std::size_t>::max();
         
         //Propagating sparsity pattern
         Sparse_Jacobian sparse_jacobian_j_i{propagate_sparsity_pattern(j_index, i_index)};
         //k_index split_position.
         //Problem instance (j_index, i_index)
         for(std::size_t aux_var_k= 1; aux_var_k <= j_index - i_index; aux_var_k++){
            //Decreasing order. From j_index - 1 to i_index
            k_index = j_index - aux_var_k;

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
